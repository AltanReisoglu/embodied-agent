"""Pre-execution verification of proposed skill calls.

This is the highest-leverage component in HumanCLAW's harness (arXiv 2607.27180). Their
ablation on 100 episodes, removing only the verifier:

    NavSR       27.0%  ->  2.0%
    InteractSR  18.9%  ->  0.0%
    FindSR      58.0%  -> 51.0%   (barely moves)

The effect is on acting, not on seeing. Their stated motivation is that VLM spatial
reasoning degrades as the rollout grows, so the planner starts hallucinating progress --
claiming it has reached an object that is still far away -- and a short, skill-specific
check before execution catches that.

Two differences from their design, both deliberate:

* HumanCLAW's verifier is a second VLM call. Most of what goes wrong here is
  *kinematically* decidable, so those checks run as exact code: no latency, no tokens,
  no chance of the verifier hallucinating too. `LLMVerifier` remains available for the
  semantic cases.
* A rejected call is never executed, and the reason is returned to the model as the tool
  result, so the step is spent learning why rather than on a motion that cannot work.

Everything here uses only what a real robot could sense about itself -- joint limits,
reachability, gripper state, whether it is holding something. Object poses are
deliberately not consulted; that would smuggle privileged information into the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from embodied_agent.envs.base import Observation
from embodied_agent.reasoner.base import ToolCall

#: Table surface. The fingertips hang ~1.5cm below the grasp site, so the site itself
#: must stay above this by that margin plus a little clearance.
TABLE_Z = 0.40
FINGERTIP_OFFSET = 0.015
MIN_SITE_Z = TABLE_Z + FINGERTIP_OFFSET + 0.005


@dataclass(frozen=True)
class Verdict:
    accept: bool
    reason: str = ""

    @staticmethod
    def ok() -> "Verdict":
        return Verdict(True)

    @staticmethod
    def reject(reason: str) -> "Verdict":
        return Verdict(False, reason)


class Verifier(Protocol):
    def check(self, call: ToolCall, obs: Observation) -> Verdict: ...


class PreconditionVerifier:
    """Rejects skill calls whose preconditions the body already contradicts."""

    def __init__(self, env) -> None:
        self.env = env

    def check(self, call: ToolCall, obs: Observation) -> Verdict:
        handler = getattr(self, f"_check_{call.name}", None)
        if handler is None:
            return Verdict.ok()
        return handler(call, obs)

    # ------------------------------------------------------------------ movement

    def _check_move_to(self, call: ToolCall, obs: Observation) -> Verdict:
        waypoints = call.arguments.get("waypoints")
        if all(isinstance(v, (int, float)) for v in (waypoints or [])):
            waypoints = [waypoints]
        if not isinstance(waypoints, list) or not waypoints:
            return Verdict.reject("waypoints must be a non-empty list of [x, y, z] triples.")

        for index, waypoint in enumerate(waypoints, start=1):
            if not isinstance(waypoint, list) or len(waypoint) != 3:
                return Verdict.reject(
                    f"waypoint {index} is not an [x, y, z] triple: {waypoint!r}."
                )
            if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in waypoint):
                return Verdict.reject(f"waypoint {index} contains a non-numeric value.")

            if waypoint[2] < MIN_SITE_Z:
                return Verdict.reject(
                    f"waypoint {index} is at z={waypoint[2]:.3f}, which would drive the "
                    f"fingertips into the table (surface is at z={TABLE_Z:.2f}). Keep z at "
                    f"or above {MIN_SITE_Z:.3f}; z=0.445 is the height for grasping an "
                    f"object resting on the table."
                )

        # Reachability is the single most common failure, and the IK solver already knows
        # the answer. Asking it here costs nothing and turns a wasted motion into a
        # correction the model can act on.
        unreachable = self._first_unreachable(waypoints, float(call.arguments.get("yaw", 0.0)))
        if unreachable is not None:
            index, error_cm = unreachable
            return Verdict.reject(
                f"waypoint {index} of {len(waypoints)} is outside the arm's workspace -- "
                f"the closest the gripper can get is {error_cm:.1f}cm away. Choose a point "
                f"nearer the centre of the table and try again."
            )
        return Verdict.ok()

    def _first_unreachable(
        self, waypoints: list[list[float]], yaw: float
    ) -> tuple[int, float] | None:
        from embodied_agent.skills.ik import solve_ik, yaw_down_quat

        quat = yaw_down_quat(yaw)
        for index, waypoint in enumerate(waypoints, start=1):
            result = solve_ik(
                self.env.model,
                self.env.data,
                self.env._site_id,
                np.asarray(waypoint, dtype=float),
                quat,
            )
            if not result.success:
                return index, result.pos_err * 100
        return None

    # ------------------------------------------------------------------- gripper

    def _check_grasp(self, call: ToolCall, obs: Observation) -> Verdict:
        if obs.holding:
            return Verdict.reject(
                f"you are already holding {obs.holding}. Move it where it belongs and "
                f"release() before grasping anything else."
            )
        if obs.gripper_opening < 0.008:
            return Verdict.reject(
                "the gripper is already closed. release() first, position it over the "
                "object, then grasp again."
            )
        return Verdict.ok()

    # ---------------------------------------------------------------- termination

    def _check_done(self, call: ToolCall, obs: Observation) -> Verdict:
        """Guard against declaring victory from the plan rather than from the body.

        This is the goal-detection failure mode: the model issues the actions it meant to
        and calls done() without checking that the world agrees.
        """
        if not call.arguments.get("success"):
            return Verdict.ok()
        if obs.holding:
            return Verdict.reject(
                f"you are still holding {obs.holding}, so it has not been placed anywhere. "
                f"Move it to the destination and release() before declaring success."
            )
        if obs.last_action is not None and not obs.last_action.ok:
            return Verdict.reject(
                f"your last action failed ({obs.last_action.reason}), so the task is very "
                f"unlikely to be complete. Fix that first, or call done(success=false) if "
                f"you are genuinely stuck."
            )
        return Verdict.ok()


class LLMVerifier:
    """HumanCLAW's original design: a short-context second opinion from the model.

    Kept for the semantic cases the deterministic checks cannot decide -- "is this
    trajectory going to knock the green cube over on the way?" -- and deliberately given
    only the current state and the proposal, never the episode history, since the whole
    point is that long context is what degraded the judgement.
    """

    PROMPT = (
        "You are a safety check on a robot arm. Given the body state and one proposed "
        "action, decide whether executing it now would produce an unintended consequence "
        "-- colliding with an object that is not the target, dropping a held object, or "
        "acting before the gripper is positioned.\n\n"
        "Reply with exactly one line: 'ACCEPT' or 'REJECT: <one sentence of why>'."
    )

    def __init__(self, reasoner, *, state_renderer) -> None:
        self.reasoner = reasoner
        self.state_renderer = state_renderer

    def check(self, call: ToolCall, obs: Observation) -> Verdict:
        messages = [
            {"role": "system", "content": self.PROMPT},
            {
                "role": "user",
                "content": (
                    f"{self.state_renderer(obs)}\n\n"
                    f"Proposed action: {call.name}({call.arguments})"
                ),
            },
        ]
        try:
            step = self.reasoner.act(messages, [])
        except Exception:
            # A verifier that cannot answer must not block the episode.
            return Verdict.ok()

        answer = (step.text or "").strip()
        if answer.upper().startswith("REJECT"):
            return Verdict.reject(answer.split(":", 1)[-1].strip() or "rejected by verifier")
        return Verdict.ok()


class ChainVerifier:
    """Runs verifiers in order, stopping at the first rejection."""

    def __init__(self, *verifiers: Verifier) -> None:
        self.verifiers = verifiers

    def check(self, call: ToolCall, obs: Observation) -> Verdict:
        for verifier in self.verifiers:
            verdict = verifier.check(call, obs)
            if not verdict.accept:
                return verdict
        return Verdict.ok()
