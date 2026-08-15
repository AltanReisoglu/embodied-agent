"""The perceive -> reason -> act loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from embodied_agent.envs.base import Observation
from embodied_agent.envs.mujoco_tabletop import TabletopEnv
from embodied_agent.history import History
from embodied_agent.memory import Memory
from embodied_agent.perception.render import draw_pixel_grid
from embodied_agent.progress import Progress
from embodied_agent.reasoner.base import AgentStep, Reasoner
from embodied_agent.reasoner.prompts import JSON_MODE_SUFFIX, system_prompt
from embodied_agent.state import state_block
from embodied_agent.tools.registry import Registry, ToolResult
from embodied_agent.trace import StepRecord, Trace, classify_failure
from embodied_agent.verifier import Verifier

#: Perception tools are side-effect free, so repeating one with identical arguments
#: while the world has not changed produces no new information.
PERCEPTION_TOOLS = frozenset({"measure", "look", "zoom", "list_objects"})

SuccessCheck = Callable[[TabletopEnv], tuple[bool, str]]


class RedundancyDetector:
    """Counts repeat perception calls (Act Wisely, arXiv 2604.08545)."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()

    def is_redundant(self, name: str, arguments: dict[str, Any]) -> bool:
        if name not in PERCEPTION_TOOLS:
            return False
        key = (name, json.dumps(arguments, sort_keys=True))
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    def world_changed(self) -> None:
        """A world-mutating action makes every earlier reading potentially stale."""
        self._seen.clear()


def run_episode(
    env: TabletopEnv,
    reasoner: Reasoner,
    registry: Registry,
    memory: Memory,
    *,
    task: str,
    max_steps: int = 15,
    trace: Trace | None = None,
    # HumanCLAW's ablation: 1 image scores best, 2 is comparable, and 10 collapses
    # NavSR from 27% to 13%. More visual context is actively worse, so keep this small.
    image_window: int = 2,
    pixel_grid: bool = True,
    json_mode: bool = False,
    success_check: SuccessCheck | None = None,
    verifier: Verifier | None = None,
    # EvoHarness-RL's Progress (arXiv 2608.05446). Optional so an episode can be run
    # without it, which is what makes the component ablatable the way their table is.
    progress: Progress | None = None,
    # The benchmark places objects for a specific task and seed before calling in, so it
    # must be able to keep that arrangement rather than have it reset out from under it.
    reset_env: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    prompt = system_prompt() + (JSON_MODE_SUFFIX if json_mode else "")
    history = History(prompt, image_window=image_window)
    redundancy = RedundancyDetector()

    obs = env.reset() if reset_env else env.observe()
    _post_observation(history, obs, memory, task, max_steps, pixel_grid, progress, first=True)
    if trace:
        trace.save_frame(0, obs.rgb)

    claimed_success: bool | None = None
    claimed_reason = "ran out of steps before finishing"

    for step in range(1, max_steps + 1):
        agent_step: AgentStep = reasoner.act(history.messages, registry.schemas())

        if verbose:
            _print_step(step, agent_step)

        record = StepRecord(
            step=step,
            thinking=agent_step.thinking,
            text=agent_step.text,
            tool_calls=[{"name": c.name, "arguments": c.arguments} for c in agent_step.tool_calls],
            state_block=state_block(obs, task=task, max_steps=max_steps),
            usage=agent_step.usage,
        )

        if not agent_step.tool_calls:
            record.tool_results.append("(no tool calls -- episode ended by the model)")
            if trace:
                trace.log_step(record)
            claimed_reason = agent_step.text or "model stopped without calling a tool"
            break

        _append_assistant(history, agent_step, json_mode)

        world_changed = False
        pending_images: list[tuple[str, np.ndarray]] = []
        finished = False

        for call in agent_step.tool_calls:
            if redundancy.is_redundant(call.name, call.arguments):
                if trace:
                    trace.note_redundant(call.name)

            # Verify before executing. A rejected proposal never runs; the reason goes
            # back as the tool result so the step is spent correcting rather than on a
            # motion that could not have worked.
            if verifier is not None:
                verdict = verifier.check(call, obs)
                if not verdict.accept:
                    text = f"rejected before execution: {verdict.reason}"
                    record.tool_results.append(text)
                    _append_tool_result(history, call.id, text, json_mode)
                    if trace:
                        trace.note_verifier_rejection(call.name)
                    if verbose:
                        print(f"    -x {text[:150]}")
                    continue

            result: ToolResult = registry.dispatch(call)
            record.tool_results.append(result.text)
            _append_tool_result(history, call.id, result.text, json_mode)
            if result.is_error and trace:
                trace.note_tool_error(call.name)

            if verbose:
                print(f"    -> {result.text.splitlines()[0][:150]}")

            if result.image is not None:
                pending_images.append((result.image_caption or call.name, result.image))
            if result.mutates_world:
                world_changed = True
                redundancy.world_changed()
                # The environment adapter's job: sub-goal status comes from the measured
                # outcome of a skill, never from the model's own account of it.
                if progress is not None:
                    progress.note_action(not result.is_error, result.text.splitlines()[0])
            if call.name == "done" and not result.is_error:
                claimed_success = bool(call.arguments.get("success", False))
                claimed_reason = str(call.arguments.get("reason", ""))
                finished = True

        if world_changed:
            obs = env.observe()
            if trace:
                record.frame = trace.save_frame(step, obs.rgb)
            _post_observation(history, obs, memory, task, max_steps, pixel_grid, progress)
        for caption, image in pending_images:
            history.add_user(f"Requested view: {caption}", _maybe_grid(image, pixel_grid))

        if trace:
            trace.log_step(record)
        if finished:
            break

    verified, verdict = (None, "no verifier configured")
    if success_check is not None:
        raw_verified, verdict = success_check(env)
        # Verifiers built from numpy comparisons hand back np.bool_, which is not JSON
        # serialisable and would blow up when the trace summary is written.
        verified = bool(raw_verified)

    success = verified if verified is not None else bool(claimed_success)
    failure_kind = None if success else classify_failure(trace.steps if trace else [])

    summary: dict[str, Any] = {
        "agent_claimed_success": claimed_success,
        "agent_reason": claimed_reason,
        "verified_success": verified,
        "verifier_verdict": verdict,
        # EvoHarness-RL's harness annealing: after training their agent settles at about
        # one harness call per episode. We cannot train, but we can measure the rate.
        "progress_commits": progress.commits if progress is not None else 0,
    }
    if trace:
        summary = trace.finish(
            success=success, reason=claimed_reason, failure_kind=failure_kind, extra=summary
        )
    else:
        summary.update({"success": success, "failure_kind": failure_kind})
    return summary


# ----------------------------------------------------------------------- helpers


def _maybe_grid(image: np.ndarray, pixel_grid: bool) -> np.ndarray:
    return draw_pixel_grid(image) if pixel_grid else image


def _post_observation(
    history: History,
    obs: Observation,
    memory: Memory,
    task: str,
    max_steps: int,
    pixel_grid: bool,
    progress: Progress | None = None,
    *,
    first: bool = False,
) -> None:
    """Deliver one observation: body state, plan, memory, then the image.

    The three text blocks are EvoHarness-RL's H_t = (Belief, Progress, Experience)
    rendered together, which is how their harness presents external state to the policy.

    A `role:"tool"` message cannot carry an image on this API, so the frame that results
    from an action must arrive as a user message. Skipping it is what leaves an agent
    acting blind -- it issues commands and never sees their effect.
    """
    header = f"TASK: {task}\n\n" if first else ""
    text = header + state_block(obs, task=None if first else task, max_steps=max_steps)
    if progress is not None:
        text += "\n\n" + progress.render()
    text += "\n\n" + memory.render()
    if not first:
        text += "\n\nThis image is the result of your last action."
    history.add_user(text, _maybe_grid(obs.rgb, pixel_grid))


def _append_assistant(history: History, step: AgentStep, json_mode: bool) -> None:
    if json_mode:
        payload = {
            "thought": step.thinking,
            "actions": [{"tool": c.name, "arguments": c.arguments} for c in step.tool_calls],
        }
        history.add_assistant_text(json.dumps(payload))
    else:
        history.add_assistant(step)


def _append_tool_result(history: History, call_id: str, text: str, json_mode: bool) -> None:
    if json_mode:
        # There is no tool role to reply to when we never sent `tools`.
        history.add_user(f"Result of your action:\n{text}")
    else:
        history.add_tool_result(call_id, text)


def _print_step(step: int, agent_step: AgentStep) -> None:
    print(f"\n[step {step}]")
    if agent_step.thinking:
        thought = agent_step.thinking.strip().replace("\n", " ")
        print(f"  think: {thought[:400]}{'...' if len(thought) > 400 else ''}")
    if agent_step.text:
        print(f"  say:   {agent_step.text.strip()[:300]}")
    for call in agent_step.tool_calls:
        print(f"  call:  {call.name}({json.dumps(call.arguments)[:200]})")
