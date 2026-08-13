"""The verifier was the decisive component in HumanCLAW's ablation, so it gets the
same scrutiny as the loop it protects."""

from __future__ import annotations

import numpy as np
import pytest

from embodied_agent.envs.mujoco_tabletop import TabletopEnv
from embodied_agent.loop import run_episode
from embodied_agent.memory import Memory
from embodied_agent.reasoner.base import AgentStep, ToolCall
from embodied_agent.tools.builtin import build_registry
from embodied_agent.trace import Trace
from embodied_agent.verifier import ChainVerifier, PreconditionVerifier, Verdict


@pytest.fixture(scope="module")
def env():
    e = TabletopEnv(image_size=(240, 320))
    yield e
    e.close()


@pytest.fixture
def checker(env):
    env.reset()
    return PreconditionVerifier(env), env


def call(name: str, **arguments) -> ToolCall:
    return ToolCall("id", name, arguments, "{}")


# ------------------------------------------------------------------------- move_to


def test_waypoint_below_the_table_is_rejected(checker):
    verifier, env = checker
    verdict = verifier.check(call("move_to", waypoints=[[0.0, 0.15, 0.40]]), env.observe())
    assert not verdict.accept
    assert "table" in verdict.reason


def test_unreachable_waypoint_is_caught_before_the_arm_moves(checker):
    """The IK solver already knows the answer, so asking it here turns a wasted motion
    into a correction the model can act on."""
    verifier, env = checker
    before = env.ee_pos.copy()

    verdict = verifier.check(call("move_to", waypoints=[[0.9, 0.9, 0.9]]), env.observe())

    assert not verdict.accept
    assert "workspace" in verdict.reason
    assert np.allclose(env.ee_pos, before), "verification must not disturb the simulation"


def test_reachable_waypoint_is_accepted(checker):
    verifier, env = checker
    assert verifier.check(
        call("move_to", waypoints=[[0.13, 0.18, 0.55], [0.13, 0.18, 0.445]]), env.observe()
    ).accept


def test_malformed_waypoints_are_named(checker):
    verifier, env = checker
    obs = env.observe()
    assert not verifier.check(call("move_to", waypoints=[[0.1, 0.2]]), obs).accept
    assert not verifier.check(call("move_to", waypoints="over there"), obs).accept
    assert not verifier.check(call("move_to", waypoints=[[0.1, 0.2, "high"]]), obs).accept


def test_a_bare_triple_is_tolerated(checker):
    """Models emit a flat [x, y, z] often enough that rejecting it would waste steps on
    a formatting slip rather than a real mistake."""
    verifier, env = checker
    assert verifier.check(call("move_to", waypoints=[0.13, 0.18, 0.55]), env.observe()).accept


# --------------------------------------------------------------------------- grasp


def test_grasp_while_already_holding_is_rejected(env):
    env.reset()
    verifier = PreconditionVerifier(env)
    env.move_to([[0.13, 0.18, 0.55], [0.13, 0.18, 0.445]])
    assert env.grasp().ok

    verdict = verifier.check(call("grasp"), env.observe())
    assert not verdict.accept
    assert "already holding" in verdict.reason


# ---------------------------------------------------------------------------- done


def test_claiming_success_while_still_holding_is_rejected(env):
    """The goal-detection failure mode: declaring victory from the plan, not the body."""
    env.reset()
    verifier = PreconditionVerifier(env)
    env.move_to([[0.13, 0.18, 0.55], [0.13, 0.18, 0.445]])
    env.grasp()

    verdict = verifier.check(call("done", success=True, reason="placed it"), env.observe())
    assert not verdict.accept
    assert "still holding" in verdict.reason


def test_claiming_success_after_a_failed_action_is_rejected(env):
    env.reset()
    verifier = PreconditionVerifier(env)
    env.move_to([[0.9, 0.9, 0.9]])  # fails

    verdict = verifier.check(call("done", success=True, reason="all good"), env.observe())
    assert not verdict.accept


def test_giving_up_is_always_allowed(env):
    env.reset()
    verifier = PreconditionVerifier(env)
    env.move_to([[0.9, 0.9, 0.9]])
    assert verifier.check(call("done", success=False, reason="stuck"), env.observe()).accept


# ---------------------------------------------------------------------- integration


class Scripted:
    def __init__(self, turns):
        self.turns, self.i = turns, 0

    def act(self, messages, tools):
        turn = self.turns[self.i] if self.i < len(self.turns) else []
        self.i += 1
        return AgentStep("", "", [ToolCall(f"c{self.i}_{k}", n, a, "{}") for k, (n, a) in enumerate(turn)])


def test_rejected_call_does_not_execute_and_is_explained(tmp_path):
    env = TabletopEnv(image_size=(240, 320))
    try:
        memory = Memory()
        registry = build_registry(env, memory)
        trace = Trace(root=tmp_path, task="t", model="scripted")
        reasoner = Scripted(
            [
                [("move_to", {"waypoints": [[0.9, 0.9, 0.9]]})],
                [("done", {"success": False, "reason": "stop"})],
            ]
        )

        run_episode(
            env, reasoner, registry, memory, task="t", max_steps=3, trace=trace,
            verifier=PreconditionVerifier(env), verbose=False,
        )

        metrics = trace.metrics()
        assert metrics["verifier_rejections"] == 1
        # The rejection replaces execution, so the skill never ran and never errored.
        assert metrics["tool_errors"] == 0
        assert any("rejected before execution" in r for s in trace.steps for r in s.tool_results)
    finally:
        env.close()


def test_chain_stops_at_the_first_rejection(env):
    env.reset()
    calls: list[str] = []

    class Recording:
        def __init__(self, name, accept):
            self.name, self.accept = name, accept

        def check(self, call, obs):
            calls.append(self.name)
            return Verdict.ok() if self.accept else Verdict.reject("no")

    chain = ChainVerifier(Recording("first", True), Recording("second", False), Recording("third", True))
    assert not chain.check(call("grasp"), env.observe()).accept
    assert calls == ["first", "second"], "a verifier after a rejection must not run"
