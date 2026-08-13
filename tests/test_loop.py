"""End-to-end loop proof, with a scripted reasoner standing in for the model.

This is the P2 gate. It shows the loop actually closes -- an action changes the world,
the environment is re-observed, and the *measured* result comes back into context -- and
it runs with no API key, so a provider outage can never hide a regression here.
"""

from __future__ import annotations

import numpy as np
import pytest

from embodied_agent.envs.mujoco_tabletop import TabletopEnv
from embodied_agent.loop import RedundancyDetector, run_episode
from embodied_agent.memory import Memory
from embodied_agent.reasoner.base import AgentStep, ToolCall
from embodied_agent.tools.builtin import build_registry


class ScriptedReasoner:
    """Replays a fixed list of turns and records the context it was handed."""

    def __init__(self, turns: list[list[tuple[str, dict]]]) -> None:
        self.turns = turns
        self.seen_messages: list[list[dict]] = []

    def act(self, messages, tools):
        self.seen_messages.append([dict(m) for m in messages])
        index = len(self.seen_messages) - 1
        if index >= len(self.turns):
            return AgentStep("out of script", "", [])
        calls = [
            ToolCall(f"c{index}_{i}", name, args, "{}")
            for i, (name, args) in enumerate(self.turns[index])
        ]
        return AgentStep(f"turn {index}", "", calls)


@pytest.fixture
def world():
    env = TabletopEnv(image_size=(240, 320))
    memory = Memory()
    yield env, memory, build_registry(env, memory, allow_privileged=True)
    env.close()


def test_loop_closes_pick_and_place(world):
    env, memory, registry = world
    start = None

    reasoner = ScriptedReasoner(
        [
            [("list_objects", {})],
            [("move_to", {"waypoints": [[0.13, 0.18, 0.55], [0.13, 0.18, 0.445]]})],
            [("grasp", {})],
            [("move_to", {"waypoints": [[0.13, 0.18, 0.58], [-0.02, 0.30, 0.56], [-0.02, 0.30, 0.50]]})],
            [("release", {})],
            [("done", {"success": True, "reason": "cube is on the plate"})],
        ]
    )

    env.reset()
    start = env.object_pose("red_cube").copy()

    def verify(e):
        cube, plate = e.object_pose("red_cube"), e.object_pose("blue_plate")
        ok = float(np.linalg.norm(cube[:2] - plate[:2])) < 0.055 and cube[2] > 0.425
        return ok, "checked"

    summary = run_episode(
        env, reasoner, registry, memory,
        task="Put the red cube on the blue plate.",
        max_steps=8, success_check=verify, verbose=False,
    )

    assert summary["verified_success"] is True
    assert summary["agent_claimed_success"] is True
    assert np.linalg.norm(env.object_pose("red_cube")[:2] - start[:2]) > 0.15


def test_action_result_reaches_the_model_as_a_fresh_observation(world):
    """The load-bearing detail: `role:"tool"` cannot carry an image, so after a
    world-changing action the loop must inject a new user message with the new frame."""
    env, memory, registry = world
    reasoner = ScriptedReasoner(
        [[("move_to", {"waypoints": [[0.13, 0.18, 0.55]]})], [("done", {"success": False, "reason": "stop"})]]
    )

    run_episode(env, reasoner, registry, memory, task="move", max_steps=3, verbose=False)

    # The context for turn 2 must contain a user message carrying both a new image and
    # the measured outcome of the move.
    second_turn = reasoner.seen_messages[1]
    user_blocks = [m for m in second_turn if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(user_blocks) >= 2, "no new observation was posted after the action"

    latest = user_blocks[-1]
    assert any(b.get("type") == "image_url" for b in latest["content"]), "no new frame"
    assert "result of your last action" in latest["content"][0]["text"]
    assert "last_action: move_to -> OK" in latest["content"][0]["text"]


def test_failed_action_reports_the_measured_reason(world):
    env, memory, registry = world
    reasoner = ScriptedReasoner(
        [[("move_to", {"waypoints": [[0.9, 0.9, 0.9]]})], [("done", {"success": False, "reason": "stuck"})]]
    )

    run_episode(env, reasoner, registry, memory, task="move", max_steps=3, verbose=False)

    context = reasoner.seen_messages[1]
    tool_replies = [m["content"] for m in context if m["role"] == "tool"]
    assert any("unreachable" in reply for reply in tool_replies)


def test_agent_claiming_success_falsely_is_recorded_as_such(world):
    """Goal-detection failure: the model says done(success=True) having done nothing."""
    env, memory, registry = world
    reasoner = ScriptedReasoner([[("done", {"success": True, "reason": "I am confident"})]])

    def verify(e):
        cube, plate = e.object_pose("red_cube"), e.object_pose("blue_plate")
        return float(np.linalg.norm(cube[:2] - plate[:2])) < 0.055, "checked"

    summary = run_episode(
        env, reasoner, registry, memory, task="t", max_steps=3,
        success_check=verify, verbose=False,
    )

    assert summary["agent_claimed_success"] is True
    assert summary["verified_success"] is False
    assert summary["success"] is False


def test_redundant_perception_calls_are_detected():
    detector = RedundancyDetector()
    args = {"pixel_x": 100, "pixel_y": 200}

    assert detector.is_redundant("measure", args) is False
    assert detector.is_redundant("measure", args) is True, "repeat reading adds nothing"
    assert detector.is_redundant("measure", {"pixel_x": 5, "pixel_y": 5}) is False

    # Acting invalidates prior readings, so re-measuring afterwards is legitimate.
    detector.world_changed()
    assert detector.is_redundant("measure", args) is False


def test_actions_are_never_counted_as_redundant():
    detector = RedundancyDetector()
    assert detector.is_redundant("grasp", {}) is False
    assert detector.is_redundant("grasp", {}) is False
