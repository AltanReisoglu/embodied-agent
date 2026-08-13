"""The metrics are the reason this is a system and not a demo, so they get tests too."""

from __future__ import annotations

import json

import pytest

from embodied_agent.envs.mujoco_tabletop import TabletopEnv
from embodied_agent.loop import run_episode
from embodied_agent.memory import Memory
from embodied_agent.reasoner.base import AgentStep, ToolCall
from embodied_agent.tools.builtin import build_registry
from embodied_agent.trace import StepRecord, Trace, classify_failure


class Scripted:
    def __init__(self, turns):
        self.turns, self.i = turns, 0

    def act(self, messages, tools):
        turn = self.turns[self.i] if self.i < len(self.turns) else []
        self.i += 1
        return AgentStep("", "", [ToolCall(f"c{self.i}_{k}", n, a, "{}") for k, (n, a) in enumerate(turn)])


@pytest.fixture
def world(tmp_path):
    env = TabletopEnv(image_size=(240, 320))
    memory = Memory()
    yield env, memory, build_registry(env, memory), tmp_path
    env.close()


def test_tool_errors_are_counted_from_the_flag_not_the_text(world):
    """A tool that explains a problem in prose is still a failed call; counting by
    string prefix silently reported zero errors."""
    env, memory, registry, tmp = world
    trace = Trace(root=tmp, task="t", model="scripted")

    reasoner = Scripted(
        [
            [("measure", {"pixel_x": 4, "pixel_y": 4})],  # lands on the floor -> error
            [("done", {"success": False, "reason": "stop"})],
        ]
    )
    run_episode(env, reasoner, registry, memory, task="t", max_steps=3, trace=trace, verbose=False)

    metrics = trace.metrics()
    assert metrics["tool_errors"] == 1
    assert metrics["tool_errors_by_name"] == {"measure": 1}


def test_summary_is_json_serialisable(world):
    env, memory, registry, tmp = world
    trace = Trace(root=tmp, task="t", model="scripted")
    reasoner = Scripted([[("done", {"success": True, "reason": "done"})]])

    summary = run_episode(
        env, reasoner, registry, memory, task="t", max_steps=2, trace=trace,
        success_check=lambda e: (e.object_pose("red_cube")[2] > 0.4, "check"),
        verbose=False,
    )

    json.dumps(summary)  # numpy bools would raise here
    assert (trace.dir / "summary.json").exists()
    assert json.loads((trace.dir / "summary.json").read_text())["task"] == "t"


@pytest.mark.parametrize(
    "result_text, expected",
    [
        ("grasp -> FAILED (gripper closed on nothing -- no object between the fingers)", "interaction"),
        ("move_to -> FAILED (IK unreachable at waypoint 1/2)", "localisation"),
        ("Pixel (4, 4) lands on the floor beside the table", "localisation"),
        ("move_to -> FAILED (stopped 6.0cm short of the last waypoint (likely blocked))", "body_awareness"),
        ("move_to -> OK", "goal_detection"),
    ],
)
def test_failure_taxonomy_buckets(result_text, expected):
    """HumanCLAW's categories, so the body-state block can be judged by whether it moves
    the body_awareness bucket rather than by overall success alone."""
    steps = [StepRecord(step=1, thinking="", text="", tool_results=[result_text])]
    assert classify_failure(steps) == expected
