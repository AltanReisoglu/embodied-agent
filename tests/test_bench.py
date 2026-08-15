"""The benchmark exists to stop this repo making the mistake arXiv 2607.12227 documents,
so its controls are tested rather than assumed."""

from __future__ import annotations

import numpy as np
import pytest

from embodied_agent.bench import BenchReport, run_benchmark
from embodied_agent.bench import Attempt
from embodied_agent.envs.mujoco_tabletop import TabletopEnv
from embodied_agent.reasoner.base import AgentStep, ToolCall
from embodied_agent.tasks import ALL_TASKS, get_split, get_task, prepare

SOLVE_RED_ON_PLATE = [
    ("list_objects", {}),
    ("move_to", {"waypoints": [[0.13, 0.18, 0.55], [0.13, 0.18, 0.445]]}),
    ("grasp", {}),
    ("move_to", {"waypoints": [[0.13, 0.18, 0.58], [-0.02, 0.30, 0.56], [-0.02, 0.30, 0.50]]}),
    ("release", {}),
    ("move_to", {"waypoints": [[-0.22, 0.02, 0.62]]}),
    ("done", {"success": True, "reason": "placed"}),
]


class Replay:
    def __init__(self, plan):
        self.plan, self.i = plan, 0

    def act(self, messages, tools):
        self.i += 1
        if self.i > len(self.plan):
            return AgentStep("", "", [])
        name, args = self.plan[self.i - 1]
        return AgentStep(f"turn {self.i}", "", [ToolCall(f"c{self.i}", name, args, "{}")])


# ------------------------------------------------------------------------- the tasks


def test_every_task_starts_unsolved():
    """A verifier that is already satisfied at reset would report free success."""
    env = TabletopEnv(image_size=(240, 320))
    try:
        for task in ALL_TASKS:
            prepare(env, task, seed=0)
            solved, why = task.verify(env)
            assert not solved, f"{task.id} is already solved at reset: {why}"
    finally:
        env.close()


def test_layout_is_deterministic_per_seed_and_varies_across_seeds():
    """pass@k compares repeated attempts at the *same* problem, so a seed must reproduce
    its layout exactly; and tasks must vary across seeds or the split measures memory."""
    env = TabletopEnv(image_size=(240, 320))
    task = get_task("red_on_plate")
    try:
        prepare(env, task, seed=3)
        first = env.object_pose("red_cube").copy()
        prepare(env, task, seed=3)
        again = env.object_pose("red_cube").copy()
        prepare(env, task, seed=4)
        other = env.object_pose("red_cube").copy()
    finally:
        env.close()

    assert np.allclose(first, again, atol=1e-6)
    assert not np.allclose(first, other, atol=1e-3)


def test_splits_are_disjoint():
    train = {t.id for t in get_split("train")}
    test = {t.id for t in get_split("test")}
    assert train and test
    assert not (train & test), "held-out evaluation requires disjoint splits"


# --------------------------------------------------------------------------- metrics


def _attempt(task_id, split, seed, attempt, success, claimed=True):
    return Attempt(
        task_id=task_id, split=split, seed=seed, attempt=attempt, success=success,
        verdict="", agent_claimed=claimed, failure_kind=None if success else "interaction",
        steps=1, tool_calls=1, tool_errors=0, verifier_rejections=0, redundant_calls=0,
    )


def test_pass_at_k_counts_problems_not_attempts():
    """The gap between pass@1 and pass@k is exactly the advantage plain retrying buys."""
    report = BenchReport(config={"budget": 2})
    report.attempts = [
        _attempt("t1", "test", 0, 0, False),
        _attempt("t1", "test", 0, 1, True),   # same problem, second try succeeds
        _attempt("t2", "test", 0, 0, False),
        _attempt("t2", "test", 0, 1, False),
    ]
    assert report.pass_at_1("test") == 0.25
    assert report.pass_at_k("test") == 0.5


def test_overclaim_rate_isolates_goal_detection_failures():
    report = BenchReport(config={"budget": 1})
    report.attempts = [
        _attempt("t1", "test", 0, 0, success=False, claimed=True),   # overclaim
        _attempt("t2", "test", 0, 0, success=True, claimed=True),
        _attempt("t3", "test", 0, 0, success=False, claimed=False),  # honest give-up
        _attempt("t4", "test", 0, 0, success=True, claimed=True),
    ]
    assert report.overclaim_rate("test") == 0.25


def test_metrics_are_reported_per_split():
    report = BenchReport(config={"budget": 1})
    report.attempts = [
        _attempt("a", "train", 0, 0, True),
        _attempt("b", "test", 0, 0, False),
    ]
    summary = report.summary()
    assert summary["train"]["pass@1"] == 1.0
    assert summary["test"]["pass@1"] == 0.0


# ----------------------------------------------------------------------- integration


@pytest.mark.parametrize("use_verifier", [True, False])
def test_benchmark_runs_end_to_end(use_verifier):
    report = run_benchmark(
        [get_task("red_on_plate")],
        lambda: Replay(SOLVE_RED_ON_PLATE),
        seeds=[0],
        budget=1,
        use_verifier=use_verifier,
        max_steps=8,
        privileged=True,
    )
    assert len(report.attempts) == 1
    assert report.attempts[0].success, report.attempts[0].verdict


def test_benchmark_does_not_reset_away_the_task_layout():
    """run_episode resets by default; the benchmark must keep the layout it just placed,
    or every task would be run from the same default scene."""
    task = get_task("red_left")
    report = run_benchmark(
        [task], lambda: Replay([("done", {"success": False, "reason": "stop"})]),
        seeds=[5], budget=1, max_steps=3, privileged=True,
    )
    assert len(report.attempts) == 1

    env = TabletopEnv(image_size=(240, 320))
    try:
        prepare(env, task, seed=5)
        seeded = env.object_pose("red_cube").copy()
        env.reset()
        default = env.object_pose("red_cube").copy()
    finally:
        env.close()
    assert not np.allclose(seeded, default, atol=1e-3), "seed 5 must differ from the default scene"


def test_a_policy_that_only_solves_one_task_scores_below_one():
    """Guards against a verifier that accidentally passes everything."""
    report = run_benchmark(
        get_split("all"),
        lambda: Replay(SOLVE_RED_ON_PLATE),
        seeds=[0],
        budget=1,
        max_steps=8,
        privileged=True,
    )
    assert 0.0 < report.pass_at_1() < 1.0
    assert report.pass_at_1("test") == 0.0, "the red-cube plan must not satisfy held-out tasks"
