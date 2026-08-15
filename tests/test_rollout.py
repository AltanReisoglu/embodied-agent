"""Rollout groups against the real simulator, with scripted policies standing in for a
model. No API key and no GPU: the data pipeline is verified end to end on CPU."""

from __future__ import annotations

import json

from embodied_agent.envs.mujoco_tabletop import TabletopEnv
from embodied_agent.reasoner.base import AgentStep, ToolCall
from embodied_agent.rl.rollout import collect_group, dataset_report, export_jsonl
from embodied_agent.tasks import get_task

SOLVE = [
    ("commit", {"subgoal": "pick up the red cube"}),
    ("list_objects", {}),
    ("move_to", {"waypoints": [[0.13, 0.18, 0.55], [0.13, 0.18, 0.445]]}),
    ("grasp", {}),
    ("commit", {"subgoal": "place it on the plate"}),
    ("move_to", {"waypoints": [[0.13, 0.18, 0.58], [-0.02, 0.30, 0.56], [-0.02, 0.30, 0.50]]}),
    ("release", {}),
    ("move_to", {"waypoints": [[-0.22, 0.02, 0.62]]}),
    ("done", {"success": True, "reason": "placed"}),
]

LIE = [("done", {"success": True, "reason": "I believe it is done"})]


class Replay:
    def __init__(self, plan):
        self.plan, self.i = plan, 0

    def act(self, messages, tools):
        self.i += 1
        if self.i > len(self.plan):
            return AgentStep("", "", [])
        name, args = self.plan[self.i - 1]
        return AgentStep(f"turn {self.i}", "", [ToolCall(f"c{self.i}", name, args, "{}")])


def _group(plan, size=3, **kwargs):
    env = TabletopEnv(image_size=(240, 320))
    try:
        return collect_group(
            env,
            get_task("red_on_plate"),
            seed=0,
            reasoner_factory=lambda: Replay(plan),
            group_size=size,
            max_steps=12,
            privileged=True,
            **kwargs,
        )
    finally:
        env.close()


def test_a_solving_policy_earns_the_success_term():
    group = _group(SOLVE)
    assert all(r.solved for r in group.rollouts)
    assert all(r.reward_terms["success"] > 0 for r in group.rollouts)


def test_a_lying_policy_is_scored_below_zero():
    """The reward has to disagree with the agent, and this is where that is proved
    against the real environment rather than a hand-written summary."""
    group = _group(LIE)
    assert not any(r.solved for r in group.rollouts)
    assert all(r.claimed for r in group.rollouts)
    assert all(r.reward < 0 for r in group.rollouts)
    assert all(r.reward_terms["overclaim"] < 0 for r in group.rollouts)


def test_a_solving_policy_outscores_a_lying_one_end_to_end():
    solving = _group(SOLVE)
    lying = _group(LIE)
    assert min(r.reward for r in solving.rollouts) > max(r.reward for r in lying.rollouts)


def test_a_deterministic_policy_collapses_the_group():
    """Every rollout identical means zero advantage -- no gradient. This is the signal
    that a real policy needs sampling temperature, not a bug to be hidden."""
    group = _group(SOLVE)
    assert group.collapsed
    assert all(r.advantage == 0.0 for r in group.rollouts)


def test_the_dataset_report_separates_learnable_groups_from_collapsed_ones():
    groups = [_group(SOLVE, size=2), _group(LIE, size=2)]
    report = dataset_report(groups)

    assert report["groups"] == 2
    assert report["rollouts"] == 4
    assert report["collapse_rate"] == 1.0
    assert report["learnable_groups"] == 0
    assert report["collapsed_all_solved"] == 1
    assert report["collapsed_none_solved"] == 1
    assert report["overclaim_rate"] == 0.5


def test_export_writes_one_json_line_per_rollout_with_its_advantage(tmp_path):
    group = _group(SOLVE, size=2)
    path = export_jsonl([group], tmp_path / "rollouts.jsonl")

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert {"task_id", "seed", "reward", "advantage", "reward_terms"} <= set(rows[0])
    assert rows[0]["task_id"] == "red_on_plate"


def test_progress_can_be_ablated_inside_a_rollout():
    """The RL path must keep every component ablatable, or an ablation cannot be trained
    against its own baseline."""
    group = _group(SOLVE, size=1, use_progress=False)
    assert len(group.rollouts) == 1
