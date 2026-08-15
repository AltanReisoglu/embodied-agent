"""Collecting GRPO rollout groups from the simulator.

GRPO needs several rollouts of the *same* problem so that reward can be normalised within
the group. `tasks.prepare` is already deterministic per (task, seed), which is what makes
a group well defined here: every member faces an identical scene, and the only variation
is the policy's own sampling.

Nothing in this module needs a GPU. The point is to have the data pipeline, the reward
and its diagnostics correct and tested before any trainer is attached, because a reward
bug is invisible in aggregate metrics and expensive to find on rented hardware.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from embodied_agent.envs.mujoco_tabletop import TabletopEnv
from embodied_agent.loop import run_episode
from embodied_agent.memory import Memory
from embodied_agent.progress import Progress
from embodied_agent.reasoner.base import Reasoner
from embodied_agent.rl.reward import GroupStats, RewardConfig, episode_reward, group_advantages
from embodied_agent.tasks import Task, prepare
from embodied_agent.tools.builtin import build_registry
from embodied_agent.trace import Trace
from embodied_agent.verifier import PreconditionVerifier

ReasonerFactory = Callable[[], Reasoner]


@dataclass
class Rollout:
    task_id: str
    split: str
    seed: int
    index: int
    solved: bool
    claimed: bool | None
    reward: float
    reward_terms: dict[str, float]
    advantage: float = 0.0
    steps: int = 0
    tool_calls: int = 0
    trace_dir: str | None = None


@dataclass
class RolloutGroup:
    """One problem, sampled `group_size` times."""

    task_id: str
    split: str
    seed: int
    rollouts: list[Rollout] = field(default_factory=list)
    mean_reward: float = 0.0
    std_reward: float = 0.0
    #: True when every rollout scored alike, so this group contributes no gradient.
    collapsed: bool = True

    def apply(self, stats: GroupStats) -> None:
        for rollout, advantage in zip(self.rollouts, stats.advantages, strict=True):
            rollout.advantage = advantage
        self.mean_reward = stats.mean
        self.std_reward = stats.std
        self.collapsed = stats.collapsed


def collect_group(
    env: TabletopEnv,
    task: Task,
    seed: int,
    reasoner_factory: ReasonerFactory,
    *,
    group_size: int = 8,
    max_steps: int = 15,
    epoch: int = 0,
    reward_config: RewardConfig | None = None,
    use_verifier: bool = True,
    use_progress: bool = True,
    image_window: int = 2,
    json_mode: bool = False,
    privileged: bool = False,
    trace_root: Path | str | None = None,
    verbose: bool = False,
) -> RolloutGroup:
    config = reward_config or RewardConfig(max_steps=max_steps)
    group = RolloutGroup(task_id=task.id, split=task.split, seed=seed)

    for index in range(group_size):
        memory = Memory()
        progress = Progress() if use_progress else None
        registry = build_registry(env, memory, progress=progress, allow_privileged=privileged)
        trace = (
            Trace(root=trace_root, task=task.instruction, model=f"{task.id}/{seed}/{index}")
            if trace_root
            else None
        )

        prepare(env, task, seed)
        summary = run_episode(
            env,
            reasoner_factory(),
            registry,
            memory,
            task=task.instruction,
            max_steps=max_steps,
            trace=trace,
            image_window=image_window,
            json_mode=json_mode,
            success_check=task.verify,
            verifier=PreconditionVerifier(env) if use_verifier else None,
            progress=progress,
            verbose=verbose,
            reset_env=False,
        )

        breakdown = episode_reward(summary, epoch=epoch, config=config)
        group.rollouts.append(
            Rollout(
                task_id=task.id,
                split=task.split,
                seed=seed,
                index=index,
                solved=bool(summary.get("success")),
                claimed=summary.get("agent_claimed_success"),
                reward=breakdown.total,
                reward_terms=breakdown.as_dict(),
                steps=int(summary.get("steps", 0)),
                tool_calls=int(summary.get("tool_calls", 0)),
                trace_dir=str(trace.dir) if trace else None,
            )
        )

    group.apply(group_advantages([r.reward for r in group.rollouts]))
    return group


def collect(
    tasks: list[Task],
    reasoner_factory: ReasonerFactory,
    *,
    seeds: list[int],
    group_size: int = 8,
    **kwargs: Any,
) -> list[RolloutGroup]:
    env = TabletopEnv()
    groups: list[RolloutGroup] = []
    try:
        for task in tasks:
            for seed in seeds:
                groups.append(
                    collect_group(env, task, seed, reasoner_factory, group_size=group_size, **kwargs)
                )
    finally:
        env.close()
    return groups


# ----------------------------------------------------------------------- diagnostics


def dataset_report(groups: list[RolloutGroup]) -> dict[str, Any]:
    """What you need to know before paying for a training run.

    The number to look at is `collapse_rate`. A collapsed group -- every rollout solving,
    or every rollout failing -- normalises to zero advantage and contributes no gradient
    at all. A dataset that is mostly collapsed will train slowly or not at all no matter
    how good the trainer is, and the fix is the task mix, not the hyperparameters.
    """
    rollouts = [r for g in groups for r in g.rollouts]
    if not rollouts:
        return {"groups": 0, "rollouts": 0}

    collapsed = [g for g in groups if g.collapsed]
    all_solved = [g for g in collapsed if all(r.solved for r in g.rollouts)]
    none_solved = [g for g in collapsed if not any(r.solved for r in g.rollouts)]
    overclaims = [r for r in rollouts if r.claimed and not r.solved]

    return {
        "groups": len(groups),
        "rollouts": len(rollouts),
        "solve_rate": round(sum(r.solved for r in rollouts) / len(rollouts), 4),
        "mean_reward": round(sum(r.reward for r in rollouts) / len(rollouts), 4),
        "collapse_rate": round(len(collapsed) / len(groups), 4),
        "collapsed_all_solved": len(all_solved),
        "collapsed_none_solved": len(none_solved),
        "learnable_groups": len(groups) - len(collapsed),
        "overclaim_rate": round(len(overclaims) / len(rollouts), 4),
        "mean_steps": round(sum(r.steps for r in rollouts) / len(rollouts), 2),
    }


def export_jsonl(groups: list[RolloutGroup], path: Path | str) -> Path:
    """One line per rollout, with its group-relative advantage already attached.

    Frames and the message history stay in the trace directory rather than being inlined:
    a base64 image per step would make this file unusable at any real group size.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for group in groups:
            for rollout in group.rollouts:
                fh.write(json.dumps(asdict(rollout)) + "\n")
    return path
