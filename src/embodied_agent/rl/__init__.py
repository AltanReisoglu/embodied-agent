"""The RL side: reward, group statistics, and rollout collection.

Deliberately trainer-agnostic. GRPO itself is commodity -- slime, prime-rl, verl and TRL
all implement it -- so what lives here is only the part this environment defines: what a
trajectory is worth, and how to draw groups of them.
"""

from embodied_agent.rl.reward import (
    GroupStats,
    RewardBreakdown,
    RewardConfig,
    diversity_weight,
    episode_reward,
    group_advantages,
)
from embodied_agent.rl.rollout import (
    Rollout,
    RolloutGroup,
    collect,
    collect_group,
    dataset_report,
    export_jsonl,
)

__all__ = [
    "GroupStats",
    "RewardBreakdown",
    "RewardConfig",
    "Rollout",
    "RolloutGroup",
    "collect",
    "collect_group",
    "dataset_report",
    "diversity_weight",
    "episode_reward",
    "export_jsonl",
    "group_advantages",
]
