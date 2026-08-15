"""The trajectory reward, and the GRPO group statistics built on it.

This is the part of an RL setup that is specific to this problem, and the part you cannot
debug on rented GPUs. The trainer is commodity -- GRPO is implemented in slime, prime-rl,
verl and TRL alike -- but the reward is where a policy learns the wrong lesson, and it is
pure CPU code that a scripted policy can exercise in milliseconds.

The shape follows EvoHarness-RL (arXiv 2608.05446):

    R(t) = R_succ + l_eff * R_eff + l_div(u) * R_div - l_spam * R_spam - l_inv * R_inv

with task success as the dominant sparse signal and the rest as dense shaping. Two of
their choices are load-bearing and are kept exactly:

* **The efficiency bonus is only paid on success.** Otherwise the shortest trajectory is
  an immediate give-up, and the policy learns to quit.
* **The diversity bonus is cosine-annealed to zero.** Its job is to stop the policy
  collapsing early -- either ignoring the harness actions entirely or looping on one verb
  -- and then to get out of the way so the policy can specialise.

Two departures, both consequences of the same difference in environment. In ALFWorld the
environment decides when an episode ends; here `done` is an action the *policy* chooses,
which makes trivially short trajectories reachable and opens two holes their formula
never has to cover.

* **`overclaim`, a term they do not have.** A policy trained only on verified success
  learns that calling `done(success=true)` early is free: it ends the episode, dodges the
  spam penalty, and costs nothing. This is the goal-detection failure mode the repo
  exists to measure, so it is priced rather than met with indifference.
* **Diversity is normalised by the available vocabulary, not by trajectory length.**
  Their `R_div = |{verb(a)}| / |t|` rewards breadth, but dividing by length also rewards
  brevity: a one-step give-up trivially scores 1.0, which made quitting immediately worth
  more than trying for fifteen steps and failing -- a policy trained on that converges to
  doing nothing. Dividing by the number of tools on offer measures the breadth the term
  was meant to measure, and is length-neutral.

The single most important line in this file is that `R_succ` reads `verified_success`.
Reading `agent_claimed_success` instead would train a policy to lie, and would look fine
in every aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: EvoHarness-RL's published coefficients, kept as the defaults so any change here is a
#: deliberate departure from a configuration that is known to have trained.
DEFAULT_SUCCESS = 10.0
DEFAULT_EFFICIENCY = 1.0
DEFAULT_DIVERSITY_MAX = 0.5
DEFAULT_SPAM = 0.1
DEFAULT_INVALID = 0.1
DEFAULT_SPAM_CAP = 10


@dataclass(frozen=True)
class RewardConfig:
    success: float = DEFAULT_SUCCESS
    efficiency: float = DEFAULT_EFFICIENCY
    diversity_max: float = DEFAULT_DIVERSITY_MAX
    spam: float = DEFAULT_SPAM
    invalid: float = DEFAULT_INVALID
    #: Not in their reward; see the module docstring. Priced above the efficiency bonus
    #: so that quitting with a false claim can never beat honestly running out of steps.
    overclaim: float = 2.0
    spam_cap: int = DEFAULT_SPAM_CAP
    #: T_max in R_eff. Must match the episode budget or the bonus is miscalibrated.
    max_steps: int = 15
    #: U, the annealing horizon for the diversity weight, in epochs.
    annealing_horizon: int = 150
    #: Denominator for R_div when the episode summary does not report how many tools were
    #: on offer. The default registry exposes measure/look/zoom/move_to/grasp/release/
    #: done/remember/commit.
    default_vocabulary: int = 9


@dataclass(frozen=True)
class RewardBreakdown:
    """Every term kept separately, because a total alone cannot be debugged."""

    total: float
    success: float = 0.0
    efficiency: float = 0.0
    diversity: float = 0.0
    spam: float = 0.0
    invalid: float = 0.0
    overclaim: float = 0.0

    def to_line(self) -> str:
        parts = [f"total={self.total:+.3f}"]
        for name in ("success", "efficiency", "diversity", "spam", "invalid", "overclaim"):
            value = getattr(self, name)
            if value:
                parts.append(f"{name}={value:+.3f}")
        return "  ".join(parts)

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "success": self.success,
            "efficiency": self.efficiency,
            "diversity": self.diversity,
            "spam": self.spam,
            "invalid": self.invalid,
            "overclaim": self.overclaim,
        }


def diversity_weight(epoch: int, config: RewardConfig | None = None) -> float:
    """Cosine-annealed l_div(u) = (l_max / 2) * (1 + cos(pi * u / U)).

    Full weight at epoch 0, zero at the horizon and beyond.
    """
    config = config or RewardConfig()
    if config.annealing_horizon <= 0:
        return 0.0
    u = min(max(epoch, 0), config.annealing_horizon)
    return (config.diversity_max / 2.0) * (1.0 + math.cos(math.pi * u / config.annealing_horizon))


def episode_reward(
    summary: dict[str, Any],
    *,
    epoch: int = 0,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score one finished episode from the summary `run_episode` returns.

    `summary` is expected to carry the fields `Trace.finish` writes: `verified_success`,
    `agent_claimed_success`, `steps`, `tool_calls`, `tool_calls_by_name`, `tool_errors`,
    `verifier_rejections` and `redundant_tool_calls`.
    """
    config = config or RewardConfig()

    # Ground truth only. See the module docstring.
    verified = summary.get("verified_success")
    solved = bool(summary.get("success", False)) if verified is None else bool(verified)

    steps = int(summary.get("steps", 0) or 0)
    calls = int(summary.get("tool_calls", 0) or 0)
    by_name = summary.get("tool_calls_by_name") or {}

    success = config.success if solved else 0.0

    # Paid only on success, so that quitting immediately is never the cheapest win.
    efficiency = 0.0
    if solved and config.max_steps > 0:
        efficiency = config.efficiency * max(0.0, 1.0 - steps / config.max_steps)

    # How much of the action vocabulary the trajectory used. Normalised by the tools on
    # offer rather than by trajectory length, so that a one-step give-up cannot score a
    # perfect ratio -- see the module docstring.
    diversity = 0.0
    if calls > 0:
        vocabulary = int(summary.get("available_tools") or config.default_vocabulary)
        used = min(len(by_name), vocabulary)
        diversity = diversity_weight(epoch, config) * (used / max(vocabulary, 1))

    # Degenerate repetition. Act Wisely's redundant perception calls are exactly that,
    # and they are already counted for us.
    redundant = int(summary.get("redundant_tool_calls", 0) or 0)
    spam = -config.spam * min(redundant, config.spam_cap)

    # Malformed or impossible actions. A verifier rejection is an invalid action caught
    # one moment before it would have wasted a motion; it belongs in the same term.
    invalid_count = int(summary.get("tool_errors", 0) or 0) + int(
        summary.get("verifier_rejections", 0) or 0
    )
    invalid = -config.invalid * invalid_count

    claimed = summary.get("agent_claimed_success")
    overclaim = -config.overclaim if (bool(claimed) and not solved) else 0.0

    total = success + efficiency + diversity + spam + invalid + overclaim
    return RewardBreakdown(
        total=round(total, 6),
        success=round(success, 6),
        efficiency=round(efficiency, 6),
        diversity=round(diversity, 6),
        spam=round(spam, 6),
        invalid=round(invalid, 6),
        overclaim=round(overclaim, 6),
    )


# --------------------------------------------------------------------------- GRPO


@dataclass
class GroupStats:
    """Group-relative advantages, the whole of what makes GRPO group-relative.

    GRPO drops the value network and normalises reward within a group of rollouts drawn
    from the same prompt. The consequence worth watching is that a group whose rollouts
    all score alike produces zero advantage and therefore no gradient: when every attempt
    fails, or every attempt succeeds, that prompt teaches nothing that step. Tracking the
    collapse rate is how you find out that a task set is too hard or too easy *before*
    spending a training run on it.
    """

    rewards: list[float]
    advantages: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    collapsed: bool = False


def group_advantages(rewards: list[float], *, eps: float = 1e-6) -> GroupStats:
    if not rewards:
        return GroupStats(rewards=[], advantages=[], collapsed=True)

    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(variance)

    if std < eps:
        # No spread: every rollout is equally good, so there is nothing to prefer.
        return GroupStats(
            rewards=list(rewards),
            advantages=[0.0] * n,
            mean=round(mean, 6),
            std=0.0,
            collapsed=True,
        )

    return GroupStats(
        rewards=list(rewards),
        advantages=[round((r - mean) / std, 6) for r in rewards],
        mean=round(mean, 6),
        std=round(std, 6),
        collapsed=False,
    )
