"""Reward-hacking tests.

A reward function is not "correct" in isolation -- it is correct if every degenerate
policy scores below an honest one. These tests are that ordering, written down. They are
cheap, they run without a GPU, and they are the only chance to catch a reward bug before
it becomes a training run that quietly optimises the wrong thing.
"""

from __future__ import annotations

import math

import pytest

from embodied_agent.rl.reward import (
    RewardConfig,
    diversity_weight,
    episode_reward,
    group_advantages,
)

CONFIG = RewardConfig(max_steps=15)


def summary(
    *,
    solved=False,
    claimed=None,
    steps=10,
    calls=10,
    by_name=None,
    errors=0,
    rejections=0,
    redundant=0,
):
    return {
        "success": solved,
        "verified_success": solved,
        "agent_claimed_success": claimed,
        "steps": steps,
        "tool_calls": calls,
        "tool_calls_by_name": by_name if by_name is not None else {"move_to": calls},
        "tool_errors": errors,
        "verifier_rejections": rejections,
        "redundant_tool_calls": redundant,
    }


# ------------------------------------------------------- the orderings that must hold


def test_an_honest_solve_beats_every_degenerate_policy():
    """The whole ordering in one assertion, because the ordering is the specification."""
    good = episode_reward(
        summary(
            solved=True,
            claimed=True,
            steps=7,
            calls=7,
            by_name={"commit": 2, "measure": 1, "move_to": 2, "grasp": 1, "release": 1},
        ),
        config=CONFIG,
    ).total

    liar = episode_reward(summary(solved=False, claimed=True, steps=1, calls=1), config=CONFIG).total
    quitter = episode_reward(
        summary(solved=False, claimed=False, steps=1, calls=1), config=CONFIG
    ).total
    spammer = episode_reward(
        summary(solved=False, claimed=False, steps=15, calls=15, redundant=15), config=CONFIG
    ).total
    thrasher = episode_reward(
        summary(solved=False, claimed=False, steps=15, calls=15, rejections=15), config=CONFIG
    ).total

    assert good > quitter > liar
    assert good > spammer
    assert good > thrasher


def test_success_is_read_from_the_world_not_from_the_agent():
    """The single most dangerous line in the reward: reading the claim would train lying."""
    claimed_but_false = episode_reward(
        {**summary(steps=3, calls=3), "verified_success": False, "agent_claimed_success": True},
        config=CONFIG,
    )
    really_solved = episode_reward(
        {**summary(steps=3, calls=3), "verified_success": True, "agent_claimed_success": True},
        config=CONFIG,
    )
    assert claimed_but_false.success == 0.0
    assert really_solved.success == CONFIG.success
    assert claimed_but_false.total < 0


def test_overclaiming_costs_more_than_the_efficiency_bonus_it_could_buy():
    """Otherwise a fast lie outscores a slow honest attempt, which is the exact failure
    mode this repo measures as overclaim_rate."""
    fast_lie = episode_reward(
        summary(solved=False, claimed=True, steps=1, calls=1), config=CONFIG
    ).total
    honest_timeout = episode_reward(
        summary(solved=False, claimed=False, steps=15, calls=15), config=CONFIG
    ).total
    assert fast_lie < honest_timeout


def test_trying_and_failing_beats_giving_up_immediately():
    """The failure mode that the first version of this reward actually had.

    EvoHarness-RL's R_div divides distinct verbs by trajectory length, so a one-step
    give-up scores a perfect 1.0 ratio. In ALFWorld that is unreachable -- the environment
    ends the episode, not the policy -- but here `done` is an action, and the consequence
    was that quitting on step one outscored fifteen honest steps. A policy trained on
    that converges to doing nothing.
    """
    gave_up = episode_reward(
        summary(solved=False, claimed=False, steps=1, calls=1, by_name={"done": 1}),
        config=CONFIG,
    ).total
    kept_trying = episode_reward(
        summary(
            solved=False,
            claimed=False,
            steps=15,
            calls=15,
            by_name={"measure": 5, "move_to": 5, "grasp": 3, "look": 2},
        ),
        config=CONFIG,
    ).total
    assert kept_trying > gave_up


def test_diversity_is_length_neutral():
    """Same vocabulary, different trajectory lengths -- the term must not move."""
    short = episode_reward(
        summary(calls=4, by_name={"measure": 1, "move_to": 1, "grasp": 1, "done": 1}),
        config=CONFIG,
    ).diversity
    long = episode_reward(
        summary(calls=12, by_name={"measure": 3, "move_to": 3, "grasp": 3, "done": 3}),
        config=CONFIG,
    ).diversity
    assert short == pytest.approx(long)


def test_efficiency_is_paid_only_on_success():
    """A bonus for short trajectories that pays on failure teaches the policy to quit."""
    quick_failure = episode_reward(summary(solved=False, steps=1, calls=1), config=CONFIG)
    assert quick_failure.efficiency == 0.0

    quick_success = episode_reward(summary(solved=True, steps=1, calls=1), config=CONFIG)
    slow_success = episode_reward(summary(solved=True, steps=15, calls=15), config=CONFIG)
    assert quick_success.efficiency > slow_success.efficiency
    assert quick_success.total > slow_success.total


def test_a_shorter_solve_never_scores_below_a_longer_one():
    scores = [
        episode_reward(summary(solved=True, steps=n, calls=n), config=CONFIG).total
        for n in range(1, 16)
    ]
    assert scores == sorted(scores, reverse=True)


# ----------------------------------------------------------------- the shaping terms


def test_verifier_rejections_are_priced_as_invalid_actions():
    """A rejected proposal is an impossible action caught one moment early; it still
    spent a step, so it cannot be free."""
    clean = episode_reward(summary(solved=True, steps=5, calls=5), config=CONFIG).total
    rejected = episode_reward(
        summary(solved=True, steps=5, calls=5, rejections=3), config=CONFIG
    ).total
    assert rejected < clean


def test_redundant_perception_is_penalised_and_capped():
    """Act Wisely's redundant calls are the degenerate repetition EvoHarness-RL penalises,
    but the cap stops one pathological episode dominating a whole group."""
    few = episode_reward(summary(redundant=2), config=CONFIG).spam
    many = episode_reward(summary(redundant=8), config=CONFIG).spam
    absurd = episode_reward(summary(redundant=500), config=CONFIG).spam
    assert few > many > absurd
    assert absurd == pytest.approx(-CONFIG.spam * CONFIG.spam_cap)


def test_diversity_rewards_using_the_action_vocabulary_not_looping_on_one_verb():
    looping = episode_reward(summary(calls=8, by_name={"measure": 8}), config=CONFIG).diversity
    varied = episode_reward(
        summary(calls=8, by_name={"measure": 2, "move_to": 2, "grasp": 2, "commit": 2}),
        config=CONFIG,
    ).diversity
    assert varied > looping > 0


def test_diversity_weight_anneals_to_zero_and_stays_there():
    """Its job is to prevent early collapse, then get out of the way."""
    assert diversity_weight(0, CONFIG) == pytest.approx(CONFIG.diversity_max)
    assert diversity_weight(CONFIG.annealing_horizon // 2, CONFIG) == pytest.approx(
        CONFIG.diversity_max / 2
    )
    assert diversity_weight(CONFIG.annealing_horizon, CONFIG) == pytest.approx(0.0, abs=1e-9)
    assert diversity_weight(10_000, CONFIG) == pytest.approx(0.0, abs=1e-9)


def test_diversity_does_not_survive_to_dominate_late_training():
    late = episode_reward(
        summary(calls=8, by_name={"a": 2, "b": 2, "c": 2, "d": 2}),
        epoch=CONFIG.annealing_horizon,
        config=CONFIG,
    )
    assert late.diversity == pytest.approx(0.0, abs=1e-9)


def test_every_term_is_reported_separately():
    """A total alone cannot be debugged."""
    breakdown = episode_reward(
        summary(solved=True, claimed=True, steps=5, calls=5, errors=1, redundant=1),
        config=CONFIG,
    )
    assert breakdown.total == pytest.approx(
        breakdown.success
        + breakdown.efficiency
        + breakdown.diversity
        + breakdown.spam
        + breakdown.invalid
        + breakdown.overclaim
    )
    assert "success=" in breakdown.to_line()


# ------------------------------------------------------------------ GRPO group maths


def test_advantages_are_group_relative_and_zero_centred():
    stats = group_advantages([0.0, 10.0, 5.0, 5.0])
    assert sum(stats.advantages) == pytest.approx(0.0, abs=1e-6)
    assert stats.advantages[1] > 0 > stats.advantages[0]
    assert not stats.collapsed


def test_a_group_where_every_rollout_scores_alike_produces_no_gradient():
    """The practical failure mode of GRPO: a task everybody solves, or nobody solves,
    teaches nothing that step. It has to be visible, not silently zero."""
    all_failed = group_advantages([-0.5, -0.5, -0.5, -0.5])
    assert all_failed.collapsed
    assert all_failed.advantages == [0.0, 0.0, 0.0, 0.0]

    all_solved = group_advantages([10.6, 10.6, 10.6])
    assert all_solved.collapsed


def test_normalisation_is_by_standard_deviation():
    stats = group_advantages([0.0, 2.0])
    assert stats.mean == pytest.approx(1.0)
    assert stats.std == pytest.approx(1.0)
    assert stats.advantages == [pytest.approx(-1.0), pytest.approx(1.0)]


def test_empty_group_is_collapsed_rather_than_a_crash():
    stats = group_advantages([])
    assert stats.collapsed and stats.advantages == []


def test_advantage_magnitude_is_finite_for_a_near_degenerate_group():
    stats = group_advantages([1.0, 1.0, 1.0, 1.0000001])
    assert all(math.isfinite(a) for a in stats.advantages)
