#!/usr/bin/env python
"""Collect GRPO rollout groups and report whether they could train anything.

    python scripts/collect_rollouts.py --split train --seeds 2 --group 8

Run this before renting a GPU. The number that decides whether a training run is worth
starting is `collapse_rate`: a group whose rollouts all score alike normalises to zero
advantage and contributes no gradient, so a mostly-collapsed dataset will not train no
matter how good the trainer is. The fix for that is the task mix and the sampling
temperature, and both are cheaper to find here than on rented hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embodied_agent.reasoner.hf_chat import HFReasoner  # noqa: E402
from embodied_agent.rl.reward import RewardConfig  # noqa: E402
from embodied_agent.rl.rollout import collect, dataset_report, export_jsonl  # noqa: E402
from embodied_agent.tasks import get_split  # noqa: E402


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--seeds", type=int, default=2, help="distinct layouts per task")
    parser.add_argument("--group", type=int, default=8, help="rollouts per problem (GRPO's G)")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--epoch", type=int, default=0, help="sets the annealed diversity weight")
    parser.add_argument("--model", default=os.environ.get("HF_MODEL"))
    parser.add_argument("--provider", default=os.environ.get("HF_PROVIDER"))
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--privileged", action="store_true")
    parser.add_argument("--out", default="runs/rollouts.jsonl")
    parser.add_argument("--traces", default=None, help="directory for per-rollout traces")
    args = parser.parse_args()

    tasks = get_split(args.split)
    seeds = list(range(args.seeds))
    total = len(tasks) * len(seeds) * args.group

    def factory():
        return HFReasoner(
            model=args.model,
            provider=args.provider,
            mode="json_schema" if args.json_mode else "tools",
        )

    probe = factory()
    print(f"model:  {probe.model} ({probe.mode})")
    print(f"split:  {args.split} -- {len(tasks)} tasks x {len(seeds)} seeds x G={args.group}")
    print(f"budget: {total} episodes\n")

    groups = collect(
        tasks,
        factory,
        seeds=seeds,
        group_size=args.group,
        max_steps=args.steps,
        epoch=args.epoch,
        reward_config=RewardConfig(max_steps=args.steps),
        use_verifier=not args.no_verifier,
        use_progress=not args.no_progress,
        json_mode=args.json_mode,
        privileged=args.privileged,
        trace_root=args.traces,
    )

    report = dataset_report(groups)
    print(json.dumps(report, indent=2))

    path = export_jsonl(groups, args.out)
    print(f"\nrollouts: {path}")

    if report["learnable_groups"] == 0:
        print(
            "\nEvery group collapsed -- no gradient anywhere in this dataset. Raise the "
            "sampling temperature, or pick tasks the policy solves some of the time."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
