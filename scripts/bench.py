#!/usr/bin/env python
"""Run the benchmark with the controls a harness claim needs.

    # headline number on held-out tasks
    python scripts/bench.py --split test --seeds 3 --budget 3

    # the ablation: does the verifier carry the result?
    python scripts/bench.py --split test --seeds 3 --no-verifier

Always reports pass@1 next to pass@k at the same budget. If a change moves pass@1 but not
pass@k, it bought retries rather than capability -- the confound that arXiv 2607.12227
found accounts for most published harness-evolution gains.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embodied_agent.bench import run_benchmark, save_report  # noqa: E402
from embodied_agent.reasoner.hf_chat import HFReasoner  # noqa: E402
from embodied_agent.tasks import get_split  # noqa: E402


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--seeds", type=int, default=3, help="distinct object layouts per task")
    parser.add_argument("--budget", type=int, default=1, help="attempts per problem (k)")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL"))
    parser.add_argument("--provider", default=os.environ.get("HF_PROVIDER"))
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--no-verifier", action="store_true", help="ablate the verifier")
    parser.add_argument("--privileged", action="store_true")
    parser.add_argument("--image-window", type=int, default=2)
    parser.add_argument("--out", default=None, help="where to write the report JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tasks = get_split(args.split)
    seeds = list(range(args.seeds))

    def factory():
        return HFReasoner(
            model=args.model,
            provider=args.provider,
            mode="json_schema" if args.json_mode else "tools",
        )

    probe = factory()
    print(f"model:    {probe.model} ({probe.mode})")
    print(f"split:    {args.split} -- {len(tasks)} tasks x {len(seeds)} seeds x {args.budget} attempts")
    print(f"verifier: {'off (ablation)' if args.no_verifier else 'on'}\n")

    report = run_benchmark(
        tasks,
        factory,
        seeds=seeds,
        budget=args.budget,
        use_verifier=not args.no_verifier,
        max_steps=args.steps,
        image_window=args.image_window,
        json_mode=args.json_mode,
        privileged=args.privileged,
        verbose=not args.quiet,
    )

    print("\n" + report.to_table())

    out = Path(args.out) if args.out else Path("runs") / f"bench_{args.split}.json"
    print(f"\nreport: {save_report(report, out)}")
    if args.no_verifier:
        print("Compare against the same command without --no-verifier to size the effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
