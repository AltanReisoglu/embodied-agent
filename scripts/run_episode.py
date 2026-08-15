#!/usr/bin/env python
"""Run one episode of the embodied agent.

    python scripts/run_episode.py --task "put the red cube on the blue plate"
    python scripts/run_episode.py --json-mode --steps 20 --privileged
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embodied_agent.envs.mujoco_tabletop import TabletopEnv  # noqa: E402
from embodied_agent.loop import run_episode  # noqa: E402
from embodied_agent.memory import Memory  # noqa: E402
from embodied_agent.progress import Progress  # noqa: E402
from embodied_agent.reasoner.hf_chat import HFReasoner  # noqa: E402
from embodied_agent.tools.builtin import build_registry  # noqa: E402
from embodied_agent.trace import Trace  # noqa: E402
from embodied_agent.verifier import PreconditionVerifier  # noqa: E402

DEFAULT_TASK = "Put the red cube on the blue plate."


def cube_on_plate(env: TabletopEnv) -> tuple[bool, str]:
    """Environment-side ground truth, independent of what the agent claims.

    Keeping this separate is the point: an agent that calls done(success=True) without
    having achieved anything is a goal-detection failure, and we can only see that by
    checking the world ourselves.
    """
    cube = env.object_pose("red_cube")
    plate = env.object_pose("blue_plate")
    horizontal = float(np.linalg.norm(cube[:2] - plate[:2]))
    resting_on_plate = bool(cube[2] > 0.425)
    if horizontal < 0.055 and resting_on_plate:
        return True, f"red cube is on the plate ({horizontal * 100:.1f}cm from its centre)"
    if not resting_on_plate:
        return False, f"red cube is still at table height (z={cube[2]:.3f})"
    return False, f"red cube is {horizontal * 100:.1f}cm from the plate centre"


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL"))
    parser.add_argument("--provider", default=os.environ.get("HF_PROVIDER"))
    parser.add_argument("--json-mode", action="store_true", help="no native tool calling")
    parser.add_argument("--privileged", action="store_true", help="enable list_objects")
    parser.add_argument("--image-window", type=int, default=2)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="drop the committed-plan block (ablation of arXiv 2608.05446 Progress)",
    )
    parser.add_argument(
        "--no-verifier",
        action="store_true",
        help="disable pre-execution checks (an ablation, not a normal setting)",
    )
    parser.add_argument("--no-grid", action="store_true", help="do not overlay a pixel grid")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomize", action="store_true")
    args = parser.parse_args()

    env = TabletopEnv(seed=args.seed)
    memory = Memory()
    progress = None if args.no_progress else Progress()
    registry = build_registry(
        env, memory, progress=progress, allow_privileged=args.privileged
    )
    reasoner = HFReasoner(
        model=args.model,
        provider=args.provider,
        mode="json_schema" if args.json_mode else "tools",
        reasoning_effort=args.reasoning_effort,
    )
    trace = Trace(task=args.task, model=reasoner.model)

    print(f"model:   {reasoner.model}  (mode: {reasoner.mode})")
    print(f"task:    {args.task}")
    print(f"trace:   {trace.dir}")
    print(f"tools:   {', '.join(t.name for t in registry.available())}")
    print(f"verifier: {'off (ablation)' if args.no_verifier else 'on'}")

    try:
        summary = run_episode(
            env,
            reasoner,
            registry,
            memory,
            task=args.task,
            max_steps=args.steps,
            trace=trace,
            image_window=args.image_window,
            pixel_grid=not args.no_grid,
            json_mode=args.json_mode,
            success_check=cube_on_plate,
            verifier=None if args.no_verifier else PreconditionVerifier(env),
            progress=progress,
        )
    finally:
        env.close()

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    print(f"\nReplay: python scripts/replay_trace.py {trace.dir}")
    return 0 if summary.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
