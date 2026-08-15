"""Benchmark runner with the controls that make a harness claim believable.

arXiv 2607.12227 showed that harness-evolution results collapse under two controls, and
that most published work applies neither:

1. **A matched-budget baseline.** Any harness change costs inference. Spending the same
   budget on plain repeated sampling was, on their benchmark, strictly better -- so a
   harness claim is only meaningful next to `pass@k` from independent retries.
2. **A held-out split.** Tuning and reporting on the same tasks turned a large apparent
   gain into +0.6 points on disjoint tasks.

So this runner always reports pass@1 alongside pass@k at the same budget, and always
labels which split a number came from. `--no-verifier` reproduces their ablation shape on
a component this repo actually has.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from embodied_agent.envs.mujoco_tabletop import TabletopEnv
from embodied_agent.loop import run_episode
from embodied_agent.memory import Memory
from embodied_agent.reasoner.base import Reasoner
from embodied_agent.tasks import Task, prepare
from embodied_agent.tools.builtin import build_registry
from embodied_agent.trace import Trace
from embodied_agent.verifier import PreconditionVerifier

ReasonerFactory = Callable[[], Reasoner]


@dataclass
class Attempt:
    task_id: str
    split: str
    seed: int
    attempt: int
    success: bool
    verdict: str
    agent_claimed: bool | None
    failure_kind: str | None
    steps: int
    tool_calls: int
    tool_errors: int
    verifier_rejections: int
    redundant_calls: int


@dataclass
class BenchReport:
    config: dict[str, Any]
    attempts: list[Attempt] = field(default_factory=list)

    # ------------------------------------------------------------------- metrics

    def pass_at_1(self, split: str | None = None) -> float:
        rows = self._rows(split)
        return round(sum(a.success for a in rows) / len(rows), 4) if rows else 0.0

    def pass_at_k(self, split: str | None = None) -> float:
        """Fraction of (task, seed) problems solved by at least one attempt.

        The gap to pass@1 is the part of any harness's apparent advantage that plain
        repeated sampling would have bought anyway.
        """
        rows = self._rows(split)
        if not rows:
            return 0.0
        problems: dict[tuple[str, int], bool] = {}
        for attempt in rows:
            key = (attempt.task_id, attempt.seed)
            problems[key] = problems.get(key, False) or attempt.success
        return round(sum(problems.values()) / len(problems), 4)

    def overclaim_rate(self, split: str | None = None) -> float:
        """How often the agent declared success the world does not support."""
        rows = [a for a in self._rows(split) if a.agent_claimed is not None]
        if not rows:
            return 0.0
        return round(sum(1 for a in rows if a.agent_claimed and not a.success) / len(rows), 4)

    def failure_kinds(self, split: str | None = None) -> dict[str, int]:
        return dict(Counter(a.failure_kind for a in self._rows(split) if not a.success))

    def _rows(self, split: str | None) -> list[Attempt]:
        return [a for a in self.attempts if split is None or a.split == split]

    def summary(self) -> dict[str, Any]:
        splits = sorted({a.split for a in self.attempts})
        out: dict[str, Any] = {"config": self.config, "n_attempts": len(self.attempts)}
        for split in splits:
            rows = self._rows(split)
            out[split] = {
                "pass@1": self.pass_at_1(split),
                f"pass@{self.config.get('budget', 1)}": self.pass_at_k(split),
                "overclaim_rate": self.overclaim_rate(split),
                "failure_kinds": self.failure_kinds(split),
                "verifier_rejections": sum(a.verifier_rejections for a in rows),
                "redundant_calls": sum(a.redundant_calls for a in rows),
                "tool_errors": sum(a.tool_errors for a in rows),
            }
        return out

    def to_table(self) -> str:
        summary = self.summary()
        budget = self.config.get("budget", 1)
        lines = [
            f"{'split':<8}{'pass@1':>9}{f'pass@{budget}':>9}{'overclaim':>11}{'rejects':>9}{'redundant':>11}",
            "-" * 57,
        ]
        for split in [s for s in summary if s not in ("config", "n_attempts")]:
            row = summary[split]
            lines.append(
                f"{split:<8}{row['pass@1']:>9.3f}{row[f'pass@{budget}']:>9.3f}"
                f"{row['overclaim_rate']:>11.3f}{row['verifier_rejections']:>9}"
                f"{row['redundant_calls']:>11}"
            )
        kinds = summary.get(list(summary)[-1], {}).get("failure_kinds", {})
        if kinds:
            lines.append("")
            lines.append("failures: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
        return "\n".join(lines)


def run_benchmark(
    tasks: list[Task],
    reasoner_factory: ReasonerFactory,
    *,
    seeds: list[int],
    budget: int = 1,
    use_verifier: bool = True,
    max_steps: int = 15,
    image_window: int = 2,
    json_mode: bool = False,
    privileged: bool = False,
    trace_root: Path | str | None = None,
    verbose: bool = False,
) -> BenchReport:
    config = {
        "tasks": [t.id for t in tasks],
        "seeds": seeds,
        "budget": budget,
        "use_verifier": use_verifier,
        "max_steps": max_steps,
        "image_window": image_window,
        "privileged": privileged,
    }
    report = BenchReport(config=config)
    env = TabletopEnv()

    try:
        for task in tasks:
            for seed in seeds:
                for attempt_index in range(budget):
                    memory = Memory()
                    registry = build_registry(env, memory, allow_privileged=privileged)
                    trace = (
                        Trace(root=trace_root, task=task.instruction, model=str(config))
                        if trace_root
                        else None
                    )

                    # Deterministic per (task, seed): every attempt at a problem faces
                    # the same problem, which is what makes pass@k meaningful.
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
                        verbose=verbose,
                        reset_env=False,
                    )

                    report.attempts.append(
                        Attempt(
                            task_id=task.id,
                            split=task.split,
                            seed=seed,
                            attempt=attempt_index,
                            success=bool(summary.get("success")),
                            verdict=str(summary.get("verifier_verdict", "")),
                            agent_claimed=summary.get("agent_claimed_success"),
                            failure_kind=summary.get("failure_kind"),
                            steps=int(summary.get("steps", 0)),
                            tool_calls=int(summary.get("tool_calls", 0)),
                            tool_errors=int(summary.get("tool_errors", 0)),
                            verifier_rejections=int(summary.get("verifier_rejections", 0)),
                            redundant_calls=int(summary.get("redundant_tool_calls", 0)),
                        )
                    )
                    if verbose:
                        last = report.attempts[-1]
                        mark = "ok " if last.success else "FAIL"
                        print(f"  [{mark}] {task.id} seed={seed} try={attempt_index}: {last.verdict}")
    finally:
        env.close()

    return report


def save_report(report: BenchReport, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"summary": report.summary(), "attempts": [asdict(a) for a in report.attempts]},
            indent=2,
        )
    )
    return path
