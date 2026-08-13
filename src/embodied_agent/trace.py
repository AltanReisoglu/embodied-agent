"""Episode recording and the two metrics that keep this honest.

Success rate alone hides the thing we actually changed. We also record:

* a failure taxonomy in the shape HumanCLAW used (localisation / body awareness / goal
  detection / interaction), so the body-state block can be judged by whether it moves the
  body-awareness bucket;
* the redundant tool-call rate from Act Wisely, which degrades silently when nobody
  counts it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FailureKind = str  # "localisation" | "body_awareness" | "goal_detection" | "interaction"


@dataclass
class StepRecord:
    step: int
    thinking: str
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    state_block: str = ""
    frame: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class Trace:
    def __init__(self, root: Path | str = "runs", *, task: str = "", model: str = "") -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / stamp
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.task = task
        self.model = model
        self.steps: list[StepRecord] = []
        self._redundant: list[str] = []
        self._errors: list[str] = []

    # ------------------------------------------------------------------ recording

    def save_frame(self, step: int, rgb: np.ndarray, tag: str = "obs") -> str:
        name = f"{step:03d}_{tag}.jpg"
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
            self.dir / "frames" / name, quality=88
        )
        return f"frames/{name}"

    def log_step(self, record: StepRecord) -> None:
        self.steps.append(record)
        with (self.dir / "trace.jsonl").open("a") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")

    def note_redundant(self, tool_name: str) -> None:
        self._redundant.append(tool_name)

    def note_tool_error(self, tool_name: str) -> None:
        """Recorded from the dispatcher's `is_error` flag rather than sniffed out of the
        result text -- a tool that explains a problem in prose is still a failed call."""
        self._errors.append(tool_name)

    # -------------------------------------------------------------------- metrics

    def metrics(self) -> dict[str, Any]:
        calls = [c for s in self.steps for c in s.tool_calls]
        by_tool = Counter(c["name"] for c in calls)
        return {
            "steps": len(self.steps),
            "tool_calls": len(calls),
            "tool_calls_by_name": dict(by_tool),
            "tool_errors": len(self._errors),
            "tool_errors_by_name": dict(Counter(self._errors)),
            # Act Wisely (arXiv 2604.08545): tracked explicitly because it degrades quietly.
            "redundant_tool_calls": len(self._redundant),
            "redundant_rate": round(len(self._redundant) / len(calls), 3) if calls else 0.0,
            "prompt_tokens": sum(s.usage.get("prompt_tokens", 0) for s in self.steps),
            "completion_tokens": sum(s.usage.get("completion_tokens", 0) for s in self.steps),
        }

    def finish(
        self,
        *,
        success: bool,
        reason: str,
        failure_kind: FailureKind | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary = {
            "task": self.task,
            "model": self.model,
            "success": success,
            "reason": reason,
            # HumanCLAW-style bucket; None when the episode succeeded.
            "failure_kind": failure_kind,
            **self.metrics(),
            **(extra or {}),
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def classify_failure(steps: list[StepRecord]) -> FailureKind:
    """Bucket an unsuccessful episode by what the last failing tool result blamed.

    A coarse first pass -- the point is to see whether body-awareness failures dominate
    the way HumanCLAW reported, not to be perfectly precise.
    """
    for step in reversed(steps):
        for result in reversed(step.tool_results):
            low = result.lower()
            if "closed on nothing" in low:
                return "interaction"
            if "unreachable" in low or "empty space" in low or "floor beside" in low:
                return "localisation"
            if "blocked" in low or "collision" in low:
                return "body_awareness"
    return "goal_detection"
