"""Environment-facing types shared by every backend (MuJoCo today, a real arm later)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class SkillResult:
    """Outcome of one skill execution, as *measured* by the environment.

    HumanCLAW (arXiv 2607.27180) found that the dominant failure mode for harnessed
    VLMs is not perception but losing track of their own body -- whether they moved,
    arrived, or collided. So skills never return a bare success flag: they report what
    actually happened, and `to_line()` is rendered verbatim into the model's context.
    """

    ok: bool
    skill: str
    reason: str = "ok"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_line(self) -> str:
        status = "OK" if self.ok else "FAILED"
        line = f"{self.skill} -> {status}"
        if self.reason and self.reason != "ok":
            line += f" ({self.reason})"
        return line


@dataclass
class Observation:
    """One perception step: pixels plus the proprioceptive facts behind them."""

    step: int
    camera: str
    rgb: np.ndarray
    depth: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    gripper_opening: float
    holding: str | None
    contacts: list[str]
    last_action: SkillResult | None

    @property
    def gripper_state(self) -> str:
        if self.gripper_opening > 0.030:
            return "open"
        if self.gripper_opening < 0.008:
            return "closed"
        return f"partially closed ({self.gripper_opening * 100:.1f}cm)"


class Env(Protocol):
    """Minimal contract the agent loop depends on."""

    def reset(self) -> Observation: ...

    def observe(self, camera: str | None = None) -> Observation: ...

    def move_to(self, waypoints: list[list[float]]) -> SkillResult: ...

    def grasp(self) -> SkillResult: ...

    def release(self) -> SkillResult: ...

    def pixel_to_world(self, u: int, v: int, camera: str | None = None) -> np.ndarray | None: ...
