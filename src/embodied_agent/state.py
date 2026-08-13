"""Rendering the agent's own body state as text.

HumanCLAW (arXiv 2607.27180) evaluated exactly this architecture -- an off-the-shelf VLM
issuing skill commands that a controller executes -- across 1,218 episodes and nine
frontier VLMs. The best reached 16.8%. The bottleneck was not perception: up to 81% of
interaction-stage failures traced to egocentric self-localisation and body awareness,
i.e. the model could not tell where its body was, whether it had arrived, or whether it
had collided.

So we never ask the model to infer those facts from pixels. Every step it receives this
block, measured from the simulator, immediately before the image.
"""

from __future__ import annotations

import numpy as np

from embodied_agent.envs.base import Observation


def _fmt_vec(v: np.ndarray, digits: int = 3) -> str:
    return "(" + ", ".join(f"{x:.{digits}f}" for x in v) + ")"


def state_block(obs: Observation, *, task: str | None = None, max_steps: int | None = None) -> str:
    lines: list[str] = ["=== BODY STATE (measured, authoritative) ==="]
    if task:
        lines.append(f"task: {task}")

    lines.append(
        f"gripper_position: {_fmt_vec(obs.ee_pos)}   "
        f"gripper: {obs.gripper_state}   holding: {obs.holding or 'nothing'}"
    )

    if obs.last_action is None:
        lines.append("last_action: none yet (this is the first observation)")
    else:
        lines.append(f"last_action: {obs.last_action.to_line()}")
        detail = obs.last_action.detail
        if detail.get("final_error_cm") is not None:
            lines.append(f"  distance from commanded waypoint: {detail['final_error_cm']}cm")
        if detail.get("collisions"):
            lines.append(f"  COLLISION during motion: {', '.join(detail['collisions'])}")

    lines.append(f"contacts_now: {', '.join(obs.contacts) if obs.contacts else 'none'}")

    if max_steps is not None:
        lines.append(f"step: {obs.step}/{max_steps}")
    lines.append(f"camera: {obs.camera}")
    lines.append("=== END BODY STATE ===")
    return "\n".join(lines)


def workspace_hint() -> str:
    """Static facts about the reachable volume, so the model does not have to discover
    them by failing repeatedly."""
    return (
        "Workspace: the table surface is at z=0.40. Objects rest with their centre at "
        "z=0.42. The arm can reach roughly x in [-0.25, 0.25], y in [0.05, 0.32], "
        "z in [0.43, 0.70], with the gripper always pointing straight down. "
        "To pick an object at (x, y): move to (x, y, 0.55), then (x, y, 0.445), then grasp."
    )
