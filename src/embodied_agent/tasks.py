"""A task family with a held-out split.

"Rethinking the Evaluation of Harness Evolution for Agents" (arXiv 2607.12227) found
that reported gains from evolving an agent's harness largely vanish under two controls:
comparing against plain test-time scaling at a matched budget, and evaluating on tasks
disjoint from the ones the harness was tuned on. On Terminal-Bench 2.1 an evolved harness
transferred for +0.6 points on average -- and +0.0 on GPT-5.4.

Any claim made from a single hardcoded task here would be exactly the kind of result that
paper is about. So: a family of tasks, split into train and test, each with an
environment-side verifier that never consults what the agent claims.

Their own closing conditions are also why this domain is worth measuring at all. Harness
evolution should matter where (1) there is real headroom and (2) performance genuinely
depends on the harness. Terminal-Bench met neither. Embodied manipulation meets both:
HumanCLAW's best model reached 16.8%, and removing one harness component took interaction
success from 18.9% to 0%.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from embodied_agent.envs.mujoco_tabletop import TabletopEnv

Verifier = Callable[[TabletopEnv], tuple[bool, str]]
Setup = Callable[[TabletopEnv, np.random.Generator], None]

#: Objects rest with their centre here; the plate's top face sits slightly higher.
TABLE_TOP_Z = 0.42
ON_PLATE_Z = 0.425
PLATE_RADIUS = 0.055
CUBE_HALF = 0.02


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    verify: Verifier
    split: str
    setup: Setup | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


# ------------------------------------------------------------------ verifier helpers


def _xy(env: TabletopEnv, name: str) -> np.ndarray:
    return env.object_pose(name)[:2]


def _z(env: TabletopEnv, name: str) -> float:
    return float(env.object_pose(name)[2])


def _on_plate(cube: str) -> Verifier:
    def verify(env: TabletopEnv) -> tuple[bool, str]:
        gap = float(np.linalg.norm(_xy(env, cube) - _xy(env, "blue_plate")))
        lifted = _z(env, cube) > ON_PLATE_Z
        if gap < PLATE_RADIUS and lifted:
            return True, f"{cube} rests on the plate, {gap * 100:.1f}cm from its centre"
        if not lifted:
            return False, f"{cube} is still at table height (z={_z(env, cube):.3f})"
        return False, f"{cube} is {gap * 100:.1f}cm from the plate centre"

    return verify


def _stacked(top: str, bottom: str) -> Verifier:
    def verify(env: TabletopEnv) -> tuple[bool, str]:
        gap = float(np.linalg.norm(_xy(env, top) - _xy(env, bottom)))
        height = _z(env, top) - _z(env, bottom)
        if gap < 0.025 and height > 0.025:
            return True, f"{top} sits on {bottom} ({gap * 100:.1f}cm off centre)"
        if height <= 0.025:
            return False, f"{top} is not on top of {bottom} (height gap {height * 100:.1f}cm)"
        return False, f"{top} is {gap * 100:.1f}cm off the centre of {bottom}"

    return verify


def _moved_to_side(obj: str, side: str) -> Verifier:
    """Region goals have no object to align to, so the agent cannot solve them by
    homing on a visible target -- it has to reason about table coordinates."""
    sign = -1.0 if side == "left" else 1.0

    def verify(env: TabletopEnv) -> tuple[bool, str]:
        x = float(env.object_pose(obj)[0])
        if sign * x > 0.10:
            return True, f"{obj} is on the {side} of the table (x={x:.3f})"
        return False, f"{obj} is at x={x:.3f}, not far enough {side}"

    return verify


def _both_on_plate() -> Verifier:
    red, green = _on_plate("red_cube"), _on_plate("green_cube")

    def verify(env: TabletopEnv) -> tuple[bool, str]:
        red_ok, red_why = red(env)
        green_ok, green_why = green(env)
        if red_ok and green_ok:
            return True, "both cubes are on the plate"
        return False, f"red: {red_why}; green: {green_why}"

    return verify


def _untouched(obj: str, tolerance: float = 0.03) -> Verifier:
    """Checks a distractor stayed put. Composed onto a goal to test whether the agent
    achieves it without knocking the rest of the scene around."""
    start: dict[str, np.ndarray] = {}

    def verify(env: TabletopEnv) -> tuple[bool, str]:
        reference = start.get(obj)
        if reference is None:
            return True, f"no baseline recorded for {obj}"
        drift = float(np.linalg.norm(_xy(env, obj) - reference))
        if drift < tolerance:
            return True, f"{obj} undisturbed"
        return False, f"{obj} was displaced by {drift * 100:.1f}cm"

    verify.record_baseline = lambda env: start.update({obj: _xy(env, obj).copy()})  # type: ignore[attr-defined]
    return verify


def _both(first: Verifier, second: Verifier) -> Verifier:
    def verify(env: TabletopEnv) -> tuple[bool, str]:
        ok_a, why_a = first(env)
        if not ok_a:
            return False, why_a
        ok_b, why_b = second(env)
        return (True, f"{why_a}; {why_b}") if ok_b else (False, why_b)

    return verify


# --------------------------------------------------------------------- setup helpers


def _jitter(*names: str, spread: float = 0.035) -> Setup:
    """Perturb starting positions so a policy cannot succeed on memorised coordinates."""

    def setup(env: TabletopEnv, rng: np.random.Generator) -> None:
        import mujoco

        for name in names:
            adr = env.model.jnt_qposadr[
                mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            env.data.qpos[adr] += rng.uniform(-spread, spread)
            env.data.qpos[adr + 1] += rng.uniform(-spread * 0.7, spread * 0.7)

    return setup


# --------------------------------------------------------------------------- the set

TRAIN_TASKS: list[Task] = [
    Task(
        id="red_on_plate",
        instruction="Put the red cube on the blue plate.",
        verify=_on_plate("red_cube"),
        split="train",
        setup=_jitter("red_cube"),
        tags=("pick_place",),
    ),
    Task(
        id="green_on_plate",
        instruction="Put the green cube on the blue plate.",
        verify=_on_plate("green_cube"),
        split="train",
        setup=_jitter("green_cube"),
        tags=("pick_place",),
    ),
    Task(
        id="red_left",
        instruction="Move the red cube to the left side of the table.",
        verify=_moved_to_side("red_cube", "left"),
        split="train",
        setup=_jitter("red_cube"),
        tags=("region",),
    ),
    Task(
        id="red_on_plate_keep_green",
        instruction="Put the red cube on the blue plate without disturbing the green cube.",
        verify=_both(_on_plate("red_cube"), _untouched("green_cube")),
        split="train",
        setup=_jitter("red_cube", "green_cube"),
        tags=("pick_place", "distractor"),
    ),
]

TEST_TASKS: list[Task] = [
    Task(
        id="green_right",
        instruction="Move the green cube to the right side of the table.",
        verify=_moved_to_side("green_cube", "right"),
        split="test",
        setup=_jitter("green_cube"),
        tags=("region",),
    ),
    Task(
        id="stack_red_on_green",
        instruction="Stack the red cube on top of the green cube.",
        verify=_stacked("red_cube", "green_cube"),
        split="test",
        setup=_jitter("red_cube", "green_cube"),
        tags=("stack",),
    ),
    Task(
        id="both_on_plate",
        instruction="Put both the red cube and the green cube on the blue plate.",
        verify=_both_on_plate(),
        split="test",
        setup=_jitter("red_cube", "green_cube"),
        tags=("pick_place", "multi_object"),
    ),
    Task(
        id="green_on_plate_keep_red",
        instruction="Put the green cube on the blue plate without disturbing the red cube.",
        verify=_both(_on_plate("green_cube"), _untouched("red_cube")),
        split="test",
        setup=_jitter("red_cube", "green_cube"),
        tags=("pick_place", "distractor"),
    ),
]

ALL_TASKS = TRAIN_TASKS + TEST_TASKS


def get_split(split: str) -> list[Task]:
    if split == "train":
        return list(TRAIN_TASKS)
    if split == "test":
        return list(TEST_TASKS)
    if split == "all":
        return list(ALL_TASKS)
    raise ValueError(f"unknown split {split!r}; expected train, test or all")


def get_task(task_id: str) -> Task:
    for task in ALL_TASKS:
        if task.id == task_id:
            return task
    known = ", ".join(t.id for t in ALL_TASKS)
    raise KeyError(f"no task {task_id!r}. Known tasks: {known}")


def prepare(env: TabletopEnv, task: Task, seed: int) -> None:
    """Reset the environment for one attempt at `task` under a given seed.

    Baselines re-run the same (task, seed) pair, so this must place the objects
    identically every time -- otherwise repeated attempts would be measuring different
    problems and pass@k would be meaningless.
    """
    import mujoco

    env.reset()
    if task.setup is not None:
        task.setup(env, np.random.default_rng(seed))
        mujoco.mj_forward(env.model, env.data)
        env._settle(steps=200)

    # Distractor checks need the post-setup pose as their baseline.
    recorder = getattr(task.verify, "record_baseline", None)
    if recorder is not None:
        recorder(env)
    for attr in ("__closure__",):  # composed verifiers hide their parts in closures
        for cell in getattr(task.verify, attr, None) or []:
            inner = getattr(cell.cell_contents, "record_baseline", None)
            if inner is not None:
                inner(env)
