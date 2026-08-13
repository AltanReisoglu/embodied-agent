"""MuJoCo tabletop environment: a 5-DOF arm, a gripper, cubes and a plate.

Everything the agent is told about its own body comes from here, measured after the
physics has run -- never predicted.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# A GL backend must be selected before mujoco is imported, and headless boxes have no
# display. EGL is the fast path; osmesa is the software fallback.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from embodied_agent.envs.base import Observation, SkillResult  # noqa: E402
from embodied_agent.skills.ik import (  # noqa: E402
    arm_qpos_indices,
    solve_ik,
    yaw_down_quat,
)

ASSET = Path(__file__).resolve().parent.parent / "assets" / "tabletop.xml"

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

#: Geoms that belong to the arm itself; contact between two of these is self-collision
#: and contact with anything else during a move is a report-worthy event.
ARM_GEOM_PREFIXES = ("link", "wrist", "base")
FINGER_GEOMS = ("fingerpad_left", "fingerpad_right")
MANIPULABLE = ("red_cube", "green_cube", "blue_plate")


class TabletopEnv:
    def __init__(
        self,
        *,
        image_size: tuple[int, int] = (480, 640),
        default_camera: str = "front",
        seed: int | None = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(ASSET))
        self.data = mujoco.MjData(self.model)
        self.height, self.width = image_size
        self.default_camera = default_camera
        self.rng = np.random.default_rng(seed)

        self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        self._site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._arm_qpos = arm_qpos_indices(self.model)
        self._gripper_actuator = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
        )
        self._finger_qpos = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_left")
        ]

        self.step_count = 0
        self.last_action: SkillResult | None = None

    # ---------------------------------------------------------------- lifecycle

    def reset(self, randomize: bool = False) -> Observation:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        if randomize:
            for name in ("red_cube", "green_cube"):
                adr = self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                self.data.qpos[adr] += self.rng.uniform(-0.04, 0.04)
                self.data.qpos[adr + 1] += self.rng.uniform(-0.03, 0.03)
        mujoco.mj_forward(self.model, self.data)
        self._settle(steps=200)
        self.step_count = 0
        self.last_action = None
        return self.observe()

    def close(self) -> None:
        renderer, self._renderer = getattr(self, "_renderer", None), None
        if renderer is not None:
            try:
                renderer.close()
            except Exception:  # GL teardown races at interpreter shutdown; nothing to salvage
                pass

    # -------------------------------------------------------------- proprioception

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._site_id].copy()

    @property
    def ee_quat(self) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._site_id])
        return quat

    @property
    def gripper_opening(self) -> float:
        return float(self.data.qpos[self._finger_qpos])

    def object_pose(self, name: str) -> np.ndarray:
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[body].copy()

    def object_poses(self) -> dict[str, np.ndarray]:
        return {name: self.object_pose(name) for name in MANIPULABLE}

    def held_object(self) -> str | None:
        """An object is held when both finger pads touch it and the gripper is not open."""
        if self.gripper_opening > 0.035:
            return None
        touching: dict[str, set[str]] = {}
        for pad, other in self._contact_pairs():
            if pad in FINGER_GEOMS and other in MANIPULABLE:
                touching.setdefault(other, set()).add(pad)
            elif other in FINGER_GEOMS and pad in MANIPULABLE:
                touching.setdefault(pad, set()).add(other)
        for obj, pads in touching.items():
            if len(pads) == 2:
                return obj
        return None

    def _geom_label(self, geom_id: int) -> str:
        """Name of a geom, falling back to its body -- the arm capsules are unnamed."""
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name:
            return name
        body = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[geom_id]
        )
        return body or f"geom{geom_id}"

    def _contact_pairs(self) -> list[tuple[str, str]]:
        return [
            (self._geom_label(self.data.contact[i].geom1), self._geom_label(self.data.contact[i].geom2))
            for i in range(self.data.ncon)
        ]

    def contact_summary(self) -> list[str]:
        out = []
        for g1, g2 in self._contact_pairs():
            if g1 == "floor" or g2 == "floor":
                continue
            if {g1, g2} <= set(MANIPULABLE) | {"table"}:
                continue  # objects resting on the table is not news
            out.append(f"{g1}<->{g2}")
        return sorted(set(out))

    def _unexpected_arm_contacts(self) -> list[str]:
        """Arm-body (not fingertip) contact with the world -- i.e. a collision."""
        out = []
        for g1, g2 in self._contact_pairs():
            for a, b in ((g1, g2), (g2, g1)):
                if a.startswith(ARM_GEOM_PREFIXES) and (b == "table" or b in MANIPULABLE):
                    out.append(f"{a}<->{b}")
        return sorted(set(out))

    # ------------------------------------------------------------------ observation

    def observe(self, camera: str | None = None) -> Observation:
        cam = camera or self.default_camera
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self.data, camera=cam)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=cam)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()

        return Observation(
            step=self.step_count,
            camera=cam,
            rgb=rgb,
            depth=depth,
            ee_pos=self.ee_pos,
            ee_quat=self.ee_quat,
            gripper_opening=self.gripper_opening,
            holding=self.held_object(),
            contacts=self.contact_summary(),
            last_action=self.last_action,
        )

    # -------------------------------------------------------------------- geometry

    def camera_intrinsics(self, camera: str) -> tuple[float, float, float, float]:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        fovy = np.deg2rad(self.model.cam_fovy[cam_id])
        fy = (self.height / 2.0) / np.tan(fovy / 2.0)
        return fy, fy, self.width / 2.0, self.height / 2.0

    def pixel_to_world(
        self, u: int, v: int, camera: str | None = None, depth: np.ndarray | None = None
    ) -> np.ndarray | None:
        """Back-project pixel (u, v) to a world point using the rendered depth buffer.

        Returns None when the pixel has no geometry behind it (depth at the far plane).
        """
        cam = camera or self.default_camera
        if depth is None:
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(self.data, camera=cam)
            depth = self._renderer.render().copy()
            self._renderer.disable_depth_rendering()

        if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
            return None
        z = float(depth[v, u])
        if not np.isfinite(z) or z <= 0 or z > 20.0:
            return None

        fx, fy, cx, cy = self.camera_intrinsics(cam)
        # MuJoCo cameras look down -z with +y up, while image rows grow downward.
        p_cam = np.array([(u - cx) * z / fx, -(v - cy) * z / fy, -z])
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        rot = self.data.cam_xmat[cam_id].reshape(3, 3)
        return self.data.cam_xpos[cam_id] + rot @ p_cam

    def world_to_pixel(self, point: np.ndarray, camera: str | None = None) -> tuple[int, int] | None:
        cam = camera or self.default_camera
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        rot = self.data.cam_xmat[cam_id].reshape(3, 3)
        p_cam = rot.T @ (np.asarray(point, dtype=float) - self.data.cam_xpos[cam_id])
        if p_cam[2] >= -1e-6:
            return None  # behind the camera
        fx, fy, cx, cy = self.camera_intrinsics(cam)
        z = -p_cam[2]
        return int(round(p_cam[0] * fx / z + cx)), int(round(-p_cam[1] * fy / z + cy))

    # ---------------------------------------------------------------------- skills

    def move_to(
        self,
        waypoints: list[list[float]],
        *,
        yaw: float = 0.0,
        seconds_per_waypoint: float = 1.2,
    ) -> SkillResult:
        """Follow a coarse trajectory with the gripper pointing down.

        Taking a waypoint list rather than a single point follows ACoT-VLA
        (arXiv 2601.11404): the model reasons in the language of actions, so its plan
        arrives as a trajectory rather than prose.
        """
        self.step_count += 1
        if not waypoints:
            return self._record(SkillResult(False, "move_to", "no waypoints given"))

        target_quat = yaw_down_quat(yaw)
        collisions: set[str] = set()

        for idx, wp in enumerate(waypoints):
            point = np.asarray(wp, dtype=float)
            if point.shape != (3,):
                return self._record(
                    SkillResult(
                        False, "move_to", f"waypoint {idx + 1} is not an [x, y, z] triple"
                    )
                )

            ik = solve_ik(self.model, self.data, self._site_id, point, target_quat)
            if not ik.success:
                return self._record(
                    SkillResult(
                        False,
                        "move_to",
                        f"IK unreachable at waypoint {idx + 1}/{len(waypoints)}: "
                        f"closest approach {ik.pos_err * 100:.1f}cm",
                        {
                            "reached_waypoints": idx,
                            "ee_pos": self.ee_pos.round(4).tolist(),
                            "collisions": sorted(collisions),
                        },
                    )
                )

            collisions |= self._ramp_to(ik.qpos, seconds_per_waypoint)

        self._settle(steps=150)
        collisions |= set(self._unexpected_arm_contacts())

        final_err = float(np.linalg.norm(self.ee_pos - np.asarray(waypoints[-1], dtype=float)))
        detail = {
            "ee_pos": self.ee_pos.round(4).tolist(),
            "final_error_cm": round(final_err * 100, 2),
            "collisions": sorted(collisions),
        }
        if final_err > 0.03:
            return self._record(
                SkillResult(
                    False,
                    "move_to",
                    f"stopped {final_err * 100:.1f}cm short of the last waypoint "
                    f"(likely blocked)",
                    detail,
                )
            )
        reason = "ok" if not collisions else f"reached, but touched {', '.join(sorted(collisions))}"
        return self._record(SkillResult(True, "move_to", reason, detail))

    def grasp(self) -> SkillResult:
        self.step_count += 1
        self.data.ctrl[self._gripper_actuator] = GRIPPER_CLOSED
        self._advance(seconds=1.0)
        self._settle(steps=100)
        held = self.held_object()
        detail = {
            "gripper_opening_cm": round(self.gripper_opening * 100, 2),
            "holding": held,
        }
        if held is None:
            return self._record(
                SkillResult(
                    False,
                    "grasp",
                    "gripper closed on nothing -- no object between the fingers",
                    detail,
                )
            )
        return self._record(SkillResult(True, "grasp", f"holding {held}", detail))

    def release(self) -> SkillResult:
        self.step_count += 1
        was_holding = self.held_object()
        self.data.ctrl[self._gripper_actuator] = GRIPPER_OPEN
        self._advance(seconds=1.0)
        self._settle(steps=200)
        detail = {"released": was_holding, "holding": self.held_object()}
        if was_holding is None:
            return self._record(
                SkillResult(True, "release", "gripper opened (was not holding anything)", detail)
            )
        return self._record(SkillResult(True, "release", f"released {was_holding}", detail))

    # -------------------------------------------------------------------- internals

    def _record(self, result: SkillResult) -> SkillResult:
        self.last_action = result
        return result

    def _ramp_to(self, arm_target: np.ndarray, seconds: float) -> set[str]:
        """Interpolate the position targets so the arm sweeps rather than teleports."""
        n_steps = max(1, int(seconds / self.model.opt.timestep))
        start = self.data.ctrl[:5].copy()
        collisions: set[str] = set()
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            self.data.ctrl[:5] = start + alpha * (arm_target - start)
            mujoco.mj_step(self.model, self.data)
            if i % 25 == 0:
                collisions |= set(self._unexpected_arm_contacts())
        return collisions

    def _advance(self, seconds: float) -> None:
        for _ in range(max(1, int(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def _settle(self, steps: int = 100) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if np.abs(self.data.qvel).max() < 1e-3:
                break
