"""Damped least-squares inverse kinematics for the tabletop arm.

We solve a weighted 6-DOF task (position weighted far above orientation) and let the
damped pseudo-inverse settle on the least-squares compromise, which keeps the gripper
close to the requested approach direction while hitting the position exactly. On an arm
with fewer than six joints that compromise is the whole point; on a redundant arm the
damping picks a well-conditioned solution out of the null space.

Which joints those are is read from the model rather than hardcoded. It used to be a
fixed tuple of five names, which silently solved over five of the Panda's seven joints
when the arm was swapped -- every target came back "unreachable" by 3 to 9cm with the
solver never touching two of the joints it had available.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

#: The fingers extend along the wrist's +z axis, so pointing them straight down means
#: rotating the wrist frame 180 degrees about x: (w, x, y, z) = (0, 1, 0, 0).
GRIPPER_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def arm_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Every hinge joint in the model, in model order.

    Gripper fingers are slide joints and object free joints are free joints, so on both
    scenes here this is exactly the arm: five on the hand-written one, seven on the Panda.
    """
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    )


@dataclass(frozen=True)
class IKResult:
    success: bool
    qpos: np.ndarray
    pos_err: float
    rot_err: float
    reason: str


def arm_dof_indices(model: mujoco.MjModel) -> np.ndarray:
    return np.array(
        [
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in arm_joint_names(model)
        ]
    )


def arm_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    return np.array(
        [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in arm_joint_names(model)
        ]
    )


def arm_joint_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in arm_joint_names(model)]
    ranges = model.jnt_range[ids]
    return ranges[:, 0].copy(), ranges[:, 1].copy()


def yaw_down_quat(yaw: float) -> np.ndarray:
    """Gripper pointing down, rotated by `yaw` radians about the world z axis."""
    half = yaw / 2.0
    q_yaw = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q_yaw, GRIPPER_DOWN_QUAT)
    return out


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None = None,
    *,
    max_iters: int = 300,
    pos_tol: float = 5e-3,
    damping: float = 5e-2,
    rot_weight: float = 0.25,
    step_scale: float = 0.6,
) -> IKResult:
    """Solve for arm joint angles that place `site_id` at `target_pos`.

    Runs on a scratch copy of `data` so the live simulation state is untouched. The
    returned qpos is clipped to the joint limits; `success` reflects the position
    tolerance only, since orientation is a soft objective.
    """
    target_pos = np.asarray(target_pos, dtype=float)
    if target_quat is None:
        target_quat = GRIPPER_DOWN_QUAT
    target_quat = np.asarray(target_quat, dtype=float)

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0

    dof_idx = arm_dof_indices(model)
    qpos_idx = arm_qpos_indices(model)
    lo, hi = arm_joint_limits(model)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    err_quat = np.zeros(4)
    rot_err_vec = np.zeros(3)

    pos_err = float("inf")
    rot_err = float("inf")

    for _ in range(max_iters):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)

        pos_delta = target_pos - scratch.site_xpos[site_id]
        pos_err = float(np.linalg.norm(pos_delta))

        mujoco.mju_mat2Quat(site_quat, scratch.site_xmat[site_id])
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(err_quat, target_quat, site_quat_conj)
        mujoco.mju_quat2Vel(rot_err_vec, err_quat, 1.0)
        rot_err = float(np.linalg.norm(rot_err_vec))

        if pos_err < pos_tol:
            break

        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        jac = np.vstack([jacp[:, dof_idx], rot_weight * jacr[:, dof_idx]])
        err = np.concatenate([pos_delta, rot_weight * rot_err_vec])

        # Damped least squares: J^T (J J^T + lambda^2 I)^-1 e
        jjt = jac @ jac.T + (damping**2) * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt, err)

        scratch.qpos[qpos_idx] = np.clip(scratch.qpos[qpos_idx] + step_scale * dq, lo, hi)

    solution = scratch.qpos[qpos_idx].copy()
    if pos_err < pos_tol:
        return IKResult(True, solution, pos_err, rot_err, "ok")
    return IKResult(
        False,
        solution,
        pos_err,
        rot_err,
        f"unreachable: converged {pos_err * 100:.1f}cm away from target",
    )
