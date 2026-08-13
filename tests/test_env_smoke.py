"""P0 gate: the environment must work end-to-end before any model is involved."""

from __future__ import annotations

import numpy as np
import pytest

from embodied_agent.envs.mujoco_tabletop import TabletopEnv


@pytest.fixture(scope="module")
def env():
    e = TabletopEnv(image_size=(240, 320))
    yield e
    e.close()


def test_reset_gives_a_clean_observation(env):
    obs = env.reset()
    assert obs.rgb.shape == (240, 320, 3)
    assert obs.depth.shape == (240, 320)
    assert obs.gripper_state == "open"
    assert obs.holding is None
    assert obs.last_action is None
    # A resting arm must not be in contact with anything; a permanent contact would
    # apply friction to a joint and silently stop it from tracking its command.
    assert obs.contacts == []


def test_arm_tracks_commanded_position(env):
    env.reset()
    result = env.move_to([[0.13, 0.18, 0.55]])
    assert result.ok, result.to_line()
    assert np.linalg.norm(env.ee_pos - np.array([0.13, 0.18, 0.55])) < 0.02


def test_unreachable_target_reports_why_instead_of_lying(env):
    env.reset()
    result = env.move_to([[0.9, 0.9, 0.9]])
    assert not result.ok
    assert "unreachable" in result.reason
    # The failure must carry the measured body state, not just a flag.
    assert "ee_pos" in result.detail


def test_grasp_on_empty_air_reports_failure(env):
    env.reset()
    assert env.move_to([[0.0, 0.15, 0.50]]).ok
    result = env.grasp()
    assert not result.ok
    assert env.held_object() is None


def test_full_pick_and_place(env):
    env.reset()
    start = env.object_pose("red_cube").copy()

    assert env.move_to([[0.13, 0.18, 0.55], [0.13, 0.18, 0.445]]).ok
    grasp = env.grasp()
    assert grasp.ok, grasp.to_line()
    assert env.held_object() == "red_cube"

    assert env.move_to([[0.13, 0.18, 0.58], [-0.02, 0.30, 0.56], [-0.02, 0.30, 0.50]]).ok
    assert env.release().ok

    end = env.object_pose("red_cube")
    assert np.linalg.norm(end[:2] - start[:2]) > 0.15, "cube did not actually move"
    # It should have landed on the plate, i.e. above the bare table surface.
    assert end[2] > 0.43


def test_pixel_to_world_round_trips_on_a_visible_object(env):
    obs = env.reset()
    cube = env.object_pose("red_cube")
    pixel = env.world_to_pixel(cube)
    assert pixel is not None

    back = env.pixel_to_world(*pixel, depth=obs.depth)
    assert back is not None
    # Depth returns the front surface while world_to_pixel projects the centre, so for a
    # 4cm cube the two differ by about half its size.
    assert np.linalg.norm(back - cube) < 0.035


def test_pixel_to_world_rejects_pixels_with_no_geometry(env):
    """A far-plane depth value means the ray hit nothing; back-projecting it would hand
    the agent a plausible-looking coordinate hundreds of metres away."""
    env.reset()
    far = np.full((env.height, env.width), 1e6, dtype=np.float32)
    assert env.pixel_to_world(160, 120, depth=far) is None


def test_pixel_to_world_rejects_out_of_frame_pixels(env):
    obs = env.reset()
    assert env.pixel_to_world(-5, 10, depth=obs.depth) is None
    assert env.pixel_to_world(10, 99999, depth=obs.depth) is None
