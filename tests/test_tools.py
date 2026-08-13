"""Dispatch must convert every foreseeable model mistake into a correctable message."""

from __future__ import annotations

import pytest

from embodied_agent.envs.mujoco_tabletop import TabletopEnv
from embodied_agent.memory import Memory
from embodied_agent.reasoner.base import ToolCall
from embodied_agent.tools.builtin import build_registry


@pytest.fixture(scope="module")
def fixtures():
    env = TabletopEnv(image_size=(240, 320))
    env.reset()
    memory = Memory()
    yield env, memory, build_registry(env, memory)
    env.close()


def test_malformed_json_arguments_do_not_raise(fixtures):
    _, _, registry = fixtures
    call = ToolCall.parse("1", "measure", '{"pixel_x": 100, "pixel_y":}')
    assert call.parse_error is not None

    result = registry.dispatch(call)
    assert result.is_error
    assert "valid JSON" in result.text


def test_unknown_tool_lists_the_real_ones(fixtures):
    _, _, registry = fixtures
    result = registry.dispatch(ToolCall("1", "teleport", {}))
    assert result.is_error
    assert "move_to" in result.text


def test_missing_required_argument_is_named(fixtures):
    _, _, registry = fixtures
    result = registry.dispatch(ToolCall("1", "measure", {"pixel_x": 10}))
    assert result.is_error
    assert "pixel_y" in result.text


def test_privileged_tool_hidden_by_default(fixtures):
    _, _, registry = fixtures
    assert "list_objects" not in [t.name for t in registry.available()]
    assert registry.dispatch(ToolCall("1", "list_objects", {})).is_error


def test_privileged_tool_available_when_enabled():
    env = TabletopEnv(image_size=(240, 320))
    env.reset()
    registry = build_registry(env, Memory(), allow_privileged=True)
    try:
        result = registry.dispatch(ToolCall("1", "list_objects", {}))
        assert not result.is_error
        assert "red_cube" in result.text
    finally:
        env.close()


def test_measure_off_the_table_explains_itself(fixtures):
    """The near-overhead camera sees floor around the table; a coordinate there is
    outside the workspace, so it must come back as a correctable error, not a number."""
    _, _, registry = fixtures
    result = registry.dispatch(ToolCall("1", "measure", {"pixel_x": 4, "pixel_y": 4}))
    assert result.is_error
    assert "floor" in result.text


def test_measure_on_an_object_returns_usable_coordinates(fixtures):
    env, _, registry = fixtures
    pixel = env.world_to_pixel(env.object_pose("red_cube"))
    assert pixel is not None

    result = registry.dispatch(
        ToolCall("1", "measure", {"pixel_x": pixel[0], "pixel_y": pixel[1]})
    )
    assert not result.is_error, result.text
    assert "world position" in result.text


def test_action_tools_are_marked_as_mutating(fixtures):
    _, _, registry = fixtures
    result = registry.dispatch(
        ToolCall("1", "move_to", {"waypoints": [[0.0, 0.15, 0.55]]})
    )
    assert result.mutates_world, "the loop relies on this to re-observe after an action"


def test_tool_crash_is_reported_not_raised(fixtures):
    _, _, registry = fixtures
    registry.tools["boom"] = registry.tools["grasp"].__class__(
        name="boom",
        description="always fails",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=lambda: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    result = registry.dispatch(ToolCall("1", "boom", {}))
    assert result.is_error
    assert "kaboom" in result.text


def test_descriptions_state_when_not_to_call(fixtures):
    """Act Wisely (arXiv 2604.08545): the abstention condition belongs in the schema the
    model reads, not only in the system prompt."""
    _, _, registry = fixtures
    for name in ("measure", "look", "zoom"):
        assert "do not" in registry.tools[name].description.lower(), name
