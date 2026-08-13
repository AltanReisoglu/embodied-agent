"""The agent's tool surface.

Every description states *when not to call* as well as when to call. "Act Wisely"
(arXiv 2604.08545) showed that agentic multimodal models reflexively invoke tools even
when the answer is already in the visual context -- redundant-call rates of 98% under a
naive objective -- which adds latency and injects noise that derails the reasoning. The
cheapest defence is to put the abstention condition in the description itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from embodied_agent.envs.mujoco_tabletop import TabletopEnv
from embodied_agent.memory import Memory
from embodied_agent.perception.render import crop_and_upscale
from embodied_agent.tools.registry import Registry, Tool, ToolResult


def _fmt_xyz(p: np.ndarray) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in p) + ")"


def build_registry(
    env: TabletopEnv,
    memory: Memory,
    *,
    allow_privileged: bool = False,
) -> Registry:
    registry = Registry(allow_privileged=allow_privileged)

    # ------------------------------------------------------------------ perception

    def measure(pixel_x: int, pixel_y: int) -> ToolResult:
        point = env.pixel_to_world(int(pixel_x), int(pixel_y))
        if point is None:
            return ToolResult(
                f"No surface at pixel ({pixel_x}, {pixel_y}) -- you are pointing past the "
                f"table into empty space. Pick a pixel that lies on an object.",
                is_error=True,
            )
        if point[2] < 0.38:
            # The table top is at z=0.40; anything well below it is the floor beyond the
            # table edge, which is outside the arm's workspace and useless to act on.
            return ToolResult(
                f"Pixel ({pixel_x}, {pixel_y}) lands on the floor beside the table "
                f"(z={point[2]:.2f}), not on the table surface. Pick a pixel inside the "
                f"table area, on the object you mean.",
                is_error=True,
            )
        return ToolResult(
            f"Pixel ({pixel_x}, {pixel_y}) is world position {_fmt_xyz(point)}. "
            f"This is the visible surface point; an object's centre sits about 2cm behind it."
        )

    registry.register(
        Tool(
            name="measure",
            description=(
                "Convert a pixel in the current image to a 3D world position, using the "
                "depth buffer. Call this once per object before you move to it -- the "
                "coordinates you need for move_to must come from here, never from your own "
                "estimate. Do NOT call it repeatedly on the same object: one reading is "
                "authoritative until the object moves."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pixel_x": {"type": "integer", "description": "Column, 0 at the left edge."},
                    "pixel_y": {"type": "integer", "description": "Row, 0 at the top edge."},
                },
                "required": ["pixel_x", "pixel_y"],
            },
            fn=measure,
        )
    )

    def look(camera: str = "front") -> ToolResult:
        if camera not in ("front", "wrist"):
            return ToolResult("error: camera must be 'front' or 'wrist'.", is_error=True)
        obs = env.observe(camera=camera)
        return ToolResult(
            f"Switched to the {camera} camera; the new view follows.",
            image=obs.rgb,
            image_caption=f"{camera} camera",
        )

    registry.register(
        Tool(
            name="look",
            description=(
                "Render the scene from another camera: 'front' (overview of the whole "
                "table) or 'wrist' (close view from the gripper, useful for checking "
                "alignment just before grasping). Do NOT call this if the current image "
                "already shows what you need."
            ),
            parameters={
                "type": "object",
                "properties": {"camera": {"type": "string", "enum": ["front", "wrist"]}},
                "required": ["camera"],
            },
            fn=look,
        )
    )

    def zoom(x0: int, y0: int, x1: int, y1: int) -> ToolResult:
        obs = env.observe()
        try:
            crop, clamped = crop_and_upscale(obs.rgb, (int(x0), int(y0), int(x1), int(y1)))
        except ValueError as exc:
            return ToolResult(f"error: {exc}", is_error=True)
        return ToolResult(
            f"Zoomed into region {clamped}. Note the crop has its own pixel coordinates -- "
            f"to use measure(), convert back by adding ({clamped[0]}, {clamped[1]}).",
            image=crop,
            image_caption=f"zoom {clamped}",
        )

    registry.register(
        Tool(
            name="zoom",
            description=(
                "Crop and magnify a rectangular region of the current front view. Use it "
                "only when an object is too small or ambiguous to localise confidently. Do "
                "NOT zoom on an object you can already identify and point at -- it costs a "
                "round trip and tells you nothing new."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "x0": {"type": "integer"},
                    "y0": {"type": "integer"},
                    "x1": {"type": "integer"},
                    "y1": {"type": "integer"},
                },
                "required": ["x0", "y0", "x1", "y1"],
            },
            fn=zoom,
        )
    )

    def list_objects() -> ToolResult:
        lines = [f"{name}: centre at {_fmt_xyz(pos)}" for name, pos in env.object_poses().items()]
        return ToolResult("Ground-truth object poses:\n" + "\n".join(lines))

    registry.register(
        Tool(
            name="list_objects",
            description=(
                "Return exact ground-truth positions of every object. This bypasses vision "
                "and is only enabled while bootstrapping."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            fn=list_objects,
            privileged=True,
        )
    )

    # ---------------------------------------------------------------------- action

    def move_to(waypoints: list[list[float]], yaw: float = 0.0) -> ToolResult:
        if not isinstance(waypoints, list) or not waypoints:
            return ToolResult(
                "error: waypoints must be a non-empty list of [x, y, z] triples.", is_error=True
            )
        # Tolerate a single flat triple, which models produce often.
        if all(isinstance(v, (int, float)) for v in waypoints):
            waypoints = [list(waypoints)]  # type: ignore[list-item]
        result = env.move_to(waypoints, yaw=float(yaw))
        memory.log_attempt(result.to_line())
        return ToolResult(result.to_line(), is_error=not result.ok, mutates_world=True)

    registry.register(
        Tool(
            name="move_to",
            description=(
                "Move the gripper along a trajectory of [x, y, z] waypoints, keeping it "
                "pointing straight down. Give the whole approach as one call -- e.g. "
                "[[x, y, 0.55], [x, y, 0.445]] to descend onto an object -- rather than one "
                "call per point. Coordinates must come from measure(). The result reports "
                "where the gripper actually ended up and any collision on the way."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "waypoints": {
                        "type": "array",
                        "description": "Ordered list of [x, y, z] positions in metres.",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    },
                    "yaw": {
                        "type": "number",
                        "description": "Optional rotation of the gripper about the vertical axis, in radians.",
                    },
                },
                "required": ["waypoints"],
            },
            fn=move_to,
            mutates_world=True,
        )
    )

    def grasp() -> ToolResult:
        result = env.grasp()
        memory.log_attempt(result.to_line())
        return ToolResult(result.to_line(), is_error=not result.ok, mutates_world=True)

    registry.register(
        Tool(
            name="grasp",
            description=(
                "Close the gripper. Succeeds only when an object is actually between the "
                "fingers, so position the gripper with move_to first. The result tells you "
                "what, if anything, you are now holding."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            fn=grasp,
            mutates_world=True,
        )
    )

    def release() -> ToolResult:
        result = env.release()
        memory.log_attempt(result.to_line())
        return ToolResult(result.to_line(), is_error=not result.ok, mutates_world=True)

    registry.register(
        Tool(
            name="release",
            description="Open the gripper, dropping whatever it holds at the current position.",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=release,
            mutates_world=True,
        )
    )

    def done(success: bool, reason: str) -> ToolResult:
        verdict = "success" if success else "give up"
        memory.log_attempt(f"done -> {verdict}: {reason}")
        return ToolResult(f"Episode ended ({verdict}): {reason}")

    registry.register(
        Tool(
            name="done",
            description=(
                "End the episode. Call this only after the body state confirms the goal is "
                "met -- not merely because you issued the actions you planned. Set "
                "success=false to stop when you are stuck."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "reason": {"type": "string", "description": "One sentence of evidence."},
                },
                "required": ["success", "reason"],
            },
            fn=done,
        )
    )

    # ---------------------------------------------------------------------- memory

    def remember(key: str, value: str) -> ToolResult:
        return ToolResult(memory.remember(str(key), str(value)))

    registry.register(
        Tool(
            name="remember",
            description=(
                "Store a fact that must survive after old images scroll out of context -- "
                "for example an object's measured position, or an approach that failed and "
                "should not be retried. Memory is shown to you on every step."
            ),
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
            },
            fn=remember,
        )
    )

    return registry


def tool_names(registry: Registry) -> list[str]:
    return [t.name for t in registry.available()]


def schemas(registry: Registry) -> list[dict[str, Any]]:
    return registry.schemas()
