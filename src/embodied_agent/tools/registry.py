"""Tool schemas and dispatch.

Two properties matter here beyond plumbing:

* ``mutates_world`` -- after such a tool runs, the frame in context is stale, so the loop
  must re-observe and inject a fresh image plus body state.
* ``image`` on a result -- a `role:"tool"` message is text-only on this API, so a tool
  that produces a picture returns it here and the loop delivers it as a user message.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from embodied_agent.reasoner.base import ToolCall


@dataclass
class ToolResult:
    text: str
    is_error: bool = False
    mutates_world: bool = False
    image: np.ndarray | None = None
    image_caption: str = ""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., ToolResult]
    mutates_world: bool = False
    privileged: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)
    allow_privileged: bool = False

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def available(self) -> list[Tool]:
        return [t for t in self.tools.values() if self.allow_privileged or not t.privileged]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.available()]

    def describe_for_prompt(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.available())

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Execute a call, converting every foreseeable failure into a message the model
        can act on. Nothing here may raise: a crashed dispatch kills the episode, while a
        returned error lets the model correct itself on the next turn."""
        if call.parse_error:
            return ToolResult(
                f"error: {call.parse_error}. You sent: {call.raw_arguments[:200]!r}. "
                f"Re-issue the call with valid JSON arguments.",
                is_error=True,
            )

        tool = self.tools.get(call.name)
        if tool is None:
            known = ", ".join(sorted(t.name for t in self.available()))
            return ToolResult(
                f"error: no tool named {call.name!r}. Available tools: {known}.", is_error=True
            )
        if tool.privileged and not self.allow_privileged:
            return ToolResult(
                f"error: {call.name!r} is disabled in this configuration.", is_error=True
            )

        required = tool.parameters.get("required", [])
        missing = [key for key in required if key not in call.arguments]
        if missing:
            return ToolResult(
                f"error: {call.name} is missing required argument(s): {', '.join(missing)}.",
                is_error=True,
            )

        try:
            result = tool.fn(**call.arguments)
        except TypeError as exc:
            return ToolResult(f"error: bad arguments for {call.name}: {exc}", is_error=True)
        except Exception as exc:  # a tool bug must not end the episode
            return ToolResult(f"error: {call.name} failed: {type(exc).__name__}: {exc}", True)

        # Trust the tool's declaration over the per-call return value.
        result.mutates_world = result.mutates_world or tool.mutates_world
        return result
