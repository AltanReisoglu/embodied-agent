"""Provider-neutral types for the reasoning layer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """One requested tool invocation.

    The wire format carries `arguments` as a JSON *string*, and models emit malformed
    JSON often enough that parsing must never raise: a bad call becomes a `parse_error`
    that we hand back to the model as a tool result so it can correct itself.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str | None = None

    @classmethod
    def parse(cls, call_id: str, name: str, raw_arguments: str) -> ToolCall:
        raw = (raw_arguments or "").strip()
        if not raw:
            return cls(call_id, name, {}, raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return cls(call_id, name, {}, raw, f"arguments are not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            return cls(call_id, name, {}, raw, "arguments must be a JSON object")
        return cls(call_id, name, parsed, raw)


@dataclass
class AgentStep:
    """What the model produced for one turn."""

    thinking: str
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class Reasoner(Protocol):
    def act(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentStep: ...
