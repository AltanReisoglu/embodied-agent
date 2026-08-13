"""Reasoner backed by Hugging Face Inference Providers.

The HF router is OpenAI-compatible, so we drive it with the official `openai` client and
only change `base_url`. That also means a local vLLM/SGLang server can be swapped in
later by changing one environment variable.

Two request modes exist because vision-capable models on HF vary in whether the serving
provider implements function calling:

* ``tools``       -- native `tools` / `tool_calls`, the preferred path;
* ``json_schema`` -- `response_format` constrains the model to emit the same call shape
                     as JSON, which we then feed into the identical dispatcher.

`scripts/check_model.py` decides which one a given model/provider actually supports.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Literal

from openai import OpenAI

from embodied_agent.reasoner.base import AgentStep, ToolCall
from embodied_agent.reasoner.thinking import extract_thinking

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

Mode = Literal["tools", "json_schema"]

#: Shape the model must emit in json_schema mode. Deliberately mirrors a tool call so
#: both modes converge on one dispatcher.
ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "Your reasoning about the current observation and what to do next.",
        },
        "actions": {
            "type": "array",
            "description": "Tools to call now. Empty when the task is finished.",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["thought", "actions"],
    "additionalProperties": False,
}


class HFReasoner:
    def __init__(
        self,
        model: str | None = None,
        *,
        mode: Mode = "tools",
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        max_tokens: int = 2048,
        temperature: float | None = 0.2,
        reasoning_effort: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        key = api_key or os.environ.get("HF_TOKEN")
        if not key:
            raise RuntimeError(
                "No Hugging Face token. Set HF_TOKEN in your environment or .env file "
                "(create one at https://huggingface.co/settings/tokens with the "
                "'Inference Providers' permission)."
            )
        self.model = model or os.environ.get("HF_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        self.provider = provider or os.environ.get("HF_PROVIDER") or None
        self.mode: Mode = mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

        # Routing to a specific provider is a URL suffix on the router.
        url = base_url or os.environ.get("HF_BASE_URL") or HF_ROUTER_BASE_URL
        if self.provider and url == HF_ROUTER_BASE_URL:
            url = f"https://router.huggingface.co/{self.provider}/v1"
        self.client = OpenAI(base_url=url, api_key=key, timeout=timeout)

    # ------------------------------------------------------------------ requests

    def _request_kwargs(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        if self.mode == "tools":
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "agent_step", "schema": ACTION_SCHEMA, "strict": True},
            }
        return kwargs

    def act(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentStep:
        response = self.client.chat.completions.create(
            messages=messages, **self._request_kwargs(tools)
        )
        choice = response.choices[0]
        message = choice.message
        thinking, text = extract_thinking(message, message.content)

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
            }

        if self.mode == "tools":
            calls = [
                ToolCall.parse(tc.id or str(uuid.uuid4()), tc.function.name, tc.function.arguments)
                for tc in (message.tool_calls or [])
            ]
            return AgentStep(thinking, text, calls, choice.finish_reason, usage)

        return self._parse_json_mode(text, thinking, choice.finish_reason, usage)

    @staticmethod
    def _parse_json_mode(
        text: str, thinking: str, finish_reason: str | None, usage: dict[str, int]
    ) -> AgentStep:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            # Surface as a no-tool step; the loop reports the problem back to the model.
            return AgentStep(
                thinking,
                f"[could not parse structured output: {exc}] {text}",
                [],
                finish_reason,
                usage,
            )

        thought = payload.get("thought", "") or thinking
        calls = []
        for item in payload.get("actions", []) or []:
            if not isinstance(item, dict) or "tool" not in item:
                continue
            args = item.get("arguments") or {}
            calls.append(
                ToolCall(
                    id=str(uuid.uuid4()),
                    name=str(item["tool"]),
                    arguments=args if isinstance(args, dict) else {},
                    raw_arguments=json.dumps(args),
                    parse_error=None if isinstance(args, dict) else "arguments must be an object",
                )
            )
        return AgentStep(thought, "", calls, finish_reason, usage)
