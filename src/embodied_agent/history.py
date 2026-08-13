"""Conversation history with a bounded number of images.

Every step appends a frame, so context grows fast and the oldest frames are the least
useful. We keep the most recent `image_window` images and replace older ones with a
placeholder. What was learned from a dropped frame is not lost, because it lives in the
Memory block that is re-rendered on every step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from embodied_agent.perception.render import to_data_uri

ELIDED = "[an earlier frame was dropped to save context -- see the MEMORY block]"


class History:
    def __init__(self, system_prompt: str, *, image_window: int = 3) -> None:
        self.image_window = image_window
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # ------------------------------------------------------------------- appending

    def add_user(self, text: str, image: np.ndarray | None = None) -> None:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": to_data_uri(image)}})
        self.messages.append({"role": "user", "content": content})
        self._prune_images()

    def add_assistant(self, step: Any) -> None:
        """Echo the assistant turn back, including tool calls, so the API sees a
        well-formed transcript on the next request."""
        message: dict[str, Any] = {"role": "assistant", "content": step.text or None}
        if step.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                }
                for call in step.tool_calls
            ]
        elif not step.text:
            message["content"] = ""
        self.messages.append(message)

    def add_tool_result(self, call_id: str, text: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    def add_assistant_text(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    # -------------------------------------------------------------------- trimming

    def _prune_images(self) -> None:
        """Keep only the newest `image_window` images; blank out the rest in place."""
        seen = 0
        for message in reversed(self.messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in reversed(content):
                if block.get("type") != "image_url":
                    continue
                seen += 1
                if seen > self.image_window and block["image_url"].get("url", "").startswith(
                    "data:"
                ):
                    block.clear()
                    block.update({"type": "text", "text": ELIDED})

    def image_count(self) -> int:
        return sum(
            1
            for message in self.messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if block.get("type") == "image_url"
        )
