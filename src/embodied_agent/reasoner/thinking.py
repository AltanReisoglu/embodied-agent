"""Recovering the model's reasoning trace, whichever way the provider exposes it.

Three routes exist in the wild and we support all of them behind one call:

1. the provider returns a dedicated field (`reasoning_content` / `reasoning`), which is
   what `reasoning_effort` turns on where it is supported;
2. the model writes `<think>...</think>` inline in the content (the Qwen/DeepSeek-style
   convention), in which case it must be stripped before the text is used;
3. neither -- and the loop falls back to requiring a `think` tool call.
"""

from __future__ import annotations

import re
from typing import Any

_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
#: A truncated response can open a think block and never close it.
_UNCLOSED_THINK = re.compile(r"<(think|thinking|reasoning)>(.*)$", re.DOTALL | re.IGNORECASE)

_REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")


def split_thinking(content: str | None) -> tuple[str, str]:
    """Return (thinking, visible_text) for inline `<think>` conventions."""
    if not content:
        return "", ""

    blocks = [match.group(2).strip() for match in _THINK_BLOCK.finditer(content)]
    text = _THINK_BLOCK.sub("", content).strip()

    if not blocks:
        unclosed = _UNCLOSED_THINK.search(text)
        if unclosed:
            return unclosed.group(2).strip(), text[: unclosed.start()].strip()

    return "\n\n".join(blocks), text


def extract_thinking(message: Any, content: str | None) -> tuple[str, str]:
    """Prefer a provider-supplied reasoning field, else fall back to inline parsing."""
    for field in _REASONING_FIELDS:
        value = getattr(message, field, None)
        if value is None and isinstance(message, dict):
            value = message.get(field)
        if isinstance(value, str) and value.strip():
            _, text = split_thinking(content)
            return value.strip(), text or (content or "").strip()

    return split_thinking(content)
