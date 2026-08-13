"""Context must stay bounded, and old frames must degrade into text rather than vanish."""

from __future__ import annotations

import numpy as np

from embodied_agent.history import ELIDED, History
from embodied_agent.memory import Memory


def frame(value: int) -> np.ndarray:
    return np.full((32, 32, 3), value, dtype=np.uint8)


def test_only_the_newest_images_survive():
    history = History("sys", image_window=2)
    for i in range(5):
        history.add_user(f"obs {i}", frame(i * 40))

    assert history.image_count() == 2

    texts = [
        block["text"]
        for message in history.messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "text"
    ]
    assert sum(1 for t in texts if t == ELIDED) == 3, "dropped frames must leave a marker"


def test_text_of_old_turns_is_preserved():
    history = History("sys", image_window=1)
    for i in range(4):
        history.add_user(f"observation number {i}", frame(i))

    joined = "".join(
        block["text"]
        for message in history.messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "text"
    )
    for i in range(4):
        assert f"observation number {i}" in joined


def test_memory_survives_frame_eviction():
    """The whole point of the external store: what a dropped frame taught is still there."""
    memory = Memory()
    memory.remember("red_cube", "measured at (0.13, 0.18, 0.42) in frame 0")

    history = History("sys", image_window=1)
    for i in range(4):
        history.add_user(f"obs {i}\n{memory.render()}", frame(i))

    assert history.image_count() == 1
    assert "0.13, 0.18, 0.42" in history.messages[-1]["content"][0]["text"]


def test_assistant_tool_calls_are_echoed_for_the_next_request():
    from embodied_agent.reasoner.base import AgentStep, ToolCall

    history = History("sys")
    step = AgentStep("thought", "", [ToolCall("call_1", "grasp", {}, "{}")], "tool_calls")
    history.add_assistant(step)

    message = history.messages[-1]
    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["function"]["name"] == "grasp"
