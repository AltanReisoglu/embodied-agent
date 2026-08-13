"""Lossless external memory for the agent.

VISTA (vista-research.github.io) makes the point that a VLM's native memory is the KV
cache: the model attends over it only implicitly, and it is compressed, lossy and
short-horizon. Once we start dropping old frames to keep the context bounded, anything
the agent learned from those frames is gone unless it lives somewhere else.

This is that somewhere else: a small key-value store the agent writes explicitly and
which is re-rendered into the context verbatim on every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Memory:
    facts: dict[str, str] = field(default_factory=dict)
    attempts: list[str] = field(default_factory=list)
    max_attempts_shown: int = 8

    def remember(self, key: str, value: str) -> str:
        key = key.strip()
        if not key:
            return "error: key must not be empty"
        previous = self.facts.get(key)
        self.facts[key] = value.strip()
        if previous is None:
            return f"remembered {key!r}"
        return f"updated {key!r} (was: {previous!r})"

    def forget(self, key: str) -> str:
        if self.facts.pop(key, None) is None:
            return f"no such key {key!r}"
        return f"forgot {key!r}"

    def log_attempt(self, line: str) -> None:
        """Record an executed action. Failed attempts are the expensive lesson to lose."""
        self.attempts.append(line)

    def render(self) -> str:
        lines = ["=== MEMORY (persists after frames are dropped) ==="]
        if self.facts:
            for key, value in self.facts.items():
                lines.append(f"{key}: {value}")
        else:
            lines.append("(no facts recorded yet)")

        if self.attempts:
            shown = self.attempts[-self.max_attempts_shown :]
            elided = len(self.attempts) - len(shown)
            lines.append("")
            header = "action history"
            if elided:
                header += f" (showing last {len(shown)} of {len(self.attempts)})"
            lines.append(f"{header}:")
            lines.extend(f"  {i + 1 + elided}. {line}" for i, line in enumerate(shown))
        lines.append("=== END MEMORY ===")
        return "\n".join(lines)
