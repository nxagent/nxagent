"""
In-process memory for agents.

Two tiers
---------
ShortTermMemory  – a fixed-size ring buffer of recent messages (conversation context).
LongTermMemory   – a simple key-value store for facts / summaries the agent should
                   always remember, regardless of context window.

Both are attached to an Agent via ``agent.memory``.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class MemoryEntry:
    role: str           # "user" | "agent" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"[{self.role}] {self.content[:80]!r}"


class ShortTermMemory:
    """
    Fixed-size sliding window of recent conversation turns.

    Parameters
    ----------
    max_entries : Maximum number of entries to keep (FIFO eviction).
    """

    def __init__(self, max_entries: int = 20) -> None:
        self._buf: Deque[MemoryEntry] = deque(maxlen=max_entries)

    def add(self, role: str, content: str, **meta) -> None:
        self._buf.append(MemoryEntry(role=role, content=content, metadata=meta))

    def get_all(self) -> List[MemoryEntry]:
        return list(self._buf)

    def to_prompt_messages(self) -> List[Dict[str, str]]:
        """Return as a list of {"role": ..., "content": ...} dicts."""
        return [{"role": e.role, "content": e.content} for e in self._buf]

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return f"ShortTermMemory(entries={len(self)})"


class LongTermMemory:
    """
    Simple key-value fact store.

    Facts are always injected into the agent's system prompt so they survive
    context window truncation.
    """

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def remember(self, key: str, value: str) -> None:
        """Store a fact under *key*. Overwrites existing values."""
        self._store[key] = value

    def recall(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def forget(self, key: str) -> None:
        self._store.pop(key, None)

    def all_facts(self) -> Dict[str, str]:
        return dict(self._store)

    def to_prompt_block(self) -> str:
        """Render as a plain-text block for injection into system prompts."""
        if not self._store:
            return ""
        lines = ["## Persistent Memory"]
        for k, v in self._store.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"LongTermMemory(facts={len(self)})"


class Memory:
    """
    Unified memory facade attached to each Agent.

    Usage
    -----
        agent.memory.short.add("user", "What is NxAgent?")
        agent.memory.long.remember("company", "Acme Corp")
    """

    def __init__(self, max_short_entries: int = 20) -> None:
        self.short = ShortTermMemory(max_entries=max_short_entries)
        self.long = LongTermMemory()

    def reset(self) -> None:
        """Clear short-term memory (long-term is preserved)."""
        self.short.clear()

    def build_context(self) -> str:
        """
        Build a single string that an LLM backend can use as extra context.
        Combines long-term facts + recent turns.
        """
        parts: List[str] = []
        lt = self.long.to_prompt_block()
        if lt:
            parts.append(lt)
        recent = self.short.get_all()
        if recent:
            parts.append("## Recent Conversation")
            for entry in recent:
                parts.append(f"{entry.role}: {entry.content}")
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return f"Memory(short={self.short}, long={self.long})"
