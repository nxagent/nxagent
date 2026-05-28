"""Public structured types used by NxAgent runtime APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from nx_agent.result import StepResult, ToolCall


@dataclass(frozen=True)
class RunConfig:
    """Per-run backend configuration.

    Values set to ``None`` are omitted when calling the backend.
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    def to_backend_kwargs(self) -> Dict[str, Any]:
        """Return only configured values for backend calls."""
        values = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass
class AgentResult(StepResult):
    """Structured result returned by ``Agent.run()``."""

    iterations: int = 0
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)

    @property
    def role(self) -> str:
        """Role label of the agent that produced this result."""
        return self.agent_role

    @property
    def instruction(self) -> str:
        """Runtime instruction originally passed to ``Agent.run()``."""
        return self.input


__all__ = ["AgentResult", "RunConfig", "ToolCall"]
