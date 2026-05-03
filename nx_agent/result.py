"""
Result objects returned by Agent.run() and Workflow.run().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Record of a single tool invocation inside a step."""
    name: str
    inputs: Dict[str, Any]
    output: Any
    error: Optional[str] = None
    duration_ms: float = 0.0

    def __repr__(self) -> str:
        status = "✓" if self.error is None else "✗"
        return f"ToolCall({status} {self.name}, duration={self.duration_ms:.0f}ms)"


@dataclass
class StepResult:
    """
    Trace of one agent's execution within a workflow.

    Attributes
    ----------
    agent_role  : Role label of the agent that ran this step.
    input       : The task / context this agent received.
    output      : The agent's final text output.
    tool_calls  : Ordered list of tool calls made during the step.
    duration_ms : Wall-clock time of the step in milliseconds.
    metadata    : Arbitrary key-value pairs for extensibility.
    """

    agent_role: str
    input: str
    output: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def succeeded(self) -> bool:
        return all(tc.error is None for tc in self.tool_calls)

    def summary(self) -> str:
        tools_used = ", ".join(tc.name for tc in self.tool_calls) or "none"
        return (
            f"[{self.agent_role}] tools={tools_used} "
            f"duration={self.duration_ms:.0f}ms succeeded={self.succeeded}"
        )

    def __repr__(self) -> str:
        return f"StepResult(agent={self.agent_role!r}, tools={len(self.tool_calls)})"


@dataclass
class WorkflowResult:
    """
    Final result returned by Workflow.run().

    Attributes
    ----------
    output  : The last agent's output — typically the final answer.
    steps   : Ordered trace of every StepResult.
    task    : The original task string.
    metadata: Arbitrary key-value pairs.
    """

    output: str
    steps: List[StepResult]
    task: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    @property
    def succeeded(self) -> bool:
        return all(s.succeeded for s in self.steps)

    @property
    def agents_used(self) -> List[str]:
        return [s.agent_role for s in self.steps]

    def pretty(self) -> str:
        lines = [
            f"Task    : {self.task}",
            f"Output  : {self.output}",
            f"Steps   : {len(self.steps)}",
            f"Duration: {self.total_duration_ms:.0f} ms",
            "",
        ]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  Step {i} — {step.summary()}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"WorkflowResult(steps={len(self.steps)}, "
            f"duration={self.total_duration_ms:.0f}ms)"
        )
