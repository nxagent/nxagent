"""Custom exceptions for NxAgent."""


class NxAgentError(Exception):
    """Base exception for all NxAgent errors."""


class ToolExecutionError(NxAgentError):
    """Raised when a tool fails during execution."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"[Tool:{tool_name}] {message}")


class AgentError(NxAgentError):
    """Raised when an agent encounters a fatal error."""

    def __init__(self, agent_role: str, message: str):
        self.agent_role = agent_role
        super().__init__(f"[Agent:{agent_role}] {message}")


class WorkflowError(NxAgentError):
    """Raised when a workflow cannot complete."""


class RouterError(NxAgentError):
    """Raised when the router cannot select a next agent."""
