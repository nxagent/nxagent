"""Custom exceptions for NxAgent."""


class NxAgentError(Exception):
    """Base exception for all NxAgent errors."""


class ToolExecutionError(NxAgentError):
    """Raised when a tool fails during execution."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"[Tool:{tool_name}] {message}")


class ToolTimeoutError(ToolExecutionError):
    """Recorded when a tool does not finish before its configured timeout."""

    def __init__(self, tool_name: str, timeout: float):
        self.timeout = timeout
        super().__init__(tool_name, f"Timed out after {timeout}s.")


class AgentError(NxAgentError):
    """Raised when an agent encounters a fatal error."""

    def __init__(self, agent_role: str, message: str):
        self.agent_role = agent_role
        super().__init__(f"[Agent:{agent_role}] {message}")


class BackendError(AgentError):
    """Raised when an agent backend fails after configured retries."""

    def __init__(self, agent_role: str, message: str, attempts: int):
        self.attempts = attempts
        super().__init__(
            agent_role,
            f"Backend failed after {attempts} attempt(s): {message}",
        )


class AgentTimeoutError(BackendError):
    """Raised when an agent backend exceeds its configured timeout."""

    def __init__(self, agent_role: str, timeout: float, attempts: int = 1):
        self.timeout = timeout
        self.attempts = attempts
        AgentError.__init__(
            self,
            agent_role,
            f"Backend timed out after {timeout}s ({attempts} attempt(s)).",
        )


class WorkflowError(NxAgentError):
    """Raised when a workflow cannot complete."""


class RouterError(NxAgentError):
    """Raised when the router cannot select a next agent."""
