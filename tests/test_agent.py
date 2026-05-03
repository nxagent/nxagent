"""Tests for Agent."""

import pytest
from nx_agent import Agent, tool
from nx_agent.result import StepResult
from nx_agent.exceptions import AgentError


# ── fixtures ──────────────────────────────────────────────────────────────────

def stub_backend(system_prompt, user_message, tools, **kw):
    return f"Answer to: {user_message[:60]}"


def tool_calling_backend(system_prompt, user_message, tools, **kw):
    """Simulates an LLM that calls 'double' then returns a final answer."""
    if "TOOL_CALL" not in user_message and tools:
        return 'TOOL_CALL:double:{"number": 7}'
    return "The doubled value was returned by the tool."


@tool
def double(number: int) -> int:
    """Return double the input number."""
    return number * 2


# ── tests ─────────────────────────────────────────────────────────────────────

class TestAgent:
    def test_basic_run(self):
        agent = Agent(role="Tester", goal="Test", llm_backend=stub_backend)
        result = agent.run("Hello")
        assert isinstance(result, StepResult)
        assert "Hello" in result.output or "Answer" in result.output

    def test_step_result_fields(self):
        agent = Agent(role="Tester", goal="Test", llm_backend=stub_backend)
        result = agent.run("Tell me something")
        assert result.agent_role == "Tester"
        assert result.input == "Tell me something"
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms >= 0

    def test_tool_execution(self):
        agent = Agent(
            role="Math",
            goal="Compute",
            tools=[double],
            llm_backend=tool_calling_backend,
        )
        result = agent.run("Double 7")
        assert len(result.tool_calls) >= 1
        tc = result.tool_calls[0]
        assert tc.name == "double"
        assert tc.output == 14

    def test_invalid_tool_raises(self):
        def not_a_tool(x):
            return x

        with pytest.raises(AgentError):
            Agent(role="X", goal="Y", tools=[not_a_tool])

    def test_memory_updated(self):
        agent = Agent(role="Memo", goal="Remember", llm_backend=stub_backend)
        agent.run("Remember this")
        assert len(agent.memory.short) == 2  # user + agent turns

    def test_context_forwarded(self):
        received = {}

        def capture_backend(system_prompt, user_message, tools, **kw):
            received["msg"] = user_message
            return "ok"

        agent = Agent(role="R", goal="G", llm_backend=capture_backend)
        agent.run("my task", context="upstream context")
        assert "upstream context" in received["msg"]
