"""Tests for Agent."""

import pytest
from nx_agent import (
    Agent,
    AgentResult,
    MaxIterationsExceeded,
    RunConfig,
    system_prompt,
    tool,
)
from nx_agent.result import StepResult
from nx_agent.exceptions import AgentError


# ── fixtures ──────────────────────────────────────────────────────────────────

def stub_backend(system_prompt, user_message, tools, **kw):
    return f"Answer to: {user_message[:60]}"


def tool_calling_backend(system_prompt, user_message, tools, **kw):
    """Simulates an LLM that calls 'double' then returns a final answer."""
    if "Tool '" not in user_message and tools:
        return 'TOOL_CALL:double:{"number": 7}'
    return "The doubled value was returned by the tool."


@tool
def double(number: int) -> int:
    """Return double the input number."""
    return number * 2


# ── tests ─────────────────────────────────────────────────────────────────────

class TestAgent:
    def test_basic_run(self):
        agent = Agent(role="Tester", system_prompt="Test", llm_backend=stub_backend)
        result = agent.run("Hello")
        assert isinstance(result, AgentResult)
        assert isinstance(result, StepResult)
        assert "Hello" in result.output or "Answer" in result.output

    def test_step_result_fields(self):
        agent = Agent(role="Tester", system_prompt="Test", llm_backend=stub_backend)
        result = agent.run("Tell me something")
        assert result.agent_role == "Tester"
        assert result.role == "Tester"
        assert result.input == "Tell me something"
        assert result.instruction == "Tell me something"
        assert result.iterations == 1
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms >= 0

    def test_tool_execution(self):
        agent = Agent(
            role="Math",
            system_prompt="Compute",
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
            Agent(role="X", system_prompt="Y", tools=[not_a_tool])

    def test_memory_updated(self):
        agent = Agent(role="Memo", system_prompt="Remember", llm_backend=stub_backend)
        agent.run("Remember this")
        assert len(agent.memory.short) == 2  # user + agent turns

    def test_context_forwarded(self):
        received = {}

        def capture_backend(system_prompt, user_message, tools, **kw):
            received["msg"] = user_message
            return "ok"

        agent = Agent(role="R", system_prompt="G", llm_backend=capture_backend)
        agent.run("my task", context="upstream context")
        assert "## Instruction\nmy task" in received["msg"]
        assert "upstream context" in received["msg"]

    def test_system_prompt_added_to_backend_system_prompt(self):
        received = {}

        def capture_backend(system_prompt, user_message, tools, **kw):
            received["system"] = system_prompt
            return "ok"

        agent = Agent(
            role="R",
            system_prompt=system_prompt(
                goal="G",
                instructions="Use exactly one sentence.",
                output_format="Plain text.",
            ),
            llm_backend=capture_backend,
        )

        agent.run("my task")

        assert "You are a R." in received["system"]
        assert "Goal: G" in received["system"]
        assert "Instructions: Use exactly one sentence." in received["system"]
        assert "Output format: Plain text." in received["system"]

    def test_system_prompt_is_optional(self):
        received = {}

        def capture_backend(system_prompt, user_message, tools, **kw):
            received["system"] = system_prompt
            return "ok"

        agent = Agent(
            role="Support Writer",
            system_prompt="Give one direct recommendation.",
            llm_backend=capture_backend,
        )

        result = agent.run("help")

        assert result.agent_role == "Support Writer"
        assert "You are a Support Writer." in received["system"]
        assert "Give one direct recommendation." in received["system"]

    def test_run_config_forwarded_to_backend(self):
        received = {}

        def capture_backend(system_prompt, user_message, tools, **kw):
            received.update(kw)
            return "ok"

        agent = Agent(role="R", llm_backend=capture_backend)

        agent.run("my task", config=RunConfig(temperature=0.1, max_tokens=12))

        assert received == {"temperature": 0.1, "max_tokens": 12}

    def test_memory_context_precedes_external_context_and_instruction(self):
        received = {}

        def capture_backend(system_prompt, user_message, tools, **kw):
            received["msg"] = user_message
            return "ok"

        agent = Agent(role="R", llm_backend=capture_backend)
        agent.memory.long.remember("project", "NxAgent")
        agent.memory.short.add("user", "previous turn")

        result = agent.run("current task", context="retrieved context")

        assert received["msg"].index("## Persistent Memory") < received["msg"].index(
            "## Context"
        )
        assert received["msg"].index("## Context") < received["msg"].index("## Instruction")
        assert result.memory_snapshot["long"]["project"] == "NxAgent"

    def test_run_async_matches_run_contract(self):
        import asyncio

        agent = Agent(role="Async", llm_backend=stub_backend)

        result = asyncio.run(agent.run_async("hello"))

        assert isinstance(result, AgentResult)
        assert result.role == "Async"

    def test_max_iterations_exceeded_raises(self):
        def looping_backend(system_prompt, user_message, tools, **kw):
            return 'TOOL_CALL:double:{"number": 1}'

        agent = Agent(
            role="Looper",
            tools=[double],
            llm_backend=looping_backend,
            max_iterations=1,
        )

        with pytest.raises(MaxIterationsExceeded):
            agent.run("loop")
