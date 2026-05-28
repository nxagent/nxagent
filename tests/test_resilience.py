"""Tests for opt-in retry and timeout behavior."""

import time

import pytest

from nx_agent import Agent, RetryPolicy, tool
from nx_agent.exceptions import AgentTimeoutError, BackendError, ProviderRateLimitError


def after_tool_backend(system_prompt, user_message, tools, **kwargs):
    if "Tool '" not in user_message:
        return 'TOOL_CALL:flaky_lookup:{"query": "nx"}'
    return "completed after tool"


class TestBackendResilience:
    def test_backend_recovers_after_retry(self):
        calls = []

        def flaky_backend(system_prompt, user_message, tools, **kwargs):
            calls.append(user_message)
            if len(calls) == 1:
                raise RuntimeError("temporary")
            return "recovered"

        agent = Agent(
            role="Retry",
            system_prompt="Recover",
            llm_backend=flaky_backend,
            retry_policy=RetryPolicy(retries=1),
        )

        result = agent.run("try")

        assert result.output == "recovered"
        assert result.metadata["backend_attempts"] == [2]

    def test_backend_failure_reports_exhausted_attempts(self):
        def failing_backend(system_prompt, user_message, tools, **kwargs):
            raise RuntimeError("unavailable")

        agent = Agent(
            role="Retry",
            system_prompt="Fail clearly",
            llm_backend=failing_backend,
            retry_policy=RetryPolicy(retries=2),
        )

        with pytest.raises(BackendError) as exc_info:
            agent.run("try")

        assert exc_info.value.attempts == 3
        assert "unavailable" in str(exc_info.value)

    def test_backend_timeout_does_not_retry_by_default(self):
        calls = []

        def slow_backend(system_prompt, user_message, tools, **kwargs):
            calls.append("called")
            time.sleep(0.05)
            return "late"

        agent = Agent(
            role="Slow",
            system_prompt="Stop waiting",
            llm_backend=slow_backend,
            retry_policy=RetryPolicy(retries=2),
            timeout=0.001,
        )

        with pytest.raises(AgentTimeoutError) as exc_info:
            agent.run("wait")

        assert calls == ["called"]
        assert exc_info.value.attempts == 1

    def test_backend_can_retry_only_provider_rate_limits(self):
        calls = []

        def throttled_backend(system_prompt, user_message, tools, **kwargs):
            calls.append(user_message)
            if len(calls) == 1:
                raise ProviderRateLimitError("openai", "try again later")
            return "recovered"

        agent = Agent(
            role="Retry",
            system_prompt="Recover from throttling",
            llm_backend=throttled_backend,
            retry_policy=RetryPolicy(
                retries=1,
                retry_on=(ProviderRateLimitError,),
            ),
        )

        result = agent.run("try")

        assert result.output == "recovered"
        assert result.metadata["backend_attempts"] == [2]

    def test_selective_rate_limit_policy_does_not_retry_other_errors(self):
        calls = []

        def failing_backend(system_prompt, user_message, tools, **kwargs):
            calls.append(user_message)
            raise RuntimeError("not throttling")

        agent = Agent(
            role="Retry",
            system_prompt="Retry only throttling",
            llm_backend=failing_backend,
            retry_policy=RetryPolicy(
                retries=2,
                retry_on=(ProviderRateLimitError,),
            ),
        )

        with pytest.raises(BackendError) as exc_info:
            agent.run("try")

        assert calls == ["## Instruction\ntry"]
        assert exc_info.value.attempts == 1


class TestToolResilience:
    def test_tool_recovers_after_retry_and_records_attempts(self):
        calls = []

        @tool(retries=1)
        def flaky_lookup(query: str) -> str:
            """Look up a query that may fail once."""
            calls.append(query)
            if len(calls) == 1:
                raise RuntimeError("temporary")
            return "found"

        agent = Agent(
            role="Tools",
            system_prompt="Use a tool",
            tools=[flaky_lookup],
            llm_backend=after_tool_backend,
        )

        result = agent.run("look up nx")

        assert result.output == "completed after tool"
        assert result.tool_calls[0].attempts == 2
        assert result.tool_calls[0].output == "found"
        assert result.tool_calls[0].timed_out is False

    def test_tool_timeout_is_recorded_in_trace(self):
        @tool(timeout=0.001)
        def flaky_lookup(query: str) -> str:
            """Run too slowly for the configured timeout."""
            time.sleep(0.05)
            return "late"

        agent = Agent(
            role="Tools",
            system_prompt="Use a tool",
            tools=[flaky_lookup],
            llm_backend=after_tool_backend,
        )

        result = agent.run("look up nx")
        tool_call = result.tool_calls[0]

        assert tool_call.output is None
        assert tool_call.timed_out is True
        assert "Timed out" in tool_call.error
