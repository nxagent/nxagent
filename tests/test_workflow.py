"""Tests for Workflow and Router."""

import json

import pytest
from nx_agent import Agent, Workflow
from nx_agent.result import WorkflowResult
from nx_agent.router import Router
from nx_agent.exceptions import WorkflowError


def make_agent(role: str, response: str = None) -> Agent:
    resp = response or f"[{role} output]"

    def backend(system_prompt, user_message, tools, **kw):
        return resp

    return Agent(role=role, goal=f"Goal of {role}", llm_backend=backend)


def make_failing_agent(role: str) -> Agent:
    def backend(system_prompt, user_message, tools, **kw):
        raise RuntimeError("provider unavailable")

    return Agent(role=role, goal=f"Goal of {role}", llm_backend=backend)


class TestWorkflow:
    def test_single_agent(self):
        wf = Workflow(agents=[make_agent("A")])
        result = wf.run("task")
        assert isinstance(result, WorkflowResult)
        assert result.output == "[A output]"
        assert len(result.steps) == 1

    def test_sequential_two_agents(self):
        wf = Workflow(agents=[make_agent("A"), make_agent("B")])
        result = wf.run("task")
        assert len(result.steps) == 2
        assert result.steps[0].agent_role == "A"
        assert result.steps[1].agent_role == "B"
        # B receives A's output as context
        assert result.output == "[B output]"

    def test_output_chaining(self):
        received_ctx = {}

        def backend_b(system_prompt, user_message, tools, **kw):
            received_ctx["msg"] = user_message
            return "B done"

        agent_a = make_agent("A", "A result")
        agent_b = Agent(role="B", goal="chain", llm_backend=backend_b)

        Workflow(agents=[agent_a, agent_b]).run("chain test")
        assert "A result" in received_ctx["msg"]

    def test_empty_agents_raises(self):
        with pytest.raises(WorkflowError):
            Workflow(agents=[])

    def test_router_string(self):
        wf = Workflow(agents=[make_agent("A")], router="sequential")
        result = wf.run("task")
        assert result.output == "[A output]"

    def test_existing_router_override_does_not_require_partial_keyword(self):
        class ExistingRouter(Router):
            def run(self, task, agents, history):
                return [agents[0].run(task)]

        result = Workflow(agents=[make_agent("A")], router=ExistingRouter()).run("task")
        assert result.output == "[A output]"

    def test_parallel_router(self):
        wf = Workflow(
            agents=[make_agent("A"), make_agent("B")],
            router=Router(strategy="parallel"),
        )
        result = wf.run("task")
        assert len(result.steps) == 2
        # Order may differ in parallel, but both agents ran
        roles = {s.agent_role for s in result.steps}
        assert roles == {"A", "B"}

    def test_hooks_called(self):
        calls = []
        wf = Workflow(agents=[make_agent("A")])
        wf.on("on_workflow_start", lambda task: calls.append("start"))
        wf.on("on_workflow_end", lambda r: calls.append("end"))
        wf.run("task")
        assert "start" in calls
        assert "end" in calls

    def test_metadata_in_result(self):
        wf = Workflow(agents=[make_agent("A")])
        result = wf.run("task", extra="hello")
        assert result.metadata["extra"] == "hello"

    def test_workflow_result_pretty(self):
        wf = Workflow(agents=[make_agent("A"), make_agent("B")])
        result = wf.run("task")
        pretty = result.pretty()
        assert "Task" in pretty
        assert "Output" in pretty
        assert "Step 1" in pretty

    def test_workflow_result_serialization(self):
        result = Workflow(agents=[make_agent("A")]).run("task", request_id="one")

        as_dict = result.to_dict()
        as_json = json.loads(result.to_json())

        assert as_dict["steps"][0]["agent_role"] == "A"
        assert as_json["metadata"]["request_id"] == "one"

    def test_agent_failure_raises_by_default(self):
        with pytest.raises(WorkflowError, match="provider unavailable"):
            Workflow(agents=[make_agent("A"), make_failing_agent("B")]).run("task")

    def test_return_partial_records_failure_and_stops_sequential_route(self):
        unused = []

        def never_run(system_prompt, user_message, tools, **kw):
            unused.append(True)
            return "unexpected"

        result = Workflow(
            agents=[
                make_agent("A", "useful partial output"),
                make_failing_agent("B"),
                Agent(role="C", goal="should not run", llm_backend=never_run),
            ],
            return_partial=True,
        ).run("task")

        assert result.output == "useful partial output"
        assert result.agents_used == ["A", "B"]
        assert not result.succeeded
        assert "BackendError" in result.steps[-1].error
        assert "provider unavailable" in result.steps[-1].error
        assert result.steps[-1].metadata["exception_type"] == "BackendError"
        assert result.to_dict()["steps"][-1]["error"] == result.steps[-1].error
        assert not unused

    def test_parallel_return_partial_records_failed_agent(self):
        result = Workflow(
            agents=[make_agent("A"), make_failing_agent("B")],
            router="parallel",
            return_partial=True,
        ).run("task")

        assert len(result.steps) == 2
        assert not result.succeeded
        failed_steps = [step for step in result.steps if step.error is not None]
        assert len(failed_steps) == 1
        assert failed_steps[0].agent_role == "B"
