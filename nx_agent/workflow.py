"""
Workflow — the top-level orchestrator.

Ties together a list of Agents and a Router into a runnable pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Union

from nx_agent.agent import Agent
from nx_agent.router import Router
from nx_agent.result import WorkflowResult, StepResult
from nx_agent.exceptions import WorkflowError


class Workflow:
    """
    Orchestrate a list of Agents through a configurable Router.

    Quick-start
    -----------
        workflow = Workflow(agents=[researcher, writer])
        result = workflow.run("Summarise agent observability.")

        print(result.output)
        print(result.steps)

    Parameters
    ----------
    agents  : Ordered list of Agent instances.
    router  : A Router instance OR a strategy string ("sequential",
              "parallel", "llm") OR a callable for custom routing.
              Defaults to sequential.
    name    : Human-readable label for logging / tracing.
    verbose : Print per-step summaries to stdout.
    hooks   : Dict of lifecycle hooks:
                  "on_step_start"  (agent, task)        → None
                  "on_step_end"    (agent, StepResult)  → None
                  "on_workflow_end"(WorkflowResult)      → None
    """

    def __init__(
        self,
        agents: List[Agent],
        router: Union[Router, str, Callable, None] = None,
        name: str = "workflow",
        verbose: bool = False,
        hooks: Optional[Dict[str, Callable]] = None,
    ) -> None:
        if not agents:
            raise WorkflowError("A Workflow must contain at least one Agent.")

        self.agents = agents
        self.name = name
        self.verbose = verbose
        self.hooks: Dict[str, Callable] = hooks or {}

        # Resolve router
        if router is None:
            self.router = Router(strategy="sequential")
        elif isinstance(router, str):
            self.router = Router(strategy=router)
        elif callable(router) and not isinstance(router, Router):
            self.router = Router(strategy=router)
        elif isinstance(router, Router):
            self.router = router
        else:
            raise WorkflowError(f"Invalid router type: {type(router)}")

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, task: str, **metadata: Any) -> WorkflowResult:
        """
        Execute the workflow on *task*.

        Parameters
        ----------
        task     : The natural-language task to solve.
        metadata : Extra key-value pairs stored in WorkflowResult.metadata.

        Returns
        -------
        WorkflowResult with .output (str) and .steps (List[StepResult]).
        """
        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"Workflow [{self.name}] → {task}")
            print(f"Agents: {[a.role for a in self.agents]}")
            print(f"Router: {self.router}")
            print(f"{'#'*60}")

        t_start = time.perf_counter()
        history: List[StepResult] = []

        # Fire hook
        self._fire("on_workflow_start", task)

        try:
            steps = self.router.run(task=task, agents=self.agents, history=history)
        except Exception as exc:
            raise WorkflowError(str(exc)) from exc

        total_ms = (time.perf_counter() - t_start) * 1000

        final_output = steps[-1].output if steps else ""

        result = WorkflowResult(
            output=final_output,
            steps=steps,
            task=task,
            metadata={
                "workflow_name": self.name,
                "total_duration_ms": total_ms,
                **metadata,
            },
        )

        if self.verbose:
            print(f"\nWorkflow complete in {total_ms:.0f} ms")
            print(result.pretty())

        self._fire("on_workflow_end", result)
        return result

    # ── lifecycle hooks ───────────────────────────────────────────────────────

    def on(self, event: str, fn: Callable) -> "Workflow":
        """
        Register a lifecycle hook and return *self* for chaining.

        Events: "on_workflow_start", "on_workflow_end",
                "on_step_start",     "on_step_end"
        """
        self.hooks[event] = fn
        return self

    def _fire(self, event: str, *args) -> None:
        fn = self.hooks.get(event)
        if fn is not None:
            try:
                fn(*args)
            except Exception:
                pass  # hooks must never crash the workflow

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        roles = [a.role for a in self.agents]
        return f"Workflow(name={self.name!r}, agents={roles}, router={self.router})"
