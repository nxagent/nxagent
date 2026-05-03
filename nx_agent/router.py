"""
Router — decides which Agent should handle the next step in a Workflow.

Three built-in strategies
--------------------------
sequential   (default) — agents run in the order they were declared.
parallel                — all agents run concurrently (ThreadPoolExecutor).
llm                     — an LLM call decides the next agent given the task.

You can also pass a plain Python callable for full custom routing:

    def my_router(task: str, agents: List[Agent], history: List[StepResult]) -> Agent:
        ...

    workflow = Workflow(agents=[a, b], router=my_router)
"""

from __future__ import annotations

import concurrent.futures
from typing import Callable, List, Optional

from nx_agent.result import StepResult
from nx_agent.exceptions import RouterError


class Router:
    """
    Configurable routing strategy for Workflow.

    Parameters
    ----------
    strategy : "sequential" | "parallel" | "llm" | Callable
    llm_backend : Required only when strategy="llm".
    max_workers : Thread pool size for parallel strategy (default 4).
    """

    STRATEGIES = {"sequential", "parallel", "llm"}

    def __init__(
        self,
        strategy: str | Callable = "sequential",
        llm_backend: Optional[Callable] = None,
        max_workers: int = 4,
    ) -> None:
        if callable(strategy):
            self._custom_fn = strategy
            self._strategy = "custom"
        elif strategy in self.STRATEGIES:
            self._strategy = strategy
            self._custom_fn = None
        else:
            raise RouterError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from {self.STRATEGIES} or pass a callable."
            )

        self._llm_backend = llm_backend
        self._max_workers = max_workers

    # ── dispatch ──────────────────────────────────────────────────────────────

    def run(self, task: str, agents: list, history: List[StepResult]) -> List[StepResult]:
        """
        Route *task* through *agents* using the configured strategy.

        Returns a list of StepResult (one per agent that ran).
        """
        if self._strategy == "custom":
            return self._run_custom(task, agents, history)
        if self._strategy == "sequential":
            return self._run_sequential(task, agents, history)
        if self._strategy == "parallel":
            return self._run_parallel(task, agents)
        if self._strategy == "llm":
            return self._run_llm(task, agents, history)
        raise RouterError(f"Unhandled strategy: {self._strategy}")

    # ── strategies ────────────────────────────────────────────────────────────

    def _run_sequential(
        self, task: str, agents: list, history: List[StepResult]
    ) -> List[StepResult]:
        """
        Pass output of each agent as context to the next.
        """
        results: List[StepResult] = []
        context = ""
        for agent in agents:
            step = agent.run(task=task, context=context)
            results.append(step)
            context = step.output  # chain outputs
        return results

    def _run_parallel(self, task: str, agents: list) -> List[StepResult]:
        """
        All agents receive the same task simultaneously.
        Results are ordered by completion time.
        """
        results: List[StepResult] = []

        def _run_one(agent):
            return agent.run(task=task)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {ex.submit(_run_one, a): a for a in agents}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        return results

    def _run_llm(
        self, task: str, agents: list, history: List[StepResult]
    ) -> List[StepResult]:
        """
        Ask an LLM which agent should run next; repeat until done.
        """
        if self._llm_backend is None:
            raise RouterError("strategy='llm' requires llm_backend to be set on Router.")

        agent_descriptions = "\n".join(
            f"{i}: role={a.role!r}, goal={a.goal!r}"
            for i, a in enumerate(agents)
        )

        results: List[StepResult] = []
        context = ""
        used: set = set()

        while len(used) < len(agents):
            history_txt = "\n".join(
                f"{s.agent_role}: {s.output[:100]}" for s in results
            )
            prompt = (
                f"Task: {task}\n\n"
                f"Available agents:\n{agent_descriptions}\n\n"
                f"History so far:\n{history_txt or 'None'}\n\n"
                f"Agents already used: {sorted(used)}\n\n"
                "Reply with ONLY the integer index of the next agent to run, "
                "or 'DONE' if the task is complete."
            )

            decision = self._llm_backend(
                system_prompt="You are a workflow orchestrator.",
                user_message=prompt,
                tools=[],
            ).strip()

            if decision.upper() == "DONE":
                break

            try:
                idx = int(decision)
            except ValueError:
                break

            if idx < 0 or idx >= len(agents) or idx in used:
                break

            agent = agents[idx]
            step = agent.run(task=task, context=context)
            results.append(step)
            context = step.output
            used.add(idx)

        return results

    def _run_custom(
        self, task: str, agents: list, history: List[StepResult]
    ) -> List[StepResult]:
        """Delegate to the user-supplied routing function."""
        results: List[StepResult] = []
        context = ""
        while True:
            next_agent = self._custom_fn(task, agents, results)  # type: ignore[misc]
            if next_agent is None:
                break
            step = next_agent.run(task=task, context=context)
            results.append(step)
            context = step.output
        return results

    def __repr__(self) -> str:
        return f"Router(strategy={self._strategy!r})"
