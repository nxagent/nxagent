"""
Agent — the single unit of work in NxAgent.

An Agent has a role label, a system prompt, and optional tools.
It is intentionally backend-agnostic: swap `llm_backend` to use any LLM.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from nx_agent.tool import ToolConfig, is_tool
from nx_agent.memory import Memory
from nx_agent.result import ToolCall
from nx_agent.prompt import system_prompt as compose_system_prompt
from nx_agent.resilience import OperationTimeoutError, RetryExhaustedError, RetryPolicy, execute_with_retry
from nx_agent.types import AgentResult, RunConfig
from nx_agent.exceptions import (
    AgentError,
    AgentTimeoutError,
    BackendError,
    MaxIterationsExceeded,
    ToolTimeoutError,
)


# ── default LLM backend (stub — replace with real implementation) ────────────

def _default_llm_backend(
    system_prompt: str,
    user_message: str,
    tools: List[dict],
    **kwargs,
) -> str:
    """
    Stub backend used when no real LLM is configured.

    Replace by passing ``llm_backend=your_function`` to the Agent, or by
    selecting a built-in provider factory from ``nx_agent.backends``.
    """
    tool_names = [t["name"] for t in tools]
    tools_txt = f" Available tools: {tool_names}." if tool_names else ""
    return (
        f"[NxAgent stub — no LLM configured]{tools_txt}\n"
        f"Role: {system_prompt[:120]}\n"
        f"Task: {user_message[:200]}"
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    A focused worker with a role label, system prompt, tools, and memory.

    Parameters
    ----------
    role        : Short label describing what this agent does; used in prompts
                  and traces. Defaults to "Assistant".
    system_prompt : Optional system instructions. Pass a plain string or use
                  ``nx_agent.system_prompt(...)`` to compose one.
    tools       : List of @tool-decorated callables this agent may invoke.
    llm_backend : Callable(system, user, tools, **kw) → str.
                  Defaults to the stub. Pass nx_agent.backends.openai_backend
                  / grok_backend / huggingface_backend / ollama_backend, or
                  your own.
    max_iterations : Safety cap on tool-call loops (default 10).
    memory_size : Number of short-term memory entries to keep.
    retry_policy : Optional retry configuration for backend calls.
    timeout     : Optional timeout in seconds for each backend call.
    verbose     : Print step traces to stdout.
    """

    def __init__(
        self,
        role: str = "Assistant",
        system_prompt: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        llm_backend: Optional[Callable] = None,
        max_iterations: int = 10,
        memory_size: int = 20,
        retry_policy: Optional[RetryPolicy] = None,
        timeout: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self.role = role
        self.system_prompt = system_prompt
        self.tools: List[Callable] = tools or []
        self._llm_backend = llm_backend or _default_llm_backend
        self.max_iterations = max_iterations
        self.memory = Memory(max_short_entries=memory_size)
        self.retry_policy = retry_policy or RetryPolicy()
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be > 0")
        self.timeout = timeout
        self.verbose = verbose

        # Validate tools
        for t in self.tools:
            if not is_tool(t):
                raise AgentError(
                    self.role,
                    f"'{getattr(t, '__name__', t)}' is not decorated with @tool. "
                    "Wrap it with @tool before passing to an Agent.",
                )

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        instruction: str,
        context: str = "",
        config: Optional[RunConfig] = None,
    ) -> AgentResult:
        """
        Execute the agent on *instruction*.

        Parameters
        ----------
        instruction : The natural-language task or question.
        context     : Optional caller-injected external knowledge.
        config      : Optional per-run backend settings.

        Returns
        -------
        AgentResult with output text and full tool-call trace.
        """
        t_start = time.perf_counter()
        tool_calls: List[ToolCall] = []
        backend_attempts: List[int] = []

        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(instruction, context)
        backend_kwargs = config.to_backend_kwargs() if config else {}

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Agent [{self.role}] ← {instruction[:100]}")

        # Agentic loop: call LLM → maybe invoke tools → repeat
        current_message = user_message
        final_output = ""
        iterations = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            tool_schemas = [t.schema.to_dict() for t in self.tools]

            try:
                raw_output, attempts = execute_with_retry(
                    lambda: self._llm_backend(
                        system_prompt=system_prompt,
                        user_message=current_message,
                        tools=tool_schemas,
                        **backend_kwargs,
                    ),
                    self.retry_policy,
                    self.timeout,
                )
            except RetryExhaustedError as exc:
                if isinstance(exc.last_error, OperationTimeoutError):
                    raise AgentTimeoutError(
                        self.role, exc.last_error.timeout, exc.attempts
                    ) from exc.last_error
                raise BackendError(
                    self.role, str(exc.last_error), exc.attempts
                ) from exc.last_error
            backend_attempts.append(attempts)

            # Check if output contains a tool call directive
            # (Format: TOOL_CALL:<name>:<json_args>)
            tool_directive = self._parse_tool_directive(raw_output)

            if tool_directive is None:
                # No tool call — LLM produced a final answer
                final_output = raw_output
                break

            # Execute the tool
            tool_name, tool_args = tool_directive
            tc = self._invoke_tool(tool_name, tool_args)
            tool_calls.append(tc)

            if self.verbose:
                print(f"  → Tool: {tc}")

            # Feed tool result back as next user turn
            tool_result = tc.output if tc.error is None else f"ERROR: {tc.error}"
            current_message = (
                f"Tool '{tool_name}' returned:\n{tool_result}\n\n"
                f"Continue completing the instruction: {instruction}"
            )

        else:
            raise MaxIterationsExceeded(self.role, self.max_iterations)

        duration_ms = (time.perf_counter() - t_start) * 1000

        # Update short-term memory
        self.memory.short.add("user", instruction)
        self.memory.short.add("agent", final_output)

        if self.verbose:
            print(f"Agent [{self.role}] → {final_output[:120]}")

        return AgentResult(
            agent_role=self.role,
            input=instruction,
            output=final_output,
            tool_calls=tool_calls,
            duration_ms=duration_ms,
            metadata={"backend_attempts": backend_attempts},
            iterations=iterations,
            memory_snapshot=self._memory_snapshot(),
        )

    async def run_async(
        self,
        instruction: str,
        context: str = "",
        config: Optional[RunConfig] = None,
    ) -> AgentResult:
        """Async variant of ``run()`` with the same contract."""
        return await asyncio.to_thread(self.run, instruction, context, config)

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        parts = [
            f"You are a {self.role}.",
        ]

        prompt = compose_system_prompt(self.system_prompt)
        if prompt:
            parts.append(prompt)

        if self.tools:
            tool_names = ", ".join(t.schema.name for t in self.tools)
            parts.append(
                f"\nYou have access to the following tools: {tool_names}.\n"
                "To call a tool, respond ONLY with a line in this exact format:\n"
                "TOOL_CALL:<tool_name>:{\"arg1\": \"value1\", ...}\n"
                "After the tool returns a result, continue normally."
            )

        return "\n".join(parts)

    def _build_user_message(self, instruction: str, context: str) -> str:
        parts = []
        memory_context = self.memory.build_context()
        if memory_context:
            parts.append(memory_context)
        if context:
            parts.append(f"## Context\n{context}")
        parts.append(f"## Instruction\n{instruction}")
        return "\n\n".join(parts)

    def _memory_snapshot(self) -> Dict[str, Any]:
        return {
            "short": self.memory.short.to_prompt_messages(),
            "long": self.memory.long.all_facts(),
        }

    def _parse_tool_directive(self, text: str):
        """
        Parse a TOOL_CALL directive from LLM output.

        Expected format (first matching line wins):
            TOOL_CALL:<name>:<json_args>

        Returns (name, args_dict) or None.
        """
        import json

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("TOOL_CALL:"):
                parts = line.split(":", 2)
                if len(parts) == 3:
                    _, name, args_raw = parts
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {"input": args_raw}
                    return name.strip(), args
        return None

    def _invoke_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolCall:
        """Find and call the named tool; return a ToolCall record."""
        t_start = time.perf_counter()

        # Lookup
        fn = next((t for t in self.tools if t.schema.name == tool_name), None)
        if fn is None:
            return ToolCall(
                name=tool_name,
                inputs=tool_args,
                output=None,
                error=f"Tool '{tool_name}' not found.",
                duration_ms=0.0,
            )

        try:
            config = getattr(fn, "config", ToolConfig())
            output, attempts = execute_with_retry(
                lambda: fn(**tool_args),
                config.retry_policy,
                config.timeout,
            )
            error = None
            timed_out = False
        except RetryExhaustedError as exc:
            output = None
            attempts = exc.attempts
            if isinstance(exc.last_error, OperationTimeoutError):
                error = str(ToolTimeoutError(tool_name, exc.last_error.timeout))
                timed_out = True
            else:
                error = str(exc.last_error)
                timed_out = False

        duration_ms = (time.perf_counter() - t_start) * 1000
        return ToolCall(
            name=tool_name,
            inputs=tool_args,
            output=output,
            error=error,
            duration_ms=duration_ms,
            attempts=attempts,
            timed_out=timed_out,
        )

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        tool_names = [t.schema.name for t in self.tools]
        return f"Agent(role={self.role!r}, tools={tool_names})"
