# NxAgent

**A production-grade agentic framework for orchestrating multi-agent workflows, tool execution, and intelligent task routing.**

[![PyPI](https://img.shields.io/pypi/v/nx-agent)](https://pypi.org/project/nx-agent/)
[![Python](https://img.shields.io/pypi/pyversions/nx-agent)](https://pypi.org/project/nx-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why NxAgent?

| Feature | NxAgent |
|---|---|
| Zero hard dependencies | Bring your own LLM |
| Pluggable backends | OpenAI · Anthropic · Hugging Face · custom |
| `@tool` decorator | Type hints → JSON schema automatically |
| Multi-agent routing | Sequential · Parallel · LLM-driven · custom |
| Built-in memory | Short-term context + long-term key-value memory |
| Full traceability | Every step, tool call, and timing captured |
| Lifecycle hooks | `on_workflow_start/end`, `on_step_start/end` |

---

## Installation

```bash
pip install nx-agent               # core (no LLM deps)

# Optional — pick your LLM backend
pip install nx-agent[openai]       # OpenAI
pip install nx-agent[anthropic]    # Anthropic Claude
pip install nx-agent[all]          # everything
```

---

## Quick Start

### 01 — Single Agent

```python
from nx_agent import Agent
from nx_agent.backends import openai_backend

agent = Agent(
    role="Researcher",
    goal="Answer clearly in one paragraph",
    llm_backend=openai_backend(),   # reads OPENAI_API_KEY from env
)

result = agent.run("What is NxAgent?")
print(result.output)
```

### 02 — Attach a Tool

```python
from nx_agent import Agent, tool
from nx_agent.backends import anthropic_backend

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    ...  # your implementation

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate source material",
    tools=[web_search],
    llm_backend=anthropic_backend(),
)
```

### 03 — Multi-Agent Workflow

```python
from nx_agent import Agent, Workflow, tool
from nx_agent.backends import openai_backend

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    ...

researcher = Agent(
    role="Research Analyst",
    goal="Collect accurate source material",
    tools=[web_search],
    llm_backend=openai_backend(),
)

writer = Agent(
    role="Technical Writer",
    goal="Turn findings into a clear brief",
    llm_backend=openai_backend(),
)

workflow = Workflow(agents=[researcher, writer])
result = workflow.run("Create a short brief on agent observability.")

print(result.output)        # final answer
print(result.steps)         # full trace
print(result.pretty())      # human-readable summary
```

---

## Routing Strategies

```python
from nx_agent import Workflow
from nx_agent.router import Router

# Sequential (default) — output of each agent feeds into the next
wf = Workflow(agents=[a, b, c])

# Parallel — all agents run concurrently on the same task
wf = Workflow(agents=[a, b], router=Router("parallel"))

# LLM-driven — the model picks the next agent at each step
wf = Workflow(agents=[a, b, c], router=Router("llm", llm_backend=my_llm))

# Custom callable
def my_router(task, agents, history):
    return agents[0] if not history else None  # return None to stop

wf = Workflow(agents=[a, b], router=my_router)
```

---

## Memory

Each agent can carry memory with short-term context and long-term key-value storage:

```python
agent.memory.long.remember("user_name", "Alice")  # long-term memory
agent.memory.short.add("user", "Hello")           # short-term context

# Full context string ready for injection
print(agent.memory.build_context())
```

---

## Lifecycle hooks

```python
workflow = (
    Workflow(agents=[researcher, writer])
    .on("on_workflow_start", lambda task: print(f"Starting: {task}"))
    .on("on_step_end",       lambda step: print(f"Done: {step.summary()}"))
    .on("on_workflow_end",   lambda r: save_to_db(r))
)
```

---

## Result Object

```python
result = workflow.run("my task")

result.output            # str  — final answer
result.steps             # List[StepResult]
result.total_duration_ms # float
result.succeeded         # bool
result.agents_used       # ["Research Analyst", "Technical Writer"]

step = result.steps[0]
step.agent_role          # "Research Analyst"
step.tool_calls          # List[ToolCall]
step.duration_ms         # float
step.succeeded           # bool
step.summary()           # "Research Analyst tools=web_search duration=234ms"
```

---

## License

NxAgent is licensed under the MIT License.