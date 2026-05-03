# NxAgent

**Production-grade agentic framework for orchestrating multi-agent workflows, tool execution, and intelligent task routing.**

[![PyPI](https://img.shields.io/pypi/v/nx-agent)](https://pypi.org/project/nx-agent/)
[![Python](https://img.shields.io/pypi/pyversions/nx-agent)](https://pypi.org/project/nx-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why NxAgent?

| Feature | NxAgent |
|---|---|
| Zero hard dependencies | ✅ Bring your own LLM |
| Pluggable backends | OpenAI · Anthropic · HuggingFace · custom |
| `@tool` decorator | Type hints → JSON schema automatically |
| Multi-agent routing | Sequential · Parallel · LLM-driven · custom |
| Built-in memory | Short-term (ring buffer) + long-term (key-value) |
| Full traceability | Every step, tool call, and timing captured |
| Lifecycle hooks | `on_workflow_start/end`, `on_step_start/end` |

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install nx-agent               # core (no LLM deps)

# Optional — pick your LLM backend
pip install nx-agent[openai]       # OpenAI
pip install nx-agent[anthropic]    # Anthropic Claude
pip install nx-agent[all]          # everything
```

---

## Quickstart

### 01 — Single agent

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

### 02 — Attach a tool

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

### 03 — Multi-agent workflow

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

## Routing strategies

```python
from nx_agent import Workflow
from nx_agent.router import Router

# Sequential (default) — output of each agent feeds into the next
wf = Workflow(agents=[a, b, c])

# Parallel — all agents run concurrently on the same task
wf = Workflow(agents=[a, b], router=Router("parallel"))

# LLM-driven — an LLM picks the next agent at each step
wf = Workflow(agents=[a, b, c], router=Router("llm", llm_backend=my_llm))

# Custom callable
def my_router(task, agents, history):
    return agents[0] if not history else None   # return None to stop

wf = Workflow(agents=[a, b], router=my_router)
```

---

## Memory

Each agent carries a `Memory` object with two tiers:

```python
agent.memory.long.remember("user_name", "Alice")    # persists in system prompt
agent.memory.short.add("user", "Hello")              # sliding window context

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

## Result object

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
step.summary()           # "Research Analyst] tools=web_search duration=234ms"
```

---

## Backends

| Backend | Factory | Extra install |
|---|---|---|
| OpenAI | `openai_backend(model="gpt-4o")` | `pip install nx-agent[openai]` |
| Anthropic | `anthropic_backend(model="claude-opus-4-5")` | `pip install nx-agent[anthropic]` |
| HuggingFace | `huggingface_backend(repo_id="...")` | `pip install nx-agent[huggingface]` |
| Custom | Any `fn(system, user, tools, **kw) → str` | — |

---

## Architecture

```
.
├── pyproject.toml    # package metadata and tool config
├── README.md         # project documentation
├── tests/            # pytest test suite
│   ├── test_agent.py
│   ├── test_tool.py
│   └── test_workflow.py
└── nx_agent/         # importable Python package
    ├── __init__.py   # public surface
    ├── agent.py      # Agent class + agentic loop
    ├── workflow.py   # Workflow orchestrator
    ├── router.py     # Sequential / Parallel / LLM / custom routing
    ├── tool.py       # @tool decorator + ToolSchema
    ├── memory.py     # ShortTermMemory + LongTermMemory
    ├── result.py     # StepResult + WorkflowResult
    ├── backends.py   # OpenAI / Anthropic / HuggingFace factories
    └── exceptions.py # Typed exception hierarchy
```

Import path:

```python
from nx_agent import Agent, Workflow, tool
```

Package distribution name:

```bash
pip install nx-agent
```

Source package name:

```
nx_agent/
```

---

## License

MIT
