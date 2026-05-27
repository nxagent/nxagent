# NxAgent

**A lightweight Python library for agent workflows, Python tools, traces, memory, and swappable LLM backends.**

[![PyPI](https://img.shields.io/pypi/v/nx-agent)](https://pypi.org/project/nx-agent/)
[![Python](https://img.shields.io/pypi/pyversions/nx-agent)](https://pypi.org/project/nx-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why NxAgent?

| Feature | NxAgent |
|---|---|
| Zero hard dependencies | Bring your own LLM |
| Pluggable backends | OpenAI · Grok (xAI) · Hugging Face · Ollama · Anthropic · custom |
| `@tool` decorator | Type hints → JSON schema automatically |
| Multi-agent routing | Sequential · Parallel · LLM-driven · custom |
| Built-in memory | Short-term context + long-term key-value memory |
| Full traceability | Every step, tool call, and timing captured |
| Opt-in resilience | Backend/tool retries and timeouts |
| Lifecycle hooks | `on_workflow_start`, `on_workflow_end` |

---

## Installation

```bash
pip install nx-agent               # core (no LLM deps)

# Optional — pick your LLM backend
pip install nx-agent[openai]       # OpenAI
pip install nx-agent[grok]         # xAI Grok (uses the OpenAI SDK)
pip install nx-agent[huggingface]  # Hugging Face Inference Providers
pip install nx-agent[ollama]       # local Ollama models
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

### 02 — Attach a Tool with Grok

```python
from nx_agent import Agent, tool
from nx_agent.backends import grok_backend

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    ...  # your implementation

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate source material",
    tools=[web_search],
    llm_backend=grok_backend(),  # reads XAI_API_KEY from env
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

## Provider Backends

Each factory returns the same callable interface, so changing providers does
not change `Agent` or `Workflow` code.

```python
from nx_agent.backends import (
    openai_backend,
    grok_backend,
    huggingface_backend,
    ollama_backend,
)

openai_llm = openai_backend(model="gpt-4o")       # OPENAI_API_KEY
grok_llm = grok_backend(model="grok-4.3")         # XAI_API_KEY
hf_llm = huggingface_backend(                     # HF_TOKEN
    repo_id="openai/gpt-oss-120b",
)
local_llm = ollama_backend(model="qwen3")          # local Ollama service
```

`openai_backend()` and `grok_backend()` use chat-completion function tools.
`grok_backend()` targets xAI's OpenAI-compatible `https://api.x.ai/v1`
endpoint. `huggingface_backend()` uses `InferenceClient.chat_completion()`;
choose a hosted model/provider that supports tool calling when attaching
NxAgent tools. `ollama_backend()` uses Ollama's native chat/tool interface and
its configured local service by default, normally `http://localhost:11434`.

### Local Agent with Ollama

After installing Ollama and pulling a tool-capable model:

```bash
ollama pull qwen3
pip install nx-agent[ollama]
```

```python
from nx_agent import Agent
from nx_agent.backends import ollama_backend

agent = Agent(
    role="Local Assistant",
    goal="Answer without a hosted API",
    llm_backend=ollama_backend(model="qwen3"),
)

print(agent.run("Summarize this project.").output)
```

Pass `host="http://another-host:11434"` to connect to another Ollama server.

---

## Retries and Timeouts

Retries and timeouts are opt-in, so existing agents retain their original
single-attempt behavior.

```python
from nx_agent import Agent, RetryPolicy, tool
from nx_agent.backends import openai_backend

@tool(retries=2, backoff=0.5, timeout=10)
def fetch_record(record_id: str) -> str:
    """Fetch a record from an external service."""
    ...

agent = Agent(
    role="Researcher",
    goal="Retrieve and explain the record",
    tools=[fetch_record],
    llm_backend=openai_backend(),
    retry_policy=RetryPolicy(retries=2, backoff=0.5),
    timeout=30,
)
```

Backend failures after all attempts raise `BackendError`; backend timeouts
raise `AgentTimeoutError`. Tool failures remain in `ToolCall.error`, while
`ToolCall.attempts` and `ToolCall.timed_out` show resilience behavior in the
trace. A timed-out operation is not retried by default because Python cannot
forcibly stop provider or tool work that has already started.

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
    .on("on_workflow_end",   lambda r: save_to_db(r))
)
```

The MVP currently fires workflow start and workflow end hooks. Per-step hooks
are reserved for a later tracing release.

---

## Result Object

```python
result = workflow.run("my task")

result.output            # str  — final answer
result.steps             # List[StepResult]
result.total_duration_ms # float
result.succeeded         # bool
result.agents_used       # ["Research Analyst", "Technical Writer"]
result.to_dict()         # serializable trace dictionary
result.to_json()         # JSON for logs or persistence

step = result.steps[0]
step.agent_role          # "Research Analyst"
step.tool_calls          # List[ToolCall]
step.duration_ms         # float
step.metadata            # {"backend_attempts": [1, ...]}
step.succeeded           # bool
step.summary()           # "Research Analyst tools=web_search duration=234ms"

call = step.tool_calls[0]
call.attempts           # int
call.timed_out          # bool
```

---

## MVP Scope

Included in `0.1.0`:

- `@tool` schema generation and a bounded agent tool loop.
- Sequential, parallel, LLM-selected, and custom workflow routing.
- Short-term and long-term in-process memory.
- Structured step/tool traces plus JSON result export.
- Optional OpenAI, Grok, Hugging Face, Ollama, and Anthropic backends with no hard core dependencies.
- Opt-in backend and tool retries/timeouts with typed backend timeout errors.

Next milestones from the design guide are partial-result workflow recovery,
streaming, deeper observability, and reusable testing helpers. They are
intentionally not presented as shipped features yet.

## Backend References

- OpenAI chat function calling: <https://platform.openai.com/docs/guides/function-calling?api-mode=chat>
- xAI chat completions and function calling: <https://docs.x.ai/developers/model-capabilities/legacy/chat-completions> and <https://docs.x.ai/developers/tools/function-calling>
- Hugging Face `InferenceClient.chat_completion`: <https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client>
- Ollama tool calling and chat API: <https://docs.ollama.com/capabilities/tool-calling> and <https://docs.ollama.com/api/chat>

---

## License

NxAgent is licensed under the MIT License.
