"""Ready-to-use LLM backend factories.

Every backend returned here follows the small NxAgent contract:

    fn(system_prompt: str, user_message: str, tools: list, **kwargs) -> str

Provider-specific tool calls are normalized into ``TOOL_CALL:name:{json}``
directives for :class:`nx_agent.Agent` to execute.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional


Backend = Callable[..., str]


def _function_tools(tools: List[dict]) -> List[dict]:
    """Convert NxAgent tool schemas to chat-completion function tools."""
    return [{"type": "function", "function": tool} for tool in tools]


def _message_output(message: Any) -> str:
    """Normalize an OpenAI-compatible response message to NxAgent output."""
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        function = tool_calls[0].function
        arguments = function.arguments
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        return f"TOOL_CALL:{function.name}:{arguments}"
    return getattr(message, "content", None) or ""


def _chat_completion_backend(
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    default_kwargs: Dict[str, Any],
) -> Backend:
    """Build a backend for clients exposing ``chat.completions.create``."""

    def _backend(
        system_prompt: str,
        user_message: str,
        tools: List[dict],
        **kwargs: Any,
    ) -> str:
        merged = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **default_kwargs,
            **kwargs,
        }
        request: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            **merged,
        }
        if tools:
            request["tools"] = _function_tools(tools)
            request["tool_choice"] = "auto"

        response = client.chat.completions.create(**request)
        return _message_output(response.choices[0].message)

    return _backend


def openai_backend(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    **default_kwargs: Any,
) -> Backend:
    """Return a backend using OpenAI Chat Completions and function tools.

    Requires ``pip install nx-agent[openai]``. The API key defaults to
    ``OPENAI_API_KEY``.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError("Install OpenAI support: pip install nx-agent[openai]") from exc

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    return _chat_completion_backend(
        client, model, temperature, max_tokens, default_kwargs
    )


def grok_backend(
    model: str = "grok-4.3",
    api_key: Optional[str] = None,
    base_url: str = "https://api.x.ai/v1",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    **default_kwargs: Any,
) -> Backend:
    """Return a backend using xAI Grok through its OpenAI-compatible API.

    Requires ``pip install nx-agent[grok]``. The API key defaults to
    ``XAI_API_KEY``.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError("Install Grok support: pip install nx-agent[grok]") from exc

    client = OpenAI(
        api_key=api_key or os.environ.get("XAI_API_KEY"),
        base_url=base_url,
    )
    return _chat_completion_backend(
        client, model, temperature, max_tokens, default_kwargs
    )


def anthropic_backend(
    model: str = "claude-opus-4-5",
    api_key: Optional[str] = None,
    max_tokens: int = 2048,
    **default_kwargs: Any,
) -> Backend:
    """Return a backend function that calls the Anthropic Messages API.

    Requires ``pip install nx-agent[anthropic]``.
    """
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install Anthropic support: pip install nx-agent[anthropic]"
        ) from exc

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def _backend(
        system_prompt: str,
        user_message: str,
        tools: List[dict],
        **kwargs: Any,
    ) -> str:
        merged = {
            "model": model,
            "max_tokens": max_tokens,
            **default_kwargs,
            **kwargs,
        }
        anthropic_tools = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for tool in tools
        ]
        response = client.messages.create(
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=anthropic_tools if anthropic_tools else [],
            **merged,
        )

        for block in response.content:
            if block.type == "tool_use":
                return f"TOOL_CALL:{block.name}:{json.dumps(block.input)}"
            if block.type == "text":
                return block.text
        return ""

    return _backend


def huggingface_backend(
    repo_id: str,
    token: Optional[str] = None,
    max_tokens: int = 512,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.2,
    **default_kwargs: Any,
) -> Backend:
    """Return a Hugging Face InferenceClient chat-completion backend.

    The selected model or endpoint must support chat completions, and must
    support function tools when tools are attached to the agent. Requires
    ``pip install nx-agent[huggingface]``. The token defaults to ``HF_TOKEN``.
    """
    try:
        from huggingface_hub import InferenceClient  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install Hugging Face support: pip install nx-agent[huggingface]"
        ) from exc

    client = InferenceClient(model=repo_id, token=token or os.environ.get("HF_TOKEN"))
    token_limit = max_new_tokens if max_new_tokens is not None else max_tokens

    def _backend(
        system_prompt: str,
        user_message: str,
        tools: List[dict],
        **kwargs: Any,
    ) -> str:
        request: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": token_limit,
            "temperature": temperature,
            **default_kwargs,
            **kwargs,
        }
        if tools:
            request["tools"] = _function_tools(tools)
            request["tool_choice"] = "auto"

        response = client.chat_completion(**request)
        return _message_output(response.choices[0].message)

    return _backend


def ollama_backend(
    model: str = "qwen3",
    host: Optional[str] = None,
    **default_kwargs: Any,
) -> Backend:
    """Return an Ollama chat backend for local or remote Ollama hosts.

    Requires ``pip install nx-agent[ollama]``. Ollama defaults to its locally
    configured host, normally ``http://localhost:11434``. The selected model
    must support tools when tools are attached to the agent.
    """
    try:
        from ollama import Client  # type: ignore
    except ImportError as exc:
        raise ImportError("Install Ollama support: pip install nx-agent[ollama]") from exc

    client = Client(host=host) if host is not None else Client()

    def _backend(
        system_prompt: str,
        user_message: str,
        tools: List[dict],
        **kwargs: Any,
    ) -> str:
        request: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            **default_kwargs,
            **kwargs,
        }
        if tools:
            request["tools"] = _function_tools(tools)

        response = client.chat(**request)
        return _message_output(response.message)

    return _backend
