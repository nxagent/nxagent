"""Ready-to-use LLM backend factories.

Every backend returned here follows the small NxAgent contract:

    fn(system_prompt: str, user_message: str, tools: list, **kwargs) -> str

Provider-specific tool calls are normalized into ``TOOL_CALL:name:{json}``
directives for :class:`nx_agent.Agent` to execute.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, NoReturn, Optional

from nx_agent.exceptions import ProviderRateLimitError


Backend = Callable[..., str]
_RATE_LIMIT_CLASS_NAMES = {
    "RateLimitError",
    "RateLimitExceeded",
    "TooManyRequests",
    "TooManyRequestsError",
}


def _status_code(error: Exception) -> Optional[int]:
    """Return a provider/HTTP status code when one is exposed."""
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _is_provider_rate_limit(error: Exception) -> bool:
    """Classify common provider SDK rate-limit errors without importing SDKs."""
    if isinstance(error, ProviderRateLimitError):
        return True
    if _status_code(error) == 429:
        return True
    return error.__class__.__name__ in _RATE_LIMIT_CLASS_NAMES


def _normalize_provider_error(provider: str, error: Exception) -> Exception:
    """Convert provider-specific rate-limit errors into NxAgent's public type."""
    if isinstance(error, ProviderRateLimitError):
        return error
    if _is_provider_rate_limit(error):
        return ProviderRateLimitError(provider, str(error))
    return error


def _raise_provider_error(provider: str, error: Exception) -> NoReturn:
    """Raise normalized provider errors while preserving other exceptions."""
    normalized = _normalize_provider_error(provider, error)
    if normalized is error:
        raise error
    raise normalized from error


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
    provider: str,
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

        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            _raise_provider_error(provider, exc)
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
        client, model, temperature, max_tokens, default_kwargs, "openai"
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
        client, model, temperature, max_tokens, default_kwargs, "grok"
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
        try:
            response = client.messages.create(
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=anthropic_tools if anthropic_tools else [],
                **merged,
            )
        except Exception as exc:
            _raise_provider_error("anthropic", exc)

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

        try:
            response = client.chat_completion(**request)
        except Exception as exc:
            _raise_provider_error("huggingface", exc)
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

        try:
            response = client.chat(**request)
        except Exception as exc:
            _raise_provider_error("ollama", exc)
        return _message_output(response.message)

    return _backend
