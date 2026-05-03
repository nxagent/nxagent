"""
Ready-to-use LLM backend factories.

Usage
-----
    from nx_agent.backends import openai_backend, anthropic_backend

    agent = Agent(
        role="Researcher",
        goal="Find accurate source material",
        llm_backend=openai_backend(model="gpt-4o"),
    )

Each factory returns a function with the signature:
    fn(system_prompt: str, user_message: str, tools: list, **kw) -> str
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional


# ── OpenAI ────────────────────────────────────────────────────────────────────

def openai_backend(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    **default_kwargs,
) -> Callable:
    """
    Return a backend function that calls the OpenAI Chat Completions API.

    Requires: ``pip install openai``
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai") from exc

    key = api_key or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    def _backend(
        system_prompt: str,
        user_message: str,
        tools: List[dict],
        **kwargs,
    ) -> str:
        merged = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **default_kwargs,
            **kwargs,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        if tools:
            response = client.chat.completions.create(
                messages=messages,
                tools=[{"type": "function", "function": t} for t in tools],
                tool_choice="auto",
                **merged,
            )
            choice = response.choices[0]
            # If the model wants to call a tool, return the directive
            if choice.finish_reason == "tool_calls":
                tc = choice.message.tool_calls[0]
                args = tc.function.arguments  # already a JSON string
                return f"TOOL_CALL:{tc.function.name}:{args}"
            return choice.message.content or ""
        else:
            response = client.chat.completions.create(messages=messages, **merged)
            return response.choices[0].message.content or ""

    return _backend


# ── Anthropic ─────────────────────────────────────────────────────────────────

def anthropic_backend(
    model: str = "claude-opus-4-5",
    api_key: Optional[str] = None,
    max_tokens: int = 2048,
    **default_kwargs,
) -> Callable:
    """
    Return a backend function that calls the Anthropic Messages API.

    Requires: ``pip install anthropic``
    """
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise ImportError("Install anthropic: pip install anthropic") from exc

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key)

    def _backend(
        system_prompt: str,
        user_message: str,
        tools: List[dict],
        **kwargs,
    ) -> str:
        merged = {
            "model": model,
            "max_tokens": max_tokens,
            **default_kwargs,
            **kwargs,
        }

        anthropic_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

        response = client.messages.create(
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=anthropic_tools if anthropic_tools else [],
            **merged,
        )

        for block in response.content:
            if block.type == "tool_use":
                args_json = json.dumps(block.input)
                return f"TOOL_CALL:{block.name}:{args_json}"
            if block.type == "text":
                return block.text

        return ""

    return _backend


# ── HuggingFace Inference (text-generation) ───────────────────────────────────

def huggingface_backend(
    repo_id: str,
    token: Optional[str] = None,
    max_new_tokens: int = 512,
    **hf_kwargs,
) -> Callable:
    """
    Return a backend using HuggingFace text-generation-inference.

    Requires: ``pip install huggingface_hub``
    """
    try:
        from huggingface_hub import InferenceClient  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub: pip install huggingface_hub"
        ) from exc

    token = token or os.environ.get("HF_TOKEN")
    client = InferenceClient(model=repo_id, token=token)

    def _backend(
        system_prompt: str,
        user_message: str,
        tools: List[dict],
        **kwargs,
    ) -> str:
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_message}\n<|assistant|>\n"
        result = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            **hf_kwargs,
            **kwargs,
        )
        return result.strip()

    return _backend
