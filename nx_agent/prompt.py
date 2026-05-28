"""Prompt composition helpers for NxAgent."""

from __future__ import annotations

from typing import Optional


def system_prompt(
    prompt: Optional[str] = None,
    *,
    goal: Optional[str] = None,
    instructions: Optional[str] = None,
    output_format: Optional[str] = None,
) -> str:
    """Compose an optional structured system prompt.

    Pass a plain string as ``prompt`` for full control, or use the structured
    fields to build a small readable prompt outside ``Agent``.
    """
    parts = []
    if prompt:
        parts.append(prompt.strip())
    if goal:
        parts.append(f"Goal: {goal.strip()}")
    if instructions:
        parts.append(f"Instructions: {instructions.strip()}")
    if output_format:
        parts.append(f"Output format: {output_format.strip()}")
    return "\n".join(part for part in parts if part)
