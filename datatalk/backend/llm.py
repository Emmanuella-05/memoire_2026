"""Adaptateur Claude API, partagé par tous les agents (classifier, sql_agent,
mongo_agent, join_planner, result_merger).

Point d'entrée unique vers le LLM : aucun agent n'appelle `anthropic`
directement, tous passent par `claude_generate`.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None


def claude_client() -> Any:
    """Create the Anthropic client from CLAUDE_API_KEY."""
    if Anthropic is None:
        raise RuntimeError("anthropic is not installed")
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("CLAUDE_API_KEY is not configured")
    return Anthropic(api_key=api_key)


def claude_generate(prompt: str, *, system: str | None = None,
                    max_tokens: int = 1200) -> str:
    """Single Claude call used by every agent in graph.py.

    The model is configurable through CLAUDE_MODEL so deployment can select
    a currently supported Anthropic model without code changes.
    """
    client = claude_client()
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system or "You are a precise assistant for the DataTalk NL-to-Query system.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in message.content
        if getattr(block, "type", None) == "text"
    )