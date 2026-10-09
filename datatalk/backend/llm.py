"""Adaptateur Claude API, partagé par tous les agents (classifier, sql_agent,
mongo_agent, join_planner, result_merger).

Point d'entrée unique vers le LLM : aucun agent n'appelle `anthropic`
directement, tous passent par `claude_generate` ou `claude_json`.

Gestion d'erreurs : toute défaillance d'infrastructure (clé absente, API
injoignable, quota, timeout) est convertie en `LLMError`. Les agents la
laissent remonter (HTTP 503) au lieu de la traiter comme une requête
« mal écrite » à corriger.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    anthropic = None
    Anthropic = None

DEFAULT_MODEL = "claude-sonnet-5-5"
REQUEST_TIMEOUT_S = float(os.getenv("CLAUDE_TIMEOUT_S", "60"))
MAX_RETRIES = int(os.getenv("CLAUDE_MAX_RETRIES", "2"))


class LLMError(RuntimeError):
    """Le LLM est indisponible ou mal configuré (≠ réponse invalide du LLM)."""


def claude_client() -> Any:
    """Create the Anthropic client from CLAUDE_API_KEY."""
    if Anthropic is None:
        raise LLMError("Le paquet 'anthropic' n'est pas installé.")
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise LLMError("CLAUDE_API_KEY n'est pas configurée.")
    return Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_S, max_retries=MAX_RETRIES)


def claude_generate(prompt: str, *, system: str | None = None,
                    max_tokens: int = 1200) -> str:
    """Single Claude call used by every agent in graph.py.

    The model is configurable through CLAUDE_MODEL so deployment can select
    a currently supported Anthropic model without code changes.
    """
    client = claude_client()
    model = os.getenv("CLAUDE_MODEL") or DEFAULT_MODEL
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a precise assistant for the DataTalk NL-to-Query system.",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        # Erreurs de l'API Anthropic (auth, quota, réseau, modèle inconnu...)
        if anthropic is not None and isinstance(exc, anthropic.APIError):
            raise LLMError(f"Appel à l'API Claude impossible ({type(exc).__name__}) : {exc}") from exc
        raise LLMError(f"Appel à l'API Claude impossible : {exc}") from exc
    return "".join(
        block.text for block in message.content
        if getattr(block, "type", None) == "text"
    )


def extract_json(raw: str) -> dict[str, Any]:
    """Parse une réponse LLM censée être un objet JSON (tolère le texte
    ou les blocs markdown autour). Lève ValueError si rien d'exploitable."""
    text = raw.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Réponse du modèle sans JSON exploitable : {raw[:200]!r}")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON invalide dans la réponse du modèle : {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("La réponse du modèle doit être un objet JSON.")
    return parsed


def claude_json(prompt: str, *, system: str | None = None, max_tokens: int = 500,
                parse_retries: int = 1) -> dict[str, Any]:
    """Appelle Claude et parse un objet JSON. Si la réponse est illisible, on
    redemande une fois en rappelant le format. Lève ValueError si l'échec
    persiste, LLMError si l'API est indisponible."""
    current = prompt
    last_error: ValueError | None = None
    for _ in range(parse_retries + 1):
        raw = claude_generate(current, system=system, max_tokens=max_tokens)
        try:
            return extract_json(raw)
        except ValueError as exc:
            last_error = exc
            current = (
                f"{prompt}\n\nTa réponse précédente était inutilisable ({exc}). "
                "Réponds UNIQUEMENT avec l'objet JSON demandé, sans autre texte."
            )
    assert last_error is not None
    raise last_error