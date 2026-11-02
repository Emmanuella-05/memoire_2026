"""Adaptateur Claude API, partagé par tous les agents (classifier, sql_agent,
mongo_agent, join_planner, result_merger).

Point d'entrée unique vers le LLM : aucun agent n'appelle `anthropic`
directement, tous passent par `claude_generate` ou `claude_json`.

Gestion d'erreurs : toute défaillance d'infrastructure (clé absente, API
injoignable, quota, timeout) est convertie en `LLMError`. Les agents la
laissent remonter (HTTP 503) au lieu de la traiter comme une requête
« mal écrite » à corriger.

Deux familles de modèles :
  - Claude (`claude_generate`, `claude_json`) : classifieur, planificateur hybride,
    validateurs et réponse finale ;
  - modèles OUVERTS (`open_chat`, `open_complete`, en bas de ce fichier) : GÉNÉRATION des
    requêtes SQL et MongoDB. Actifs seulement si LLM_BASE_URL est définie (voir
    `open_models_enabled`) ; sinon les agents génèrent avec Claude, comme avant.
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

# ---------------------------------------------------------------------------
# Modèles ouverts (génération des requêtes SQL / MongoDB)
# ---------------------------------------------------------------------------
# Serveur compatible OpenAI : colab/llm_server.py (Google Colab, GPU) ou Ollama / vLLM / llama.cpp.
# Un modèle par rôle, choisi dans le .env :
#   "sql"     : génération SQL        (SQLCoder, prompt brut -> /v1/completions)
#   "mongo"   : génération MongoDB   (Qwen2.5-Coder, chat -> /v1/chat/completions)
#   "general" : corrections et traduction de la question (Qwen2.5-Coder, chat)

OPEN_ROLES = ("sql", "mongo", "general")
OPEN_DEFAULT_MODELS = {"sql": "sqlcoder-7b-2", "mongo": "qwen2.5-coder-7b", "general": "qwen2.5-coder-7b"}
OPEN_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "180"))      # un GPU Colab est lent au premier appel
OPEN_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
_OPEN_RETRY_STATUS = {502, 503, 504}


def open_models_enabled() -> bool:
    """Génération par modèles ouverts : LLM_BASE_URL définie, sauf si DATATALK_GENERATOR=claude
    (permet de comparer Claude et les modèles ouverts sans toucher à l'URL)."""
    if os.getenv("DATATALK_GENERATOR", "open").strip().lower() == "claude":
        return False
    return bool((os.getenv("LLM_BASE_URL") or "").strip())


def fix_with_open_models() -> bool:
    """Les corrections (tentatives 2 et 3) sont faites par le modèle ouvert (défaut) ou par Claude
    (DATATALK_FIX_WITH=claude : plus fiable pour corriger, mais l'appel Claude est payant)."""
    return open_models_enabled() and os.getenv("DATATALK_FIX_WITH", "open").strip().lower() != "claude"


def open_model_for(role: str) -> str:
    if role not in OPEN_ROLES:
        raise ValueError(f"Rôle LLM inconnu : {role!r}")
    return os.getenv(f"LLM_MODEL_{role.upper()}") or OPEN_DEFAULT_MODELS[role]


def llm_status() -> dict[str, Any]:
    """Configuration visible (sans secret), pour /workspace."""
    enabled = open_models_enabled()
    return {
        "generation": "open" if enabled else "claude",
        "llm_models": {role: open_model_for(role) for role in OPEN_ROLES} if enabled else None,
    }


def _open_base_url() -> str:
    url = (os.getenv("LLM_BASE_URL") or "").strip().rstrip("/")
    if not url:
        raise LLMError("LLM_BASE_URL n'est pas configurée (URL du serveur de modèles ouverts).")
    return url[:-3] if url.endswith("/v1") else url


def _open_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    import time

    import httpx

    url = f"{_open_base_url()}{path}"
    headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}
    key = os.getenv("LLM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    last: Exception | None = None
    for attempt in range(OPEN_MAX_RETRIES + 1):
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=OPEN_TIMEOUT_S)
        except httpx.HTTPError as exc:   # réseau, timeout, DNS...
            last = exc
        else:
            if response.status_code in _OPEN_RETRY_STATUS:
                last = LLMError(f"HTTP {response.status_code}")
            elif response.status_code in (401, 403):
                raise LLMError("Le serveur de modèles ouverts refuse la clé (vérifiez LLM_API_KEY).")
            elif response.status_code >= 400:
                try:
                    detail = str(response.json().get("detail") or response.text[:300])
                except (ValueError, AttributeError):
                    detail = response.text[:300]
                hint = (" Regardez la sortie de la cellule Colab qui exécute llm_server.py (trace complète ; "
                        "mémoire GPU saturée ?).") if response.status_code >= 500 else ""
                raise LLMError(f"Le serveur de modèles ouverts a répondu HTTP {response.status_code} : {detail}.{hint}")
            else:
                try:
                    return response.json()
                except ValueError as exc:
                    raise LLMError(f"Réponse illisible du serveur de modèles ouverts : {response.text[:200]!r}") from exc
        if attempt < OPEN_MAX_RETRIES:
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"Serveur de modèles ouverts injoignable ({url}) : {last}")


def open_chat(prompt: str, *, system: str | None = None, role: str = "general",
              max_tokens: int = 1200, temperature: float = 0.0) -> str:
    """Appel chat d'un modèle ouvert instruct. Retourne le texte généré."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    data = _open_post("/v1/chat/completions", {
        "model": open_model_for(role), "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
    })
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Réponse inattendue du serveur de modèles ouverts : {str(data)[:200]}") from exc


def open_complete(prompt: str, *, role: str = "sql", max_tokens: int = 400,
                  stop: list[str] | None = None, num_beams: int = 1) -> str:
    """Appel « completion » brut (modèles non-chat comme SQLCoder), décodage déterministe ;
    `num_beams` > 1 active la recherche par faisceaux."""
    payload: dict[str, Any] = {"model": open_model_for(role), "prompt": prompt, "max_tokens": max_tokens,
                               "temperature": 0.0, "num_beams": num_beams}
    if stop:
        payload["stop"] = stop
    data = _open_post("/v1/completions", payload)
    try:
        return data["choices"][0]["text"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Réponse inattendue du serveur de modèles ouverts : {str(data)[:200]}") from exc
