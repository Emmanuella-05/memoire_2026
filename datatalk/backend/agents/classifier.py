"""Agent classifieur : détermine quelle(s) source(s) de données interroger.

Premier nœud du graphe. Regarde le catalogue (tables SQLite, collections
MongoDB, mappings) et la question de l'utilisateur pour décider de la route :
- "sql"     -> seule sql_agent doit s'exécuter
- "mongo"   -> seule mongo_agent doit s'exécuter
- "hybrid"  -> les deux, puis join_planner + result_merger
"""

from __future__ import annotations

import json
from typing import Any, Literal

from ..llm import claude_generate
from ..tools.catalog_tools import load_database_docs, load_mappings

Source = Literal["sql", "mongo", "hybrid"]

SYSTEM_PROMPT = """Tu es le classifieur du système DataTalk.
Tu dois décider quelle(s) base(s) de données interroger pour répondre à la
question de l'utilisateur : "sql" (SQLite uniquement), "mongo" (MongoDB
uniquement) ou "hybrid" (les deux, avec une jointure).

Réponds STRICTEMENT avec un JSON de la forme :
{"source": "sql" | "mongo" | "hybrid", "reason": "<justification courte>"}
Aucun texte hors de ce JSON."""


def _catalog_summary() -> str:
    sqlite_docs = load_database_docs("sqlite")
    mongo_docs = load_database_docs("mongodb")
    tables = [t.get("name", "") for t in sqlite_docs.get("tables", [])]
    collections = [c.get("name", "") for c in mongo_docs.get("collections", [])]
    mappings = [m.get("entity", "") for m in load_mappings()]
    return (
        f"Tables SQLite: {', '.join(tables) or 'aucune'}\n"
        f"Collections MongoDB: {', '.join(collections) or 'aucune'}\n"
        f"Entités jointes disponibles: {', '.join(mappings) or 'aucune'}"
    )


def classify(question: str) -> dict[str, Any]:
    """Retourne {"source": ..., "reason": ...}. Repli sur "hybrid" si le
    catalogue permet une jointure et que la réponse du LLM est invalide."""
    prompt = f"{_catalog_summary()}\n\nQuestion: {question}"
    raw = claude_generate(prompt, system=SYSTEM_PROMPT, max_tokens=200)
    try:
        parsed = json.loads(raw)
        source = parsed.get("source")
        if source not in ("sql", "mongo", "hybrid"):
            raise ValueError(f"Unexpected source: {source}")
        return {"source": source, "reason": parsed.get("reason", "")}
    except (json.JSONDecodeError, ValueError):
        fallback: Source = "hybrid" if load_mappings() else "sql"
        return {"source": fallback, "reason": "fallback: unparsable classifier output"}


def classifier_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : lit state['question'], écrit source/classifier_reason.

    Ne renvoie que les clés modifiées (pas tout `state`) : c'est requis par
    LangGraph pour permettre le fan-out parallèle plus loin dans le graphe
    (deux nœuds concurrents qui renverraient chacun 'question' inchangée
    provoqueraient un conflit d'écriture sur ce canal)."""
    result = classify(state["question"])
    return {"source": result["source"], "classifier_reason": result["reason"]}