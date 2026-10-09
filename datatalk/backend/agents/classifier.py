"""Agent classifieur : détermine quelle(s) source(s) de données interroger.

Premier nœud du graphe. Regarde le catalogue (tables SQLite, collections
MongoDB, correspondances) et la question de l'utilisateur pour décider de la route :
- "sql"     -> seule sql_agent doit s'exécuter
- "mongo"   -> seule mongo_agent doit s'exécuter
- "hybrid"  -> la question a besoin des DEUX bases ET une correspondance du
               catalogue permet de les relier ; join_planner vérifie ensuite
               la relation, puis sql_agent, mongo_agent et result_merger s'exécutent

Si le modèle ne rend pas une réponse exploitable, on ne devine pas : le
nœud renvoie une erreur explicite (state['error']) traitée par result_merger.
"""

from __future__ import annotations

from typing import Any, Literal

from ..llm import claude_json
from ..tools.catalog_tools import (
    DataSourceUnavailable,
    available_sources,
    load_database_docs,
    mapping_context,
)

Source = Literal["sql", "mongo", "hybrid"]

SYSTEM_PROMPT = """Tu es le classifieur du système DataTalk.
Tu dois décider quelle(s) base(s) de données interroger pour répondre à la
question de l'utilisateur :
- "sql"    : la réponse se trouve uniquement dans SQLite
- "mongo"  : la réponse se trouve uniquement dans MongoDB
- "hybrid" : la réponse exige de COMBINER des données de SQLite ET de MongoDB.

Règle stricte : choisis "hybrid" uniquement si la question a réellement besoin
des deux bases ET qu'une des correspondances listées permet de les relier.
Une question qui semble toucher deux bases mais pour laquelle aucune
correspondance ne les relie reste "hybrid" (le système expliquera qu'il ne sait
pas relier les données) ; n'invente jamais de correspondance.

Réponds STRICTEMENT avec un JSON de la forme :
{"source": "sql" | "mongo" | "hybrid", "reason": "<justification courte>"}
Aucun texte hors de ce JSON."""


def _catalog_summary() -> str:
    sqlite_docs = load_database_docs("sqlite")
    mongo_docs = load_database_docs("mongodb")
    tables = [
        f"{t.get('name', '')}({', '.join(c.get('name', '') for c in t.get('columns', []))})"
        for t in sqlite_docs.get("tables", [])
    ]
    collections = [
        f"{c.get('name', '')}({', '.join(f.get('name', '') for f in c.get('fields', []))})"
        for c in mongo_docs.get("collections", [])
    ]
    return (
        f"Tables SQLite: {'; '.join(tables) or 'aucune'}\n"
        f"Collections MongoDB: {'; '.join(collections) or 'aucune'}\n"
        f"Correspondances SQL ↔ MongoDB:\n{mapping_context()}"
    )


def classify(question: str) -> dict[str, Any]:
    """Retourne {"source": ..., "reason": ...} ; lève ValueError si la réponse du
    modèle reste inexploitable après une relance (LLMError si l'API est indisponible)."""
    prompt = f"{_catalog_summary()}\n\nQuestion: {question}"
    parsed = claude_json(prompt, system=SYSTEM_PROMPT, max_tokens=200)
    source = parsed.get("source")
    if source not in ("sql", "mongo", "hybrid"):
        raise ValueError(f"Source inattendue renvoyée par le classifieur : {source!r}")
    return {"source": source, "reason": str(parsed.get("reason", ""))}


def classifier_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : lit state['question'], écrit source/classifier_reason
    (ou error si la question n'a pas pu être classée)."""
    sources = available_sources()
    if not sources:
        raise DataSourceUnavailable("Aucune donnée importée : chargez une base SQLite et/ou des données MongoDB.")
    if len(sources) == 1:
        # Une seule source importée : aucune décision à prendre, pas d'appel au LLM.
        return {"source": sources[0], "classifier_reason": "Une seule source de données est disponible."}
    try:
        result = classify(state["question"])
    except ValueError as exc:
        return {"error": f"Impossible de déterminer quelle base interroger : {exc}"}
    return {"source": result["source"], "classifier_reason": result["reason"]}