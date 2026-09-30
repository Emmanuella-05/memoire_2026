"""Agent result merger : dernier nœud du graphe.

Fusionne les résultats SQL/Mongo (via join_plan si hybride) et rédige la
réponse finale en langage naturel.
"""

from __future__ import annotations

from typing import Any

from ..llm import claude_generate
from ..tools.catalog_tools import merge_on_key

SYSTEM_PROMPT = """Tu es l'agent de synthèse de DataTalk.
On te donne la question de l'utilisateur et des données brutes (résultats
de requêtes SQL et/ou MongoDB). Rédige une réponse claire et concise en
langage naturel, en français, qui répond directement à la question à
partir de ces données. Ne mentionne pas les requêtes techniques."""


def merge(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Fusionne selon la source. Ne lève jamais : une erreur de jointure
    retombe sur la concaténation brute des résultats disponibles."""
    source = state.get("source")
    sql_results = state.get("sql_results") or []
    mongo_results = state.get("mongo_results") or []

    if source == "sql":
        return sql_results
    if source == "mongo":
        return mongo_results

    join_plan = state.get("join_plan")
    if not join_plan or not sql_results or not mongo_results:
        return sql_results + mongo_results

    try:
        return merge_on_key(
            sql_results,
            mongo_results,
            left_key=join_plan["sqlite_key"],
            right_key=join_plan["mongo_key"],
        )
    except Exception:
        return sql_results + mongo_results


def write_answer(question: str, merged_results: list[dict[str, Any]]) -> str:
    if not merged_results:
        return "Je n'ai trouvé aucune donnée correspondant à cette question."
    prompt = f"Question: {question}\n\nDonnées (extrait, max 50 lignes): {merged_results[:50]}"
    return claude_generate(prompt, system=SYSTEM_PROMPT, max_tokens=800)


def result_merger_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph terminal : écrit merged_results et answer."""
    merged = merge(state)
    return {"merged_results": merged, "answer": write_answer(state["question"], merged)}