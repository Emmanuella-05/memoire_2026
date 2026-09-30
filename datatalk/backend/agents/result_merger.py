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


def merge(state: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    """Fusionne les résultats sans masquer une erreur de jointure hybride."""
    source = state.get("source")
    sql_results = state.get("sql_results") or []
    mongo_results = state.get("mongo_results") or []

    if source == "sql":
        return sql_results, None
    if source == "mongo":
        return mongo_results, None

    join_plan = state.get("join_plan")
    if not join_plan:
        return [], "Impossible de fusionner les résultats hybrides : aucune correspondance SQL ↔ MongoDB n'est disponible."

    left_key = join_plan.get("sqlite_key")
    right_key = join_plan.get("mongo_key")
    if not left_key or not right_key:
        return [], "Impossible de fusionner les résultats hybrides : les clés de jointure sont incomplètes."

    if sql_results and not any(left_key in row for row in sql_results):
        return [], f"Impossible de fusionner les résultats hybrides : la clé SQLite '{left_key}' est absente des résultats."
    if mongo_results and not any(right_key in row for row in mongo_results):
        return [], f"Impossible de fusionner les résultats hybrides : la clé MongoDB '{right_key}' est absente des résultats."

    try:
        return merge_on_key(
            sql_results,
            mongo_results,
            left_key=left_key,
            right_key=right_key,
        ), None
    except Exception as exc:
        return [], f"Erreur lors de la jointure hybride : {exc}"


def write_answer(question: str, merged_results: list[dict[str, Any]]) -> str:
    if not merged_results:
        return "Je n'ai trouvé aucune donnée correspondant à cette question."
    prompt = f"Question: {question}\n\nDonnées (extrait, max 50 lignes): {merged_results[:50]}"
    return claude_generate(prompt, system=SYSTEM_PROMPT, max_tokens=800)


def result_merger_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph terminal : écrit merged_results et answer."""
    merged, merge_error = merge(state)
    if merge_error:
        return {"merged_results": [], "answer": merge_error}
    return {"merged_results": merged, "answer": write_answer(state["question"], merged)}