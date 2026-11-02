from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import mongo_agent as _mongo
from .agents import sql_agent as _sql
from .agents.classifier import classifier_node
from .agents.join_planner import join_planner_node
from .agents.mongo_agent import mongo_agent_node
from .agents.result_merger import result_merger_node
from .agents.sql_agent import sql_agent_node
from .tools.hybrid_tools import MAX_JOIN_KEYS, extract_keys

MAX_QUESTION_CHARS = 2000


class DataTalkState(TypedDict, total=False):
    question: str
    source: str
    classifier_reason: str
    error: str | None
    status: str
    sql_query: str | None
    sql_results: list[dict[str, Any]]
    sql_error: str | None
    sql_attempts: int
    mongo_collection: str | None
    mongo_pipeline: list[dict[str, Any]] | None
    mongo_results: list[dict[str, Any]]
    mongo_error: str | None
    mongo_attempts: int
    join_plan: dict[str, Any] | None
    join_error: str | None
    join_stats: dict[str, Any] | None
    translated_question: str | None
    hybrid_keys: list[Any]
    hybrid_key_count: int
    hybrid_error: str | None
    merged_results: list[dict[str, Any]]
    answer: str


def _route_after_classifier(state: DataTalkState) -> str:
    if state.get("error"):
        return "failed"
    source = state.get("source")
    if source in ("sql", "mongo", "hybrid"):
        return source
    raise ValueError(f"Source inconnue du classifieur: {source!r}")


def _route_after_join_planner(state: DataTalkState) -> str:
    # Hybride = deux sources ET une relation connue ; sinon on n'exécute rien.
    return "plan_ok" if state.get("join_plan") else "no_relation"


def _first_source(plan: dict[str, Any]) -> str:
    """Source interrogée en premier : choisie par le planificateur (plan décomposé),
    sinon SQL (ancien comportement : la question entière aux deux agents)."""
    return plan["first"] if plan.get("decomposed") else "sql"


def _route_after_first_step(state: DataTalkState) -> str:
    # Inutile d'interroger la seconde source si la première a échoué, n'a rien trouvé
    # (aucun identifiant à lui transmettre) ou en a trouvé trop.
    plan = state["join_plan"]
    if state.get(f"{_first_source(plan)}_error") or state.get("hybrid_error"):
        return "stop"
    if plan.get("decomposed") and not state.get("hybrid_keys"):
        return "stop"
    return "continue"


def _run_sql(state: DataTalkState) -> dict[str, Any]:
    return sql_agent_node(state)


def _run_mongo(state: DataTalkState) -> dict[str, Any]:
    return mongo_agent_node(state)


def _run_source(source: str, state: DataTalkState, step: str, keys: list[Any] | None = None) -> dict[str, Any]:
    plan = state["join_plan"]
    if not plan.get("decomposed"):
        # Pas de décomposition : la question entière, sans injection (jointure en mémoire ensuite).
        return (_sql if source == "sql" else _mongo).run(state["question"], plan)
    question = plan["sql_question"] if source == "sql" else plan["mongo_question"]
    return (_sql if source == "sql" else _mongo).run(question, plan, injected_keys=keys, step=step)


def _hybrid_first(state: DataTalkState) -> dict[str, Any]:
    """Étape 1 : la source qui porte les conditions répond à SA sous-question et retourne
    les identifiants (clé de jointure) des entités concernées."""
    plan = state["join_plan"]
    source = _first_source(plan)
    update = _run_source(source, state, "first")
    if plan.get("decomposed") and not update.get(f"{source}_error"):
        key = plan["sqlite_key"] if source == "sql" else plan["mongo_key"]
        keys = extract_keys(update.get(f"{source}_results") or [], key)
        update["hybrid_key_count"] = len(keys)
        if len(keys) > MAX_JOIN_KEYS:
            update["hybrid_error"] = (
                f"L'étape 1 a retourné {len(keys)} identifiants, au-delà de la limite de {MAX_JOIN_KEYS} "
                "pour la jointure entre les deux bases. Précisez la question (filtre plus restrictif)."
            )
        else:
            update["hybrid_keys"] = keys
    return update


def _hybrid_second(state: DataTalkState) -> dict[str, Any]:
    """Étape 2 : l'autre source répond au reste de la question pour les seuls identifiants
    de l'étape 1 (injectés par le code dans sa requête)."""
    plan = state["join_plan"]
    second = "mongo" if _first_source(plan) == "sql" else "sql"
    return _run_source(second, state, "second", state.get("hybrid_keys"))


def build_graph():
    """Construit le graphe LangGraph réel de DataTalk.

        classifier ─┬─ sql ────────────► sql_agent ───────────────────────────┐
                    ├─ mongo ──────────► mongo_agent ─────────────────────────┤
                    ├─ hybrid ─► join_planner ─ relation connue ─► hybrid_sql ┤► result_merger
                    │                         └─ relation inconnue ───────────┤
                    └─ failed ────────────────────────────────────────────────┘
        hybrid_first (source qui filtre) ─► hybrid_second (autre source, restreinte aux identifiants de
        l'étape 1) ─► result_merger ; si l'étape 1 échoue ou ne trouve rien : direct au merger
    """
    workflow = StateGraph(DataTalkState)

    workflow.add_node("classifier", classifier_node)
    workflow.add_node("sql_agent", _run_sql)
    workflow.add_node("mongo_agent", _run_mongo)
    workflow.add_node("hybrid_first", _hybrid_first)
    workflow.add_node("hybrid_second", _hybrid_second)
    workflow.add_node("join_planner", join_planner_node)
    workflow.add_node("result_merger", result_merger_node)

    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges(
        "classifier",
        _route_after_classifier,
        {"sql": "sql_agent", "mongo": "mongo_agent", "hybrid": "join_planner", "failed": "result_merger"},
    )
    workflow.add_edge("sql_agent", "result_merger")
    workflow.add_edge("mongo_agent", "result_merger")
    workflow.add_conditional_edges(
        "join_planner",
        _route_after_join_planner,
        {"plan_ok": "hybrid_first", "no_relation": "result_merger"},
    )
    workflow.add_conditional_edges(
        "hybrid_first",
        _route_after_first_step,
        {"continue": "hybrid_second", "stop": "result_merger"},
    )
    workflow.add_edge("hybrid_second", "result_merger")
    workflow.add_edge("result_merger", END)

    return workflow.compile()


graph = build_graph()


def run(question: str) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("La question est vide.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"La question est trop longue ({MAX_QUESTION_CHARS} caractères maximum).")
    return graph.invoke({"question": question})