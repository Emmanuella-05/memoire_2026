from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents.classifier import classifier_node
from .agents.join_planner import join_planner_node
from .agents.mongo_agent import mongo_agent_node
from .agents.result_merger import result_merger_node
from .agents.sql_agent import sql_agent_node

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


def _route_after_hybrid_sql(state: DataTalkState) -> str:
    # Inutile d'interroger MongoDB si la partie SQL a échoué.
    return "stop" if state.get("sql_error") else "continue"


def _run_sql(state: DataTalkState) -> dict[str, Any]:
    return sql_agent_node(state)


def _run_mongo(state: DataTalkState) -> dict[str, Any]:
    return mongo_agent_node(state)


def build_graph():
    """Construit le graphe LangGraph réel de DataTalk.

        classifier ─┬─ sql ────────────► sql_agent ───────────────────────────┐
                    ├─ mongo ──────────► mongo_agent ─────────────────────────┤
                    ├─ hybrid ─► join_planner ─ relation connue ─► hybrid_sql ┤► result_merger
                    │                         └─ relation inconnue ───────────┤
                    └─ failed ────────────────────────────────────────────────┘
        (hybrid_sql ─► hybrid_mongo ─► result_merger ; si SQL échoue : direct au merger)
    """
    workflow = StateGraph(DataTalkState)

    workflow.add_node("classifier", classifier_node)
    workflow.add_node("sql_agent", _run_sql)
    workflow.add_node("mongo_agent", _run_mongo)
    workflow.add_node("hybrid_sql", _run_sql)
    workflow.add_node("hybrid_mongo", _run_mongo)
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
        {"plan_ok": "hybrid_sql", "no_relation": "result_merger"},
    )
    workflow.add_conditional_edges(
        "hybrid_sql",
        _route_after_hybrid_sql,
        {"continue": "hybrid_mongo", "stop": "result_merger"},
    )
    workflow.add_edge("hybrid_mongo", "result_merger")
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