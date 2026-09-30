from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents.classifier import classifier_node
from .agents.join_planner import join_planner_node
from .agents.mongo_agent import mongo_agent_node
from .agents.result_merger import result_merger_node
from .agents.sql_agent import sql_agent_node


class DataTalkState(TypedDict, total=False):
    question: str
    source: str
    classifier_reason: str
    sql_query: str
    sql_results: list[dict[str, Any]]
    sql_error: str | None
    sql_attempts: int
    mongo_collection: str
    mongo_pipeline: list[dict[str, Any]]
    mongo_results: list[dict[str, Any]]
    mongo_error: str | None
    mongo_attempts: int
    join_plan: dict[str, Any] | None
    merged_results: list[dict[str, Any]]
    answer: str


def _route_after_classifier(state: DataTalkState) -> str:
    source = state.get("source")
    if source in ("sql", "mongo", "hybrid"):
        return source
    raise ValueError(f"Source inconnue du classifieur: {source!r}")


def _run_sql(state: DataTalkState) -> dict[str, Any]:
    return sql_agent_node(state)


def _run_mongo(state: DataTalkState) -> dict[str, Any]:
    return mongo_agent_node(state)


def build_graph():
    """Construit le graphe LangGraph réel de DataTalk."""
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
        {"sql": "sql_agent", "mongo": "mongo_agent", "hybrid": "hybrid_sql"},
    )
    workflow.add_edge("sql_agent", "result_merger")
    workflow.add_edge("mongo_agent", "result_merger")
    workflow.add_edge("hybrid_sql", "hybrid_mongo")
    workflow.add_edge("hybrid_mongo", "join_planner")
    workflow.add_edge("join_planner", "result_merger")
    workflow.add_edge("result_merger", END)

    return workflow.compile()


graph = build_graph()


def run(question: str) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("La question est vide.")
    return graph.invoke({"question": question})
