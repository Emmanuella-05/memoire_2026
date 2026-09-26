from __future__ import annotations

from typing import Any

try:
    from .tools import claude_generate
except ImportError:
    from tools import claude_generate


class SimpleGraph:
    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        question = str(state.get("question", "")).strip()
        if not question:
            return {
                "final_answer": "Question vide.",
                "merged_result": [],
                "query_type": "fallback",
                "sources_used": [],
                "correspondences_used": [],
            }

        try:
            answer = claude_generate(
                f"Réponds en français à cette question : {question}",
                max_tokens=300,
            )
        except Exception:
            answer = f"Question reçue : {question}"

        return {
            "final_answer": answer,
            "merged_result": [],
            "query_type": "fallback",
            "sources_used": [],
            "correspondences_used": [],
        }


def build_graph() -> SimpleGraph:
    return SimpleGraph()