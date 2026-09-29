"""Agent join planner : s'exécute seulement quand classifier a choisi
"hybrid". Décide QUEL mapping du catalogue (tools/catalog_tools.py) utiliser
pour joindre les résultats SQL et Mongo.
"""

from __future__ import annotations

import json
from typing import Any

from llm import claude_generate
from tools.catalog_tools import load_mappings, mapping_context

SYSTEM_PROMPT = """Tu es le planificateur de jointure de DataTalk.
On te donne la question de l'utilisateur et la liste des correspondances
connues entre SQLite et MongoDB. Choisis la correspondance la plus
pertinente pour joindre les résultats des deux agents.

Réponds STRICTEMENT avec un JSON de la forme :
{"entity": "<entity>", "sqlite_key": "<colonne>", "mongo_key": "<champ>"}
Aucun texte hors de ce JSON."""


def plan_join(question: str) -> dict[str, Any] | None:
    mappings = load_mappings()
    if not mappings:
        return None
    if len(mappings) == 1:
        mapping = mappings[0]
        return {
            "entity": mapping.get("entity"),
            "sqlite_key": mapping["sqlite"]["key"],
            "sqlite_table": mapping["sqlite"]["table"],
            "mongo_key": mapping["mongodb"]["key"],
            "mongo_collection": mapping["mongodb"]["collection"],
        }

    prompt = f"Correspondances disponibles:\n{mapping_context()}\n\nQuestion: {question}"
    raw = claude_generate(prompt, system=SYSTEM_PROMPT, max_tokens=200)
    try:
        parsed = json.loads(raw)
        entity = parsed["entity"]
    except (json.JSONDecodeError, KeyError, TypeError):
        entity = None

    chosen = next((m for m in mappings if m.get("entity") == entity), mappings[0])
    return {
        "entity": chosen.get("entity"),
        "sqlite_key": chosen["sqlite"]["key"],
        "sqlite_table": chosen["sqlite"]["table"],
        "mongo_key": chosen["mongodb"]["key"],
        "mongo_collection": chosen["mongodb"]["collection"],
    }


def join_planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : appelé uniquement si state['source'] == 'hybrid'.
    Écrit join_plan (ou None si aucun mapping n'existe)."""
    return {"join_plan": plan_join(state["question"])}