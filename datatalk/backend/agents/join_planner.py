"""Agent join planner : s'exécute seulement quand le classifieur a choisi
"hybrid". Il vérifie que la relation entre les deux sources est CONNUE et choisit
la correspondance du catalogue (tools/catalog_tools.py) qui servira à joindre
les résultats SQL et MongoDB.

Une question « hybride » n'est exécutée que si l'on sait comment relier les
données. Sinon, aucune requête n'est lancée : join_error explique pourquoi.
"""

from __future__ import annotations

from typing import Any

from ..llm import claude_json
from ..tools.catalog_tools import load_mappings

SYSTEM_PROMPT = """Tu es le planificateur de jointure de DataTalk.
On te donne la question de l'utilisateur et la liste numérotée des
correspondances connues entre SQLite et MongoDB. Choisis la correspondance la
plus pertinente pour joindre les résultats des deux bases, et le type de jointure :
- "inner" (par défaut) : seules les lignes présentes dans les deux sources
- "left" : toutes les lignes SQLite, même sans données MongoDB (ex. « tous les
  clients, avec leurs avis s'il y en a »)

Réponds STRICTEMENT avec un JSON de la forme :
{"mapping": <numéro>, "join_type": "inner" | "left"}
Aucun texte hors de ce JSON."""


def _plan(mapping: dict[str, Any], how: str = "inner") -> dict[str, Any]:
    return {
        "entity": mapping.get("entity"),
        "sqlite_table": mapping["sqlite"]["table"],
        "sqlite_key": mapping["sqlite"]["key"],
        "mongo_collection": mapping["mongodb"]["collection"],
        "mongo_key": mapping["mongodb"]["key"],
        "how": how,
        "status": mapping.get("status"),
        "confidence": mapping.get("confidence"),
    }


def plan_join(question: str) -> tuple[dict[str, Any] | None, str | None]:
    """Retourne (plan, None) si la relation est identifiée, sinon (None, explication)."""
    mappings = load_mappings()
    if not mappings:
        return None, (
            "Cette question nécessite de combiner SQLite et MongoDB, mais aucune correspondance "
            "n'est définie entre les deux bases (ni détectée automatiquement, ni ajoutée manuellement). "
            "Ajoutez ou confirmez une correspondance (ex. clients.id ↔ avis.customer_id) puis reposez la question."
        )
    if len(mappings) == 1:
        return _plan(mappings[0]), None

    listing = "\n".join(
        f"{i}. {m.get('entity')}: SQLite {m['sqlite']['table']}({m['sqlite']['key']}) "
        f"-> MongoDB {m['mongodb']['collection']}({m['mongodb']['key']})"
        for i, m in enumerate(mappings, start=1)
    )
    try:
        parsed = claude_json(
            f"Correspondances disponibles:\n{listing}\n\nQuestion: {question}",
            system=SYSTEM_PROMPT, max_tokens=100,
        )
        number = int(parsed["mapping"])
        if not 1 <= number <= len(mappings):
            raise ValueError("numéro hors liste")
    except (ValueError, KeyError, TypeError) as exc:
        return None, f"Plusieurs correspondances sont possibles et aucune n'a pu être choisie de façon fiable ({exc})."
    how = parsed.get("join_type") if parsed.get("join_type") in ("inner", "left") else "inner"
    return _plan(mappings[number - 1], how), None


def join_planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : écrit join_plan, ou join_error si la relation est inconnue."""
    plan, error = plan_join(state["question"])
    return {"join_plan": plan, "join_error": error}