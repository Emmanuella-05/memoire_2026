"""Agent result merger : dernier nœud du graphe.

Fusionne les résultats SQL/Mongo (via join_plan si hybride), constate l'état
final (ok / aucune donnée / erreur) et rédige la réponse en langage naturel.

Les échecs ne sont jamais déguisés en « aucune donnée » : une erreur de
requête, de jointure ou de relation inconnue produit un message explicite et
status = "error".
"""

from __future__ import annotations

from typing import Any

from ..llm import LLMError, claude_generate
from ..tools.catalog_tools import get_path, merge_results, normalize_key

SYSTEM_PROMPT = """Tu es l'agent de synthèse de DataTalk.
On te donne la question de l'utilisateur et des données brutes (résultats
de requêtes SQL et/ou MongoDB). Rédige une réponse claire et concise en
langage naturel, en français, qui répond directement à la question à
partir de ces données. Ne mentionne pas les requêtes techniques.
Si des remarques techniques sont fournies, signale brièvement celles qui
limitent la fiabilité de la réponse (lignes sans correspondance, etc.)."""


def _sample_keys(rows: list[dict[str, Any]], key: str, n: int = 3) -> list[str]:
    keys: list[str] = []
    for row in rows:
        value = normalize_key(get_path(row, key))
        if value is not None and value not in keys:
            keys.append(value)
        if len(keys) == n:
            break
    return keys


def merge(state: dict[str, Any]) -> dict[str, Any]:
    """Retourne {"rows", "status", "message", "stats"}.
    status : "ok" | "no_data" | "error" ; message : explication (hors "ok")."""
    source = state.get("source")

    if state.get("error"):
        return {"rows": [], "status": "error", "message": state["error"], "stats": None}

    if source == "sql":
        if state.get("sql_error"):
            return _failure(f"La requête SQL n'a pas pu être exécutée après {state.get('sql_attempts')} tentative(s) : {state['sql_error']}")
        rows = state.get("sql_results") or []
        return _single(rows, "SQLite")

    if source == "mongo":
        if state.get("mongo_error"):
            return _failure(f"Le pipeline MongoDB n'a pas pu être exécuté après {state.get('mongo_attempts')} tentative(s) : {state['mongo_error']}")
        rows = state.get("mongo_results") or []
        return _single(rows, "MongoDB")

    # --- hybride ---
    plan = state.get("join_plan")
    if not plan:
        return _failure(state.get("join_error") or "Impossible de fusionner : aucune relation SQL ↔ MongoDB n'est disponible.")
    if state.get("hybrid_error"):
        return _failure(state["hybrid_error"])
    if plan.get("decomposed"):
        return _merge_dependent(state, plan)
    problems = []
    if state.get("sql_error"):
        problems.append(f"requête SQL : {state['sql_error']}")
    if state.get("mongo_error"):
        problems.append(f"pipeline MongoDB : {state['mongo_error']}")
    if problems:
        note = plan.get("plan_note")
        hint = (f" La question n'a pas pu être décomposée en deux étapes ({note}) : chaque base a donc reçu la "
                "question entière.") if note else ""
        return _failure("La partie hybride a échoué (" + " ; ".join(problems) + ")." + hint)

    sql_rows = state.get("sql_results") or []
    mongo_rows = state.get("mongo_results") or []
    if not sql_rows or not mongo_rows:
        empty = "SQLite" if not sql_rows else "MongoDB"
        return {"rows": [], "status": "no_data", "stats": None,
                "message": f"Aucune donnée n'a été trouvée côté {empty} pour cette question."}

    try:
        merged, stats = merge_results(
            sql_rows, mongo_rows, plan["sqlite_key"], plan["mongo_key"], plan.get("how", "inner")
        )
    except Exception as exc:
        return _failure(f"Erreur lors de la jointure hybride : {exc}")

    if not merged:
        return {
            "rows": [], "status": "no_data", "stats": stats,
            "message": (
                f"Les deux sources ont répondu ({len(sql_rows)} ligne(s) SQLite, {len(mongo_rows)} document(s) MongoDB) "
                f"mais aucune clé ne correspond entre {plan['sqlite_table']}.{plan['sqlite_key']} "
                f"et {plan['mongo_collection']}.{plan['mongo_key']}. "
                f"Exemples de clés SQLite : {_sample_keys(sql_rows, plan['sqlite_key'])} ; "
                f"MongoDB : {_sample_keys(mongo_rows, plan['mongo_key'])}. "
                "Vérifiez la correspondance, ou les filtres de la question."
            ),
        }
    return {"rows": merged, "status": "ok", "message": None, "stats": stats}


_LABELS = {"sql": "SQLite", "mongo": "MongoDB"}


def _has_key(rows: list[dict[str, Any]], key: str) -> bool:
    return any(get_path(row, key) is not None for row in rows)


def _merge_dependent(state: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Plan hybride en deux étapes : la seconde source a déjà été restreinte aux identifiants de
    la première. Son résultat EST la réponse (souvent agrégé : par ville, par catégorie...) ;
    on ne joint en mémoire que si les deux résultats portent la clé commune."""
    first = plan["first"]
    second = "mongo" if first == "sql" else "sql"
    for step, source in ((1, first), (2, second)):
        if state.get(f"{source}_error"):
            return _failure(
                f"Étape {step} ({_LABELS[source]}) en échec après {state.get(f'{source}_attempts')} tentative(s) : "
                f"{state[f'{source}_error']}"
            )
    first_rows = state.get(f"{first}_results") or []
    if not first_rows or not state.get("hybrid_keys"):
        return {"rows": [], "status": "no_data", "stats": None, "message": (
            f"Aucune donnée {_LABELS[first]} ne correspond aux critères de la première étape "
            f"(« {plan[f'{first}_question']} »)."
        )}
    second_rows = state.get(f"{second}_results") or []
    if not second_rows:
        return {"rows": [], "status": "no_data", "stats": None, "message": (
            f"{len(state['hybrid_keys'])} identifiant(s) trouvé(s) côté {_LABELS[first]}, mais aucune donnée "
            f"{_LABELS[second]} ne correspond pour la suite de la question (« {plan[f'{second}_question']} »)."
        )}

    sql_rows = state.get("sql_results") or []
    mongo_rows = state.get("mongo_results") or []
    if _has_key(sql_rows, plan["sqlite_key"]) and _has_key(mongo_rows, plan["mongo_key"]):
        try:
            merged, stats = merge_results(sql_rows, mongo_rows, plan["sqlite_key"], plan["mongo_key"],
                                          plan.get("how", "inner"))
        except Exception as exc:
            return _failure(f"Erreur lors de la jointure hybride : {exc}")
        if merged:
            return {"rows": merged, "status": "ok", "message": None, "stats": stats}
    return {"rows": second_rows, "status": "ok", "message": None, "stats": None}


def _failure(message: str) -> dict[str, Any]:
    return {"rows": [], "status": "error", "message": message, "stats": None}


def _single(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not rows:
        return {"rows": [], "status": "no_data", "stats": None,
                "message": f"Aucune donnée {label} ne correspond à cette question."}
    return {"rows": rows, "status": "ok", "message": None, "stats": None}


def _remarks(stats: dict[str, Any] | None) -> str:
    if not stats:
        return ""
    notes = []
    if stats["unmatched_left_rows"]:
        notes.append(f"{stats['unmatched_left_rows']} ligne(s) SQLite sans correspondance MongoDB")
    if stats["unmatched_right_rows"]:
        notes.append(f"{stats['unmatched_right_rows']} document(s) MongoDB sans correspondance SQLite")
    if stats["one_to_many"]:
        notes.append("relation un-à-plusieurs : une ligne SQLite peut apparaître plusieurs fois")
    return "\n\nRemarques techniques : " + " ; ".join(notes) if notes else ""


def write_answer(question: str, rows: list[dict[str, Any]], stats: dict[str, Any] | None = None) -> str:
    prompt = f"Question: {question}\n\nDonnées (extrait, max 50 lignes sur {len(rows)}): {rows[:50]}{_remarks(stats)}"
    try:
        return claude_generate(prompt, system=SYSTEM_PROMPT, max_tokens=800)
    except LLMError as exc:
        # Les données existent : on les renvoie même si la rédaction est impossible.
        return f"{len(rows)} résultat(s) trouvé(s). La rédaction automatique de la réponse est indisponible ({exc})."


def result_merger_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph terminal : écrit merged_results, answer, status, error, join_stats."""
    outcome = merge(state)
    if outcome["status"] == "ok":
        answer = write_answer(state["question"], outcome["rows"], outcome["stats"])
        error = None
    else:
        answer = outcome["message"]
        error = outcome["message"] if outcome["status"] == "error" else None
    return {
        "merged_results": outcome["rows"],
        "answer": answer,
        "status": outcome["status"],
        "error": error,
        "join_stats": outcome["stats"],
    }