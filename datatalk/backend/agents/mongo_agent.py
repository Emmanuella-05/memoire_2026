"""Agent MongoDB, à part entière : il connaît le schéma inféré, génère le
pipeline d'agrégation, le VALIDE avant de l'exécuter, l'exécute, et se
corrige en cas de rejet ou d'erreur.

Symétrique de sql_agent.py.

Pipeline par tentative (3 au maximum) :
  1. génération (ou correction de la tentative précédente)
  2. validation déterministe : collection connue, aucun opérateur d'écriture,
     de code JavaScript ou d'administration, à aucune profondeur
     (tools.mongo_tools.pipeline_rejection_reason)
  3. validation sémantique : le LLM vérifie que le pipeline répond à la question
  4. exécution, seulement si les validations passent
"""

from __future__ import annotations

import json
from typing import Any

from ..llm import LLMError, claude_json
from ..tools.catalog_tools import DataSourceUnavailable, get_path, load_database_docs
from ..tools.mongo_tools import collection_rejection_reason, execute_mongo, pipeline_rejection_reason
from ..tools.rag_tools import context as rag_context

MAX_ATTEMPTS = 3
_INFRA_ERRORS = (LLMError, DataSourceUnavailable)

GENERATE_SYSTEM_PROMPT = """Tu es l'agent MongoDB de DataTalk.
Tu reçois le schéma inféré des collections MongoDB disponibles, du contexte
documentaire optionnel et une question en langage naturel. Tu dois produire
UN pipeline d'agrégation MongoDB (lecture seule : aucune étape $out/$merge,
aucun $function/$accumulator/$where) qui répond à la question. Chaque étape
du pipeline contient un seul opérateur.

Réponds STRICTEMENT avec un JSON de la forme :
{"collection": "<nom_collection>", "pipeline": [<étapes d'agrégation>]}
Aucun texte, aucune explication, aucun bloc markdown hors de ce JSON."""

VALIDATE_SYSTEM_PROMPT = """Tu es le validateur MongoDB de DataTalk.
On te donne le schéma des collections MongoDB, une question, et un pipeline
d'agrégation généré pour y répondre. Vérifie que :
- le pipeline répond effectivement à la question posée
- les champs référencés existent dans le schéma
- les comparaisons/filtres sont cohérents avec le type de chaque champ
  (ex: pas de comparaison numérique sur un champ string)
- les étapes ($group, $lookup, $unwind...) ont un sens métier

Réponds STRICTEMENT avec un JSON de la forme :
{"valid": true} ou {"valid": false, "issues": "<explication courte>"}
Aucun texte hors de ce JSON."""

FIX_PROMPT_TEMPLATE = """Schéma MongoDB:
{schema}

Question: {question}

Le pipeline MongoDB suivant a été rejeté :
Collection: {collection}
Pipeline: {pipeline}

Raison :
{issue}
{join_requirement}
Corrige le pipeline. Réponds STRICTEMENT avec le même format JSON :
{{"collection": "<nom_collection>", "pipeline": [<étapes corrigées>]}}"""


def _known_collections() -> list[str]:
    return [c.get("name", "") for c in load_database_docs("mongodb").get("collections", []) if c.get("name")]


def _schema_summary() -> str:
    docs = load_database_docs("mongodb")
    lines = []
    for collection in docs.get("collections", []):
        fields = ", ".join(
            f"{f.get('name', '')}:{f.get('type', 'unknown')}"
            for f in collection.get("fields", [])
        )
        lines.append(f"- {collection.get('name', '')}({fields})")
    return "\n".join(lines) if lines else "Aucune collection documentée."


def _join_requirement(join_plan: dict[str, Any] | None) -> str:
    if not join_plan:
        return ""
    return (
        "\n\nContrainte HYBRID : ce pipeline sera joint à SQLite. "
        f"Il doit impérativement retourner, sans l'agréger, le champ de jointure "
        f"'{join_plan.get('mongo_key')}' (collection {join_plan.get('mongo_collection')}) dans ses résultats."
    )


def _read_pipeline(parsed: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    collection, pipeline = parsed.get("collection"), parsed.get("pipeline")
    if not isinstance(collection, str) or not isinstance(pipeline, list):
        raise ValueError("La réponse du modèle doit contenir 'collection' (texte) et 'pipeline' (liste).")
    return collection, pipeline


def generate_pipeline(question: str, join_plan: dict[str, Any] | None = None) -> tuple[str, list[dict[str, Any]]]:
    prompt = (
        f"Schéma MongoDB:\n{_schema_summary()}\n\n"
        f"Contexte documentaire:\n{rag_context(question, database='mongodb')}\n\n"
        f"Question: {question}{_join_requirement(join_plan)}"
    )
    return _read_pipeline(claude_json(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=700))


def fix_pipeline(question: str, collection: str, pipeline: list[dict[str, Any]], issue: str,
                 join_plan: dict[str, Any] | None = None) -> tuple[str, list[dict[str, Any]]]:
    prompt = FIX_PROMPT_TEMPLATE.format(
        schema=_schema_summary(), question=question, collection=collection,
        pipeline=json.dumps(pipeline, ensure_ascii=False, default=str), issue=issue,
        join_requirement=_join_requirement(join_plan),
    )
    return _read_pipeline(claude_json(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=700))


def validate_pipeline(question: str, collection: str, pipeline: list[dict[str, Any]]) -> dict[str, Any]:
    """Validation en deux temps : déterministe (collection connue, aucun
    opérateur interdit) puis sémantique (LLM)."""
    known = _known_collections()
    reason = collection_rejection_reason(collection, known) or pipeline_rejection_reason(pipeline, known)
    if reason:
        return {"valid": False, "issues": reason}

    prompt = (
        f"Schéma MongoDB:\n{_schema_summary()}\n\nQuestion: {question}\n\n"
        f"Collection: {collection}\nPipeline: {json.dumps(pipeline, ensure_ascii=False, default=str)}"
    )
    try:
        parsed = claude_json(prompt, system=VALIDATE_SYSTEM_PROMPT, max_tokens=300)
    except ValueError:
        # Validateur illisible : le pipeline a déjà passé les contrôles
        # déterministes et execute_mongo reste le garde-fou final.
        return {"valid": True, "issues": None}
    if parsed.get("valid") is True:
        return {"valid": True, "issues": None}
    return {"valid": False, "issues": parsed.get("issues") or "Pipeline jugé invalide par le validateur."}


def run(question: str, join_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Génère -> valide -> exécute, avec correction à chaque rejet ou erreur,
    jusqu'à MAX_ATTEMPTS."""
    current: tuple[str, list[dict[str, Any]]] | None = None
    issue: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            current = (
                generate_pipeline(question, join_plan) if current is None
                else fix_pipeline(question, current[0], current[1], issue or "", join_plan)
            )
        except _INFRA_ERRORS:
            raise
        except Exception as exc:
            issue = f"Génération impossible : {exc}"
            continue
        collection, pipeline = current

        validation = validate_pipeline(question, collection, pipeline)
        if not validation["valid"]:
            issue = validation["issues"]
            continue

        try:
            rows = execute_mongo(collection, pipeline, _known_collections())
        except _INFRA_ERRORS:
            raise
        except Exception as exc:
            issue = str(exc)
            continue

        key = join_plan.get("mongo_key") if join_plan else None
        if key and rows and not any(get_path(row, key) is not None for row in rows):
            issue = f"Le pipeline MongoDB ne retourne pas la clé de jointure '{key}'."
            continue

        return {
            "mongo_collection": collection,
            "mongo_pipeline": pipeline,
            "mongo_results": rows,
            "mongo_error": None,
            "mongo_attempts": attempt,
        }

    return {
        "mongo_collection": current[0] if current else None,
        "mongo_pipeline": current[1] if current else None,
        "mongo_results": [],
        "mongo_error": issue,
        "mongo_attempts": MAX_ATTEMPTS,
    }


def mongo_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : lit state['question'] (et join_plan en mode hybride),
    écrit mongo_collection/mongo_pipeline/mongo_results/mongo_error/mongo_attempts."""
    return run(state["question"], state.get("join_plan"))