"""Agent MongoDB, à part entière : il connaît le schéma inféré, génère le
pipeline d'agrégation, le VALIDE avant de l'exécuter, l'exécute, et se
corrige en cas de rejet ou d'erreur.

Symétrique de sql_agent.py.

Pipeline par tentative :
  1. génération (ou correction si tentative précédente rejetée/en échec)
  2. validation déterministe : aucune étape d'écriture/admin
     (tools.mongo_tools.is_pipeline_safe)
  3. validation sémantique : le LLM vérifie que le pipeline répond bien à la
     question et référence une collection/des champs qui existent
  4. exécution, seulement si les deux validations passent
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..llm import claude_generate
from ..tools.catalog_tools import load_database_docs
from ..tools.rag_tools import context as rag_context
from ..tools.mongo_tools import execute_mongo, is_pipeline_safe

MAX_ATTEMPTS = 3

GENERATE_SYSTEM_PROMPT = """Tu es l'agent MongoDB de DataTalk.
Tu reçois le schéma inféré des collections MongoDB disponibles, du contexte
documentaire optionnel et une question en langage naturel. Tu dois produire
UN pipeline d'agrégation MongoDB (lecture seule, aucune étape $out/$merge)
qui répond à la question.

Réponds STRICTEMENT avec un JSON de la forme :
{"collection": "<nom_collection>", "pipeline": [<étapes d'agrégation>]}
Aucun texte, aucune explication, aucun bloc markdown hors de ce JSON."""

VALIDATE_SYSTEM_PROMPT = """Tu es le validateur MongoDB de DataTalk.
On te donne le schéma des collections MongoDB, une question, et un pipeline
d'agrégation généré pour y répondre. Vérifie que :
- le pipeline répond effectivement à la question posée
- la collection et les champs référencés existent dans le schéma
- les comparaisons/filtres sont cohérents avec le type de chaque champ
  (ex: pas de comparaison numérique sur un champ string)
- le pipeline est syntaxiquement plausible

Réponds STRICTEMENT avec un JSON de la forme :
{"valid": true} ou {"valid": false, "issues": "<explication courte>"}
Aucun texte hors de ce JSON."""

FIX_PROMPT_TEMPLATE = """Le pipeline MongoDB suivant a été rejeté :
Collection: {collection}
Pipeline: {pipeline}

Raison :
{issue}

Corrige le pipeline. Réponds STRICTEMENT avec le même format JSON :
{{"collection": "<nom_collection>", "pipeline": [<étapes corrigées>]}}"""


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


def _extract_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Could not extract JSON from model output: {raw!r}")
        return json.loads(match.group(0))


def generate_pipeline(question: str) -> tuple[str, list[dict[str, Any]]]:
    prompt = (
        f"Schéma MongoDB:\n{_schema_summary()}\n\n"
        f"Contexte documentaire:\n{rag_context(question, database='mongodb')}\n\n"
        f"Question: {question}"
    )
    raw = claude_generate(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=500)
    parsed = _extract_json(raw)
    return parsed["collection"], parsed["pipeline"]


def validate_pipeline(question: str, collection: str, pipeline: list[dict[str, Any]]) -> dict[str, Any]:
    """Validation en deux temps : déterministe (pas d'étape d'écriture/admin)
    puis sémantique (LLM, contre le schéma et la question)."""
    if not is_pipeline_safe(pipeline):
        return {"valid": False, "issues": "Le pipeline contient une étape d'écriture ou d'administration interdite."}

    prompt = (
        f"Schéma MongoDB:\n{_schema_summary()}\n\nQuestion: {question}\n\n"
        f"Collection: {collection}\nPipeline: {pipeline}"
    )
    raw = claude_generate(prompt, system=VALIDATE_SYSTEM_PROMPT, max_tokens=300)
    try:
        parsed = _extract_json(raw)
        if parsed.get("valid") is True:
            return {"valid": True, "issues": None}
        return {"valid": False, "issues": parsed.get("issues", "Pipeline jugé invalide par le validateur.")}
    except ValueError:
        # Réponse du validateur illisible : on ne bloque pas indéfiniment,
        # on laisse passer à l'exécution (execute_mongo reste le garde-fou final).
        return {"valid": True, "issues": None}


def run(question: str) -> dict[str, Any]:
    """Génère -> valide -> exécute, avec correction à chaque rejet ou erreur,
    jusqu'à MAX_ATTEMPTS."""
    collection, pipeline = generate_pipeline(question)
    last_issue: str | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        validation = validate_pipeline(question, collection, pipeline)
        if not validation["valid"]:
            last_issue = validation["issues"]
            if attempt == MAX_ATTEMPTS:
                break
            raw = claude_generate(
                FIX_PROMPT_TEMPLATE.format(collection=collection, pipeline=pipeline, issue=last_issue),
                system=GENERATE_SYSTEM_PROMPT,
                max_tokens=500,
            )
            parsed = _extract_json(raw)
            collection, pipeline = parsed["collection"], parsed["pipeline"]
            continue

        try:
            rows = execute_mongo(collection, pipeline)
            return {
                "mongo_collection": collection,
                "mongo_pipeline": pipeline,
                "mongo_results": rows,
                "mongo_error": None,
                "mongo_attempts": attempt,
            }
        except Exception as exc:
            last_issue = str(exc)
            if attempt == MAX_ATTEMPTS:
                break
            raw = claude_generate(
                FIX_PROMPT_TEMPLATE.format(collection=collection, pipeline=pipeline, issue=last_issue),
                system=GENERATE_SYSTEM_PROMPT,
                max_tokens=500,
            )
            parsed = _extract_json(raw)
            collection, pipeline = parsed["collection"], parsed["pipeline"]

    return {
        "mongo_collection": collection,
        "mongo_pipeline": pipeline,
        "mongo_results": [],
        "mongo_error": last_issue,
        "mongo_attempts": MAX_ATTEMPTS,
    }


def mongo_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : lit state['question'], écrit mongo_collection/mongo_pipeline/mongo_results/mongo_error.

    Ne renvoie que ces clés (pas tout `state`) : requis par LangGraph pour le
    fan-out parallèle avec sql_agent en mode hybride (sinon conflit
    d'écriture concurrente sur 'question')."""
    return run(state["question"])