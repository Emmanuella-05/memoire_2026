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
import os
from typing import Any

from ..llm import LLMError, claude_json, fix_with_open_models, open_chat, open_models_enabled
from ..tools.catalog_tools import DataSourceUnavailable, get_path, load_database_docs, mongo_connection
from ..tools.hybrid_tools import DEDUP_NOTE, coerce_for_mongo, display_pipeline, distinct_keys_pipeline, with_selected_keys
from ..tools.mongo_fields import pipeline_field_issues
from ..tools.mongo_normalizer import normalize_mongo_output
from ..tools.mongo_tools import collection_rejection_reason, execute_mongo, pipeline_rejection_reason
from ..tools.rag_tools import context as rag_context
from ..tools.schema_tools import mongo_schema_summary
from ..tools.translate_tools import to_english, translate_context_enabled

MAX_ATTEMPTS = 3
_INFRA_ERRORS = (LLMError, DataSourceUnavailable)

# --- Modèles ouverts (Qwen2.5-Coder) : le modèle écrit une commande mongosh naturelle pour lui ; le
# convertisseur (tools/mongo_normalizer.py) en tire (collection, pipeline), même si la sortie est incomplète
# (filtre seul, étape seule, `find` absent, syntaxe JavaScript...). Les contrôles déterministes
# (collection connue, opérateurs interdits, champs existants) s'appliquent ensuite.
OPEN_GENERATE_SYSTEM_PROMPT = """You are the MongoDB query agent of DataTalk.
You receive the MongoDB collections (fields with types and exact possible values), optional business rules, and a question.
Write ONE read-only MongoDB command that answers it, in mongosh syntax:
  db.<collection>.find(<filter>, <projection>).sort(...).limit(...)
  db.<collection>.aggregate([ <stages> ])
Rules:
- ALWAYS start with db.<collection>. and use ONLY collections and fields listed in the schema.
- Use aggregate() for grouping, counting, averages, joins ($lookup) and any computed result; find() for simple filters.
- Each pipeline stage contains exactly one operator. Never use $out, $merge, $function, $accumulator or $where.
- For fields with listed possible values, use those values exactly as written.
- Fields of type date or objectId are compared with extended JSON: {"$date": "2024-01-31"} or {"$oid": "<24 hex chars>"}.
Reply with ONLY the command inside one ```javascript code block, nothing else.

Examples:
Question: customers living in Paris
```javascript
db.customers.find({"city": "Paris"})
```
Question: number of orders per customer, highest first
```javascript
db.orders.aggregate([{"$group": {"_id": "$customer_id", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}])
```"""

OPEN_FIX_PROMPT_TEMPLATE = """MongoDB schema:
{schema}

Business rules:
{rules}

Question: {question}

This command was rejected:
db.{collection}.aggregate({pipeline})

Reason:
{issue}
{requirement}
Write the corrected command (same format: only one ```javascript code block)."""

GENERATE_SYSTEM_PROMPT = """Tu es l'agent MongoDB de DataTalk.
Tu reçois le schéma inféré des collections MongoDB disponibles, du contexte
documentaire optionnel et une question en langage naturel. Tu dois produire
UN pipeline d'agrégation MongoDB (lecture seule : aucune étape $out/$merge,
aucun $function/$accumulator/$where) qui répond à la question. Chaque étape
du pipeline contient un seul opérateur.
Les champs de type « date » ou « objectId » se comparent avec le JSON étendu :
{"$date": "2024-01-31"} ou {"$date": "2024-01-31T00:00:00Z"} pour une date,
{"$oid": "<24 caractères hexadécimaux>"} pour un objectId (jamais une chaîne simple).

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
    """Collections, champs typés et valeurs exactes des champs catégoriels."""
    return mongo_schema_summary()


SELECTED_KEYS_NOTE = (
    "(Note : le filtre sur les identifiants sélectionnés par l'étape précédente est déjà appliqué "
    "automatiquement en tête du pipeline.)"
)


def _join_requirement(join_plan: dict[str, Any] | None, step: str | None = None) -> str:
    if not join_plan:
        return ""
    if step == "first":
        return (
            f"\n\nÉtape 1 d'une question hybride : retourne, pour les documents de la collection "
            f"{join_plan.get('mongo_collection')} qui répondent à la question, le champ '{join_plan.get('mongo_key')}' "
            "à la racine du résultat. Les doublons sont éliminés automatiquement : n'ajoute pas de $group pour "
            "dédoublonner. Si la condition porte sur un calcul par entité (moyenne, somme, nombre), fais ce calcul "
            f"avec $group sur '{join_plan.get('mongo_key')}', filtre dessus, puis $project pour renvoyer "
            f"'{join_plan.get('mongo_key')}' à la racine."
        )
    if step == "second":
        return (
            f"\n\nÉtape 2 d'une question hybride : utilise la collection '{join_plan.get('mongo_collection')}'. "
            f"Un filtre sur le champ '{join_plan.get('mongo_key')}' (identifiants déjà sélectionnés par l'étape "
            "précédente) est appliqué AUTOMATIQUEMENT avant ton pipeline : n'écris pas ce filtre et aucun identifiant."
        )
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


def _fields_by_collection() -> dict[str, list[str]]:
    return {
        c["name"]: [f.get("name", "") for f in c.get("fields", []) if f.get("name")]
        for c in load_database_docs("mongodb").get("collections", []) if c.get("name")
    }


def _open_requirement(join_plan: dict[str, Any] | None, step: str | None) -> str:
    if not join_plan:
        return ""
    collection, key = join_plan.get("mongo_collection"), join_plan.get("mongo_key")
    if step == "first":
        return (f"\n\nStep 1 of a cross-database question: for the documents of collection {collection} that answer the "
                f"question, return field '{key}' at the root of the result. Duplicates are removed automatically: do NOT add "
                f"a $group just to deduplicate. If the condition depends on a per-entity computation (average, sum, count), "
                f"compute it with $group on '{key}', filter on it, then $project to return '{key}' at the root.")
    if step == "second":
        return (f"\n\nStep 2 of a cross-database question: use collection '{collection}'. A filter on field '{key}' "
                "(identifiers already selected by the previous step) is applied AUTOMATICALLY before your pipeline: "
                "do not write that filter and never write identifiers.")
    return (f"\n\nThis result will be joined to SQLite: it MUST return, without aggregating it, the field '{key}' "
            f"(collection {collection}).")


def _open_rules(question: str) -> str:
    ctx = rag_context(question, database="mongodb")
    if ctx.strip() and translate_context_enabled():
        ctx = to_english(ctx)
    return ctx.strip() or "(none)"


def _convert(raw: str, join_plan: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    """Commande brute du modèle -> (collection, pipeline). Lève ValueError si inexploitable (=> correction)."""
    default = join_plan.get("mongo_collection") if join_plan else None
    return normalize_mongo_output(raw, _known_collections(), default, _fields_by_collection(),
                                  extended_json=mongo_connection() is not None)


def generate_pipeline(question: str, join_plan: dict[str, Any] | None = None,
                      step: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    if open_models_enabled():
        prompt = (
            f"MongoDB schema:\n{mongo_schema_summary('en')}\n\n"
            f"Business rules:\n{_open_rules(question)}\n\n"
            f"Question: {to_english(question)}{_open_requirement(join_plan, step)}"
        )
        return _convert(open_chat(prompt, system=OPEN_GENERATE_SYSTEM_PROMPT, role="mongo", max_tokens=700), join_plan)
    prompt = (
        f"Schéma MongoDB:\n{_schema_summary()}\n\n"
        f"Contexte documentaire:\n{rag_context(question, database='mongodb')}\n\n"
        f"Question: {question}{_join_requirement(join_plan, step)}"
    )
    return _read_pipeline(claude_json(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=700))


def fix_pipeline(question: str, collection: str, pipeline: list[dict[str, Any]], issue: str,
                 join_plan: dict[str, Any] | None = None,
                 step: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    if fix_with_open_models():
        prompt = OPEN_FIX_PROMPT_TEMPLATE.format(
            schema=mongo_schema_summary("en"), rules=_open_rules(question), question=to_english(question),
            collection=collection, pipeline=json.dumps(pipeline, ensure_ascii=False, default=str), issue=issue,
            requirement=_open_requirement(join_plan, step),
        )
        return _convert(open_chat(prompt, system=OPEN_GENERATE_SYSTEM_PROMPT, role="mongo", max_tokens=700), join_plan)
    prompt = FIX_PROMPT_TEMPLATE.format(
        schema=_schema_summary(), question=question, collection=collection,
        pipeline=json.dumps(pipeline, ensure_ascii=False, default=str), issue=issue,
        join_requirement=_join_requirement(join_plan, step),
    )
    return _read_pipeline(claude_json(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=700))


def validate_pipeline(question: str, collection: str, pipeline: list[dict[str, Any]],
                      executable: list[dict[str, Any]] | None = None, note: str | None = None) -> dict[str, Any]:
    """Validation en deux temps : déterministe (collection connue, aucun
    opérateur interdit) puis sémantique (LLM). `executable` est le pipeline réellement
    exécuté (avec le filtre sur les identifiants) quand il diffère de celui du modèle."""
    known = _known_collections()
    reason = collection_rejection_reason(collection, known) or pipeline_rejection_reason(executable or pipeline, known)
    if not reason and open_models_enabled() and os.getenv("DATATALK_MONGO_FIELD_CHECK", "1").strip().lower() not in ("0", "false", "off", "no"):
        # Un petit modèle invente des champs ; MongoDB, lui, renvoie alors zéro résultat sans rien dire.
        reason = pipeline_field_issues(collection, pipeline, _fields_by_collection())
    if reason:
        return {"valid": False, "issues": reason}

    note_text = f"\n{note}" if note else ""
    prompt = (
        f"Schéma MongoDB:\n{_schema_summary()}\n\nQuestion: {question}{note_text}\n\n"
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


def run(question: str, join_plan: dict[str, Any] | None = None, injected_keys: list[Any] | None = None,
        step: str | None = None) -> dict[str, Any]:
    """Génère -> valide -> exécute, avec correction à chaque rejet ou erreur,
    jusqu'à MAX_ATTEMPTS.

    Plan hybride en deux étapes : `question` est alors la SOUS-QUESTION de cette source.
    step="first"  : le pipeline doit retourner les identifiants distincts de la clé de jointure ;
    step="second" : `injected_keys` (identifiants de l'étape 1) est injecté par le code, sous la
                    forme d'un $match {clé: {$in: [...]}} placé en tête du pipeline du modèle."""
    current: tuple[str, list[dict[str, Any]]] | None = None
    issue: str | None = None
    keys: list[Any] = []
    if step == "second":
        keys = coerce_for_mongo(injected_keys or [], join_plan["mongo_collection"], join_plan["mongo_key"])
    translated = to_english(question)       # identique à `question` si les modèles ouverts ne sont pas utilisés
    translated_question = translated if translated != question else None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            current = (
                generate_pipeline(question, join_plan, step) if current is None
                else fix_pipeline(question, current[0], current[1], issue or "", join_plan, step)
            )
        except _INFRA_ERRORS:
            raise
        except Exception as exc:
            issue = f"Génération impossible : {exc}"
            continue
        collection, pipeline = current

        if step == "second" and collection != join_plan["mongo_collection"]:
            issue = (f"Utilise la collection '{join_plan['mongo_collection']}' : c'est elle qui porte la clé de "
                     f"jointure '{join_plan['mongo_key']}' sur laquelle les identifiants sont filtrés.")
            continue
        executable = None
        note = None
        if step == "second":
            executable, note = with_selected_keys(pipeline, join_plan["mongo_key"], keys), SELECTED_KEYS_NOTE
        elif step == "first":
            deduplicated = distinct_keys_pipeline(pipeline, join_plan["mongo_key"])
            executable, note = (deduplicated if deduplicated != pipeline else None), DEDUP_NOTE

        validation = validate_pipeline(question, collection, pipeline, executable, note)
        if not validation["valid"]:
            issue = validation["issues"]
            continue

        try:
            rows = execute_mongo(collection, executable or pipeline, _known_collections())
        except _INFRA_ERRORS:
            raise
        except Exception as exc:
            issue = str(exc)
            continue

        key = join_plan.get("mongo_key") if join_plan else None
        if key and step != "second" and rows and not any(get_path(row, key) is not None for row in rows):
            issue = f"Le pipeline MongoDB ne retourne pas la clé de jointure '{key}'."
            continue

        return {
            "mongo_collection": collection,
            "mongo_pipeline": display_pipeline(executable, keys) if executable else pipeline,
            "mongo_results": rows,
            "mongo_error": None,
            "mongo_attempts": attempt,
            "translated_question": translated_question,
        }

    return {
        "mongo_collection": current[0] if current else None,
        "mongo_pipeline": (display_pipeline(with_selected_keys(current[1], join_plan["mongo_key"], keys), keys)
                           if current and step == "second" else (current[1] if current else None)),
        "mongo_results": [],
        "mongo_error": issue,
        "mongo_attempts": MAX_ATTEMPTS,
        "translated_question": translated_question,
    }


def mongo_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : lit state['question'] (et join_plan en mode hybride),
    écrit mongo_collection/mongo_pipeline/mongo_results/mongo_error/mongo_attempts."""
    return run(state["question"], state.get("join_plan"))