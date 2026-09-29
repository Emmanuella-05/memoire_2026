"""Agent SQL, à part entière : il connaît le schéma, génère la requête, la
VALIDE avant de l'exécuter, l'exécute, et se corrige en cas de rejet ou
d'erreur.

Fusionne ce qui était auparavant trois agents séparés (schéma / génération /
évaluation) en un seul.

Pipeline par tentative :
  1. génération (ou correction si tentative précédente rejetée/en échec)
  2. validation déterministe : lecture seule uniquement (tools.sql_tools.is_read_only)
  3. validation sémantique : le LLM vérifie que la requête répond bien à la
     question et référence des tables/colonnes qui existent
  4. exécution, seulement si les deux validations passent
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm import claude_generate
from tools.catalog_tools import load_database_docs
from tools.rag_tools import context as rag_context
from tools.sql_tools import execute_sql, is_read_only

MAX_ATTEMPTS = 3

GENERATE_SYSTEM_PROMPT = """Tu es l'agent SQL de DataTalk.
Tu reçois le schéma des tables SQLite disponibles, du contexte documentaire
optionnel et une question en langage naturel. Tu dois produire UNE requête
SQL en lecture seule (SELECT ou WITH) qui répond à la question.

Réponds STRICTEMENT avec un JSON de la forme :
{"sql": "<requête SQL>"}
Aucun texte, aucune explication, aucun bloc markdown hors de ce JSON."""

VALIDATE_SYSTEM_PROMPT = """Tu es le validateur SQL de DataTalk.
On te donne le schéma des tables SQLite, une question, et une requête SQL
générée pour y répondre. Vérifie que :
- la requête répond effectivement à la question posée
- toutes les tables et colonnes référencées existent dans le schéma
- les comparaisons/filtres sont cohérents avec le type de chaque colonne
  (ex: pas de comparaison numérique sur une colonne texte, format de date
  correct pour une colonne date)
- la requête est syntaxiquement plausible

Réponds STRICTEMENT avec un JSON de la forme :
{"valid": true} ou {"valid": false, "issues": "<explication courte>"}
Aucun texte hors de ce JSON."""

FIX_PROMPT_TEMPLATE = """La requête SQL suivante a été rejetée :
{sql}

Raison :
{issue}

Corrige la requête. Réponds STRICTEMENT avec le même format JSON :
{{"sql": "<requête corrigée>"}}"""


def _schema_summary() -> str:
    docs = load_database_docs("sqlite")
    lines = []
    for table in docs.get("tables", []):
        columns = ", ".join(
            f"{c.get('name', '')}:{c.get('type', 'unknown')}"
            for c in table.get("columns", [])
        )
        lines.append(f"- {table.get('name', '')}({columns})")
    return "\n".join(lines) if lines else "Aucune table documentée."


def _extract_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Could not extract JSON from model output: {raw!r}")
        return json.loads(match.group(0))


def generate_sql(question: str) -> str:
    prompt = (
        f"Schéma SQLite:\n{_schema_summary()}\n\n"
        f"Contexte documentaire:\n{rag_context(question, database='sqlite')}\n\n"
        f"Question: {question}"
    )
    raw = claude_generate(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=500)
    return _extract_json(raw)["sql"].strip()


def validate_sql(question: str, sql: str) -> dict[str, Any]:
    """Validation en deux temps : déterministe (lecture seule) puis
    sémantique (LLM, contre le schéma et la question)."""
    if not is_read_only(sql):
        return {"valid": False, "issues": "La requête n'est pas en lecture seule (SELECT/WITH uniquement)."}

    prompt = f"Schéma SQLite:\n{_schema_summary()}\n\nQuestion: {question}\n\nRequête: {sql}"
    raw = claude_generate(prompt, system=VALIDATE_SYSTEM_PROMPT, max_tokens=300)
    try:
        parsed = _extract_json(raw)
        if parsed.get("valid") is True:
            return {"valid": True, "issues": None}
        return {"valid": False, "issues": parsed.get("issues", "Requête jugée invalide par le validateur.")}
    except ValueError:
        # Réponse du validateur illisible : on ne bloque pas indéfiniment,
        # on laisse passer à l'exécution (execute_sql reste le garde-fou final).
        return {"valid": True, "issues": None}


def run(question: str) -> dict[str, Any]:
    """Génère -> valide -> exécute, avec correction à chaque rejet ou erreur,
    jusqu'à MAX_ATTEMPTS."""
    sql = generate_sql(question)
    last_issue: str | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        validation = validate_sql(question, sql)
        if not validation["valid"]:
            last_issue = validation["issues"]
            if attempt == MAX_ATTEMPTS:
                break
            raw = claude_generate(
                FIX_PROMPT_TEMPLATE.format(sql=sql, issue=last_issue),
                system=GENERATE_SYSTEM_PROMPT,
                max_tokens=500,
            )
            sql = _extract_json(raw)["sql"].strip()
            continue

        try:
            rows = execute_sql(sql)
            return {
                "sql_query": sql,
                "sql_results": rows,
                "sql_error": None,
                "sql_attempts": attempt,
            }
        except Exception as exc:
            last_issue = str(exc)
            if attempt == MAX_ATTEMPTS:
                break
            raw = claude_generate(
                FIX_PROMPT_TEMPLATE.format(sql=sql, issue=last_issue),
                system=GENERATE_SYSTEM_PROMPT,
                max_tokens=500,
            )
            sql = _extract_json(raw)["sql"].strip()

    return {
        "sql_query": sql,
        "sql_results": [],
        "sql_error": last_issue,
        "sql_attempts": MAX_ATTEMPTS,
    }


def sql_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : lit state['question'], écrit sql_query/sql_results/sql_error.

    Ne renvoie que ces clés (pas tout `state`) : requis par LangGraph pour le
    fan-out parallèle avec mongo_agent en mode hybride (sinon conflit
    d'écriture concurrente sur 'question')."""
    return run(state["question"])