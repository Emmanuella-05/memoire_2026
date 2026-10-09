"""Agent SQL, à part entière : il connaît le schéma, génère la requête, la
VALIDE avant de l'exécuter, l'exécute, et se corrige en cas de rejet ou
d'erreur.

Fusionne ce qui était auparavant trois agents séparés (schéma / génération /
évaluation) en un seul.

Pipeline par tentative (3 au maximum) :
  1. génération (ou correction de la tentative précédente)
  2. validation déterministe : lecture seule (tools.sql_tools.sql_rejection_reason)
     puis compilation par SQLite (EXPLAIN : tables/colonnes inconnues, syntaxe)
  3. validation sémantique : le LLM vérifie que la requête répond à la question
  4. exécution, seulement si les validations passent

Gestion d'erreurs : une requête mal formée ou rejetée est une « tentative
ratée » (on corrige). Une infrastructure indisponible (LLM, base absente) n'est
pas corrigible : l'exception remonte (LLMError / DataSourceUnavailable).
"""

from __future__ import annotations

from typing import Any

from ..llm import LLMError, claude_json
from ..tools.catalog_tools import DataSourceUnavailable, load_database_docs
from ..tools.rag_tools import context as rag_context
from ..tools.sql_tools import execute_sql, sql_compile_error, sql_rejection_reason

MAX_ATTEMPTS = 3
_INFRA_ERRORS = (LLMError, DataSourceUnavailable)

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
- les comparaisons/filtres sont cohérents avec le type de chaque colonne
  (ex: pas de comparaison numérique sur une colonne texte, format de date
  correct pour une colonne date)
- les agrégations et jointures ont un sens métier

Réponds STRICTEMENT avec un JSON de la forme :
{"valid": true} ou {"valid": false, "issues": "<explication courte>"}
Aucun texte hors de ce JSON."""

FIX_PROMPT_TEMPLATE = """Schéma SQLite:
{schema}

Question: {question}

La requête SQL suivante a été rejetée :
{sql}

Raison :
{issue}
{join_requirement}
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


def _join_requirement(join_plan: dict[str, Any] | None) -> str:
    if not join_plan:
        return ""
    return (
        "\n\nContrainte HYBRID : cette requête sera jointe à MongoDB. "
        f"Elle doit impérativement retourner, sans l'agréger, la colonne de jointure "
        f"'{join_plan.get('sqlite_key')}' (table {join_plan.get('sqlite_table')}) dans ses résultats."
    )


def generate_sql(question: str, join_plan: dict[str, Any] | None = None) -> str:
    prompt = (
        f"Schéma SQLite:\n{_schema_summary()}\n\n"
        f"Contexte documentaire:\n{rag_context(question, database='sqlite')}\n\n"
        f"Question: {question}{_join_requirement(join_plan)}"
    )
    return _read_sql(claude_json(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=500))


def fix_sql(question: str, sql: str, issue: str, join_plan: dict[str, Any] | None = None) -> str:
    prompt = FIX_PROMPT_TEMPLATE.format(
        schema=_schema_summary(), question=question, sql=sql, issue=issue,
        join_requirement=_join_requirement(join_plan),
    )
    return _read_sql(claude_json(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=500))


def _read_sql(parsed: dict[str, Any]) -> str:
    sql = parsed.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("La réponse du modèle ne contient pas de champ 'sql' exploitable.")
    return sql.strip()


def validate_sql(question: str, sql: str) -> dict[str, Any]:
    """Validation en trois temps : textuelle (lecture seule), compilation par
    SQLite, puis sémantique (LLM)."""
    reason = sql_rejection_reason(sql)
    if reason:
        return {"valid": False, "issues": reason}
    compile_error = sql_compile_error(sql)
    if compile_error:
        return {"valid": False, "issues": compile_error}

    prompt = f"Schéma SQLite:\n{_schema_summary()}\n\nQuestion: {question}\n\nRequête: {sql}"
    try:
        parsed = claude_json(prompt, system=VALIDATE_SYSTEM_PROMPT, max_tokens=300)
    except ValueError:
        # Validateur illisible : on ne bloque pas, la requête a déjà passé les
        # contrôles déterministes et execute_sql reste le garde-fou final.
        return {"valid": True, "issues": None}
    if parsed.get("valid") is True:
        return {"valid": True, "issues": None}
    return {"valid": False, "issues": parsed.get("issues") or "Requête jugée invalide par le validateur."}


def run(question: str, join_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Génère -> valide -> exécute, avec correction à chaque rejet ou erreur,
    jusqu'à MAX_ATTEMPTS."""
    sql: str | None = None       # dernière requête produite (base d'une correction)
    issue: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            sql = generate_sql(question, join_plan) if sql is None else fix_sql(question, sql, issue or "", join_plan)
        except _INFRA_ERRORS:
            raise
        except Exception as exc:
            issue = f"Génération impossible : {exc}"
            continue

        validation = validate_sql(question, sql)
        if not validation["valid"]:
            issue = validation["issues"]
            continue

        try:
            rows = execute_sql(sql)
        except _INFRA_ERRORS:
            raise
        except Exception as exc:
            issue = str(exc)
            continue

        key = join_plan.get("sqlite_key") if join_plan else None
        if key and rows and not any(key in row for row in rows):
            issue = f"La requête SQL ne retourne pas la clé de jointure '{key}'."
            continue

        return {"sql_query": sql, "sql_results": rows, "sql_error": None, "sql_attempts": attempt}

    return {"sql_query": sql, "sql_results": [], "sql_error": issue, "sql_attempts": MAX_ATTEMPTS}


def sql_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : lit state['question'] (et join_plan en mode hybride),
    écrit sql_query/sql_results/sql_error/sql_attempts. Ne renvoie que ces clés."""
    return run(state["question"], state.get("join_plan"))