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
from ..tools.catalog_tools import DataSourceUnavailable
from ..tools.hybrid_tools import coerce_for_sql, display_sql, with_selected_ids
from ..tools.schema_tools import sql_schema_summary
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
    """Tables, colonnes typées et valeurs exactes des colonnes catégorielles."""
    return sql_schema_summary()


SELECTED_IDS_NOTE = (
    "(Note : la table temporaire selected_ids(id) contient déjà les identifiants sélectionnés "
    "par l'étape précédente.)"
)


def _join_requirement(join_plan: dict[str, Any] | None, step: str | None = None) -> str:
    if not join_plan:
        return ""
    if step == "first":
        return (
            "\n\nÉtape 1 d'une question hybride : retourne UNIQUEMENT les identifiants DISTINCTS "
            f"de la colonne '{join_plan.get('sqlite_key')}' (table {join_plan.get('sqlite_table')}) "
            "des lignes qui répondent à la question, sans agrégation."
        )
    if step == "second":
        return (
            "\n\nÉtape 2 d'une question hybride : les identifiants déjà sélectionnés par l'étape précédente "
            "sont dans la table temporaire selected_ids(id). Restreins-toi à eux, par exemple "
            f"WHERE {join_plan.get('sqlite_table')}.{join_plan.get('sqlite_key')} IN (SELECT id FROM selected_ids). "
            "N'écris aucun identifiant toi-même et ne crée pas selected_ids."
        )
    return (
        "\n\nContrainte HYBRID : cette requête sera jointe à MongoDB. "
        f"Elle doit impérativement retourner, sans l'agréger, la colonne de jointure "
        f"'{join_plan.get('sqlite_key')}' (table {join_plan.get('sqlite_table')}) dans ses résultats."
    )


def generate_sql(question: str, join_plan: dict[str, Any] | None = None, step: str | None = None) -> str:
    prompt = (
        f"Schéma SQLite:\n{_schema_summary()}\n\n"
        f"Contexte documentaire:\n{rag_context(question, database='sqlite')}\n\n"
        f"Question: {question}{_join_requirement(join_plan, step)}"
    )
    return _read_sql(claude_json(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=500))


def fix_sql(question: str, sql: str, issue: str, join_plan: dict[str, Any] | None = None,
            step: str | None = None) -> str:
    prompt = FIX_PROMPT_TEMPLATE.format(
        schema=_schema_summary(), question=question, sql=sql, issue=issue,
        join_requirement=_join_requirement(join_plan, step),
    )
    return _read_sql(claude_json(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=500))


def _read_sql(parsed: dict[str, Any]) -> str:
    sql = parsed.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("La réponse du modèle ne contient pas de champ 'sql' exploitable.")
    return sql.strip()


def validate_sql(question: str, sql: str, executable: str | None = None) -> dict[str, Any]:
    """Validation en trois temps : textuelle (lecture seule), compilation par
    SQLite, puis sémantique (LLM). `executable` est la requête réellement exécutée
    (avec la CTE selected_ids) quand elle diffère de celle écrite par le modèle."""
    checked = executable or sql
    reason = sql_rejection_reason(checked)
    if reason:
        return {"valid": False, "issues": reason}
    compile_error = sql_compile_error(checked)
    if compile_error:
        return {"valid": False, "issues": compile_error}

    note = f"\n{SELECTED_IDS_NOTE}" if executable else ""
    prompt = f"Schéma SQLite:\n{_schema_summary()}\n\nQuestion: {question}{note}\n\nRequête: {sql}"
    try:
        parsed = claude_json(prompt, system=VALIDATE_SYSTEM_PROMPT, max_tokens=300)
    except ValueError:
        # Validateur illisible : on ne bloque pas, la requête a déjà passé les
        # contrôles déterministes et execute_sql reste le garde-fou final.
        return {"valid": True, "issues": None}
    if parsed.get("valid") is True:
        return {"valid": True, "issues": None}
    return {"valid": False, "issues": parsed.get("issues") or "Requête jugée invalide par le validateur."}


def run(question: str, join_plan: dict[str, Any] | None = None, injected_keys: list[Any] | None = None,
        step: str | None = None) -> dict[str, Any]:
    """Génère -> valide -> exécute, avec correction à chaque rejet ou erreur,
    jusqu'à MAX_ATTEMPTS.

    Plan hybride en deux étapes : `question` est alors la SOUS-QUESTION de cette source.
    step="first"  : la requête doit retourner les identifiants distincts de la clé de jointure ;
    step="second" : `injected_keys` (identifiants de l'étape 1) est injecté par le code devant
                    la requête du modèle, sous la forme de la CTE selected_ids(id)."""
    sql: str | None = None       # dernière requête produite (base d'une correction)
    issue: str | None = None
    keys: list[Any] = []
    if step == "second":
        keys = coerce_for_sql(injected_keys or [], join_plan["sqlite_table"], join_plan["sqlite_key"])
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            sql = (generate_sql(question, join_plan, step) if sql is None
                   else fix_sql(question, sql, issue or "", join_plan, step))
        except _INFRA_ERRORS:
            raise
        except Exception as exc:
            issue = f"Génération impossible : {exc}"
            continue

        executable = with_selected_ids(sql, keys) if step == "second" else None
        validation = validate_sql(question, sql, executable)
        if not validation["valid"]:
            issue = validation["issues"]
            continue

        try:
            rows = execute_sql(executable or sql)
        except _INFRA_ERRORS:
            raise
        except Exception as exc:
            issue = str(exc)
            continue

        key = join_plan.get("sqlite_key") if join_plan else None
        if key and step != "second" and rows and not any(key in row for row in rows):
            issue = f"La requête SQL ne retourne pas la clé de jointure '{key}'."
            continue

        shown = display_sql(sql, keys) if step == "second" else sql
        return {"sql_query": shown, "sql_results": rows, "sql_error": None, "sql_attempts": attempt}

    shown = display_sql(sql, keys) if (step == "second" and sql) else sql
    return {"sql_query": shown, "sql_results": [], "sql_error": issue, "sql_attempts": MAX_ATTEMPTS}


def sql_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : lit state['question'] (et join_plan en mode hybride),
    écrit sql_query/sql_results/sql_error/sql_attempts. Ne renvoie que ces clés."""
    return run(state["question"], state.get("join_plan"))