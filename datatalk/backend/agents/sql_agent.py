"""Agent SQL, à part entière : il connaît le schéma, génère la requête, la
VALIDE avant de l'exécuter, l'exécute, et se corrige en cas de rejet ou
d'erreur.

Fusionne ce qui était auparavant trois agents séparés (schéma / génération /
évaluation) en un seul.

Qui écrit la requête :
  - LLM_BASE_URL définie  -> modèles OUVERTS : SQLCoder (prompt brut, schéma en CREATE TABLE,
    question traduite en anglais) pour la tentative 1 ; un modèle chat ouvert pour les corrections
    (DATATALK_FIX_WITH=claude pour corriger avec Claude) ;
  - sinon                 -> Claude, comme avant.
Dans tous les cas, la validation sémantique reste faite par Claude et les contrôles déterministes
(lecture seule, EXPLAIN) restent identiques.

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

import os
import re
from typing import Any

from ..llm import LLMError, claude_json, fix_with_open_models, open_chat, open_complete, open_models_enabled
from ..tools.catalog_tools import DataSourceUnavailable, load_database_docs
from ..tools.hybrid_tools import DEDUP_NOTE, coerce_for_sql, display_sql, distinct_keys_sql, with_selected_ids
from ..tools.schema_tools import sql_schema_summary
from ..tools.rag_tools import context as rag_context
from ..tools.sql_tools import execute_sql, sql_compile_error, sql_rejection_reason
from ..tools.translate_tools import to_english, translate_context_enabled

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
            f"\n\nÉtape 1 d'une question hybride : retourne la colonne '{join_plan.get('sqlite_key')}' "
            f"(table {join_plan.get('sqlite_table')}) des lignes qui répondent à la question. Les doublons sont "
            "éliminés automatiquement : n'ajoute pas de DISTINCT pour dédoublonner. Si la condition porte sur un "
            "calcul par entité (moyenne, somme, nombre), utilise GROUP BY sur cette colonne avec HAVING."
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


# --- Modèles ouverts : SQLCoder ---------------------------------------------------------------------

# Prompt de la fiche de defog/sqlcoder-7b-2 ; la section « Instructions » reçoit les règles métier du
# RAG et les contraintes de l'étape hybride. Décodage conseillé : sans échantillonnage, num_beams=4.
SQLCODER_TEMPLATE = """### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
{instructions}

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""

OPEN_FIX_SYSTEM_PROMPT = """You are an expert SQLite engineer. You fix a failing SQL query.
Rules: the database is SQLite; only use tables and columns that exist in the schema; the query must be a single
read-only SELECT (or WITH ... SELECT); use the possible values listed in the schema exactly as written.
Reply with ONLY the corrected query inside one ```sql code block."""

OPEN_FIX_PROMPT_TEMPLATE = """Database schema:
{schema}

Instructions:
{instructions}

Question: {question}

This SQL query was rejected:
{sql}

Reason:
{issue}

Write the corrected query."""


def _quote_ident(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else '"' + name.replace('"', '""') + '"'


def _schema_ddl(join_plan: dict[str, Any] | None = None, step: str | None = None) -> str:
    """Schéma en CREATE TABLE (le format que SQLCoder connaît), avec clés primaires et étrangères pour
    guider les jointures, et les valeurs possibles des colonnes catégorielles en commentaire."""
    docs = load_database_docs("sqlite")
    statements = []
    for table in docs.get("tables", []):
        entries: list[tuple[str, str | None]] = []
        pks = []
        for col in table.get("columns", []):
            name = _quote_ident(str(col.get("name", "")))
            values = col.get("distinct_values")
            comment = ("possible values: " + ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)) if values else None
            entries.append((f"{name} {col.get('type') or 'TEXT'}", comment))
            if col.get("primary_key"):
                pks.append(name)
        if pks:
            entries.append((f"PRIMARY KEY ({', '.join(pks)})", None))
        for fk in table.get("foreign_keys", []):
            entries.append((
                f"FOREIGN KEY ({_quote_ident(str(fk.get('column', '')))}) "
                f"REFERENCES {_quote_ident(str(fk.get('references_table', '')))}"
                f"({_quote_ident(str(fk.get('references_column', '')))})", None))
        body = "\n".join(
            "  " + text + ("," if i < len(entries) - 1 else "") + (f" -- {comment}" if comment else "")
            for i, (text, comment) in enumerate(entries)
        )
        statements.append(f"CREATE TABLE {_quote_ident(str(table.get('name', '')))} (\n{body}\n);")
    if step == "second" and join_plan:
        key_type = next((c.get("type") for t in docs.get("tables", []) if t.get("name") == join_plan.get("sqlite_table")
                         for c in t.get("columns", []) if c.get("name") == join_plan.get("sqlite_key")), None) or "INTEGER"
        statements.append(f"CREATE TABLE selected_ids (id {key_type}); -- identifiers already selected by the previous step")
    return "\n\n".join(statements) if statements else "-- no table documented"


def _open_requirement(join_plan: dict[str, Any] | None, step: str | None) -> list[str]:
    if not join_plan:
        return []
    table, key = join_plan.get("sqlite_table"), join_plan.get("sqlite_key")
    if step == "first":
        return [f"- Step 1 of a cross-database question: return column {table}.{key} of the rows that answer the question; "
                "duplicates are removed automatically, so do not add DISTINCT just for that. If the condition depends on a "
                f"per-entity computation (average, sum, count), GROUP BY {key} with HAVING"]
    if step == "second":
        return [f"- Table selected_ids(id) holds the identifiers already selected by the previous step: restrict to them, "
                f"e.g. WHERE {table}.{key} IN (SELECT id FROM selected_ids). Never write identifiers yourself"]
    return [f"- This query will be joined to MongoDB: it MUST return, without aggregating it, column {key} (table {table})"]


def _open_instructions(question: str, join_plan: dict[str, Any] | None, step: str | None) -> str:
    lines = [
        "- If you cannot answer the question with the available database schema, return 'I do not know'",
        "- The database is SQLite (not PostgreSQL): no ILIKE, no DATE_TRUNC, no :: casts; use strftime() for dates and LIKE for text search",
        "- Only use the tables and columns of the schema below; never invent one",
        "- For text columns with listed possible values, use those values exactly as written",
        "- Write a single read-only SELECT query",
    ] + _open_requirement(join_plan, step)
    ctx = rag_context(question, database="sqlite")      # règles métier : récupérées avec la question d'origine (français)
    if ctx.strip():
        if translate_context_enabled():
            ctx = to_english(ctx)
        lines += [f"- {line.strip()}" for line in ctx.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_sql(raw: str) -> str:
    """Isole la requête SQL de la sortie brute d'un modèle (bloc markdown, marqueur [SQL], texte après
    le « ; »). Lève ValueError si ce n'est pas un SELECT."""
    text = raw.strip()
    fence = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    text = fence.group(1).strip() if fence else re.sub(r"^\s*```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\s*(\[SQL\]|SQL\s*:)\s*", "", text, flags=re.IGNORECASE).strip()
    if re.match(r"^'?I do not know'?", text, re.IGNORECASE):
        raise ValueError("Le modèle indique qu'il ne peut pas répondre avec ce schéma.")
    quote = None
    for i, ch in enumerate(text):          # coupe au premier « ; » hors chaîne
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == ";":
            text = text[:i]
            break
    text = text.strip()
    if not re.match(r"^(SELECT|WITH)\b", text, re.IGNORECASE):
        raise ValueError(f"Le modèle n'a pas produit de requête SELECT : {raw[:120]!r}")
    return text


def _generate_open(question: str, join_plan: dict[str, Any] | None, step: str | None) -> str:
    prompt = SQLCODER_TEMPLATE.format(
        question=to_english(question),
        instructions=_open_instructions(question, join_plan, step),
        schema=_schema_ddl(join_plan, step),
    )
    return extract_sql(open_complete(prompt, role="sql", max_tokens=400, num_beams=4))


def _fix_open(question: str, sql: str, issue: str, join_plan: dict[str, Any] | None, step: str | None) -> str:
    prompt = OPEN_FIX_PROMPT_TEMPLATE.format(
        schema=_schema_ddl(join_plan, step), instructions=_open_instructions(question, join_plan, step),
        question=to_english(question), sql=sql, issue=issue,
    )
    return extract_sql(open_chat(prompt, system=OPEN_FIX_SYSTEM_PROMPT, role="general", max_tokens=400))


def generate_sql(question: str, join_plan: dict[str, Any] | None = None, step: str | None = None) -> str:
    if open_models_enabled():
        return _generate_open(question, join_plan, step)
    prompt = (
        f"Schéma SQLite:\n{_schema_summary()}\n\n"
        f"Contexte documentaire:\n{rag_context(question, database='sqlite')}\n\n"
        f"Question: {question}{_join_requirement(join_plan, step)}"
    )
    return _read_sql(claude_json(prompt, system=GENERATE_SYSTEM_PROMPT, max_tokens=500))


def fix_sql(question: str, sql: str, issue: str, join_plan: dict[str, Any] | None = None,
            step: str | None = None) -> str:
    if fix_with_open_models():
        return _fix_open(question, sql, issue, join_plan, step)
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


def validate_sql(question: str, sql: str, executable: str | None = None, note: str | None = None) -> dict[str, Any]:
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

    note_text = f"\n{note}" if note else ""
    prompt = f"Schéma SQLite:\n{_schema_summary()}\n\nQuestion: {question}{note_text}\n\nRequête: {sql}"
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
    translated = to_english(question)       # identique à `question` si les modèles ouverts ne sont pas utilisés
    translated_question = translated if translated != question else None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            sql = (generate_sql(question, join_plan, step) if sql is None
                   else fix_sql(question, sql, issue or "", join_plan, step))
        except _INFRA_ERRORS:
            raise
        except Exception as exc:
            issue = f"Génération impossible : {exc}"
            continue

        executable, note = None, None
        if step == "second":
            executable, note = with_selected_ids(sql, keys), SELECTED_IDS_NOTE
        elif step == "first":
            deduplicated = distinct_keys_sql(sql, join_plan["sqlite_key"])
            executable, note = (deduplicated if deduplicated != sql.strip().rstrip(";").strip() else None), DEDUP_NOTE
        validation = validate_sql(question, sql, executable, note)
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

        shown = display_sql(sql, keys) if step == "second" else (executable or sql)
        return {"sql_query": shown, "sql_results": rows, "sql_error": None, "sql_attempts": attempt,
                "translated_question": translated_question}

    shown = display_sql(sql, keys) if (step == "second" and sql) else sql
    return {"sql_query": shown, "sql_results": [], "sql_error": issue, "sql_attempts": MAX_ATTEMPTS,
            "translated_question": translated_question}


def sql_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : lit state['question'] (et join_plan en mode hybride),
    écrit sql_query/sql_results/sql_error/sql_attempts. Ne renvoie que ces clés."""
    return run(state["question"], state.get("join_plan"))