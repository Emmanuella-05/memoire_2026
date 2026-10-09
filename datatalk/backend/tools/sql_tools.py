"""Outils SQLite : analyse du schéma et exécution de requêtes en lecture seule.

Garde-fous (défense en profondeur) :
  1. is_read_only : contrôle textuel (retour rapide fait à l'agent pour qu'il corrige)
  2. connexion ouverte en lecture seule (mode=ro + PRAGMA query_only)
  3. set_authorizer : SQLite refuse lui-même toute opération autre que lecture
  4. timeout d'exécution et plafond du nombre de lignes retournées
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from .catalog_tools import SQLITE_DB, DataSourceUnavailable


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    """Connexion en lecture seule SANS authorizer (l'analyse a besoin des PRAGMA)."""
    path = Path(db_path)
    if not path.exists():
        raise DataSourceUnavailable("Aucune base SQLite n'a été importée.")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def analyze_sqlite_schema(db_path: str | Path = SQLITE_DB) -> dict[str, Any]:
    """Inspect SQLite tables, columns, PK/FK metadata and indexes."""
    conn = _connect_ro(db_path)
    try:
        table_names = [
            row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        tables = []
        for table in table_names:
            escaped = table.replace('"', '""')
            columns = conn.execute(f'PRAGMA table_info("{escaped}")').fetchall()
            foreign_keys = conn.execute(f'PRAGMA foreign_key_list("{escaped}")').fetchall()
            indexes = conn.execute(f'PRAGMA index_list("{escaped}")').fetchall()
            tables.append({
                "name": table,
                "columns": [
                    {
                        "name": row["name"],
                        "type": row["type"] or "unknown",
                        "nullable": not bool(row["notnull"]),
                        "primary_key": bool(row["pk"]),
                        "default": row["dflt_value"],
                    }
                    for row in columns
                ],
                "foreign_keys": [
                    {"column": row["from"], "references_table": row["table"], "references_column": row["to"]}
                    for row in foreign_keys
                ],
                "indexes": [{"name": row["name"], "unique": bool(row["unique"])} for row in indexes],
            })
        return {"database": "sqlite", "tables": tables}
    finally:
        conn.close()


def sqlite_schema(db_path: str | Path = SQLITE_DB) -> list[dict[str, Any]]:
    """Backward-compatible compact SQLite schema for agents."""
    return analyze_sqlite_schema(db_path).get("tables", [])


# --- Garde-fous -----------------------------------------------------------

MAX_FETCH_ROWS = 10_000
QUERY_TIMEOUT_S = 15.0

_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH",
    "PRAGMA", "VACUUM", "REINDEX", "TRUNCATE", "BEGIN", "COMMIT", "ROLLBACK",
    "SAVEPOINT", "RELEASE",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)

# Opérations que l'authorizer SQLite laisse passer : lecture uniquement.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def _strip_literals_and_comments(query: str) -> str:
    """Supprime commentaires, chaînes '...' et identifiants "..." / `...` / [...]
    pour que les mots-clés recherchés ne soient pas détectés à l'intérieur."""
    query = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    query = re.sub(r"--[^\n]*", " ", query)
    query = re.sub(r"'(?:[^']|'')*'", "''", query)
    query = re.sub(r'"(?:[^"]|"")*"', '""', query)
    query = re.sub(r"`[^`]*`", "``", query)
    query = re.sub(r"\[[^\]]*\]", "[]", query)
    return query


def sql_rejection_reason(query: str) -> str | None:
    """None si la requête passe le contrôle textuel, sinon la raison du rejet."""
    if not isinstance(query, str) or not query.strip():
        return "La requête SQL est vide."
    cleaned = _strip_literals_and_comments(query).strip().rstrip(";").strip()
    if ";" in cleaned:
        return "Une seule instruction SQL est autorisée."
    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return "Seules les requêtes SELECT / WITH sont autorisées."
    found = _FORBIDDEN_RE.search(cleaned)
    if found:
        return f"Mot-clé interdit dans une requête en lecture seule : {found.group(1).upper()}."
    return None


def is_read_only(query: str) -> bool:
    """True si la requête passe le contrôle textuel (SELECT/WITH, une seule
    instruction, aucun mot-clé d'écriture ou d'administration).
    Utilisé par execute_sql (garde-fou) et par l'agent SQL (validation)."""
    return sql_rejection_reason(query) is None


def _authorizer(action: int, *_args: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def open_readonly(db_path: str | Path = SQLITE_DB) -> sqlite3.Connection:
    """Ouvre la base en lecture seule, avec authorizer."""
    path = Path(db_path)
    if not path.exists():
        raise DataSourceUnavailable("Aucune base SQLite n'a été importée.")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_authorizer)
    return conn


def sql_compile_error(query: str, db_path: str | Path = SQLITE_DB) -> str | None:
    """Fait compiler la requête par SQLite (EXPLAIN, sans l'exécuter) : détecte
    gratuitement et de façon déterministe les tables/colonnes/fonctions
    inconnues et les erreurs de syntaxe. None si la requête est compilable."""
    conn = open_readonly(db_path)
    try:
        conn.execute(f"EXPLAIN {query}").fetchall()
        return None
    except sqlite3.Error as exc:
        return f"SQLite ne peut pas compiler la requête : {exc}"
    finally:
        conn.close()


def execute_sql(query: str, db_path: str | Path = SQLITE_DB,
                max_rows: int = MAX_FETCH_ROWS) -> list[dict[str, Any]]:
    reason = sql_rejection_reason(query)
    if reason:
        raise ValueError(reason)
    conn = open_readonly(db_path)
    try:
        deadline = time.monotonic() + QUERY_TIMEOUT_S

        def _abort_if_too_long() -> int:
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(_abort_if_too_long, 10_000)
        try:
            cursor = conn.execute(query)
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise TimeoutError(f"La requête SQL dépasse {QUERY_TIMEOUT_S:.0f} s.") from exc
            raise
        if len(rows) > max_rows:
            raise ValueError(
                f"Le résultat dépasse {max_rows} lignes : ajoutez un filtre, une agrégation ou un LIMIT."
            )
        return [dict(row) for row in rows]
    finally:
        conn.close()