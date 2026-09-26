"""Outils SQLite : analyse du schéma et exécution de requêtes en lecture seule."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from .catalog_tools import SQLITE_DB


def analyze_sqlite_schema(db_path: str | Path = SQLITE_DB) -> dict[str, Any]:
    """Inspect SQLite tables, columns, PK/FK metadata and indexes."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
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


def sqlite_schema(db_path: str | Path = SQLITE_DB) -> list[dict[str, Any]]:
    """Backward-compatible compact SQLite schema for agents."""
    return analyze_sqlite_schema(db_path).get("tables", [])


def is_read_only(query: str) -> bool:
    """True si la requête est un SELECT/WITH (donc sans danger pour la base).
    Utilisé à la fois par execute_sql (garde-fou final) et par l'agent SQL
    (validation avant exécution)."""
    return bool(re.match(r"^\s*(SELECT|WITH)\b", query, re.IGNORECASE))


def execute_sql(query: str, db_path: str | Path = SQLITE_DB) -> list[dict[str, Any]]:
    if not is_read_only(query):
        raise ValueError("Only read-only SELECT/WITH SQL statements are allowed")
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query).fetchall()]