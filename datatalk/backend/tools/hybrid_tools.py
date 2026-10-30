"""Outils du plan hybride en deux étapes dépendantes.

La première source répond à sa sous-question et retourne les identifiants (clé de
jointure) des entités concernées. Ces identifiants sont ensuite INJECTÉS PAR LE CODE
dans la requête de la seconde source, jamais recopiés par un LLM :
  - SQL     : CTE `selected_ids(id)` placée devant la requête du modèle ;
  - MongoDB : étape `{"$match": {<clé>: {"$in": [...]}}}` placée en tête du pipeline.
"""

from __future__ import annotations

import os
import re
from typing import Any

from .catalog_tools import get_path, load_database_docs, normalize_key

MAX_JOIN_KEYS = int(os.getenv("DATATALK_MAX_JOIN_KEYS", "5000"))
DISPLAY_KEYS = 20  # nombre d'identifiants affichés dans la requête montrée à l'utilisateur


def extract_keys(rows: list[dict[str, Any]], key: str) -> list[Any]:
    """Valeurs distinctes (ordre conservé) de la clé de jointure dans les résultats d'une étape."""
    seen: set[str] = set()
    keys: list[Any] = []
    for row in rows:
        value = get_path(row, key)
        norm = normalize_key(value)
        if norm is None or norm in seen:
            continue
        seen.add(norm)
        keys.append(value)
    return keys


def _column_type(table: str, column: str) -> str:
    for t in load_database_docs("sqlite").get("tables", []):
        if t.get("name") == table:
            for c in t.get("columns", []):
                if c.get("name") == column:
                    return str(c.get("type") or "")
    return ""


def _field_type(collection: str, field: str) -> str:
    for c in load_database_docs("mongodb").get("collections", []):
        if c.get("name") == collection:
            for f in c.get("fields", []):
                if f.get("name") == field:
                    return str(f.get("type") or "")
    return ""


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    return None


def coerce_for_sql(keys: list[Any], table: str, column: str) -> list[Any]:
    """Aligne le type des identifiants sur celui de la colonne SQLite (INTEGER, TEXT...)."""
    declared = _column_type(table, column).upper()
    out: list[Any] = []
    for key in keys:
        if "INT" in declared:
            number = _as_int(key)
            out.append(number if number is not None else str(key))
        elif any(t in declared for t in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
            try:
                out.append(float(key))
            except (TypeError, ValueError):
                out.append(str(key))
        elif declared:
            out.append(str(key))
        else:
            out.append(key if isinstance(key, (int, float)) else str(key))
    return out


def coerce_for_mongo(keys: list[Any], collection: str, field: str) -> list[Any]:
    """Aligne le type des identifiants sur celui du champ MongoDB. Type inconnu ou
    mixte : on fournit les deux formes (nombre et texte), ce qui est sans risque pour $in."""
    declared = _field_type(collection, field).lower()
    out: list[Any] = []
    for key in keys:
        number = _as_int(key)
        if declared in ("integer", "int", "number", "float", "long"):
            out.append(number if number is not None else key)
        elif declared in ("string", "text"):
            out.append(str(key))
        elif declared == "objectid":
            out.append({"$oid": str(key)})
        else:
            out.append(key)
            alt = (number if number is not None else None) if isinstance(key, str) else str(key)
            if alt is not None and alt != key:
                out.append(alt)
    return out


# --- SQL ---------------------------------------------------------------------

_WITH_RE = re.compile(r"^\s*WITH\s+(RECURSIVE\s+)?", re.IGNORECASE)


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def with_selected_ids(sql: str, keys: list[Any]) -> str:
    """Place devant `sql` la CTE selected_ids(id) contenant les identifiants de l'étape 1."""
    sql = sql.strip().rstrip(";").strip()
    cte = "selected_ids(id) AS (VALUES " + ", ".join(f"({_sql_literal(k)})" for k in keys) + ")"
    match = _WITH_RE.match(sql)
    if match:   # la requête a déjà ses propres CTE : on ajoute la nôtre en tête
        return f"WITH {match.group(1) or ''}{cte}, {sql[match.end():]}"
    return f"WITH {cte} {sql}"


def display_sql(sql: str, keys: list[Any]) -> str:
    """Requête montrée à l'utilisateur : celle du modèle, précédée de la provenance des identifiants."""
    shown = ", ".join(str(k) for k in keys[:DISPLAY_KEYS])
    more = f", … (+{len(keys) - DISPLAY_KEYS})" if len(keys) > DISPLAY_KEYS else ""
    return f"-- selected_ids = {len(keys)} identifiant(s) issus de l'étape 1 : {shown}{more}\n{sql.strip()}"


# --- MongoDB -----------------------------------------------------------------

def with_selected_keys(pipeline: list[dict[str, Any]], key: str, keys: list[Any]) -> list[dict[str, Any]]:
    return [{"$match": {key: {"$in": keys}}}] + list(pipeline)


def display_pipeline(pipeline: list[dict[str, Any]], keys: list[Any]) -> list[dict[str, Any]]:
    """Pipeline montré à l'utilisateur : la liste d'identifiants est abrégée si elle est longue."""
    if len(keys) <= DISPLAY_KEYS * 2 or not pipeline:
        return pipeline
    stage = pipeline[0]
    (field, cond), = stage["$match"].items()
    short = list(cond["$in"][:DISPLAY_KEYS]) + [f"… (+{len(cond['$in']) - DISPLAY_KEYS} identifiants)"]
    return [{"$match": {field: {"$in": short}}}] + pipeline[1:]
