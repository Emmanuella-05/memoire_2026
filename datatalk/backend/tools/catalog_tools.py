"""Helpers de catalogue et d'ingestion pour DataTalk.

Ce module centralise les chemins de données, les fichiers JSON, les loaders et
les utilitaires communs partagés par les agents et le RAG.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DATA_DIR = Path(os.getenv("DATATALK_DATA_DIR", Path(__file__).resolve().parents[2] / "data")).resolve()
SQLITE_DB = DATA_DIR / "sqlite" / "data.db"
SQLITE_DOCS = DATA_DIR / "sqlite" / "database_docs.json"
MONGO_DOCS = DATA_DIR / "mongodb" / "database_docs.json"
MAPPINGS_FILE = DATA_DIR / "mappings.json"
BUSINESS_RULES_FILE = DATA_DIR / "business_rules.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> Path:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default if default is not None else {}


def load_database_docs(database: str) -> dict[str, Any]:
    if database == "sqlite":
        return load_json(SQLITE_DOCS, {"database": "sqlite", "tables": []})
    if database == "mongodb":
        return load_json(MONGO_DOCS, {"database": "mongodb", "collections": []})
    raise ValueError(f"Unsupported database type: {database}")


def load_mappings() -> list[dict[str, Any]]:
    payload = load_json(MAPPINGS_FILE, {"mappings": []})
    if isinstance(payload, dict):
        return payload.get("mappings", [])
    return payload if isinstance(payload, list) else []


def mapping_context() -> str:
    mappings = load_mappings()
    return "\n".join(
        f"- {m.get('entity', 'unknown')}: SQLite {m.get('sqlite', {}).get('table')}({m.get('sqlite', {}).get('key')}) -> MongoDB {m.get('mongodb', {}).get('collection')}({m.get('mongodb', {}).get('key')})"
        for m in mappings
    ) or "Aucune correspondance documentée."


def merge_on_key(left: list[dict[str, Any]], right: list[dict[str, Any]], left_key: str, right_key: str) -> list[dict[str, Any]]:
    right_by_key = {}
    for row in right:
        key = row.get(right_key)
        if key is not None:
            right_by_key[str(key)] = row
    merged: list[dict[str, Any]] = []
    for left_row in left:
        left_value = left_row.get(left_key)
        right_row = right_by_key.get(str(left_value), {})
        merged_row = dict(left_row)
        merged_row.update({f"mongo_{k}": v for k, v in right_row.items() if k != right_key})
        merged.append(merged_row)
    return merged


def save_business_rules(content: bytes, filename: str = "business_rules.txt") -> Path:
    """Store optional business rules in a RAG-friendly canonical JSON format."""
    suffix = Path(filename).suffix.lower()
    try:
        text = content.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Business rules must be UTF-8 text") from exc
    if not text:
        raise ValueError("Business rules file is empty")
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid business rules JSON: {exc}") from exc
        if isinstance(payload, dict):
            rules = payload.get("rules", [payload])
        elif isinstance(payload, list):
            rules = payload
        else:
            raise ValueError("Business rules JSON must contain an object or array")
    else:
        rules = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not rules:
        raise ValueError("No business rules found")
    _write_json(BUSINESS_RULES_FILE, {"rules": rules})
    return BUSINESS_RULES_FILE


__all__ = [
    "DATA_DIR",
    "SQLITE_DB",
    "SQLITE_DOCS",
    "MONGO_DOCS",
    "MAPPINGS_FILE",
    "BUSINESS_RULES_FILE",
    "_write_json",
    "load_json",
    "load_database_docs",
    "load_mappings",
    "mapping_context",
    "merge_on_key",
    "save_business_rules",
]