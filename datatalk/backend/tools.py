"""Deterministic data/RAG tools plus the Claude API adapter for DataTalk.

LLM calls decide *what* to query. These helpers handle retrieval, data access,
uploaded workspaces, cross-source mappings and deterministic pandas fusion.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover
    MongoClient = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DATA_DIR = Path(os.getenv("DATATALK_DATA_DIR", str(DEFAULT_DATA_DIR)))
SQLITE_DOCS = DATA_DIR / "sqlite" / "database_docs.json"
MONGO_DOCS = DATA_DIR / "mongodb" / "database_docs.json"
MAPPINGS_FILE = DATA_DIR / "mappings.json"
SQLITE_DB = DATA_DIR / "datatalk.db"
UPLOADS_DIR = DATA_DIR / "uploads"


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------


def claude_client() -> Any:
    """Create the Anthropic client from CLAUDE_API_KEY."""
    if Anthropic is None:
        raise RuntimeError("anthropic is not installed")
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("CLAUDE_API_KEY is not configured")
    return Anthropic(api_key=api_key)


def claude_generate(prompt: str, *, system: str | None = None,
                    max_tokens: int = 1200) -> str:
    """Single Claude call used by LangGraph agents.

    Model can be changed with CLAUDE_MODEL without changing application code.
    """
    client = claude_client()
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system or "You are a precise assistant for the DataTalk NL-to-Query system.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")


# ---------------------------------------------------------------------------
# JSON documentation / mapping catalog
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_database_docs(payload: dict[str, Any], database: str) -> None:
    """Validate the small JSON contract consumed by the RAG index."""
    database = database.lower()
    if not isinstance(payload, dict):
        raise ValueError("Database documentation must be a JSON object")
    if database == "sqlite":
        items = payload.get("tables")
        item_label = "tables"
        child_label = "columns"
    elif database == "mongodb":
        items = payload.get("collections")
        item_label = "collections"
        child_label = "fields"
    else:
        raise ValueError(f"Unsupported database documentation type: {database}")
    if not isinstance(items, list):
        raise ValueError(f"Database documentation must contain a '{item_label}' array")
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise ValueError(f"Each {item_label[:-1]} must have a non-empty 'name'")
        children = item.get(child_label, [])
        if not isinstance(children, list):
            raise ValueError(f"'{child_label}' must be an array")
        for child in children:
            if not isinstance(child, dict) or not isinstance(child.get("name"), str) or not child["name"].strip():
                raise ValueError(f"Each {child_label[:-1]} must have a non-empty 'name'")


def validate_mappings(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("mappings"), list):
        raise ValueError("Mappings must be a JSON object containing a 'mappings' array")
    for mapping in payload["mappings"]:
        if not isinstance(mapping, dict) or not mapping.get("entity"):
            raise ValueError("Each mapping must contain an 'entity'")
        sqlite = mapping.get("sqlite")
        mongo = mapping.get("mongodb", mapping.get("mongo"))
        if not isinstance(sqlite, dict) or not sqlite.get("table") or not sqlite.get("key"):
            raise ValueError("Each mapping.sqlite must contain 'table' and 'key'")
        if not isinstance(mongo, dict) or not mongo.get("collection") or not mongo.get("key"):
            raise ValueError("Each mapping.mongodb must contain 'collection' and 'key'")


def load_database_docs(database: str) -> dict[str, Any]:
    path = SQLITE_DOCS if database.lower() == "sqlite" else MONGO_DOCS
    return load_json(path)


def load_mappings() -> list[dict[str, Any]]:
    return load_json(MAPPINGS_FILE).get("mappings", [])


def find_mapping(entity: str | None = None, sqlite_key: str | None = None,
                 mongo_key: str | None = None) -> list[dict[str, Any]]:
    results = []
    entity_norm = (entity or "").lower()
    for mapping in load_mappings():
        sqlite = mapping.get("sqlite", {})
        mongo = mapping.get("mongodb", mapping.get("mongo", {}))
        if entity_norm and str(mapping.get("entity", "")).lower() != entity_norm:
            continue
        if sqlite_key and sqlite.get("key") != sqlite_key:
            continue
        if mongo_key and mongo.get("key") != mongo_key:
            continue
        results.append(mapping)
    return results


# ---------------------------------------------------------------------------
# Lightweight RAG over JSON database documentation
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    database: str
    name: str
    text: str
    metadata: dict[str, Any]


def _flatten_docs(database: str, payload: dict[str, Any]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    if database == "sqlite":
        for table in payload.get("tables", []):
            columns = table.get("columns", [])
            col_text = " ".join(
                f"{c.get('name', '')} {c.get('type', '')} {c.get('description', '')}"
                for c in columns
            )
            text = f"table {table.get('name', '')} {table.get('description', '')} {col_text}"
            chunks.append(DocumentChunk(database, table.get("name", ""), text, table))
    else:
        for collection in payload.get("collections", []):
            fields = collection.get("fields", [])
            field_text = " ".join(
                f"{c.get('name', '')} {c.get('type', '')} {c.get('description', '')}"
                for c in fields
            )
            text = f"collection {collection.get('name', '')} {collection.get('description', '')} {field_text}"
            chunks.append(DocumentChunk(database, collection.get("name", ""), text, collection))
    return chunks


class DatabaseRAG:
    """Local TF-IDF retriever over uploaded/committed JSON documentation."""

    def __init__(self) -> None:
        self.chunks = (
            _flatten_docs("sqlite", load_database_docs("sqlite"))
            + _flatten_docs("mongodb", load_database_docs("mongodb"))
        )
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks]) if self.chunks else None

    def retrieve(self, question: str, database: str | None = None, k: int = 3) -> list[dict[str, Any]]:
        candidates = [(i, c) for i, c in enumerate(self.chunks)
                      if database is None or c.database == database.lower()]
        if not candidates or self.matrix is None:
            return []
        candidate_indices = [i for i, _ in candidates]
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix[candidate_indices])[0]
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:k]
        return [
            {"database": c.database, "name": c.name, "score": float(score),
             "metadata": c.metadata, "text": c.text}
            for score, (_, c) in ranked if score > 0
        ]

    def context(self, question: str, database: str | None = None, k: int = 3) -> str:
        docs = self.retrieve(question, database, k)
        return "\n\n".join(f"[{d['database']}.{d['name']}] {d['text']}" for d in docs)


_rag: DatabaseRAG | None = None


def reset_rag() -> None:
    global _rag
    _rag = None


def get_rag() -> DatabaseRAG:
    global _rag
    if _rag is None:
        _rag = DatabaseRAG()
    return _rag


# ---------------------------------------------------------------------------
# Uploaded workspace helpers
# ---------------------------------------------------------------------------


def save_upload(filename: str, content: bytes, target: str) -> Path:
    """Save and validate a user upload into the active DataTalk workspace.

    target: sqlite-db | sqlite-docs | mongodb-docs | mappings
    """
    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("Invalid filename")
    targets = {
        "sqlite-db": (DATA_DIR, {".db", ".sqlite", ".sqlite3"}),
        "sqlite-docs": (DATA_DIR / "sqlite", {".json"}),
        "mongodb-docs": (DATA_DIR / "mongodb", {".json"}),
        "mappings": (DATA_DIR, {".json"}),
    }
    if target not in targets:
        raise ValueError("Unknown upload target")
    directory, allowed = targets[target]
    suffix = Path(safe_name).suffix.lower()
    if suffix not in allowed:
        raise ValueError(f"Unsupported file type for {target}: {suffix}")

    if target == "sqlite-db":
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(content)
                tmp.flush()
                with sqlite3.connect(tmp.name) as conn:
                    conn.execute("PRAGMA schema_version").fetchone()
        except sqlite3.Error as exc:
            raise ValueError(f"Invalid SQLite database: {exc}") from exc
    elif target in {"sqlite-docs", "mongodb-docs", "mappings"}:
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON file: {exc}") from exc
        if target == "sqlite-docs":
            validate_database_docs(payload, "sqlite")
        elif target == "mongodb-docs":
            validate_database_docs(payload, "mongodb")
        else:
            validate_mappings(payload)

    directory.mkdir(parents=True, exist_ok=True)
    if target == "sqlite-db":
        destination = SQLITE_DB
    elif target == "sqlite-docs":
        destination = SQLITE_DOCS
    elif target == "mongodb-docs":
        destination = MONGO_DOCS
    else:
        destination = MAPPINGS_FILE
    destination.write_bytes(content)
    if target in {"sqlite-docs", "mongodb-docs", "mappings"}:
        reset_rag()
    return destination


def workspace_status() -> dict[str, Any]:
    return {
        "data_dir": str(DATA_DIR),
        "sqlite_database": SQLITE_DB.exists(),
        "sqlite_docs": SQLITE_DOCS.exists(),
        "mongodb_docs": MONGO_DOCS.exists(),
        "mappings": MAPPINGS_FILE.exists(),
        "rag_documents": len(get_rag().chunks),
        "claude_configured": bool(os.getenv("CLAUDE_API_KEY")),
        "claude_model": os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
    }


def sqlite_schema(db_path: str | Path = SQLITE_DB) -> list[dict[str, Any]]:
    """Return SQLite table/column metadata for agents and diagnostics."""
    with sqlite3.connect(str(db_path)) as conn:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()]
        result = []
        for table in tables:
            columns = conn.execute(f'PRAGMA table_info("{table.replace(chr(34), chr(34) + chr(34))}")').fetchall()
            result.append({
                "table": table,
                "columns": [
                    {"name": row[1], "type": row[2], "nullable": not bool(row[3]), "primary_key": bool(row[5])}
                    for row in columns
                ],
            })
        return result


# ---------------------------------------------------------------------------
# Database execution
# ---------------------------------------------------------------------------


def execute_sql(query: str, db_path: str | Path = SQLITE_DB) -> list[dict[str, Any]]:
    if not re.match(r"^\s*(SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError("Only read-only SELECT/WITH SQL statements are allowed")
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query).fetchall()]


def execute_mongo(collection: str, pipeline: list[dict[str, Any]],
                  mongo_uri: str | None = None, database: str | None = None) -> list[dict[str, Any]]:
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed")
    uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = database or os.getenv("MONGO_DB", "datatalk")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    try:
        return list(client[db_name][collection].aggregate(pipeline))
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Deterministic hybrid fusion
# ---------------------------------------------------------------------------


def merge_on_key(left: Iterable[dict[str, Any]], right: Iterable[dict[str, Any]],
                 left_key: str, right_key: str, how: str = "inner") -> list[dict[str, Any]]:
    left_df = pd.DataFrame(list(left))
    right_df = pd.DataFrame(list(right))
    if left_df.empty or right_df.empty:
        return []
    merged = left_df.merge(left_df if False else right_df, left_on=left_key, right_on=right_key,
                           how=how, suffixes=("_sql", "_mongo"))
    return merged.where(pd.notna(merged), None).to_dict(orient="records")


def hybrid_correspondences() -> list[dict[str, Any]]:
    return load_mappings()
