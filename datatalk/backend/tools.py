"""Deterministic data/RAG tools used by the DataTalk LangGraph.

The LLM decides *what* to query; these helpers handle data access, retrieval,
mapping lookup and deterministic pandas fusion.
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
    from pymongo import MongoClient
except ImportError:  # pragma: no cover - keeps SQL/RAG usable without Mongo installed
    MongoClient = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SQLITE_DOCS = DATA_DIR / "sqlite" / "database_docs.json"
MONGO_DOCS = DATA_DIR / "mongodb" / "database_docs.json"
MAPPINGS_FILE = DATA_DIR / "mappings.json"
SQLITE_DB = DATA_DIR / "datatalk.db"


# ---------------------------------------------------------------------------
# JSON documentation / mapping catalog
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_database_docs(database: str) -> dict[str, Any]:
    """Load the structured documentation for one database."""
    path = SQLITE_DOCS if database.lower() == "sqlite" else MONGO_DOCS
    return load_json(path)


def load_mappings() -> list[dict[str, Any]]:
    return load_json(MAPPINGS_FILE).get("mappings", [])


def find_mapping(entity: str | None = None, sqlite_key: str | None = None,
                 mongo_key: str | None = None) -> list[dict[str, Any]]:
    """Return explicit cross-source correspondences from mappings.json."""
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
            text = (
                f"collection {collection.get('name', '')} "
                f"{collection.get('description', '')} {field_text}"
            )
            chunks.append(DocumentChunk(database, collection.get("name", ""), text, collection))
    return chunks


class DatabaseRAG:
    """Small, local retriever; no external vector DB is required for the MVP."""

    def __init__(self) -> None:
        self.chunks = (
            _flatten_docs("sqlite", load_database_docs("sqlite"))
            + _flatten_docs("mongodb", load_database_docs("mongodb"))
        )
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.matrix = (
            self.vectorizer.fit_transform([c.text for c in self.chunks])
            if self.chunks else None
        )

    def retrieve(self, question: str, database: str | None = None, k: int = 3) -> list[dict[str, Any]]:
        candidates = [
            (i, c) for i, c in enumerate(self.chunks)
            if database is None or c.database == database.lower()
        ]
        if not candidates or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix[candidates_i := [i for i, _ in candidates]])[0]
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:k]
        return [
            {"database": c.database, "name": c.name, "score": float(score), "metadata": c.metadata, "text": c.text}
            for score, (_, c) in ranked
            if score > 0
        ]

    def context(self, question: str, database: str | None = None, k: int = 3) -> str:
        docs = self.retrieve(question, database, k)
        return "\n\n".join(
            f"[{d['database']}.{d['name']}] {d['text']}" for d in docs
        )


_rag: DatabaseRAG | None = None


def get_rag() -> DatabaseRAG:
    global _rag
    if _rag is None:
        _rag = DatabaseRAG()
    return _rag


# ---------------------------------------------------------------------------
# Database execution
# ---------------------------------------------------------------------------


def execute_sql(query: str, db_path: str | Path = SQLITE_DB) -> list[dict[str, Any]]:
    """Execute one read-only SQL statement and return JSON-friendly rows."""
    if not re.match(r"^\s*(SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError("Only read-only SELECT/WITH SQL statements are allowed")
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query).fetchall()]


def execute_mongo(collection: str, pipeline: list[dict[str, Any]],
                  mongo_uri: str | None = None, database: str | None = None) -> list[dict[str, Any]]:
    """Execute a MongoDB aggregation pipeline."""
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
    """Fuse SQL/Mongo results in memory with pandas; no LLM is involved."""
    left_df = pd.DataFrame(list(left))
    right_df = pd.DataFrame(list(right))
    if left_df.empty or right_df.empty:
        return []
    merged = left_df.merge(
        right_df,
        left_on=left_key,
        right_on=right_key,
        how=how,
        suffixes=("_sql", "_mongo"),
    )
    return merged.where(pd.notna(merged), None).to_dict(orient="records")


def hybrid_correspondences() -> list[dict[str, Any]]:
    """Expose mappings for Join Planner and explainability."""
    return load_mappings()
