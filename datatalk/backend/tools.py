"""Deterministic data/RAG tools plus the Claude API adapter for DataTalk.

The workspace is self-describing: users provide the databases/data and DataTalk
automatically inspects schemas, generates documentation and proposes cross-source
correspondences. Business rules are optional complementary knowledge.
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
BUSINESS_RULES_FILE = DATA_DIR / "business_rules.json"
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

    The model is configurable through CLAUDE_MODEL so deployment can select
    a currently supported Anthropic model without code changes.
    """
    client = claude_client()
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system or "You are a precise assistant for the DataTalk NL-to-Query system.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in message.content
        if getattr(block, "type", None) == "text"
    )


# ---------------------------------------------------------------------------
# JSON documentation / mapping catalog
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_database_docs(payload: dict[str, Any], database: str) -> None:
    database = database.lower()
    if not isinstance(payload, dict):
        raise ValueError("Database documentation must be a JSON object")
    if database == "sqlite":
        items, item_label, child_label = payload.get("tables"), "tables", "columns"
    elif database == "mongodb":
        items, item_label, child_label = payload.get("collections"), "collections", "fields"
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


def mapping_context() -> str:
    """Compact mapping catalog context for the Join Planner."""
    lines = []
    for mapping in load_mappings():
        sqlite = mapping.get("sqlite", {})
        mongo = mapping.get("mongodb", mapping.get("mongo", {}))
        lines.append(
            f"{mapping.get('entity', 'entity')}: "
            f"SQLite {sqlite.get('table')}.{sqlite.get('key')} <-> "
            f"MongoDB {mongo.get('collection')}.{mongo.get('key')} "
            f"({mapping.get('relation', 'correspondence')})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Automatic database analysis and generated documentation
# ---------------------------------------------------------------------------

def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _merge_type(types: set[str]) -> str:
    clean = sorted(t for t in types if t != "null")
    if not clean:
        return "unknown"
    if len(clean) == 1:
        return clean[0]
    return "mixed"


def _flatten_mongo_fields(value: Any, prefix: str = "") -> dict[str, set[str]]:
    fields: dict[str, set[str]] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(child, dict):
                fields.setdefault(path, set()).add("object")
                nested = _flatten_mongo_fields(child, path)
                for name, types in nested.items():
                    fields.setdefault(name, set()).update(types)
            elif isinstance(child, list):
                fields.setdefault(path, set()).add("array")
                if child:
                    first = child[0]
                    fields.setdefault(path, set()).add(_json_type(first))
                    if isinstance(first, dict):
                        nested = _flatten_mongo_fields(first, path)
                        for name, types in nested.items():
                            fields.setdefault(name, set()).update(types)
            else:
                fields.setdefault(path, set()).add(_json_type(child))
    return fields


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


def analyze_mongodb_schema(
    mongo_uri: str | None = None,
    database: str | None = None,
    sample_size: int = 50,
) -> dict[str, Any]:
    """Inspect MongoDB collections and infer field paths/types from samples."""
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed")
    uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = database or os.getenv("MONGO_DB", "datatalk")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    try:
        db = client[db_name]
        collections = []
        for name in sorted(db.list_collection_names()):
            docs = list(db[name].find({}, {"_id": 0}).limit(sample_size))
            field_types: dict[str, set[str]] = {}
            for doc in docs:
                for field, types in _flatten_mongo_fields(doc).items():
                    field_types.setdefault(field, set()).update(types)
            fields = [
                {"name": field, "type": _merge_type(types), "sampled": len(docs)}
                for field, types in sorted(field_types.items())
            ]
            collections.append({
                "name": name,
                "document_count": db[name].count_documents({}),
                "sample_size": len(docs),
                "fields": fields,
            })
        return {"database": "mongodb", "database_name": db_name, "collections": collections}
    finally:
        client.close()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def generate_sqlite_docs(schema: dict[str, Any]) -> dict[str, Any]:
    payload = {"database": "sqlite", "tables": []}
    for table in schema.get("tables", []):
        columns = []
        for col in table.get("columns", []):
            description = "primary key" if col.get("primary_key") else ""
            columns.append({**col, "description": description})
        payload["tables"].append({
            "name": table["name"],
            "description": f"SQLite table {table['name']}.",
            "columns": columns,
            "foreign_keys": table.get("foreign_keys", []),
        })
    validate_database_docs(payload, "sqlite")
    _write_json(SQLITE_DOCS, payload)
    return payload


def generate_mongodb_docs(schema: dict[str, Any]) -> dict[str, Any]:
    payload = {"database": "mongodb", "collections": []}
    for collection in schema.get("collections", []):
        payload["collections"].append({
            "name": collection["name"],
            "description": f"MongoDB collection {collection['name']}.",
            "fields": [
                {**field, "description": ""}
                for field in collection.get("fields", [])
            ],
        })
    validate_database_docs(payload, "mongodb")
    _write_json(MONGO_DOCS, payload)
    return payload


def _norm_name(value: str) -> str:
    value = value.split(".")[-1]
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _is_key_name(value: str) -> bool:
    normalized = _norm_name(value)
    return normalized == "id" or normalized.endswith("id") or normalized.endswith("key")


def analyze_correspondences(
    sqlite_schema: dict[str, Any],
    mongo_schema: dict[str, Any],
) -> dict[str, Any]:
    """Generate conservative SQL<->Mongo key candidates.

    Exact normalized field-name matches receive the strongest confidence.
    We only emit likely key fields to avoid turning arbitrary similarly named
    attributes into join keys.
    """
    mappings: list[dict[str, Any]] = []
    mongo_fields = []
    for collection in mongo_schema.get("collections", []):
        for field in collection.get("fields", []):
            mongo_fields.append((collection["name"], field["name"], field.get("type", "unknown")))

    for table in sqlite_schema.get("tables", []):
        sql_columns = table.get("columns", [])
        for sql_col in sql_columns:
            sql_name = sql_col["name"]
            if not _is_key_name(sql_name):
                continue
            for collection, mongo_field, mongo_type in mongo_fields:
                if not _is_key_name(mongo_field):
                    continue
                if _norm_name(sql_name) != _norm_name(mongo_field):
                    continue
                sql_type = str(sql_col.get("type", "")).lower()
                compatible = (
                    mongo_type in {"string", "integer", "number", "unknown", "mixed"}
                    or any(token in sql_type for token in ("char", "text", "int", "real", "numeric"))
                )
                if not compatible:
                    continue
                entity = table["name"].rstrip("s")
                mappings.append({
                    "entity": entity,
                    "sqlite": {"table": table["name"], "key": sql_name},
                    "mongodb": {"collection": collection, "key": mongo_field},
                    "relation": "same_normalized_key",
                    "confidence": 0.98,
                    "method": "deterministic_name_match",
                })

    unique = {}
    for mapping in mappings:
        key = (
            mapping["sqlite"]["table"],
            mapping["sqlite"]["key"],
            mapping["mongodb"]["collection"],
            mapping["mongodb"]["key"],
        )
        unique[key] = mapping
    payload = {"mappings": list(unique.values())}
    validate_mappings(payload)
    _write_json(MAPPINGS_FILE, payload)
    return payload


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
    normalized = {"rules": rules}
    _write_json(BUSINESS_RULES_FILE, normalized)
    reset_rag()
    return BUSINESS_RULES_FILE


# ---------------------------------------------------------------------------
# Lightweight RAG over generated documentation + business rules
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
    elif database == "mongodb":
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
    """Local TF-IDF retriever over generated database docs and optional rules."""

    def __init__(self) -> None:
        self.chunks = (
            _flatten_docs("sqlite", load_database_docs("sqlite"))
            + _flatten_docs("mongodb", load_database_docs("mongodb"))
        )
        rules = load_json(BUSINESS_RULES_FILE).get("rules", [])
        for index, rule in enumerate(rules):
            text = rule if isinstance(rule, str) else json.dumps(rule, ensure_ascii=False)
            self.chunks.append(
                DocumentChunk("business", f"rule_{index + 1}", text, {"rule": rule})
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
        candidate_indices = [i for i, _ in candidates]
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix[candidate_indices])[0]
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:k]
        return [
            {
                "database": c.database,
                "name": c.name,
                "score": float(score),
                "metadata": c.metadata,
                "text": c.text,
            }
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
# Workspace lifecycle
# ---------------------------------------------------------------------------

def analyze_workspace() -> dict[str, Any]:
    """Inspect available databases, regenerate docs/mappings and refresh RAG."""
    result: dict[str, Any] = {
        "sqlite": {"available": SQLITE_DB.exists()},
        "mongodb": {"available": False},
        "correspondences": [],
        "business_rules": BUSINESS_RULES_FILE.exists(),
    }

    sqlite_schema_data = None
    mongo_schema_data = None

    if SQLITE_DB.exists():
        sqlite_schema_data = analyze_sqlite_schema()
        sqlite_docs = generate_sqlite_docs(sqlite_schema_data)
        result["sqlite"].update({
            "tables": len(sqlite_schema_data.get("tables", [])),
            "documentation_generated": True,
            "documentation_path": str(SQLITE_DOCS),
            "schema": sqlite_schema_data,
        })

    if MongoClient is not None:
        try:
            mongo_schema_data = analyze_mongodb_schema()
            generate_mongodb_docs(mongo_schema_data)
            result["mongodb"] = {
                "available": True,
                "collections": len(mongo_schema_data.get("collections", [])),
                "documentation_generated": True,
                "documentation_path": str(MONGO_DOCS),
                "schema": mongo_schema_data,
            }
        except Exception as exc:
            result["mongodb"] = {"available": False, "error": str(exc)}

    if sqlite_schema_data and mongo_schema_data:
        mappings = analyze_correspondences(sqlite_schema_data, mongo_schema_data)
        result["correspondences"] = mappings["mappings"]

    reset_rag()
    result["rag_documents"] = len(get_rag().chunks)
    return result


def save_upload(filename: str, content: bytes, target: str) -> Path:
    """Save a database upload or optional business rules.

    target: sqlite-db | business-rules
    Documentation and mappings are generated automatically.
    """
    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("Invalid filename")

    if target == "sqlite-db":
        suffix = Path(safe_name).suffix.lower()
        if suffix not in {".db", ".sqlite", ".sqlite3"}:
            raise ValueError(f"Unsupported file type for sqlite-db: {suffix}")
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(content)
                tmp.flush()
                with sqlite3.connect(tmp.name) as conn:
                    conn.execute("PRAGMA schema_version").fetchone()
        except sqlite3.Error as exc:
            raise ValueError(f"Invalid SQLite database: {exc}") from exc
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SQLITE_DB.write_bytes(content)
        analyze_workspace()
        return SQLITE_DB

    if target == "business-rules":
        return save_business_rules(content, safe_name)

    raise ValueError("Unknown upload target")


def workspace_status() -> dict[str, Any]:
    return {
        "data_dir": str(DATA_DIR),
        "sqlite_database": SQLITE_DB.exists(),
        "sqlite_docs": SQLITE_DOCS.exists(),
        "mongodb_docs": MONGO_DOCS.exists(),
        "mappings": MAPPINGS_FILE.exists(),
        "business_rules": BUSINESS_RULES_FILE.exists(),
        "rag_documents": len(get_rag().chunks),
        "claude_configured": bool(os.getenv("CLAUDE_API_KEY")),
        "claude_model": os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
    }


def sqlite_schema(db_path: str | Path = SQLITE_DB) -> list[dict[str, Any]]:
    """Backward-compatible compact SQLite schema for agents."""
    return analyze_sqlite_schema(db_path).get("tables", [])


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
    merged = left_df.merge(
        right_df,
        left_on=left_key,
        right_on=right_key,
        how=how,
        suffixes=("_sql", "_mongo"),
    )
    return merged.where(pd.notna(merged), None).to_dict(orient="records")


def hybrid_correspondences() -> list[dict[str, Any]]:
    return load_mappings()
