"""Outils MongoDB : inférence de schéma et exécution de pipelines d'agrégation."""

from __future__ import annotations

import os
from typing import Any

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover
    MongoClient = None


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


FORBIDDEN_STAGES = {"$out", "$merge", "$currentOp", "$collStats", "$indexStats", "$function", "$accumulator"}


def is_pipeline_safe(pipeline: list[dict[str, Any]]) -> bool:
    """True si aucune étape du pipeline n'écrit ni n'inspecte le serveur.
    Utilisé à la fois par execute_mongo (garde-fou final) et par l'agent
    Mongo (validation avant exécution)."""
    if not isinstance(pipeline, list):
        return False
    for stage in pipeline:
        if not isinstance(stage, dict):
            return False
        if FORBIDDEN_STAGES & stage.keys():
            return False
    return True


def execute_mongo(collection: str, pipeline: list[dict[str, Any]],
                  mongo_uri: str | None = None, database: str | None = None) -> list[dict[str, Any]]:
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed")
    if not is_pipeline_safe(pipeline):
        raise ValueError("Pipeline contains a forbidden (write/admin) stage")
    uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = database or os.getenv("MONGO_DB", "datatalk")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    try:
        return list(client[db_name][collection].aggregate(pipeline))
    finally:
        client.close()