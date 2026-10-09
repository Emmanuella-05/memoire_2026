"""Outils MongoDB : inférence de schéma, garde-fous et exécution de pipelines
d'agrégation.

Deux moteurs, choisis automatiquement :
  - MONGO_URI défini  -> serveur MongoDB réel (les données importées y sont chargées)
  - sinon             -> moteur embarqué `mongomock`, alimenté par data.json
                         (permet d'utiliser DataTalk sans installer MongoDB)

Garde-fous : parcours RÉCURSIF du pipeline (opérateurs d'écriture, de code
JavaScript et d'administration interdits à toute profondeur), noms de
collection contrôlés, plafond de lignes et timeout côté serveur.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import re
from typing import Any

from .catalog_tools import MONGO_DATA, DataSourceUnavailable

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:  # pragma: no cover
    MongoClient = None
    PyMongoError = Exception

MAX_FETCH_ROWS = 10_000
MAX_TIME_MS = 15_000
MAX_STAGES = 30
MAX_PIPELINE_CHARS = 20_000
SAMPLE_VALUES_PER_FIELD = 3

_COLLECTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,119}$")


# ---------------------------------------------------------------------------
# Inférence de schéma
# ---------------------------------------------------------------------------

def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
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
                for name, types in _flatten_mongo_fields(child, path).items():
                    fields.setdefault(name, set()).update(types)
            elif isinstance(child, list):
                fields.setdefault(path, set()).add("array")
                if child:
                    first = child[0]
                    fields.setdefault(path, set()).add(_json_type(first))
                    if isinstance(first, dict):
                        for name, types in _flatten_mongo_fields(first, path).items():
                            fields.setdefault(name, set()).update(types)
            else:
                fields.setdefault(path, set()).add(_json_type(child))
    return fields


def iter_scalar_values(value: Any, prefix: str = ""):
    """Génère (chemin, valeur scalaire) pour tout un document (objets imbriqués
    inclus ; les tableaux ne sont pas descendus)."""
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(child, dict):
                yield from iter_scalar_values(child, path)
            elif not isinstance(child, list):
                yield path, child


def _short(value: Any) -> Any:
    return value[:40] + "…" if isinstance(value, str) and len(value) > 40 else value


def _docs_to_collection_entry(name: str, docs: list[dict[str, Any]], total: int, sample_size: int) -> dict[str, Any]:
    field_types: dict[str, set[str]] = {}
    samples: dict[str, list[Any]] = {}
    for doc in docs:
        for field, types in _flatten_mongo_fields(doc).items():
            field_types.setdefault(field, set()).update(types)
        for path, scalar in iter_scalar_values(doc):
            bucket = samples.setdefault(path, [])
            if scalar is not None and len(bucket) < SAMPLE_VALUES_PER_FIELD and scalar not in bucket:
                bucket.append(_short(scalar))
    fields = []
    for field, types in sorted(field_types.items()):
        entry = {"name": field, "type": _merge_type(types), "sampled": len(docs)}
        if samples.get(field):
            entry["sample_values"] = samples[field]
        fields.append(entry)
    return {
        "name": name,
        "description": "",
        "document_count": total,
        "sample_size": len(docs),
        "fields": fields,
    }


def analyze_mongo_payload(payload: dict[str, list[dict[str, Any]]], sample_size: int = 200,
                          database_name: str | None = None) -> dict[str, Any]:
    """Documentation des collections à partir des documents eux-mêmes (aucun
    serveur MongoDB requis)."""
    collections = [
        _docs_to_collection_entry(name, docs[:sample_size], len(docs), sample_size)
        for name, docs in sorted(payload.items())
    ]
    return {
        "database": "mongodb",
        "database_name": database_name or os.getenv("MONGO_DB", "datatalk"),
        "collections": collections,
    }


def analyze_mongodb_schema(mongo_uri: str | None = None, database: str | None = None,
                           sample_size: int = 200) -> dict[str, Any]:
    """Documente un serveur MongoDB existant (échantillon de chaque collection)."""
    payload = sample_live_collections(mongo_uri, database, sample_size)
    return analyze_mongo_payload(payload, sample_size, database)


def sample_live_collections(mongo_uri: str | None = None, database: str | None = None,
                            limit: int = 1000) -> dict[str, list[dict[str, Any]]]:
    if MongoClient is None:
        raise DataSourceUnavailable("Le paquet 'pymongo' n'est pas installé.")
    uri = mongo_uri or os.getenv("MONGO_URI")
    if not uri:
        raise DataSourceUnavailable("MONGO_URI n'est pas défini.")
    db_name = database or os.getenv("MONGO_DB", "datatalk")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    try:
        db = client[db_name]
        return {
            name: list(db[name].find({}, {"_id": 0}).limit(limit))
            for name in sorted(db.list_collection_names())
            if not name.startswith("system.")
        }
    except PyMongoError as exc:
        raise DataSourceUnavailable(f"MongoDB injoignable ({type(exc).__name__}).") from exc
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Validation des données importées
# ---------------------------------------------------------------------------

def _has_bad_key(value: Any, depth: int = 0) -> str | None:
    if depth > 30:
        return "<profondeur excessive>"
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key.startswith("$") or "\x00" in key:
                return str(key)
            bad = _has_bad_key(child, depth + 1)
            if bad:
                return bad
    elif isinstance(value, list):
        for child in value:
            bad = _has_bad_key(child, depth + 1)
            if bad:
                return bad
    return None


def validate_mongo_payload(payload: Any) -> dict[str, list[dict[str, Any]]]:
    """Vérifie le format {collection: [documents]} ; lève ValueError sinon."""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Le JSON MongoDB doit être un objet non vide {collection: [documents]}.")
    if len(payload) > 200:
        raise ValueError("Trop de collections (200 maximum).")
    for name, docs in payload.items():
        if not _COLLECTION_RE.match(name) or name.startswith("system."):
            raise ValueError(f"Nom de collection invalide : {name!r}.")
        if not isinstance(docs, list):
            raise ValueError(f"La collection '{name}' doit être une liste de documents.")
        for index, doc in enumerate(docs):
            if not isinstance(doc, dict):
                raise ValueError(f"Collection '{name}', élément {index} : un document (objet JSON) est attendu.")
        bad = _has_bad_key(docs)
        if bad:
            raise ValueError(f"Collection '{name}' : nom de champ invalide {bad!r} (les noms ne peuvent pas commencer par '$').")
    return payload


# ---------------------------------------------------------------------------
# Garde-fous sur les pipelines
# ---------------------------------------------------------------------------

FORBIDDEN_OPERATORS = {
    # écriture
    "$out", "$merge",
    # exécution de code JavaScript
    "$function", "$accumulator", "$where",
    # administration / introspection du serveur
    "$currentOp", "$collStats", "$indexStats", "$planCacheStats",
    "$listSessions", "$listLocalSessions", "$listSampledQueries", "$changeStream",
}
FORBIDDEN_STAGES = FORBIDDEN_OPERATORS  # compatibilité


def collection_rejection_reason(collection: Any, known: list[str] | None = None) -> str | None:
    if not isinstance(collection, str) or not _COLLECTION_RE.match(collection) or collection.startswith("system."):
        return f"Nom de collection invalide : {collection!r}."
    if known and collection not in known:
        return f"La collection '{collection}' n'existe pas (collections disponibles : {', '.join(known)})."
    return None


def _walk(value: Any, found: set[str], referenced: list[str], depth: int = 0) -> None:
    if depth > 40:
        found.add("<profondeur excessive>")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_OPERATORS:
                found.add(key)
            if key in ("$lookup", "$graphLookup") and isinstance(child, dict) and isinstance(child.get("from"), str):
                referenced.append(child["from"])
            if key == "$unionWith":
                referenced.append(child if isinstance(child, str) else (child.get("coll") if isinstance(child, dict) else None))
            _walk(child, found, referenced, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _walk(child, found, referenced, depth + 1)


def pipeline_rejection_reason(pipeline: Any, known_collections: list[str] | None = None) -> str | None:
    """None si le pipeline est acceptable, sinon la raison du rejet."""
    if not isinstance(pipeline, list) or not pipeline:
        return "Le pipeline doit être une liste non vide d'étapes."
    if len(pipeline) > MAX_STAGES:
        return f"Pipeline trop long ({MAX_STAGES} étapes maximum)."
    if any(not isinstance(stage, dict) or len(stage) != 1 for stage in pipeline):
        return "Chaque étape du pipeline doit être un objet avec un seul opérateur ($match, $group...)."
    try:
        if len(json.dumps(pipeline, default=str)) > MAX_PIPELINE_CHARS:
            return "Pipeline trop volumineux."
    except (TypeError, ValueError):
        return "Pipeline non sérialisable en JSON."
    found: set[str] = set()
    referenced: list[str] = []
    _walk(pipeline, found, referenced)
    if found:
        return f"Opérateur interdit (écriture, code JavaScript ou administration) : {', '.join(sorted(found))}."
    for name in referenced:
        reason = collection_rejection_reason(name, known_collections)
        if reason:
            return reason
    return None


def is_pipeline_safe(pipeline: Any) -> bool:
    """True si le pipeline ne contient, à aucune profondeur, d'opérateur
    d'écriture, de code JavaScript ou d'administration."""
    return pipeline_rejection_reason(pipeline) is None


# ---------------------------------------------------------------------------
# Moteurs d'exécution
# ---------------------------------------------------------------------------

_mock_cache: dict[str, Any] = {"stamp": None, "client": None}


def _use_real_server() -> bool:
    return bool(os.getenv("MONGO_URI"))


def _db_name() -> str:
    return os.getenv("MONGO_DB", "datatalk")


def _mock_client():
    """Client mongomock alimenté par data.json (rechargé si le fichier change)."""
    try:
        import mongomock
    except ImportError as exc:
        raise DataSourceUnavailable(
            "Le paquet 'mongomock' est requis sans MONGO_URI (pip install mongomock)."
        ) from exc
    if not MONGO_DATA.exists():
        raise DataSourceUnavailable("Aucune donnée MongoDB n'a été importée.")
    stat = MONGO_DATA.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    if _mock_cache["stamp"] != stamp:
        payload = json.loads(MONGO_DATA.read_text(encoding="utf-8"))
        client = mongomock.MongoClient()
        db = client[_db_name()]
        for name, docs in payload.items():
            if docs:
                db[name].insert_many(copy.deepcopy(docs))
        _mock_cache.update(stamp=stamp, client=client)
    return _mock_cache["client"]


def load_payload_into_server(payload: dict[str, list[dict[str, Any]]]) -> None:
    """Charge les collections importées dans le serveur MongoDB (MONGO_URI).
    Les collections portant le même nom sont remplacées."""
    if MongoClient is None:
        raise DataSourceUnavailable("Le paquet 'pymongo' n'est pas installé.")
    client = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=3000)
    try:
        db = client[_db_name()]
        for name, docs in payload.items():
            db.drop_collection(name)
            if docs:
                db[name].insert_many(copy.deepcopy(docs))
    except PyMongoError as exc:
        raise DataSourceUnavailable(f"Import dans MongoDB impossible ({type(exc).__name__}).") from exc
    finally:
        client.close()


def execute_mongo(collection: str, pipeline: list[dict[str, Any]],
                  known_collections: list[str] | None = None,
                  max_rows: int = MAX_FETCH_ROWS) -> list[dict[str, Any]]:
    reason = collection_rejection_reason(collection, known_collections) or pipeline_rejection_reason(
        pipeline, known_collections
    )
    if reason:
        raise ValueError(reason)

    if _use_real_server():
        if MongoClient is None:
            raise DataSourceUnavailable("Le paquet 'pymongo' n'est pas installé.")
        client = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=3000)
        kwargs: dict[str, Any] = {"maxTimeMS": MAX_TIME_MS, "allowDiskUse": False}
        close = client.close
    else:
        client = _mock_client()
        kwargs = {}
        close = lambda: None  # le client mongomock est conservé en cache

    try:
        try:
            cursor = client[_db_name()][collection].aggregate(pipeline, **kwargs)
            rows = list(itertools.islice(cursor, max_rows + 1))
        except PyMongoError as exc:
            name = type(exc).__name__
            if name in ("ServerSelectionTimeoutError", "AutoReconnect", "NetworkTimeout", "ConnectionFailure"):
                raise DataSourceUnavailable(f"MongoDB injoignable ({name}).") from exc
            raise ValueError(f"Erreur MongoDB ({name}) : {exc}") from exc
        except NotImplementedError as exc:
            raise ValueError(f"Opérateur MongoDB non supporté par le moteur embarqué : {exc}") from exc
        if len(rows) > max_rows:
            raise ValueError(
                f"Le résultat dépasse {max_rows} documents : ajoutez un filtre, une agrégation ou un $limit."
            )
        return rows
    finally:
        close()