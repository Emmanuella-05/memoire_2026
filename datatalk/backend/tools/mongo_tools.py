"""Outils MongoDB : inférence de schéma, garde-fous et exécution de pipelines
d'agrégation.

Deux moteurs, choisis automatiquement (voir catalog_tools.mongo_connection) :
  - connexion saisie dans l'interface -> serveur MongoDB réel, en LECTURE SEULE
  - MONGO_URI défini                  -> serveur MongoDB réel (les données importées y sont chargées)
  - sinon                             -> moteur embarqué `mongomock`, alimenté par data.json
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
from datetime import datetime, timezone
from typing import Any

from .catalog_tools import MONGO_DATA, DataSourceUnavailable, mongo_connection

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:  # pragma: no cover
    MongoClient = None
    PyMongoError = Exception

try:
    from bson import Decimal128, ObjectId
except ImportError:  # pragma: no cover
    Decimal128 = ObjectId = None

MAX_FETCH_ROWS = 10_000
MAX_TIME_MS = 15_000
MAX_STAGES = 30
MAX_PIPELINE_CHARS = 20_000
SAMPLE_VALUES_PER_FIELD = 3
MAX_ENUM_VALUES = 20  # au plus 20 valeurs distinctes pour qu'un champ soit « catégoriel »
# champs de référence / identifiants / données personnelles : jamais énumérés (aligné sur ingest_tools.KEY_LIKE_TOKENS)
_ENUM_SKIP_TOKENS = {"id", "ids", "key", "ref", "reference", "code", "uuid", "guid", "number", "num", "no", "pk", "fk",
                     "idx", "sku", "isbn", "slug", "username", "login", "matricule", "email", "mail", "phone", "tel",
                     "telephone", "password", "token"}

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
    if isinstance(value, datetime):
        return "date"
    if ObjectId is not None and isinstance(value, ObjectId):
        return "objectId"
    if Decimal128 is not None and isinstance(value, Decimal128):
        return "decimal"
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
    """Valeur d'exemple affichable ET sérialisable en JSON (dates, ObjectId et
    Decimal128 d'une vraie base MongoDB deviennent du texte)."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return f"<binaire {len(value)} octets>"
    if not isinstance(value, (str, int, float, bool, type(None))):
        value = str(value)
    return value[:40] + "…" if isinstance(value, str) and len(value) > 40 else value


def _enum_values(field: str, seen: dict[str, int] | None) -> list[str]:
    """Valeurs d'un champ texte catégoriel (peu de valeurs qui se répètent), sinon []."""
    if not seen:
        return []
    last = re.split(r"[.]", field)[-1].lower()
    if last.endswith("_id") or last in _ENUM_SKIP_TOKENS or set(re.split(r"[^a-z0-9]+", last)) & _ENUM_SKIP_TOKENS:
        return []
    if any(len(v) > 40 for v in seen) or sum(seen.values()) < 2 * len(seen):
        return []
    return [v for v, _ in sorted(seen.items(), key=lambda item: (-item[1], item[0]))]


def _docs_to_collection_entry(name: str, docs: list[dict[str, Any]], total: int, sample_size: int) -> dict[str, Any]:
    field_types: dict[str, set[str]] = {}
    samples: dict[str, list[Any]] = {}
    counts: dict[str, dict[str, int]] = {}      # champ texte -> {valeur: occurrences} (abandonné au-delà de 20 valeurs)
    overflow: set[str] = set()
    for doc in docs:
        for field, types in _flatten_mongo_fields(doc).items():
            field_types.setdefault(field, set()).update(types)
        for path, scalar in iter_scalar_values(doc):
            bucket = samples.setdefault(path, [])
            if scalar is not None and len(bucket) < SAMPLE_VALUES_PER_FIELD and scalar not in bucket:
                bucket.append(_short(scalar))
            if isinstance(scalar, str) and path not in overflow:
                seen = counts.setdefault(path, {})
                seen[scalar] = seen.get(scalar, 0) + 1
                if len(seen) > MAX_ENUM_VALUES:
                    overflow.add(path)
                    counts.pop(path, None)
    fields = []
    for field, types in sorted(field_types.items()):
        entry = {"name": field, "type": _merge_type(types), "sampled": len(docs)}
        if samples.get(field):
            entry["sample_values"] = samples[field]
        enum = _enum_values(field, counts.get(field))
        if enum:
            entry["distinct_values"] = enum
        fields.append(entry)
    return {
        "name": name,
        "description": "",
        "document_count": total,
        "sample_size": len(docs),
        "fields": fields,
    }


def analyze_mongo_payload(payload: dict[str, list[dict[str, Any]]], sample_size: int = 200,
                          database_name: str | None = None,
                          totals: dict[str, int] | None = None) -> dict[str, Any]:
    """Documentation des collections à partir des documents eux-mêmes (aucun
    serveur MongoDB requis). `totals` donne le vrai nombre de documents de chaque
    collection quand `payload` n'en contient qu'un échantillon (base connectée)."""
    totals = totals or {}
    collections = [
        _docs_to_collection_entry(name, docs[:sample_size], totals.get(name, len(docs)), sample_size)
        for name, docs in sorted(payload.items())
    ]
    conn = mongo_connection()
    return {
        "database": "mongodb",
        "database_name": database_name or (conn or {}).get("database") or os.getenv("MONGO_DB", "datatalk"),
        "collections": collections,
    }


def analyze_mongodb_schema(mongo_uri: str | None = None, database: str | None = None,
                           sample_size: int = 200) -> dict[str, Any]:
    """Documente un serveur MongoDB existant (échantillon de chaque collection)."""
    payload, totals = read_live_mongodb(mongo_uri, database, sample_size)
    return analyze_mongo_payload(payload, sample_size, database, totals)


_DB_NAME_RE = re.compile(r'^[^/\\. "$*<>:|?\x00]{1,63}$')


def _friendly_mongo_error(exc: Exception) -> str:
    """Message lisible pour l'utilisateur, sans jamais citer l'URI (mot de passe)."""
    name = type(exc).__name__
    text = str(exc).lower()
    if name in ("ServerSelectionTimeoutError", "AutoReconnect", "NetworkTimeout", "ConnectionFailure"):
        return ("MongoDB est injoignable à cette adresse. Vérifiez que le serveur MongoDB est démarré "
                "(sous Windows : service « MongoDB ») et que l'adresse et le port sont corrects.")
    if name == "OperationFailure" and ("auth" in text or "unauthorized" in text or "not authorized" in text):
        return "Accès refusé par MongoDB : vérifiez l'identifiant, le mot de passe et les droits de lecture."
    if name in ("InvalidURI", "ConfigurationError"):
        return "URI MongoDB invalide. Exemple : mongodb://localhost:27017"
    return f"Erreur MongoDB ({name})."


def _open_client(uri: str, timeout_ms: int = 4000):
    if MongoClient is None:
        raise DataSourceUnavailable("Le paquet 'pymongo' n'est pas installé.")
    try:
        return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
    except Exception as exc:  # URI mal formée
        raise ValueError(_friendly_mongo_error(exc)) from exc


def check_mongo_connection(uri: str, database: str) -> dict[str, Any]:
    """Teste la connexion et liste les collections (avec leur nombre de documents).
    Lève ValueError avec un message lisible si la base n'est pas utilisable."""
    if not re.match(r"^mongodb(?:\+srv)?://\S+$", uri.strip()):
        raise ValueError("L'URI doit commencer par mongodb:// (ou mongodb+srv://). Exemple : mongodb://localhost:27017")
    if not _DB_NAME_RE.match(database):
        raise ValueError("Nom de base invalide (63 caractères maximum, sans espace ni / \\ . \" $).")
    client = _open_client(uri.strip())
    try:
        client.admin.command("ping")
        db = client[database]
        names = sorted(n for n in db.list_collection_names() if not n.startswith("system."))
        if not names:
            others: list[str] = []
            try:
                others = [d for d in client.list_database_names() if d not in ("admin", "config", "local")]
            except PyMongoError:
                pass
            hint = f" Bases disponibles : {', '.join(others)}." if others else ""
            raise ValueError(f"La base '{database}' n'existe pas ou ne contient aucune collection.{hint}")
        collections = []
        for name in names:
            try:
                count: int | None = db[name].estimated_document_count()
            except PyMongoError:
                count = None  # vue ou compte non disponible
            collections.append({"name": name, "documents": count})
        return {"collections": collections}
    except ValueError:
        raise
    except PyMongoError as exc:
        raise ValueError(_friendly_mongo_error(exc)) from exc
    finally:
        client.close()


def read_live_mongodb(mongo_uri: str | None = None, database: str | None = None,
                      limit: int = 1000) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Lecture seule d'un serveur MongoDB : (échantillon de documents par collection,
    vrai nombre de documents par collection). Le champ _id est omis de l'échantillon."""
    conn = mongo_connection()
    uri = mongo_uri or (conn or {}).get("uri")
    if not uri:
        raise DataSourceUnavailable("Aucune connexion MongoDB n'est configurée.")
    db_name = database or (conn or {}).get("database") or os.getenv("MONGO_DB", "datatalk")
    client = _open_client(uri, 3000)
    payload: dict[str, list[dict[str, Any]]] = {}
    totals: dict[str, int] = {}
    try:
        db = client[db_name]
        for name in sorted(db.list_collection_names()):
            if name.startswith("system."):
                continue
            payload[name] = list(db[name].find({}, {"_id": 0}).limit(limit))
            try:
                totals[name] = db[name].estimated_document_count()
            except PyMongoError:
                totals[name] = len(payload[name])
        return payload, totals
    except PyMongoError as exc:
        raise DataSourceUnavailable(f"MongoDB injoignable ({type(exc).__name__}).") from exc
    finally:
        client.close()


def sample_live_collections(mongo_uri: str | None = None, database: str | None = None,
                            limit: int = 1000) -> dict[str, list[dict[str, Any]]]:
    return read_live_mongodb(mongo_uri, database, limit)[0]


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
    return mongo_connection() is not None


def _db_name() -> str:
    conn = mongo_connection()
    return conn["database"] if conn else os.getenv("MONGO_DB", "datatalk")


def _to_bson(value: Any) -> Any:
    """Convertit le JSON étendu écrit par le LLM en vrais types BSON, pour comparer
    des champs de type date ou objectId d'un vrai serveur :
    {"$date": "2024-01-31"} -> datetime ; {"$oid": "..."} -> ObjectId."""
    if isinstance(value, dict):
        if set(value) == {"$date"} and isinstance(value["$date"], str):
            text = value["$date"].strip().replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError as exc:
                raise ValueError(f"Date invalide : {value['$date']!r} (format AAAA-MM-JJ ou AAAA-MM-JJTHH:MM:SSZ).") from exc
            # datetime naïf en UTC : c'est ce que PyMongo écrit et relit pour les dates BSON
            return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
        if set(value) == {"$oid"} and isinstance(value["$oid"], str) and ObjectId is not None:
            try:
                return ObjectId(value["$oid"])
            except Exception as exc:
                raise ValueError(f"ObjectId invalide : {value['$oid']!r}.") from exc
        return {key: _to_bson(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_bson(child) for child in value]
    return value


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
    Les collections portant le même nom sont remplacées. Refusé pour une base
    connectée depuis l'interface : elle est en lecture seule."""
    conn = mongo_connection()
    if not conn:
        raise DataSourceUnavailable("Aucun serveur MongoDB n'est configuré.")
    if conn["source"] == "connected":
        raise ValueError("La base MongoDB connectée est en lecture seule : aucune donnée n'y est écrite.")
    if MongoClient is None:
        raise DataSourceUnavailable("Le paquet 'pymongo' n'est pas installé.")
    client = MongoClient(conn["uri"], serverSelectionTimeoutMS=3000)
    try:
        db = client[conn["database"]]
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

    conn = mongo_connection()
    if conn:
        if MongoClient is None:
            raise DataSourceUnavailable("Le paquet 'pymongo' n'est pas installé.")
        client = MongoClient(conn["uri"], serverSelectionTimeoutMS=3000)
        pipeline = _to_bson(pipeline)
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