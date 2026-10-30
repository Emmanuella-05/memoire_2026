"""Tests de la connexion à une base MongoDB existante (serveur simulé par mongomock).

Lancer :  python -m pytest tests -q      (depuis le dossier datatalk/)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="datatalk-mongo-test-")
os.environ["DATATALK_DATA_DIR"] = _TMP          # avant tout import de backend
os.environ.pop("MONGO_URI", None)
os.environ.pop("MONGO_DB", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mongomock  # noqa: E402
import pytest  # noqa: E402
from bson import Decimal128, ObjectId  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError  # noqa: E402

from backend.tools import catalog_tools, ingest_tools, mongo_tools  # noqa: E402

URI = "mongodb://localhost:27017"
DB = "beninterroir"


class _Admin:
    def __init__(self, error: Exception | None) -> None:
        self.error = error

    def command(self, name: str):
        if self.error:
            raise self.error
        return {"ok": 1}


class _BrokenClient:
    def __init__(self, error: Exception) -> None:
        self.admin = _Admin(error)

    def close(self) -> None:
        pass


@pytest.fixture()
def server(monkeypatch):
    """Faux serveur MongoDB « beninterroir » (types BSON réels : dates, ObjectId, Decimal128)."""
    client = mongomock.MongoClient()
    db = client[DB]
    db.products.insert_many([
        {"_id": ObjectId(), "name": "Karité", "producer_id": i, "price": Decimal128(f"{10 + i}.5"),
         "created_at": datetime(2024, i, 15), "tags": ["bio"], "origin": {"city": "Natitingou"}}
        for i in range(1, 7)
    ])
    db.producers.insert_many([{"_id": ObjectId(), "id": i, "name": f"Coopérative {i}"} for i in range(1, 7)])
    monkeypatch.setattr(mongo_tools, "MongoClient", lambda uri, **kw: client)
    monkeypatch.setattr(client, "close", lambda: None, raising=False)
    monkeypatch.setattr(ingest_tools, "MAX_KEY_VALUES", 3)          # échantillon plus petit que la base
    catalog_tools.clear_mongo_connection()
    for path in (catalog_tools.MONGO_DATA, catalog_tools.MONGO_DOCS, catalog_tools.MAPPINGS_FILE):
        path.unlink(missing_ok=True)
    yield client
    catalog_tools.clear_mongo_connection()


def _before(client) -> dict:
    return {name: client[DB][name].count_documents({}) for name in client[DB].list_collection_names()}


# --------------------------------------------------------------------------
# Connexion et documentation
# --------------------------------------------------------------------------

def test_connect_documents_the_real_database(server):
    result = ingest_tools.connect_mongodb(URI, DB)
    assert {c["name"]: c["documents"] for c in result["collections"]} == {"products": 6, "producers": 6}
    assert result["persisted"] is True                       # URI sans mot de passe : enregistrée

    docs = json.loads(catalog_tools.MONGO_DOCS.read_text(encoding="utf-8"))   # sérialisable malgré les types BSON
    assert docs["database_name"] == DB
    products = next(c for c in docs["collections"] if c["name"] == "products")
    assert products["document_count"] == 6 and products["sample_size"] == 3      # vrai total, pas celui de l'échantillon
    types = {f["name"]: f["type"] for f in products["fields"]}
    assert types["created_at"] == "date" and types["price"] == "decimal"
    assert types["origin.city"] == "string" and "tags" in types
    assert "_id" not in types
    assert "mongo" in catalog_tools.available_sources()


def test_workspace_never_exposes_the_password(server):
    result = ingest_tools.connect_mongodb("mongodb://admin:S3cret!@localhost:27017", DB)
    assert result["persisted"] is False
    assert not catalog_tools.MONGO_CONNECTION.exists()       # rien d'écrit sur le disque
    public = catalog_tools.public_mongo_connection()
    assert public == {"source": "connected", "server": "mongodb://localhost:27017", "database": DB}
    assert "S3cret" not in json.dumps(public)
    assert catalog_tools.mongo_connection()["uri"].endswith("@localhost:27017")   # mais utilisable en mémoire


def test_mappings_are_suggested_from_the_live_database(server):
    catalog_tools.SQLITE_DB.parent.mkdir(parents=True, exist_ok=True)
    catalog_tools.SQLITE_DB.unlink(missing_ok=True)
    conn = sqlite3.connect(catalog_tools.SQLITE_DB)
    conn.executescript("CREATE TABLE producers (id INTEGER PRIMARY KEY, name TEXT);"
                       "INSERT INTO producers VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d'),(5,'e'),(6,'f');")
    conn.commit(); conn.close()
    ingest_tools.connect_mongodb(URI, DB)
    links = {(m["sqlite"]["table"], m["mongodb"]["collection"]) for m in catalog_tools.load_all_mappings()}
    assert ("producers", "products") in links or ("producers", "producers") in links


# --------------------------------------------------------------------------
# Erreurs lisibles, rien n'est enregistré
# --------------------------------------------------------------------------

@pytest.mark.parametrize("uri, database, expected", [
    ("localhost:27017", DB, "mongodb://"),
    (URI, "ma base", "Nom de base invalide"),
    (URI, "inexistante", "n'existe pas"),
])
def test_invalid_input_is_explained(server, uri, database, expected):
    with pytest.raises(ValueError, match=expected):
        ingest_tools.connect_mongodb(uri, database)
    assert catalog_tools.mongo_connection() is None


def test_unknown_database_lists_the_available_ones(server):
    with pytest.raises(ValueError, match="beninterroir"):
        ingest_tools.connect_mongodb(URI, "inexistante")


@pytest.mark.parametrize("error, expected", [
    (ServerSelectionTimeoutError("localhost:27017: connection refused"), "injoignable"),
    (OperationFailure("Authentication failed."), "Accès refusé"),
])
def test_server_errors_are_explained_without_leaking_the_uri(monkeypatch, error, expected):
    catalog_tools.clear_mongo_connection()
    monkeypatch.setattr(mongo_tools, "MongoClient", lambda uri, **kw: _BrokenClient(error))
    with pytest.raises(ValueError, match=expected) as info:
        ingest_tools.connect_mongodb("mongodb://admin:S3cret!@localhost:27017", DB)
    assert "S3cret" not in str(info.value)
    assert catalog_tools.mongo_connection() is None and not catalog_tools.MONGO_CONNECTION.exists()


# --------------------------------------------------------------------------
# Lecture seule
# --------------------------------------------------------------------------

def test_connected_database_is_read_only(server):
    ingest_tools.connect_mongodb(URI, DB)
    before = _before(server)
    with pytest.raises(ValueError, match="déconnectez"):
        ingest_tools.ingest_mongo_payload({"products": [{"name": "x"}]})       # l'import JSON est refusé
    with pytest.raises(ValueError, match="lecture seule"):
        mongo_tools.load_payload_into_server({"products": [{"name": "x"}]})    # aucun chemin d'écriture
    for stage in ({"$out": "copie"}, {"$merge": {"into": "x"}}):
        with pytest.raises(ValueError, match="interdit"):
            mongo_tools.execute_mongo("products", [{"$match": {}}, stage], ["products", "producers"])
    assert _before(server) == before                                            # la base n'a pas bougé


# --------------------------------------------------------------------------
# Requêtes sur les types BSON
# --------------------------------------------------------------------------

def test_date_filter_uses_extended_json(server):
    ingest_tools.connect_mongodb(URI, DB)
    known = ["products", "producers"]
    rows = mongo_tools.execute_mongo(
        "products", [{"$match": {"created_at": {"$gte": {"$date": "2024-04-01"}}}}, {"$project": {"_id": 0, "producer_id": 1}}],
        known)
    assert sorted(r["producer_id"] for r in rows) == [4, 5, 6]
    rows = mongo_tools.execute_mongo(
        "products", [{"$match": {"created_at": {"$lt": {"$date": "2024-03-01T00:00:00Z"}}}}, {"$count": "n"}], known)
    assert rows == [{"n": 2}]
    with pytest.raises(ValueError, match="Date invalide"):
        mongo_tools.execute_mongo("products", [{"$match": {"created_at": {"$gte": {"$date": "hier"}}}}], known)


def test_oid_filter_and_json_safe_rows(server):
    ingest_tools.connect_mongodb(URI, DB)
    first = server[DB].products.find_one()
    rows = mongo_tools.execute_mongo("products", [{"$match": {"_id": {"$oid": str(first["_id"])}}}], ["products"])
    assert len(rows) == 1 and rows[0]["_id"] == first["_id"]
    from backend.main import _json_safe
    safe = _json_safe(rows)                                     # ObjectId / date / Decimal128 -> texte
    assert isinstance(safe[0]["_id"], str) and isinstance(safe[0]["created_at"], str)


# --------------------------------------------------------------------------
# Déconnexion et retour au mode fichier
# --------------------------------------------------------------------------

def test_disconnect_returns_to_the_uploaded_json(server):
    ingest_tools.connect_mongodb(URI, DB)
    catalog_tools.MONGO_DATA.parent.mkdir(parents=True, exist_ok=True)
    catalog_tools.MONGO_DATA.write_text(json.dumps({"villes": [{"nom": "Cotonou"}]}), encoding="utf-8")
    ingest_tools.disconnect_mongodb()
    assert catalog_tools.mongo_connection() is None
    docs = json.loads(catalog_tools.MONGO_DOCS.read_text(encoding="utf-8"))
    assert [c["name"] for c in docs["collections"]] == ["villes"]          # le JSON redevient la source
    assert "mongo" in catalog_tools.available_sources()


def test_disconnect_without_any_file_removes_the_mongo_source(server):
    ingest_tools.connect_mongodb(URI, DB)
    ingest_tools.disconnect_mongodb()
    assert not catalog_tools.MONGO_DOCS.exists()
    assert "mongo" not in catalog_tools.available_sources()


def test_json_upload_still_works_when_not_connected(server):
    summary = ingest_tools.ingest_mongo_payload({"villes": [{"nom": "Cotonou"}]})
    assert summary["mongodb"] is not None and catalog_tools.MONGO_DATA.exists()


# --------------------------------------------------------------------------
# API HTTP
# --------------------------------------------------------------------------

def test_http_connect_and_disconnect(server):
    from backend.main import app
    client = TestClient(app)
    ok = client.post("/connect/mongodb", json={"uri": URI, "database": DB})
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["workspace"]["mongodb_engine"] == "server"
    assert body["workspace"]["mongodb_connection"]["database"] == DB
    assert {c["name"] for c in body["collections"]} == {"products", "producers"}

    refused = client.post("/upload/mongodb-data", files={"file": ("d.json", b'{"a": [{"x": 1}]}', "application/json")})
    assert refused.status_code == 400 and "déconnectez" in refused.json()["detail"]

    bad = client.post("/connect/mongodb", json={"uri": URI, "database": "inexistante"})
    assert bad.status_code == 400 and "n'existe pas" in bad.json()["detail"]
    assert client.get("/workspace").json()["mongodb_connection"]["database"] == DB   # la connexion valide est conservée

    gone = client.delete("/connect/mongodb")
    assert gone.status_code == 200 and gone.json()["workspace"]["mongodb_connection"] is None
