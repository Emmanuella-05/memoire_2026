"""Fiabilité des correspondances SQL ↔ MongoDB : pas de fausse correspondance entre identifiants
numériques sans rapport, et seules les correspondances sûres sont utilisées par le mode hybride.

Lancer :  python -m pytest tests -q      (depuis le dossier datatalk/)
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATATALK_DATA_DIR", tempfile.mkdtemp(prefix="datatalk-mapping-test-"))
os.environ.pop("MONGO_URI", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.tools import catalog_tools  # noqa: E402


def links(mappings) -> set[tuple[str, str, str, str]]:
    return {(m["sqlite"]["table"], m["sqlite"]["key"], m["mongodb"]["collection"], m["mongodb"]["key"]) for m in mappings}


@pytest.fixture(scope="module", autouse=True)
def workspace():
    from backend.tools.ingest_tools import ingest_mongo_payload, ingest_sqlite_file

    catalog_tools.clear_mongo_connection()
    catalog_tools.MAPPINGS_FILE.unlink(missing_ok=True)
    staging = Path(os.environ["DATATALK_DATA_DIR"]) / "mapping_upload.db"
    staging.unlink(missing_ok=True)
    conn = sqlite3.connect(staging)
    conn.executescript("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL);
        CREATE TABLE products (sku TEXT PRIMARY KEY, label TEXT);
        CREATE TABLE warehouses (wh_code TEXT PRIMARY KEY, city TEXT);
        CREATE TABLE fournisseurs (fournisseur_id INTEGER PRIMARY KEY, nom TEXT);
    """)
    conn.executemany("INSERT INTO customers VALUES (?,?)", [(i, f"Client {i}") for i in range(1, 41)])
    conn.executemany("INSERT INTO orders VALUES (?,?,?)", [(i, (i % 40) + 1, 10.0 * i) for i in range(1, 61)])
    conn.executemany("INSERT INTO products VALUES (?,?)", [(f"SKU-{i:03d}", f"Produit {i}") for i in range(1, 21)])
    conn.executemany("INSERT INTO warehouses VALUES (?,?)", [(f"WH-{i}", "Cotonou") for i in range(1, 9)])
    conn.executemany("INSERT INTO fournisseurs VALUES (?,?)", [(i, f"Fournisseur {i}") for i in range(1, 16)])
    conn.commit(); conn.close()
    ingest_sqlite_file(staging)
    ingest_mongo_payload({
        # identifiants numériques qui se recoupent PAR COÏNCIDENCE avec customer_id / order_id
        "support_tickets": [{"ticket_id": i, "customer_id": (i % 40) + 1, "category": "retard"} for i in range(1, 61)],
        "reviews": [{"review_id": i, "item_ref": f"SKU-{(i % 20) + 1:03d}", "rating": i % 5} for i in range(1, 41)],
        # 8 codes seulement en commun et aucun nom de champ commun : preuve trop faible pour valider seul
        "stock": [{"location_ref": f"WH-{(i % 8) + 1}", "quantity": i} for i in range(1, 25)],
        # « fournisseur » (SQL, français) = « supplier » (Mongo, anglais)
        "deliveries": [{"delivery_id": 900 + i, "supplier_id": (i % 15) + 1, "status": "ok"} for i in range(1, 31)],
    })


def test_numeric_ids_that_merely_overlap_are_not_mapped():
    found = links(catalog_tools.load_all_mappings())
    assert ("customers", "customer_id", "support_tickets", "customer_id") in found      # vrai lien : même nom, mêmes valeurs
    for table, key in (("customers", "customer_id"), ("orders", "order_id")):
        assert (table, key, "support_tickets", "ticket_id") not in found                # coïncidence d'entiers
        assert (table, key, "reviews", "review_id") not in found


def _by_keys(mappings, sql_key, mongo_key):
    return next(m for m in mappings if (m["sqlite"]["key"], m["mongodb"]["key"]) == (sql_key, mongo_key))


def test_system_validates_strong_mappings_by_itself():
    """Aucune confirmation humaine : les correspondances sûres sont utilisées d'emblée."""
    usable = links(catalog_tools.load_mappings())
    assert ("customers", "customer_id", "support_tickets", "customer_id") in usable     # même nom + mêmes valeurs
    assert ("fournisseurs", "fournisseur_id", "deliveries", "supplier_id") in usable    # fournisseur = supplier
    sku = _by_keys(catalog_tools.load_all_mappings(), "sku", "item_ref")                # aucun nom commun...
    assert ("products", "sku", "reviews", "item_ref") in usable                         # ...mais 20 codes sur 20 en commun
    assert sku["status"] == "suggested" and sku["confidence"] == 0.85 and sku["evidence"]["proven_by_values"] is True


def test_weak_evidence_is_not_used_but_never_blocks_the_user():
    usable = links(catalog_tools.load_mappings())
    weak = _by_keys(catalog_tools.load_all_mappings(), "wh_code", "location_ref")      # 8 codes communs, pas de nom commun
    assert weak["confidence"] < catalog_tools.MIN_AUTO_CONFIDENCE and weak["evidence"]["proven_by_values"] is False
    assert ("warehouses", "wh_code", "stock", "location_ref") not in usable            # reste inutilisée, sans rien demander


def test_synonyms_and_accents_in_names():
    from backend.tools.ingest_tools import _signature
    assert _signature("client_id", "x") == _signature("customer_id", "y") == {"customer"}
    assert _signature("employé_id", "x") == {"employee"} and _signature("id", "avis") == {"review"}
    assert _signature("account_id", "x") != _signature("user_id", "x")                 # mot ambigu : pas d'équivalence


def test_an_administrator_can_still_confirm_or_reject():
    client = TestClient(__import__("backend.main", fromlist=["app"]).app)
    listed = client.get("/mappings").json()["mappings"]
    flags = {(m["sqlite"]["key"], m["mongodb"]["key"]): m["usable"] for m in listed}
    assert flags[("wh_code", "location_ref")] is False and flags[("customer_id", "customer_id")] is True
    assert flags[("sku", "item_ref")] is True

    for m in listed:
        if (m["sqlite"]["key"], m["mongodb"]["key"]) == ("wh_code", "location_ref"):
            m["status"] = "confirmed"
    saved = client.put("/mappings", json={"mappings": listed}).json()["mappings"]
    assert _by_keys(saved, "wh_code", "location_ref")["usable"] is True

    for m in saved:                                        # une correspondance fausse que le système avait validée : rejetable
        if (m["sqlite"]["key"], m["mongodb"]["key"]) == ("sku", "item_ref"):
            m["status"] = "rejected"
    client.put("/mappings", json={"mappings": saved})
    usable = links(catalog_tools.load_mappings())
    assert ("products", "sku", "reviews", "item_ref") not in usable and ("warehouses", "wh_code", "stock", "location_ref") in usable


def test_planner_prompt_shows_how_reliable_each_mapping_is(monkeypatch):
    from backend.agents import join_planner
    catalog_tools.MAPPINGS_FILE.unlink(missing_ok=True)
    from backend.tools.ingest_tools import analyze_workspace
    analyze_workspace()                                    # recalcule les correspondances (l'état de l'étape précédente est effacé)
    seen = {}
    monkeypatch.setattr(join_planner, "claude_json", lambda prompt, system="", max_tokens=0: seen.setdefault("p", prompt) and {})
    join_planner.plan_join("Quelle est la note moyenne des clients ?")
    listing = seen["p"].split("Correspondances disponibles :")[1]
    assert "[détectée automatiquement, confiance 1.0]" in listing and "reviews(item_ref) [détectée automatiquement, confiance 0.85]" in listing
    assert "location_ref" not in listing                   # la preuve trop faible n'est même pas proposée au modèle
    fields = {f["name"]: f for c in catalog_tools.load_database_docs("mongodb")["collections"] for f in c["fields"]}
    assert "distinct_values" not in fields["item_ref"]     # une référence n'est jamais énumérée dans les prompts
    assert not catalog_tools.mapping_is_trusted({"status": "suggested", "confidence": 0.79,
        "sqlite": {"table": "a", "key": "b"}, "mongodb": {"collection": "c", "key": "d"}})
    assert catalog_tools.mapping_is_trusted({"status": "confirmed", "confidence": 0.1,
        "sqlite": {"table": "a", "key": "b"}, "mongodb": {"collection": "c", "key": "d"}})
