"""Ingestion des données importées par l'utilisateur.

Chaîne complète  upload → analyse → documentation → correspondances :

  1. validation du fichier reçu (SQLite valide, JSON {collection: [documents]})
  2. installation dans l'espace de travail (remplacement atomique)
  3. génération de la documentation structurée des schémas
     (tables/colonnes/valeurs d'exemple, collections/champs/types)
  4. détection automatique des correspondances SQL ↔ MongoDB à partir des noms
     ET des valeurs (recouvrement des clés), enregistrées dans mappings.json
  5. invalidation de l'index RAG

Aucune donnée « réelle » n'est requise pour développer ou tester cette chaîne.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from ..rag.retriever import reset_index
from .catalog_tools import (
    MAPPINGS_FILE,
    MONGO_DATA,
    MONGO_DOCS,
    SQLITE_DB,
    SQLITE_DOCS,
    DataSourceUnavailable,
    _write_json,
    load_all_mappings,
    mapping_is_valid,
    normalize_key,
)
from .mongo_tools import (
    analyze_mongo_payload,
    iter_scalar_values,
    load_payload_into_server,
    sample_live_collections,
    validate_mongo_payload,
)
from .sql_tools import _connect_ro, analyze_sqlite_schema

SQLITE_MAGIC = b"SQLite format 3\x00"
SAMPLE_VALUES_PER_COLUMN = 3
MAX_KEY_VALUES = 2000

GENERIC_TOKENS = {"id", "ids", "key", "ref", "reference", "code", "uuid", "guid",
                  "number", "num", "no", "pk", "fk", "idx"}
KEY_LIKE_TOKENS = GENERIC_TOKENS | {"email", "sku", "isbn", "slug", "username", "login", "matricule"}
NON_KEY_TYPES = ("BLOB", "REAL", "FLOA", "DOUB", "DATE", "TIME", "BOOL")


# ---------------------------------------------------------------------------
# Noms : tokens, singulier, « ressemble à une clé »
# ---------------------------------------------------------------------------

def _tokens(name: str) -> list[str]:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t]


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _signature(field: str, container: str) -> set[str]:
    """Mots significatifs d'un champ ; si le champ est générique (id, code...),
    on emprunte le nom de la table/collection (customers.id -> {customer})."""
    sig = {_singular(t) for t in _tokens(field) if t not in GENERIC_TOKENS}
    if not sig:
        sig = {_singular(t) for t in _tokens(container) if t not in GENERIC_TOKENS}
    return sig


def _key_like(field: str) -> bool:
    return any(t in KEY_LIKE_TOKENS for t in _tokens(field))


# ---------------------------------------------------------------------------
# Documentation SQLite
# ---------------------------------------------------------------------------

def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _short(value: Any) -> Any:
    return value[:40] + "…" if isinstance(value, str) and len(value) > 40 else value


def build_sqlite_docs(db_path: Path = SQLITE_DB) -> dict[str, Any]:
    """Schéma SQLite enrichi : nombre de lignes et valeurs d'exemple par colonne."""
    docs = analyze_sqlite_schema(db_path)
    conn = _connect_ro(db_path)
    try:
        for table in docs["tables"]:
            table.setdefault("description", "")
            try:
                table["row_count"] = conn.execute(f"SELECT COUNT(*) FROM {_quote(table['name'])}").fetchone()[0]
            except Exception:
                table["row_count"] = None
            for column in table["columns"]:
                if any(t in column["type"].upper() for t in ("BLOB",)):
                    continue
                try:
                    values = [
                        _short(row[0]) for row in conn.execute(
                            f"SELECT DISTINCT {_quote(column['name'])} FROM {_quote(table['name'])} "
                            f"WHERE {_quote(column['name'])} IS NOT NULL LIMIT {SAMPLE_VALUES_PER_COLUMN}"
                        )
                    ]
                except Exception:
                    values = []
                if values:
                    column["sample_values"] = values
    finally:
        conn.close()
    return docs


# ---------------------------------------------------------------------------
# Détection des correspondances SQL ↔ MongoDB
# ---------------------------------------------------------------------------

def _sqlite_key_candidates(docs: dict[str, Any], db_path: Path) -> list[dict[str, Any]]:
    candidates = []
    conn = _connect_ro(db_path)
    try:
        for table in docs.get("tables", []):
            fk_columns = {fk["column"] for fk in table.get("foreign_keys", [])}
            for column in table.get("columns", []):
                ctype = column.get("type", "").upper()
                if any(t in ctype for t in NON_KEY_TYPES):
                    continue
                if not (column.get("primary_key") or column["name"] in fk_columns or _key_like(column["name"])):
                    continue
                try:
                    raw = conn.execute(
                        f"SELECT DISTINCT {_quote(column['name'])} FROM {_quote(table['name'])} "
                        f"WHERE {_quote(column['name'])} IS NOT NULL LIMIT {MAX_KEY_VALUES}"
                    ).fetchall()
                except Exception:
                    continue
                values = {k for k in (normalize_key(r[0]) for r in raw) if k is not None}
                if values:
                    candidates.append({
                        "table": table["name"], "column": column["name"], "values": values,
                        "sig": _signature(column["name"], table["name"]),
                    })
    finally:
        conn.close()
    return candidates


def _mongo_key_candidates(payload: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    candidates = []
    for collection, docs in payload.items():
        values_by_path: dict[str, set[str]] = {}
        for doc in docs[:MAX_KEY_VALUES]:
            for path, scalar in iter_scalar_values(doc):
                key = normalize_key(scalar)
                if key is not None:
                    values_by_path.setdefault(path, set()).add(key)
        for path, values in values_by_path.items():
            if _key_like(path.split(".")[-1]) or _key_like(path):
                candidates.append({
                    "collection": collection, "field": path, "values": values,
                    "sig": _signature(path.split(".")[-1], collection),
                })
    return candidates


def suggest_mappings(sqlite_docs: dict[str, Any], db_path: Path,
                     mongo_payload: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Propose des correspondances (table.colonne ↔ collection.champ) quand les
    noms se ressemblent ET que les valeurs de clés se recoupent réellement."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    sql_candidates = _sqlite_key_candidates(sqlite_docs, db_path)
    for s in sql_candidates:
        for m in _mongo_key_candidates(mongo_payload):
            common = s["values"] & m["values"]
            if not common:
                continue
            overlap = len(common) / min(len(s["values"]), len(m["values"]))
            shared_words = s["sig"] & m["sig"]
            name_match = bool(shared_words)
            strong_values = overlap >= 0.9 and len(common) >= 5
            if overlap < 0.5 or not (name_match or strong_values):
                continue
            confidence = round(0.6 * overlap + 0.4 * (1.0 if name_match else 0.0), 2)
            pair = (s["table"], m["collection"])
            if pair in best and best[pair]["confidence"] >= confidence:
                continue
            entity = sorted(shared_words)[0] if shared_words else _singular(s["table"])
            best[pair] = {
                "entity": entity,
                "sqlite": {"table": s["table"], "key": s["column"]},
                "mongodb": {"collection": m["collection"], "key": m["field"]},
                "status": "suggested",
                "source": "auto",
                "confidence": confidence,
                "evidence": {
                    "common_values": len(common),
                    "sqlite_distinct": len(s["values"]),
                    "mongodb_distinct": len(m["values"]),
                    "overlap": round(overlap, 2),
                    "names_match": name_match,
                },
            }
    return sorted(best.values(), key=lambda x: -x["confidence"])


def _same_link(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["sqlite"]["table"] == b["sqlite"]["table"] and a["sqlite"]["key"] == b["sqlite"]["key"]
        and a["mongodb"]["collection"] == b["mongodb"]["collection"] and a["mongodb"]["key"] == b["mongodb"]["key"]
    )


def mapping_matches_docs(mapping: dict[str, Any], sqlite_docs: dict[str, Any], mongo_docs: dict[str, Any]) -> bool:
    """La correspondance référence-t-elle des tables/colonnes/collections/champs existants ?"""
    if not mapping_is_valid(mapping):
        return False
    table = next((t for t in sqlite_docs.get("tables", []) if t["name"] == mapping["sqlite"]["table"]), None)
    if not table or mapping["sqlite"]["key"] not in {c["name"] for c in table.get("columns", [])}:
        return False
    coll = next((c for c in mongo_docs.get("collections", []) if c["name"] == mapping["mongodb"]["collection"]), None)
    return bool(coll and mapping["mongodb"]["key"] in {f["name"] for f in coll.get("fields", [])})


def refresh_mappings(sqlite_docs: dict[str, Any] | None, mongo_docs: dict[str, Any] | None,
                     db_path: Path, mongo_payload: dict[str, list[dict[str, Any]]] | None) -> list[dict[str, Any]]:
    """Recalcule les suggestions en conservant les choix de l'utilisateur
    (manuels, confirmés ou rejetés) tant qu'ils restent valides."""
    existing = load_all_mappings()
    kept = [
        m for m in existing
        if (m.get("source") == "manual" or m.get("status") in ("confirmed", "rejected"))
        and sqlite_docs and mongo_docs and mapping_matches_docs(m, sqlite_docs, mongo_docs)
    ]
    suggestions: list[dict[str, Any]] = []
    if sqlite_docs and sqlite_docs.get("tables") and mongo_payload:
        for s in suggest_mappings(sqlite_docs, db_path, mongo_payload):
            if not any(_same_link(s, k) for k in kept):
                suggestions.append(s)
    mappings = kept + suggestions
    _write_json(MAPPINGS_FILE, {"mappings": mappings})
    return mappings


def save_user_mappings(mappings: Any) -> list[dict[str, Any]]:
    """Remplace les correspondances par celles de l'utilisateur (après
    vérification contre la documentation existante)."""
    if not isinstance(mappings, list):
        raise ValueError("Les correspondances doivent être une liste.")
    sqlite_docs = _read_docs(SQLITE_DOCS, "tables")
    mongo_docs = _read_docs(MONGO_DOCS, "collections")
    cleaned = []
    for index, m in enumerate(mappings, start=1):
        if not isinstance(m, dict) or not mapping_is_valid(m):
            raise ValueError(f"Correspondance {index} incomplète : sqlite.table/key et mongodb.collection/key sont requis.")
        if not mapping_matches_docs(m, sqlite_docs, mongo_docs):
            raise ValueError(f"Correspondance {index} : table, colonne, collection ou champ introuvable dans les schémas.")
        status = m.get("status", "confirmed")
        if status not in ("confirmed", "suggested", "rejected"):
            raise ValueError(f"Correspondance {index} : statut invalide {status!r}.")
        cleaned.append({
            "entity": str(m.get("entity") or m["sqlite"]["table"]),
            "sqlite": {"table": m["sqlite"]["table"], "key": m["sqlite"]["key"]},
            "mongodb": {"collection": m["mongodb"]["collection"], "key": m["mongodb"]["key"]},
            "status": status,
            "source": "manual" if m.get("source") != "auto" else "auto",
            **({"confidence": m["confidence"]} if "confidence" in m else {}),
            **({"evidence": m["evidence"]} if "evidence" in m else {}),
        })
    _write_json(MAPPINGS_FILE, {"mappings": cleaned})
    return cleaned


# ---------------------------------------------------------------------------
# Points d'entrée : upload et ré-analyse
# ---------------------------------------------------------------------------

def _read_docs(path: Path, key: str) -> dict[str, Any]:
    if not path.exists():
        return {key: []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {key: []}


def _read_mongo_payload() -> dict[str, list[dict[str, Any]]] | None:
    """Données MongoDB disponibles pour l'analyse : fichier importé, sinon échantillon du serveur."""
    if MONGO_DATA.exists():
        return json.loads(MONGO_DATA.read_text(encoding="utf-8"))
    if os.getenv("MONGO_URI"):
        return sample_live_collections(limit=MAX_KEY_VALUES)
    return None


def _summary(sqlite_docs: dict[str, Any] | None, mongo_docs: dict[str, Any] | None,
             mappings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sqlite": None if not sqlite_docs else [
            {"table": t["name"], "columns": len(t["columns"]), "rows": t.get("row_count")}
            for t in sqlite_docs.get("tables", [])
        ],
        "mongodb": None if not mongo_docs else [
            {"collection": c["name"], "fields": len(c["fields"]), "documents": c.get("document_count")}
            for c in mongo_docs.get("collections", [])
        ],
        "mappings": mappings,
    }


def validate_sqlite_file(path: Path) -> None:
    """Lève ValueError si le fichier n'est pas une base SQLite exploitable."""
    with path.open("rb") as handle:
        if handle.read(16) != SQLITE_MAGIC:
            raise ValueError("Ce fichier n'est pas une base SQLite valide.")
    try:
        conn = _connect_ro(path)
    except DataSourceUnavailable as exc:
        raise ValueError(str(exc)) from exc
    try:
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("La base SQLite est corrompue (quick_check a échoué).")
        tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
        if tables == 0:
            raise ValueError("La base SQLite ne contient aucune table.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Base SQLite illisible : {exc}") from exc
    finally:
        conn.close()


def ingest_sqlite_file(uploaded: Path) -> dict[str, Any]:
    """Installe la base reçue (déjà écrite dans `uploaded`, à côté de la cible),
    puis génère sa documentation et rafraîchit les correspondances."""
    validate_sqlite_file(uploaded)
    SQLITE_DB.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(uploaded, SQLITE_DB)       # atomique si même disque
    except OSError:
        shutil.move(str(uploaded), str(SQLITE_DB))
    return analyze_workspace()


def ingest_mongo_payload(payload: Any) -> dict[str, Any]:
    """Valide, installe les données MongoDB, puis génère documentation et correspondances."""
    payload = validate_mongo_payload(payload)
    _write_json(MONGO_DATA, payload)
    if os.getenv("MONGO_URI"):
        load_payload_into_server(payload)
    return analyze_workspace()


def analyze_workspace() -> dict[str, Any]:
    """(Ré)génère toute la documentation à partir des données présentes :
    schémas SQLite et MongoDB, correspondances suggérées, index RAG."""
    sqlite_docs = None
    if SQLITE_DB.exists():
        sqlite_docs = build_sqlite_docs(SQLITE_DB)
        _write_json(SQLITE_DOCS, sqlite_docs)

    mongo_payload = _read_mongo_payload()
    mongo_docs = None
    if mongo_payload is not None:
        mongo_docs = analyze_mongo_payload(mongo_payload)
        _write_json(MONGO_DOCS, mongo_docs)

    mappings = refresh_mappings(sqlite_docs, mongo_docs, SQLITE_DB, mongo_payload)
    reset_index()
    return _summary(sqlite_docs, mongo_docs, mappings)