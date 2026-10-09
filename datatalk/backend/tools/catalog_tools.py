"""Helpers de catalogue et d'ingestion pour DataTalk.

Ce module centralise les chemins de données, les fichiers JSON, les loaders et
les utilitaires communs partagés par les agents et le RAG.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


DATA_DIR = Path(os.getenv("DATATALK_DATA_DIR", Path(__file__).resolve().parents[2] / "data")).resolve()
SQLITE_DB = DATA_DIR / "sqlite" / "data.db"
SQLITE_DOCS = DATA_DIR / "sqlite" / "database_docs.json"
MONGO_DATA = DATA_DIR / "mongodb" / "data.json"
MONGO_DOCS = DATA_DIR / "mongodb" / "database_docs.json"
MAPPINGS_FILE = DATA_DIR / "mappings.json"
BUSINESS_RULES_FILE = DATA_DIR / "business_rules.json"


class DataSourceUnavailable(RuntimeError):
    """Une source de données n'est pas utilisable (aucune base importée, serveur
    MongoDB injoignable...). Ce n'est pas une requête à corriger : les agents
    laissent remonter l'erreur au lieu de relancer le LLM."""


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> Path:
    """Écriture atomique : fichier temporaire puis remplacement, pour ne jamais
    laisser un JSON tronqué si le processus s'arrête en cours d'écriture."""
    _ensure_parent(path)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
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


def available_sources() -> list[str]:
    """Sources réellement utilisables ("sql", "mongo") : données présentes ET
    documentation générée. L'utilisateur peut importer l'une, l'autre ou les deux."""
    sources = []
    if SQLITE_DB.exists() and load_database_docs("sqlite").get("tables"):
        sources.append("sql")
    if (MONGO_DATA.exists() or os.getenv("MONGO_URI")) and load_database_docs("mongodb").get("collections"):
        sources.append("mongo")
    return sources


def load_all_mappings() -> list[dict[str, Any]]:
    """Toutes les correspondances du fichier, y compris celles rejetées."""
    payload = load_json(MAPPINGS_FILE, {"mappings": []})
    if isinstance(payload, dict):
        payload = payload.get("mappings", [])
    return [m for m in payload if isinstance(m, dict)] if isinstance(payload, list) else []


def mapping_is_valid(mapping: dict[str, Any]) -> bool:
    return bool(
        mapping.get("sqlite", {}).get("table")
        and mapping.get("sqlite", {}).get("key")
        and mapping.get("mongodb", {}).get("collection")
        and mapping.get("mongodb", {}).get("key")
    )


def load_mappings() -> list[dict[str, Any]]:
    """Correspondances SQL ↔ MongoDB utilisables par le mode hybride :
    complètes et non rejetées par l'utilisateur."""
    return [
        m for m in load_all_mappings()
        if m.get("status") != "rejected" and mapping_is_valid(m)
    ]


def mapping_context() -> str:
    lines = []
    for m in load_mappings():
        extra = ""
        if m.get("status") == "suggested" and m.get("confidence") is not None:
            extra = f" [détectée automatiquement, confiance {m['confidence']}]"
        lines.append(
            f"- {m.get('entity', 'unknown')}: SQLite {m['sqlite']['table']}({m['sqlite']['key']}) "
            f"-> MongoDB {m['mongodb']['collection']}({m['mongodb']['key']}){extra}"
        )
    return "\n".join(lines) or "Aucune correspondance documentée."


# ---------------------------------------------------------------------------
# Fusion SQL ↔ MongoDB
# ---------------------------------------------------------------------------

def get_path(row: dict[str, Any], path: str) -> Any:
    """Lit une valeur, avec support des chemins pointés (ex. 'customer.id')."""
    if path in row:
        return row[path]
    current: Any = row
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def normalize_key(value: Any) -> str | None:
    """Forme canonique d'une valeur de clé pour comparer SQL et MongoDB :
    1, 1.0 et '1' désignent la même clé ; espaces ignorés ; None/listes/objets
    ne sont pas joignables."""
    if value is None or isinstance(value, (list, dict, set, tuple)):
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def merge_results(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_key: str,
    right_key: str,
    how: str = "inner",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Jointure en mémoire entre résultats SQL (left) et MongoDB (right).

    - 'inner' : seules les lignes présentes des deux côtés (défaut)
    - 'left'  : toutes les lignes SQL, complétées par MongoDB quand elles existent
    Les relations un-à-plusieurs produisent une ligne par couple (aucune
    ligne n'est écrasée). Les champs MongoDB qui entrent en collision avec un
    champ SQL sont préfixés par 'mongo_'.

    Retourne (lignes, statistiques) ; les statistiques permettent d'expliquer
    et de diagnostiquer la jointure (clés sans correspondance, doublons...).
    """
    if how not in ("inner", "left"):
        raise ValueError(f"Type de jointure inconnu : {how!r}")

    index: dict[str, list[dict[str, Any]]] = {}
    right_without_key = 0
    for row in right:
        key = normalize_key(get_path(row, right_key))
        if key is None:
            right_without_key += 1
            continue
        index.setdefault(key, []).append(row)

    merged: list[dict[str, Any]] = []
    matched_left = 0
    left_without_key = 0
    left_keys: set[str] = set()
    for left_row in left:
        key = normalize_key(get_path(left_row, left_key))
        if key is None:
            left_without_key += 1
            continue
        left_keys.add(key)
        matches = index.get(key, [])
        if not matches:
            if how == "left":
                merged.append(dict(left_row))
            continue
        matched_left += 1
        for right_row in matches:
            out = dict(left_row)
            for name, value in right_row.items():
                if name == right_key and right_key == left_key:
                    continue  # même colonne de jointure des deux côtés
                out[f"mongo_{name}" if name in left_row else name] = value
            merged.append(out)

    unmatched_right = sum(len(rows) for key, rows in index.items() if key not in left_keys)
    stats = {
        "how": how,
        "left_rows": len(left),
        "right_rows": len(right),
        "joined_rows": len(merged),
        "matched_left_rows": matched_left,
        "unmatched_left_rows": len(left) - matched_left - left_without_key,
        "unmatched_right_rows": unmatched_right,
        "left_rows_without_key": left_without_key,
        "right_rows_without_key": right_without_key,
        "one_to_many": any(len(rows) > 1 for rows in index.values()),
    }
    return merged, stats


def save_business_rules(content: bytes, filename: str = "business_rules.txt") -> Path:
    """Store optional business rules in a RAG-friendly canonical JSON format."""
    suffix = Path(filename).suffix.lower()
    try:
        text = content.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Les règles métier doivent être un fichier texte UTF-8.") from exc
    if not text:
        raise ValueError("Le fichier de règles métier est vide.")
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON de règles métier invalide : {exc}") from exc
        if isinstance(payload, dict):
            rules = payload.get("rules", [payload])
        elif isinstance(payload, list):
            rules = payload
        else:
            raise ValueError("Le JSON de règles métier doit contenir un objet ou une liste.")
    else:
        rules = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not rules:
        raise ValueError("Aucune règle métier trouvée dans le fichier.")
    _write_json(BUSINESS_RULES_FILE, {"rules": rules})
    return BUSINESS_RULES_FILE


__all__ = [
    "DataSourceUnavailable",
    "DATA_DIR",
    "SQLITE_DB",
    "SQLITE_DOCS",
    "MONGO_DATA",
    "MONGO_DOCS",
    "MAPPINGS_FILE",
    "BUSINESS_RULES_FILE",
    "_write_json",
    "load_json",
    "load_database_docs",
    "available_sources",
    "load_all_mappings",
    "load_mappings",
    "mapping_is_valid",
    "mapping_context",
    "get_path",
    "normalize_key",
    "merge_results",
    "save_business_rules",
]