"""Description du schéma pour les prompts des agents (SQL, MongoDB, planificateur).

Chaque colonne / champ est décrit avec son type ET, quand c'est utile, ses
valeurs EXACTES : l'énumération complète pour une colonne à peu de valeurs
(« produit_endommage », sans accent, avec tiret bas), sinon quelques exemples.
Sans cela, un modèle écrit « produit endommagé » et le filtre ne trouve rien.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .catalog_tools import load_database_docs

# Colonnes dont on n'envoie jamais d'exemples au LLM (données personnelles / secrets).
_PRIVATE_TOKENS = {"email", "mail", "phone", "tel", "telephone", "password", "passwd", "motdepasse",
                   "token", "secret", "iban", "card", "carte", "cvv"}
_EXAMPLE_TYPES = {"string", "text", "date", "datetime", "varchar", "char", "unknown", "mixed"}


def _is_private(name: str) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
    return bool(tokens & _PRIVATE_TOKENS)


def _fmt(values: list[Any]) -> str:
    return ", ".join(json.dumps(v, ensure_ascii=False) for v in values)


def describe_field(name: str, type_: str, sample_values: list[Any] | None = None,
                   distinct_values: list[Any] | None = None, lang: str = "fr") -> str:
    base = f"{name}:{type_ or 'unknown'}"
    if _is_private(name):
        return base
    if distinct_values:
        label = "possible values" if lang == "en" else "valeurs possibles"
        return f"{base} ({label}: {' | '.join(json.dumps(v, ensure_ascii=False) for v in distinct_values)})"
    if sample_values and (type_ or "unknown").lower().split("(")[0] in _EXAMPLE_TYPES:
        return f"{base} ({'e.g.' if lang == 'en' else 'ex :'} {_fmt(sample_values[:3])})"
    return base


def sql_schema_summary(lang: str = "fr") -> str:
    docs = load_database_docs("sqlite")
    lines = []
    for table in docs.get("tables", []):
        columns = ", ".join(
            describe_field(c.get("name", ""), c.get("type", "unknown"), c.get("sample_values"), c.get("distinct_values"), lang)
            for c in table.get("columns", [])
        )
        lines.append(f"- {table.get('name', '')}({columns})")
    return "\n".join(lines) if lines else ("No table documented." if lang == "en" else "Aucune table documentée.")


def mongo_schema_summary(lang: str = "fr") -> str:
    docs = load_database_docs("mongodb")
    lines = []
    for collection in docs.get("collections", []):
        fields = ", ".join(
            describe_field(f.get("name", ""), f.get("type", "unknown"), f.get("sample_values"), f.get("distinct_values"), lang)
            for f in collection.get("fields", [])
        )
        lines.append(f"- {collection.get('name', '')}({fields})")
    return "\n".join(lines) if lines else ("No collection documented." if lang == "en" else "Aucune collection documentée.")
