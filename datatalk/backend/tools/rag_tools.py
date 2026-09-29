"""Facade RAG exposée aux agents DataTalk."""

from __future__ import annotations

from pathlib import Path

from backend.rag import retriever
from .catalog_tools import BUSINESS_RULES_FILE, _write_json
import json


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
    _write_json(BUSINESS_RULES_FILE, {"rules": rules})
    retriever.reset_index()
    return BUSINESS_RULES_FILE


def retrieve(question: str, database: str | None = None, k: int = 3) -> list[dict]:
    return retriever.retrieve(question, database, k)


def context(question: str, database: str | None = None, k: int = 3) -> str:
    return retriever.context(question, database, k)


def reset_rag() -> None:
    retriever.reset_index()


def rag_document_count() -> int:
    return len(retriever.get_index().chunks)