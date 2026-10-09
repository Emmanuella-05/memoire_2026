"""Facade RAG exposée aux agents DataTalk."""

from __future__ import annotations

from ..rag import retriever


def retrieve(question: str, database: str | None = None, k: int = 3) -> list[dict]:
    return retriever.retrieve(question, database, k)


def context(question: str, database: str | None = None, k: int = 3) -> str:
    return retriever.context(question, database, k)


def reset_rag() -> None:
    retriever.reset_index()


def rag_document_count() -> int:
    return len(retriever.get_index().chunks)