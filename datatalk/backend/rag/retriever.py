"""Retriever RAG : construit les chunks à partir du catalogue, les indexe et
répond aux requêtes de similarité.

Remplace l'ancien TF-IDF local par des embeddings (rag/embeddings.py),
persistés dans rag/vectorstore/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..tools.catalog_tools import (
    BUSINESS_RULES_FILE,
    load_database_docs,
    load_json,
)
from .embeddings import embed_query, embed_texts

VECTORSTORE_DIR = Path(__file__).resolve().parent / "vectorstore"
INDEX_FILE = VECTORSTORE_DIR / "index.npz"
METADATA_FILE = VECTORSTORE_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    database: str
    name: str
    text: str
    metadata: dict[str, Any]


def _flatten_docs(database: str, payload: dict[str, Any]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    if database == "sqlite":
        for table in payload.get("tables", []):
            columns = table.get("columns", [])
            col_text = " ".join(
                f"{c.get('name', '')} {c.get('type', '')} {c.get('description', '')}"
                for c in columns
            )
            text = f"table {table.get('name', '')} {table.get('description', '')} {col_text}"
            chunks.append(DocumentChunk(database, table.get("name", ""), text, table))
    elif database == "mongodb":
        for collection in payload.get("collections", []):
            fields = collection.get("fields", [])
            field_text = " ".join(
                f"{c.get('name', '')} {c.get('type', '')} {c.get('description', '')}"
                for c in fields
            )
            text = f"collection {collection.get('name', '')} {collection.get('description', '')} {field_text}"
            chunks.append(DocumentChunk(database, collection.get("name", ""), text, collection))
    return chunks


def build_chunks() -> list[DocumentChunk]:
    """Documentation SQLite + MongoDB + règles métier, à ré-indexer."""
    chunks = (
        _flatten_docs("sqlite", load_database_docs("sqlite"))
        + _flatten_docs("mongodb", load_database_docs("mongodb"))
    )
    rules = load_json(BUSINESS_RULES_FILE).get("rules", [])
    for index, rule in enumerate(rules):
        text = rule if isinstance(rule, str) else json.dumps(rule, ensure_ascii=False)
        chunks.append(DocumentChunk("business", f"rule_{index + 1}", text, {"rule": rule}))
    return chunks


# ---------------------------------------------------------------------------
# Index vectoriel (persisté sur disque, numpy simple — pas de dépendance
# FAISS/Chroma pour rester léger ; à remplacer ici si le volume grandit)
# ---------------------------------------------------------------------------

class VectorIndex:
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def build(self, chunks: list[DocumentChunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else build_chunks()
        self.vectors = embed_texts([c.text for c in self.chunks]) if self.chunks else np.zeros((0, 0), dtype=np.float32)
        self._save()

    def _save(self) -> None:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(INDEX_FILE, vectors=self.vectors)
        METADATA_FILE.write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> bool:
        if not INDEX_FILE.exists() or not METADATA_FILE.exists():
            return False
        data = np.load(INDEX_FILE)
        self.vectors = data["vectors"]
        raw = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        self.chunks = [DocumentChunk(**item) for item in raw]
        return True

    def search(self, query: str, database: str | None = None, k: int = 3) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        candidate_indices = [
            i for i, c in enumerate(self.chunks)
            if database is None or c.database == database.lower()
        ]
        if not candidate_indices:
            return []
        query_vector = embed_query(query)
        candidate_vectors = self.vectors[candidate_indices]
        scores = candidate_vectors @ query_vector  # vecteurs normalisés -> cosinus
        ranked = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)[:k]
        return [
            {
                "database": self.chunks[i].database,
                "name": self.chunks[i].name,
                "score": float(score),
                "metadata": self.chunks[i].metadata,
                "text": self.chunks[i].text,
            }
            for i, score in ranked if score > 0
        ]


_index: VectorIndex | None = None


def get_index(force_rebuild: bool = False) -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex()
        if force_rebuild or not _index.load():
            _index.build()
    return _index


def reset_index() -> None:
    """Invalide l'index en mémoire ; il sera reconstruit et resauvegardé
    au prochain appel de get_index()."""
    global _index
    _index = None


def retrieve(question: str, database: str | None = None, k: int = 3) -> list[dict[str, Any]]:
    return get_index().search(question, database, k)


def context(question: str, database: str | None = None, k: int = 3) -> str:
    docs = retrieve(question, database, k)
    return "\n\n".join(f"[{d['database']}.{d['name']}] {d['text']}" for d in docs)