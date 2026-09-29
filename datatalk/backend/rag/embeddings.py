"""Encodeur d'embeddings pour le RAG.

Isolé de retriever.py pour que le choix du modèle (et son remplacement
éventuel, ex. un modèle hébergé plutôt que local) ne touche qu'un seul
fichier.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

DEFAULT_MODEL_NAME = os.getenv("DATATALK_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_model: Any = None
_model_name: str | None = None


def get_embedder(model_name: str | None = None) -> Any:
    """Charge (une seule fois) le modèle d'embeddings."""
    global _model, _model_name
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    name = model_name or DEFAULT_MODEL_NAME
    if _model is None or _model_name != name:
        _model = SentenceTransformer(name)
        _model_name = name
    return _model


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Encode une liste de textes en vecteurs normalisés (pour cosinus = dot)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_embedder(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype(np.float32)


def embed_query(text: str, model_name: str | None = None) -> np.ndarray:
    return embed_texts([text], model_name)[0]


def reset_embedder() -> None:
    """Force le rechargement du modèle au prochain appel (tests, changement de modèle)."""
    global _model, _model_name
    _model = None
    _model_name = None