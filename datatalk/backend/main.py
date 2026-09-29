"""API FastAPI de DataTalk.

Lancement (depuis backend/) :
    uvicorn main:app --reload

Endpoints :
    POST /ask     -> pose une question en langage naturel, exécute le graphe
    GET  /status  -> état du workspace (bases, catalogue, RAG, clé Claude)
    GET  /health  -> simple ping

Les endpoints d'upload (base SQLite, règles métier) viendront avec
ingestion/.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Doit s'exécuter AVANT les imports du projet : catalog_tools lit
# DATATALK_DATA_DIR dès son import. Sans .env (ex. Colab), on ignore.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import graph
from tools.catalog_tools import (
    BUSINESS_RULES_FILE,
    DATA_DIR,
    MAPPINGS_FILE,
    MONGO_DOCS,
    SQLITE_DB,
    SQLITE_DOCS,
)

MAX_RESULT_ROWS = 200

app = FastAPI(title="DataTalk API", version="0.1.0")

# Frontend en développement : à restreindre à l'origine réelle en production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DATATALK_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question en langage naturel")


class AskResponse(BaseModel):
    question: str
    source: str | None = None
    answer: str | None = None
    results: list[dict[str, Any]] = []
    results_truncated: bool = False
    sql_query: str | None = None
    sql_error: str | None = None
    mongo_collection: str | None = None
    mongo_pipeline: list[dict[str, Any]] | None = None
    mongo_error: str | None = None
    join_plan: dict[str, Any] | None = None


def _json_safe(value: Any) -> Any:
    """Les résultats Mongo peuvent contenir ObjectId, datetime, Decimal...
    On convertit tout type non sérialisable en chaîne."""
    return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, Any]:
    info: dict[str, Any] = {
        "data_dir": str(DATA_DIR),
        "sqlite_database": SQLITE_DB.exists(),
        "sqlite_docs": SQLITE_DOCS.exists(),
        "mongodb_docs": MONGO_DOCS.exists(),
        "mappings": MAPPINGS_FILE.exists(),
        "business_rules": BUSINESS_RULES_FILE.exists(),
        "claude_configured": bool(os.getenv("CLAUDE_API_KEY")),
        "claude_model": os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
    }
    # Le comptage RAG charge le modèle d'embeddings : on n'échoue pas /status
    # si sentence-transformers est absent ou si l'index ne peut pas se construire.
    try:
        from tools.rag_tools import rag_document_count
        info["rag_documents"] = rag_document_count()
    except Exception as exc:
        info["rag_documents"] = None
        info["rag_error"] = str(exc)
    return info


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    # `def` (et non `async def`) : graph.run est bloquant (appels LLM + bases),
    # FastAPI l'exécute donc dans un thread sans bloquer la boucle d'événements.
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="La question est vide.")

    try:
        state = graph.run(question)
    except RuntimeError as exc:
        # Configuration manquante (CLAUDE_API_KEY, anthropic, pymongo...)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {exc}") from exc

    merged = state.get("merged_results") or []
    return AskResponse(
        question=question,
        source=state.get("source"),
        answer=state.get("answer"),
        results=_json_safe(merged[:MAX_RESULT_ROWS]),
        results_truncated=len(merged) > MAX_RESULT_ROWS,
        sql_query=state.get("sql_query"),
        sql_error=state.get("sql_error"),
        mongo_collection=state.get("mongo_collection"),
        mongo_pipeline=_json_safe(state.get("mongo_pipeline")),
        mongo_error=state.get("mongo_error"),
        join_plan=state.get("join_plan"),
    )