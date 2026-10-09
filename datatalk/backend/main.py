"""FastAPI entrypoint for DataTalk.

Flux utilisateur :
  1. POST /upload/sqlite-db, /upload/mongodb-data (et /upload/business-rules, optionnel)
     -> validation, installation, génération automatique de la documentation
        et des correspondances SQL ↔ MongoDB
  2. GET /docs, GET/PUT /mappings : consulter la documentation générée, confirmer
     ou corriger les correspondances
  3. POST /query : poser une question
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .graph import run
from .llm import LLMError
from .tools.catalog_tools import (
    BUSINESS_RULES_FILE,
    DATA_DIR,
    MAPPINGS_FILE,
    MONGO_DATA,
    MONGO_DOCS,
    SQLITE_DB,
    SQLITE_DOCS,
    DataSourceUnavailable,
    available_sources,
    load_all_mappings,
    load_database_docs,
    save_business_rules,
)
from .tools.ingest_tools import (
    analyze_workspace,
    ingest_mongo_payload,
    ingest_sqlite_file,
    save_user_mappings,
)
from .tools.rag_tools import rag_document_count, reset_rag

logger = logging.getLogger("datatalk")

MAX_RESULT_ROWS = 200
MAX_SQLITE_BYTES = int(os.getenv("DATATALK_MAX_SQLITE_MB", "200")) * 1024 * 1024
MAX_JSON_BYTES = int(os.getenv("DATATALK_MAX_JSON_MB", "100")) * 1024 * 1024
MAX_RULES_BYTES = 2 * 1024 * 1024

app = FastAPI(title="DataTalk API", version="0.3.0")

_cors = os.getenv("DATATALK_CORS_ORIGINS") or os.getenv("FRONTEND_ORIGIN") or "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _cors.split(",") if origin.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Un seul import/analyse à la fois : évite que deux uploads simultanés
# s'écrasent ou laissent une documentation incohérente.
_ingest_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Gestion d'erreurs : message clair, code HTTP adapté, jamais de trace brute
# ---------------------------------------------------------------------------

@app.exception_handler(LLMError)
async def _llm_error(_request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": f"Service LLM indisponible : {exc}"})


@app.exception_handler(DataSourceUnavailable)
async def _source_error(_request, exc: DataSourceUnavailable) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unexpected_error(_request, exc: Exception) -> JSONResponse:
    error_id = uuid.uuid4().hex[:8]
    logger.exception("Erreur interne [%s]", error_id)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Erreur interne (référence {error_id}). Consultez les journaux du serveur."},
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    question: str
    source: str | None = None
    status: str = "ok"            # ok | no_data | error
    error: str | None = None
    answer: str | None = None
    data: list[dict[str, Any]] = []
    execution: dict[str, Any] = {}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _safe_rag_count() -> int | None:
    try:
        return rag_document_count()
    except Exception:
        return None


def _workspace() -> dict[str, Any]:
    return {
        "data_dir": str(DATA_DIR),
        "sqlite_database": SQLITE_DB.exists(),
        "sqlite_docs": SQLITE_DOCS.exists(),
        "mongodb_data": MONGO_DATA.exists(),
        "mongodb_docs": MONGO_DOCS.exists(),
        "mongodb_engine": "server" if os.getenv("MONGO_URI") else "embedded",
        "available_sources": available_sources(),
        "mappings": MAPPINGS_FILE.exists(),
        "business_rules": BUSINESS_RULES_FILE.exists(),
        "claude_configured": bool(os.getenv("CLAUDE_API_KEY")),
        "claude_model": os.getenv("CLAUDE_MODEL"),
        "rag_documents": _safe_rag_count(),
    }


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Lit un upload par blocs en refusant de dépasser `limit` octets."""
    chunks, size = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=f"Fichier trop volumineux (maximum {limit // (1024 * 1024)} Mo).")
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/workspace")
@app.get("/status")
def workspace() -> dict[str, Any]:
    return _workspace()


@app.post("/analyze")
def analyze() -> dict[str, Any]:
    """Régénère la documentation des schémas et les correspondances à partir
    des données actuellement importées."""
    with _ingest_lock:
        summary = analyze_workspace()
    return {"summary": summary, "workspace": _workspace()}


@app.get("/docs/schemas")
def schemas() -> dict[str, Any]:
    """Documentation générée des deux bases (/docs est réservé à Swagger)."""
    return {"sqlite": load_database_docs("sqlite"), "mongodb": load_database_docs("mongodb")}


@app.get("/mappings")
def get_mappings() -> dict[str, Any]:
    return {"mappings": load_all_mappings()}


@app.put("/mappings")
def put_mappings(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Remplace les correspondances (confirmer, corriger, rejeter une suggestion)."""
    try:
        with _ingest_lock:
            mappings = save_user_mappings(body.get("mappings"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"mappings": mappings, "workspace": _workspace()}


@app.post("/query", response_model=QueryResponse)
@app.post("/ask", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="La question est vide.")
    try:
        state = run(question)
    except ValueError as exc:  # question vide / trop longue
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # LLMError, DataSourceUnavailable et erreurs inattendues : gestionnaires globaux.

    results = (
        state.get("merged_results")
        or state.get("sql_results")
        or state.get("mongo_results")
        or []
    )
    execution = {
        "classifier_reason": state.get("classifier_reason"),
        "sql_query": state.get("sql_query"),
        "sql_error": state.get("sql_error"),
        "mongo_collection": state.get("mongo_collection"),
        "mongo_pipeline": state.get("mongo_pipeline"),
        "mongo_error": state.get("mongo_error"),
        "join_plan": state.get("join_plan"),
        "join_error": state.get("join_error"),
        "join_stats": state.get("join_stats"),
        "attempts": {
            "sql": state.get("sql_attempts"),
            "mongo": state.get("mongo_attempts"),
        },
    }
    return QueryResponse(
        question=question,
        source=state.get("source"),
        status=state.get("status", "ok"),
        error=state.get("error"),
        answer=state.get("answer"),
        data=_json_safe(results[:MAX_RESULT_ROWS]),
        execution=_json_safe(execution),
    )


@app.post("/upload/sqlite-db")
async def upload_sqlite_db(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in {".db", ".sqlite", ".sqlite3"}:
        raise HTTPException(status_code=400, detail="Format SQLite non supporté (.db, .sqlite ou .sqlite3).")
    content = await _read_limited(file, MAX_SQLITE_BYTES)

    def _install() -> dict[str, Any]:
        SQLITE_DB.parent.mkdir(parents=True, exist_ok=True)
        staging = SQLITE_DB.with_name(f"upload-{uuid.uuid4().hex}.tmp")
        staging.write_bytes(content)
        try:
            with _ingest_lock:
                return ingest_sqlite_file(staging)   # valide, installe, documente
        finally:
            staging.unlink(missing_ok=True)

    try:
        summary = await run_in_threadpool(_install)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Base SQLite importée et analysée.", "summary": summary, "workspace": _workspace()}


@app.post("/upload/mongodb-data")
async def upload_mongodb_data(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await _read_limited(file, MAX_JSON_BYTES)
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="JSON MongoDB invalide (UTF-8 attendu).") from exc

    def _install() -> dict[str, Any]:
        with _ingest_lock:
            return ingest_mongo_payload(payload)     # valide, installe, documente

    try:
        summary = await run_in_threadpool(_install)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Données MongoDB importées et analysées.", "summary": summary, "workspace": _workspace()}


@app.post("/upload/business-rules")
async def upload_business_rules(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await _read_limited(file, MAX_RULES_BYTES)
    try:
        save_business_rules(content, file.filename or "business_rules.txt")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    reset_rag()
    return {"message": "Règles métier importées.", "workspace": _workspace()}


@app.post("/claude/test")
def claude_test() -> dict[str, Any]:
    from .llm import claude_generate
    return {"ok": True, "response": claude_generate("Réponds uniquement par OK.", max_tokens=20)}