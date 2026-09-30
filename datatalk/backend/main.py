"""FastAPI entrypoint for DataTalk."""
from __future__ import annotations

import json
import os
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .graph import run
from .tools.catalog_tools import (
    BUSINESS_RULES_FILE,
    DATA_DIR,
    MAPPINGS_FILE,
    MONGO_DOCS,
    SQLITE_DB,
    SQLITE_DOCS,
    save_business_rules,
)
from .tools.rag_tools import rag_document_count, reset_rag

MAX_RESULT_ROWS = 200
app = FastAPI(title="DataTalk API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DATATALK_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

class QueryResponse(BaseModel):
    question: str
    source: str | None = None
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
        "mongodb_docs": MONGO_DOCS.exists(),
        "mappings": MAPPINGS_FILE.exists(),
        "business_rules": BUSINESS_RULES_FILE.exists(),
        "claude_configured": bool(os.getenv("CLAUDE_API_KEY")),
        "claude_model": os.getenv("CLAUDE_MODEL"),
        "rag_documents": _safe_rag_count(),
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/workspace")
@app.get("/status")
def workspace() -> dict[str, Any]:
    return _workspace()

@app.post("/analyze")
def analyze() -> dict[str, Any]:
    reset_rag()
    return {"workspace": _workspace()}

@app.post("/query", response_model=QueryResponse)
@app.post("/ask", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="La question est vide.")
    try:
        state = run(question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {exc}") from exc

    results = (
        state.get("merged_results")
        or state.get("sql_results")
        or state.get("mongo_results")
        or []
    )
    execution = {
        "sql_query": state.get("sql_query"),
        "sql_error": state.get("sql_error"),
        "mongo_collection": state.get("mongo_collection"),
        "mongo_pipeline": state.get("mongo_pipeline"),
        "mongo_error": state.get("mongo_error"),
        "join_plan": state.get("join_plan"),
        "attempts": {
            "sql": state.get("sql_attempts"),
            "mongo": state.get("mongo_attempts"),
        },
    }
    return QueryResponse(
        question=question,
        source=state.get("source"),
        answer=state.get("answer"),
        data=_json_safe(results[:MAX_RESULT_ROWS]),
        execution=_json_safe(execution),
    )

@app.post("/upload/sqlite-db")
async def upload_sqlite_db(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in {".db", ".sqlite", ".sqlite3"}:
        raise HTTPException(status_code=400, detail="Format SQLite non supporté.")
    target = SQLITE_DB
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(await file.read())
    return {"message": "Base SQLite importée.", "workspace": _workspace()}

@app.post("/upload/mongodb-data")
async def upload_mongodb_data(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        payload = json.loads((await file.read()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="JSON MongoDB invalide.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Le JSON MongoDB doit être un objet collection -> documents.",
        )
    target = DATA_DIR / "mongodb" / "data.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"message": "Données MongoDB importées.", "workspace": _workspace()}

@app.post("/upload/business-rules")
async def upload_business_rules(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        save_business_rules(await file.read(), file.filename or "business_rules.txt")
        reset_rag()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Règles métier importées.", "workspace": _workspace()}

@app.post("/claude/test")
def claude_test() -> dict[str, Any]:
    from .llm import claude_generate
    try:
        return {"ok": True, "response": claude_generate("Réponds uniquement par OK.", max_tokens=20)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
