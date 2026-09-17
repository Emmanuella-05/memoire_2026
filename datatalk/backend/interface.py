"""FastAPI interface layer for DataTalk.

This module is intentionally separate so it can be imported by the team's main.py
without changing the LangGraph implementation owned by the other branch member.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .tools import claude_generate, save_upload, workspace_status
except ImportError:
    from tools import claude_generate, save_upload, workspace_status

app = FastAPI(title="DataTalk API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

_graph = None

def get_graph() -> Any:
    global _graph
    if _graph is None:
        try:
            from .graph import build_graph
        except ImportError:
            from graph import build_graph
        _graph = build_graph()
    return _graph

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "workspace": workspace_status()}

@app.post("/query")
def query(payload: QueryRequest) -> dict[str, Any]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        state = get_graph().invoke({"question": question})
        return {
            "answer": state.get("final_answer", state.get("answer", "")),
            "data": state.get("merged_result", state.get("sql_result", state.get("mongo_result", []))),
            "execution": {
                "route": state.get("query_type", "UNKNOWN"),
                "sources": state.get("sources_used", []),
                "correspondences": state.get("correspondences_used", state.get("join_plan", [])),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/upload/{target}")
async def upload(target: str, file: UploadFile = File(...)) -> dict[str, Any]:
    allowed = {"sqlite-db", "sqlite-docs", "mongodb-docs", "mappings"}
    if target not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported upload target")
    try:
        content = await file.read()
        destination = save_upload(file.filename or "upload", content, target)
        return {"ok": True, "target": target, "filename": destination.name, "workspace": workspace_status()}
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/claude/test")
def claude_test() -> dict[str, str]:
    try:
        text = claude_generate("Reply with exactly: Claude OK", max_tokens=20)
        return {"status": "ok", "response": text}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
