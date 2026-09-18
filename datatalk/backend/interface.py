"""FastAPI interface layer for DataTalk.

Keeps upload/configuration concerns separate from the LangGraph implementation.
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


@app.post("/upload/mongodb-data")
async def upload_mongodb_data(file: UploadFile = File(...)) -> dict[str, Any]:
    """Import a JSON MongoDB export into the configured MongoDB database.

    Accepted shape: {"collection_name": [{...}, {...}], ...}.
    This keeps MongoDB server-side while allowing the web UI to load test data.
    """
    try:
        from pymongo import MongoClient

        filename = file.filename or "upload.json"
        if not filename.lower().endswith(".json"):
            raise ValueError("MongoDB data upload must be a .json file")
        payload = json.loads((await file.read()).decode("utf-8"))
        if not isinstance(payload, dict) or not payload:
            raise ValueError("MongoDB upload must be a non-empty JSON object mapping collection names to arrays")

        client = MongoClient(
            os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            serverSelectionTimeoutMS=3000,
        )
        db = client[os.getenv("MONGO_DB", "datatalk")]
        imported = 0
        collections = 0
        try:
            for collection_name, documents in payload.items():
                if not isinstance(collection_name, str) or not collection_name.strip():
                    raise ValueError("MongoDB collection names must be non-empty strings")
                if not isinstance(documents, list):
                    raise ValueError(f"Collection '{collection_name}' must contain an array")
                if any(not isinstance(document, dict) for document in documents):
                    raise ValueError(f"Collection '{collection_name}' must contain JSON objects")

                db[collection_name].delete_many({})
                if documents:
                    db[collection_name].insert_many(documents)
                imported += len(documents)
                collections += 1
        finally:
            client.close()

        return {
            "ok": True,
            "target": "mongodb-data",
            "collections_imported": collections,
            "documents_imported": imported,
            "workspace": workspace_status(),
        }
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MongoDB import failed: {exc}") from exc


@app.post("/upload/{target}")
async def upload(target: str, file: UploadFile = File(...)) -> dict[str, Any]:
    allowed = {"sqlite-db", "sqlite-docs", "mongodb-docs", "mappings"}
    if target not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported upload target")
    try:
        content = await file.read()
        destination = save_upload(file.filename or "upload", content, target)
        return {
            "ok": True,
            "target": target,
            "filename": destination.name,
            "workspace": workspace_status(),
        }
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
