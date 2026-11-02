"""Serveur LLM pour Google Colab (GPU T4) : héberge SQLCoder et Qwen2.5-Coder
derrière une API compatible OpenAI, exposée par un tunnel ngrok.

Pourquoi : le PC (8 Go de RAM, sans GPU) ne peut pas charger deux modèles de
7 milliards de paramètres. Le backend DataTalk reste sur le PC et appelle ce
serveur via LLM_BASE_URL (voir le README à la racine, étape 7).

Un seul tunnel (le plan gratuit de ngrok n'en autorise qu'un) : un seul serveur
héberge les deux modèles, choisis par le champ "model" de la requête.

Endpoints :
  POST /v1/chat/completions   modèle instruct (Qwen)   -> choices[0].message.content
  POST /v1/completions        prompt brut (SQLCoder)   -> choices[0].text
  GET  /health, GET /v1/models

Variables d'environnement (toutes facultatives) :
  LLM_API_KEY        clé exigée dans « Authorization: Bearer ... » (À DÉFINIR : l'URL ngrok est publique)
  NGROK_AUTHTOKEN    jeton ngrok (sinon, serveur local seulement)
  SQL_MODEL_REPO     dépôt Hugging Face de SQLCoder     (défaut : version 4 bits pré-quantifiée)
  CHAT_MODEL_REPO    dépôt Hugging Face de Qwen-Coder   (défaut : version 4 bits pré-quantifiée)
  HF_TOKEN           jeton Hugging Face (seulement si un dépôt est protégé)
  PORT               port local (défaut 8000)
"""

from __future__ import annotations

import gc
import os
import threading
import time
import traceback
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# alias utilisé par le backend (LLM_MODEL_*)  ->  dépôt Hugging Face + type d'appel
REGISTRY: dict[str, dict[str, str]] = {
    "sqlcoder-7b-2": {
        "repo": os.getenv("SQL_MODEL_REPO", "RichardErkhov/defog_-_sqlcoder-7b-2-4bits"),
        "kind": "raw",
    },
    "qwen2.5-coder-7b": {
        "repo": os.getenv("CHAT_MODEL_REPO", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"),
        "kind": "chat",
    },
}

app = FastAPI(title="DataTalk LLM server")


@app.exception_handler(Exception)
async def _explain_errors(request: Request, exc: Exception) -> JSONResponse:
    """Une erreur du serveur est renvoyée avec son type et son message (le backend DataTalk les affiche),
    et la trace complète est imprimée dans la cellule Colab."""
    traceback.print_exception(exc)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {str(exc)[:400]}"})
_models: dict[str, tuple[Any, Any]] = {}
_load_lock = threading.Lock()
_gpu_lock = threading.Lock()  # une génération à la fois (un seul GPU)


# --- chargement --------------------------------------------------------------

def _is_prequantized(repo: str) -> bool:
    return any(tag in repo.lower() for tag in ("bnb", "4bit", "awq", "gptq"))


def _load(alias: str) -> tuple[Any, Any]:
    with _load_lock:
        if alias in _models:
            return _models[alias]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        repo = REGISTRY[alias]["repo"]
        print(f"[llm_server] chargement de {alias} ({repo}) ...", flush=True)
        started = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo)
        kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": torch.float16}
        if not _is_prequantized(repo):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            )
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        _models[alias] = (tokenizer, model)
        print(f"[llm_server] {alias} prêt en {time.time() - started:.0f} s", flush=True)
        return _models[alias]


def _resolve(alias: str) -> str:
    wanted = (alias or "").strip().lower()
    for name in REGISTRY:
        if wanted == name.lower():
            return name
    raise HTTPException(status_code=404, detail=f"Modèle inconnu : {alias!r}. Disponibles : {', '.join(REGISTRY)}")


def _check_key(authorization: str | None) -> None:
    key = os.getenv("LLM_API_KEY")
    if key and authorization != f"Bearer {key}":
        raise HTTPException(status_code=401, detail="Clé invalide.")


# --- génération --------------------------------------------------------------

MAX_INPUT_TOKENS = int(os.getenv("LLM_MAX_INPUT_TOKENS", "6000"))


def _is_oom(exc: Exception) -> bool:
    return "outofmemory" in type(exc).__name__.lower() or "out of memory" in str(exc).lower()


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # pas de GPU / torch absent : rien à libérer
        pass


def _with_oom_fallback(run, num_beams: int):
    """Exécute run(beams). Si la mémoire du GPU est saturée (la recherche par faisceaux multiplie le cache par
    le nombre de faisceaux), libère le cache et réessaie avec moitié moins de faisceaux, jusqu'à 1."""
    beams = max(1, num_beams)
    while True:
        try:
            return run(beams)
        except Exception as exc:
            if not _is_oom(exc) or beams <= 1:
                raise
            _free_gpu()
            beams = beams // 2 if beams > 2 else 1
            print(f"[llm_server] mémoire GPU saturée : nouvel essai avec {beams} faisceau(x)", flush=True)


def _generate(alias: str, prompt: str, *, max_tokens: int, temperature: float,
              num_beams: int, stop: list[str] | None) -> tuple[str, int, int]:
    import torch

    tokenizer, model = _load(alias)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_in = int(inputs["input_ids"].shape[1])
    if n_in > MAX_INPUT_TOKENS:
        raise HTTPException(status_code=413, detail=(
            f"Prompt trop long ({n_in} tokens, maximum {MAX_INPUT_TOKENS}) : le schéma ou les règles métier "
            "dépassent ce que le GPU peut traiter. Réduisez-les ou relevez LLM_MAX_INPUT_TOKENS."))
    sampling = temperature > 0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max(1, min(int(max_tokens), 1500)),
        "do_sample": sampling,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if sampling:
        gen_kwargs["temperature"] = temperature
    if stop:
        gen_kwargs["stop_strings"] = stop
        gen_kwargs["tokenizer"] = tokenizer

    def run(beams: int):
        with _gpu_lock, torch.inference_mode():
            return model.generate(**inputs, **gen_kwargs, num_beams=1 if sampling else beams)

    output = _with_oom_fallback(run, 1 if sampling else min(int(num_beams), 4))
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    for marker in stop or []:
        cut = text.find(marker)
        if cut != -1:
            text = text[:cut]
    return text, int(inputs["input_ids"].shape[1]), int(len(new_tokens))


# --- API ---------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 700
    temperature: float = 0.0
    stop: list[str] | str | None = None
    response_format: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 400
    temperature: float = 0.0
    num_beams: int = 1
    stop: list[str] | str | None = None


def _stops(stop: list[str] | str | None) -> list[str] | None:
    if not stop:
        return None
    return [stop] if isinstance(stop, str) else list(stop)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "loaded": sorted(_models), "available": list(REGISTRY)}


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_key(authorization)
    return {"object": "list", "data": [{"id": name, "object": "model"} for name in REGISTRY]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_key(authorization)
    alias = _resolve(req.model)
    tokenizer, _ = _load(alias)
    prompt = tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
    json_prefix = ""
    if (req.response_format or {}).get("type") == "json_object":
        json_prefix = "{"          # amorce la réponse : le modèle continue l'objet JSON
        prompt += json_prefix
    text, n_in, n_out = _generate(
        alias, prompt, max_tokens=req.max_tokens, temperature=req.temperature,
        num_beams=1, stop=_stops(req.stop),
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "model": alias,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": json_prefix + text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_key(authorization)
    alias = _resolve(req.model)
    text, n_in, n_out = _generate(
        alias, req.prompt, max_tokens=req.max_tokens, temperature=req.temperature,
        num_beams=req.num_beams, stop=_stops(req.stop),
    )
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}", "object": "text_completion", "model": alias,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
    }


# --- démarrage (Colab) -------------------------------------------------------

def _preload() -> None:
    for alias in REGISTRY:
        try:
            _load(alias)
        except Exception as exc:  # le serveur reste utilisable : chargement à la 1re requête
            print(f"[llm_server] préchargement de {alias} impossible : {exc}", flush=True)


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    if not os.getenv("LLM_API_KEY"):
        print("[llm_server] ATTENTION : LLM_API_KEY n'est pas définie, l'URL publique sera ouverte à tous.", flush=True)
    token = os.getenv("NGROK_AUTHTOKEN")
    if token:
        from pyngrok import conf, ngrok

        conf.get_default().auth_token = token
        tunnel = ngrok.connect(port)  # sans bind_tls : refusé par certaines versions de l'agent ngrok
        public_url = tunnel.public_url.replace("http://", "https://", 1)  # toujours en https (un POST ne survit pas à une redirection)
        print(f"\n[llm_server] URL publique : {public_url}\n"
              f"[llm_server] à mettre dans le .env du PC :  LLM_BASE_URL={public_url}\n", flush=True)
    threading.Thread(target=_preload, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
