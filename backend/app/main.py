import json
import re
import threading
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import benchmark, compressor, lint, lm_pruner, pricing, risk, security, tokenizers

app = FastAPI(title="PromptToken Analyzer & Compressor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TokenizeRequest(BaseModel):
    text: str
    tokenizers: list[str]


class Span(BaseModel):
    start: int
    end: int


class CompressRequest(BaseModel):
    text: str
    protected_spans: list[Span] = []
    tokenizers: list[str] = ["gpt-4o", "llama"]
    method: Literal["rules", "lm", "query"] = "rules"
    level: Literal["light", "medium", "aggressive"] = "medium"
    keep_risky: bool = True
    query: str = ""


@app.post("/compress")
def compress(req: CompressRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    spans = [s.model_dump() for s in req.protected_spans]
    start = time.perf_counter()
    try:
        if req.method == "lm":
            result = lm_pruner.prune(req.text, req.level, spans, req.keep_risky)
        elif req.method == "query":
            if not req.query.strip():
                raise HTTPException(status_code=400, detail="query is required for method 'query'")
            result = lm_pruner.select_sentences(req.text, req.query, req.level, spans)
        else:
            result = compressor.compress(req.text, spans)
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail="Model-based compression needs: pip install -r requirements-ml.txt"
        ) from exc
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    result["lost"] = risk.find_lost(result["compressed"], result["segments"])

    counts = {}
    for tokenizer_id in req.tokenizers:
        try:
            before = tokenizers.tokenize(req.text, tokenizer_id)["count"]
            after = tokenizers.tokenize(result["compressed"], tokenizer_id)["count"]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        saved = before - after
        counts[tokenizer_id] = {
            "before": before,
            "after": after,
            "saved": saved,
            "saved_pct": round(100 * saved / before, 1) if before else 0.0,
            "cost_saved_per_1k_calls_usd": round(pricing.cost_saved_per_1k_calls(saved), 4),
        }

    return {
        **result,
        "tokens": counts,
        "latency_ms": latency_ms,
        "price_note": pricing.PRICE_NOTE,
        "usd_per_mtok": pricing.INPUT_USD_PER_MTOK,
    }


class LintRequest(BaseModel):
    text: str


@app.post("/lint")
def lint_prompt(req: LintRequest):
    return lint.lint(req.text)


@app.get("/tokenizers")
def list_tokenizers():
    return {"tokenizers": list(tokenizers.SUPPORTED.keys())}


@app.post("/tokenize")
def tokenize(req: TokenizeRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if not req.tokenizers:
        raise HTTPException(status_code=400, detail="tokenizers must not be empty")

    results = {}
    for tokenizer_id in req.tokenizers:
        try:
            results[tokenizer_id] = tokenizers.tokenize(req.text, tokenizer_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"results": results}


# ---------- benchmark results and runs ----------

ALLOWED_MODELS = {
    "distilbert-base-cased-distilled-squad",
    "deepset/roberta-base-squad2",
    "Qwen/Qwen2.5-0.5B-Instruct",
}
_RESULT_NAME = re.compile(r"^[A-Za-z0-9_.\-]+\.json$")
JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


@app.get("/benchmark/results")
def list_results():
    items = []
    for path in sorted(benchmark.RESULTS_DIR.glob("*.json"), reverse=True):
        meta = json.loads(path.read_text()).get("meta", {})
        kind = "security" if path.name.startswith("security_") else "benchmark"
        items.append({"name": path.name, "kind": kind, "meta": meta})
    return {"results": items}


@app.get("/benchmark/results/{name}")
def get_result(name: str):
    path = benchmark.RESULTS_DIR / name
    if not _RESULT_NAME.match(name) or not path.is_file():
        raise HTTPException(status_code=404, detail="result not found")
    return json.loads(path.read_text())


class RunRequest(BaseModel):
    task: Literal["squad", "gsm8k", "security"] = "squad"
    n: int = Field(50, ge=5, le=500)
    seed: int = 0
    model: str | None = None


def _run_job(job_id, req):
    def progress(done, total, stage):
        JOBS[job_id].update(done=done, total=total, stage=stage)

    try:
        if req.task == "security":
            _, path = security.run(req.n, req.seed, progress=progress)
        else:
            _, path = benchmark.run(req.task, req.n, req.seed, model=req.model, progress=progress)
        JOBS[job_id].update(status="done", result_name=path.name)
    except Exception as exc:  # surfaced to the UI instead of dying silently in the thread
        JOBS[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")


@app.post("/benchmark")
def start_run(req: RunRequest):
    if req.model is not None and req.model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"model must be one of {sorted(ALLOWED_MODELS)}")
    with _JOBS_LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            raise HTTPException(status_code=409, detail="a run is already in progress")
        job_id = uuid.uuid4().hex[:8]
        JOBS[job_id] = {"status": "running", "stage": "starting", "done": 0, "total": req.n,
                        "task": req.task, "n": req.n, "result_name": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"id": job_id}


@app.get("/benchmark/runs/{job_id}")
def get_run(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="run not found")
    return JOBS[job_id]
