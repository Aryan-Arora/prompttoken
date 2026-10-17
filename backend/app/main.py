import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import compressor, lint, pricing, risk, tokenizers

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


@app.post("/compress")
def compress(req: CompressRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    start = time.perf_counter()
    result = compressor.compress(req.text, [s.model_dump() for s in req.protected_spans])
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
