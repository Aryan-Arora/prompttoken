from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import tokenizers

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
