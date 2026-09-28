"""Tokenizer registry: wraps tiktoken (GPT-style) and a Hugging Face
tokenizer (Llama-style) behind one interface so /tokenize can compare them
side by side.
"""
from functools import lru_cache

import tiktoken
from transformers import AutoTokenizer

# TinyLlama's tokenizer is Llama-style BPE and, unlike most Llama checkpoints,
# is not gated behind a Hugging Face access request.
LLAMA_STYLE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SUPPORTED = {
    "gpt-4o": {"family": "tiktoken", "encoding": "o200k_base"},
    "gpt-3.5-4": {"family": "tiktoken", "encoding": "cl100k_base"},
    "llama": {"family": "hf", "model": LLAMA_STYLE_MODEL},
}


@lru_cache(maxsize=8)
def _tiktoken_encoding(name: str):
    return tiktoken.get_encoding(name)


@lru_cache(maxsize=4)
def _hf_tokenizer(model: str):
    return AutoTokenizer.from_pretrained(model)


def tokenize(text: str, tokenizer_id: str) -> dict:
    """Return {tokens: [{id, text}], count} for one tokenizer."""
    if tokenizer_id not in SUPPORTED:
        raise ValueError(f"unknown tokenizer '{tokenizer_id}'")

    spec = SUPPORTED[tokenizer_id]

    if spec["family"] == "tiktoken":
        enc = _tiktoken_encoding(spec["encoding"])
        ids = enc.encode(text)
        tokens = [{"id": tid, "text": enc.decode([tid])} for tid in ids]
        return {"tokens": tokens, "count": len(tokens)}

    if spec["family"] == "hf":
        tok = _hf_tokenizer(spec["model"])
        ids = tok.encode(text, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        # Llama-style BPE marks a leading space with '▁'; swap it back in so
        # the highlighter can render something readable.
        tokens = [
            {"id": tid, "text": piece.replace("▁", " ")}
            for tid, piece in zip(ids, pieces)
        ]
        return {"tokens": tokens, "count": len(tokens)}

    raise ValueError(f"unhandled tokenizer family for '{tokenizer_id}'")
