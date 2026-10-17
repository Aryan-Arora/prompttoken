"""Layer 2: model-based pruning.

`lm` method: score every word by its information content (sum of token
surprisal under a small causal LM, in the spirit of LLMLingua) and drop the
least informative words until the level's target is reached. Protected spans
are never touched. With keep_risky=True, words that carry numbers, names,
negations, or constraint words are never dropped either.

`query` method: for RAG-style context trimming, keep the sentences that share
the most (IDF-weighted) vocabulary with a query and drop the rest.

Pure logic takes a `scorer` so it can be tested without a model.
"""
import bisect
import math
import re
from functools import lru_cache

from . import compressor, risk

LM_MODEL = "distilgpt2"
LEVELS = {"light": 0.2, "medium": 0.4, "aggressive": 0.6}
CHUNK = 512
_WORD = re.compile(r"\S+")
_STOP = set(
    "a an the of in on at to for and or is are was were be been by with as from that this "
    "it its which who whom what when where why how did does do has have had not".split()
)


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LM_MODEL).eval()
    return tok, model


def token_surprisals(text):
    """[(start, end, surprisal_nats)] per LM token. Long text is scored in chunks."""
    import torch

    tok, model = _load()
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    out = []
    for lo in range(0, len(ids), CHUNK):
        chunk = ids[lo:lo + CHUNK]
        inp = torch.tensor([[tok.bos_token_id] + chunk])
        with torch.no_grad():
            logits = model(inp).logits[0, :-1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        sur = -logp[torch.arange(len(chunk)), torch.tensor(chunk)]
        for j, value in enumerate(sur.tolist()):
            out.append((offsets[lo + j][0], offsets[lo + j][1], value))
    return out


def words_with_scores(text, scorer=token_surprisals):
    words = [
        {"start": m.start(), "end": m.end(), "text": m.group(), "score": 0.0}
        for m in _WORD.finditer(text)
    ]
    starts = [w["start"] for w in words]
    for s, e, surprisal in scorer(text):
        seg = text[s:e]
        first = s + (len(seg) - len(seg.lstrip()))
        i = bisect.bisect_right(starts, first) - 1
        if 0 <= i < len(words) and first < words[i]["end"]:
            words[i]["score"] += surprisal
    return words


def _overlaps(a, b, spans):
    return any(a < e and b > s for s, e in spans)


def _risky_word_mask(text, words):
    keys = risk.risky_tokens(text)
    lowered = {k.lower() for k in keys}
    for k in list(lowered):
        if " " in k:
            lowered.update(k.split())
    mask = []
    for w in words:
        subs = [t.rstrip(".'") for t in risk._WORD.findall(w["text"])]
        mask.append(any(t in keys or t.lower() in lowered for t in subs))
    return mask


def _protected_result(text, spans, compressed):
    result = compressor.protected_stats(text, spans, compressed)
    segments = compressor._diff_segments(text, compressed)
    removed = [s["text"].strip() for s in segments if s["kind"] == "removed" and s["text"].strip()]
    return {"compressed": compressed, "segments": segments, "removed": removed, "protected": result}


def prune(text, level="medium", protected_spans=(), keep_risky=True, scorer=token_surprisals):
    ratio = LEVELS[level]
    spans = compressor._merge_spans([(s["start"], s["end"]) for s in protected_spans], len(text))
    words = words_with_scores(text, scorer)

    locked = [_overlaps(w["start"], w["end"], spans) for w in words]
    if keep_risky:
        locked = [a or b for a, b in zip(locked, _risky_word_mask(text, words))]

    droppable = [i for i, is_locked in enumerate(locked) if not is_locked]
    n_drop = min(round(ratio * len(words)), len(droppable))
    dropped = set(sorted(droppable, key=lambda i: (words[i]["score"], i))[:n_drop])

    out, prev_end, prev_kept = "", 0, True
    for i, w in enumerate(words):
        sep = text[prev_end:w["start"]]
        if i not in dropped:
            if not out and not _overlaps(prev_end, w["start"], spans):
                sep = sep if i == 0 else ""
            elif not prev_kept and not _overlaps(prev_end, w["start"], spans):
                sep = "\n" if "\n" in sep else " "
            out += sep + w["text"]
        prev_kept = i not in dropped
        prev_end = w["end"]
    if prev_kept or _overlaps(prev_end, len(text), spans):
        out += text[prev_end:]
    return _protected_result(text, spans, out)


def _terms(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


def select_sentences(text, query, level="medium", protected_spans=()):
    ratio = LEVELS[level]
    spans = compressor._merge_spans([(s["start"], s["end"]) for s in protected_spans], len(text))
    units = [(m.start(), m.end(), m.group()) for m in compressor._SENTENCE.finditer(text) if m.group().strip()]
    if len(units) <= 1:
        return _protected_result(text, spans, text)

    sets = [set(_terms(u[2])) for u in units]
    n = len(units)
    df = {}
    for s in sets:
        for t in s:
            df[t] = df.get(t, 0) + 1
    q_terms = set(_terms(query))
    scores = [
        sum(math.log((n + 1) / (df.get(t, 0) + 0.5)) for t in q_terms & s) for s in sets
    ]
    counts = [len(u[2].split()) for u in units]
    total = sum(counts)
    locked = [_overlaps(u[0], u[1], spans) for u in units]

    dropped, dropped_words = set(), 0
    for i in sorted(range(n), key=lambda i: (scores[i], -i)):
        if locked[i] or len(dropped) >= n - 1:
            continue
        if dropped_words + counts[i] > ratio * total:
            continue
        dropped.add(i)
        dropped_words += counts[i]

    kept = "".join(u[2] for i, u in enumerate(units) if i not in dropped)
    kept = re.sub(r"[ \t]+$", "", kept)
    return _protected_result(text, spans, kept)
