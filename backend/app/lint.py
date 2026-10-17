"""Prompt linter: flag patterns that inflate token counts.

Token counts use the GPT-4o tokenizer (o200k_base); other tokenizers differ,
so treat `tokens` and `wasted` as indicative, not exact. `wasted` is measured
in context (a window around the match), because tokenizers merge neighbouring
characters: "a   b" costs more than "a b" even though "   " alone is one token.
"""
import re
import unicodedata

from . import compressor, tokenizers

MAX_FLAGS = 100
WINDOW = 24
_WS = re.compile(r"[ \t]{2,}|[ \t]+(?=\n)|\n{3,}")
_LONG_NUMBER = re.compile(r"\d{7,}")
_LONG_STRING = re.compile(r"[A-Za-z0-9+/=_\-]{32,}")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]+")


def _count(s):
    return len(tokenizers._tiktoken_encoding("o200k_base").encode(s))


def _flag(kind, full, start, end, message, replacement=None):
    text = full[start:end]
    wasted = None
    if replacement is not None:
        left = full[max(0, start - WINDOW):start]
        right = full[end:end + WINDOW]
        wasted = _count(left + text + right) - _count(left + replacement + right)
    return {
        "kind": kind, "start": start, "end": end, "text": text,
        "message": message, "tokens": _count(text), "wasted": wasted,
    }


def _is_non_latin(ch):
    return ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN")


def _non_latin_runs(text):
    start = None
    for i, ch in enumerate(text):
        if _is_non_latin(ch):
            if start is None:
                start = i
        elif start is not None:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if ch == " " and nxt and _is_non_latin(nxt):
                continue
            yield start, i
            start = None
    if start is not None:
        yield start, len(text)


def lint(text):
    flags = []

    for m in _WS.finditer(text):
        run = m.group()
        if run.startswith("\n"):
            repl = "\n\n"
        elif text[m.end():m.end() + 1] in ("\n", ""):
            repl = ""
        else:
            repl = " "
        f = _flag("whitespace", text, m.start(), m.end(), "Extra whitespace can cost extra tokens and adds nothing.", repl)
        if f["wasted"] and f["wasted"] > 0:
            flags.append(f)

    for m in _LONG_NUMBER.finditer(text):
        flags.append(_flag("long_number", text, m.start(), m.end(), "Long numbers split into many tokens. Shorten or drop digits the model does not need."))

    for m in _LONG_STRING.finditer(text):
        flags.append(_flag("long_string", text, m.start(), m.end(), "Hashes, keys, and encoded blobs tokenize poorly. Pass them by reference if you can."))

    for m in _EMOJI.finditer(text):
        flags.append(_flag("emoji", text, m.start(), m.end(), "Emoji can cost several tokens each."))

    for s, e in _non_latin_runs(text):
        f = _flag("non_latin", text, s, e, "")
        f["message"] = f"Non-Latin text: {f['tokens']} tokens for {e - s} characters. Some tokenizers split it into many pieces."
        flags.append(f)

    for pattern, repl in compressor._FILLER:
        for m in pattern.finditer(text):
            if m.group().strip():
                flags.append(_flag("filler", text, m.start(), m.end(), "Filler phrase. The rule-based compressor removes it.", repl))

    seen = []
    for m in compressor._SENTENCE.finditer(text):
        if not m.group().strip():
            continue
        sig = compressor._signature(m.group())
        if any(compressor._near_duplicate(prev, sig) for prev in seen):
            flags.append(_flag("repeated", text, m.start(), m.end(), "Near-duplicate of an earlier sentence.", ""))
        else:
            seen.append(sig)

    flags.sort(key=lambda f: (f["start"], f["kind"]))
    total = len(flags)
    shown = flags[:MAX_FLAGS]
    # overlapping flags (e.g. filler + its whitespace) must not be double counted
    wasted, covered_end = 0, 0
    for f in shown:
        if f["wasted"] is None or f["start"] < covered_end:
            continue
        wasted += f["wasted"]
        covered_end = f["end"]
    return {
        "flags": shown,
        "flag_count": total,
        "truncated": total > MAX_FLAGS,
        "tokens_wasted_estimate": wasted,
        "tokenizer": "gpt-4o",
    }
