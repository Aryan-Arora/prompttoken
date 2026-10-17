"""Layer 1: rule-based compressor.

Pipeline: split text into protected / unprotected pieces -> split unprotected
pieces into sentence units -> strip filler -> drop near-duplicate sentences ->
normalize whitespace. Protected text is copied through byte-for-byte.
"""
import re
from difflib import SequenceMatcher

# (pattern, replacement). Order matters: longer phrases first.
FILLER_RULES = [
    (r"\bif you would be so kind,?\s*", ""),
    (r"\bif you (?:could|can) please,?\s*", ""),
    (r"\b(?:I would like you to|I want you to|I need you to)\s+", ""),
    (r"\bplease,?\s*", ""),
    (r"\bkindly\s+", ""),
    (r"\bfeel free to\s+", ""),
    (r"\bbasically,?\s*", ""),
    (r"\b(?:thanks|thank you) in advance[.!]?\s*", ""),
    (r"\bI hope this (?:message )?finds you well[.,!]?\s*", ""),
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bin the event that\b", "if"),
    (r"\bwith regard to\b", "about"),
]
_FILLER = [(re.compile(p, re.IGNORECASE), r) for p, r in FILLER_RULES]

NEGATIONS = {"not", "no", "never", "cannot", "without", "none", "nor"}
DUP_CONTAINMENT = 0.8
DUP_MIN_WORDS = 4

_SENTENCE = re.compile(r".*?(?:[.!?]+(?:\s+|$)|\n+|$)", re.S)
_DIFF_TOKEN = re.compile(r"\s+|\w+|[^\w\s]")


def _merge_spans(spans, n):
    clean = sorted((max(0, s), min(n, e)) for s, e in spans if min(n, e) > max(0, s))
    merged = []
    for s, e in clean:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _words(sentence):
    return re.findall(r"[a-z0-9']+", sentence.lower())


def _is_negation(word):
    return word in NEGATIONS or word.endswith("n't")


def _signature(sentence):
    words = _words(sentence)
    return {
        "set": set(words),
        "neg": {w for w in words if _is_negation(w)},
        "nums": {w for w in words if any(c.isdigit() for c in w)},
        "names": set(re.findall(r"\b[A-Z][A-Za-z]+\b", sentence)[1:]),
        "n": len(words),
    }


def _near_duplicate(a, b):
    if a["n"] < DUP_MIN_WORDS or b["n"] < DUP_MIN_WORDS:
        return False
    # Never merge sentences that differ in negation, numbers, or names: that is
    # how compression silently flips or loses meaning.
    if a["neg"] != b["neg"] or a["nums"] != b["nums"] or a["names"] != b["names"]:
        return False
    overlap = len(a["set"] & b["set"]) / min(len(a["set"]), len(b["set"]))
    return overlap >= DUP_CONTAINMENT


def _strip_filler(sentence):
    out = sentence
    for pattern, repl in _FILLER:
        out = pattern.sub(repl, out)
    if out == sentence:
        return sentence
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"^(\s*)[,;:]\s*", r"\1", out)
    orig_first = re.search(r"[A-Za-z]", sentence)
    new_first = re.search(r"[A-Za-z]", out)
    if orig_first and new_first and orig_first.group().isupper() and new_first.group().islower():
        i = new_first.start()
        out = out[:i] + out[i].upper() + out[i + 1 :]
    if not re.search(r"\w", out):
        return ""
    return out


def _normalize_whitespace(run):
    run = re.sub(r"[ \t]+\n", "\n", run)
    run = re.sub(r"\n{3,}", "\n\n", run)
    run = re.sub(r"(?<=\S)[ \t]{2,}", " ", run)
    return run


def _diff_segments(original, compressed):
    a = _DIFF_TOKEN.findall(original)
    b = _DIFF_TOKEN.findall(compressed)
    segments = []

    def add(kind, toks):
        text = "".join(toks)
        if not text:
            return
        if segments and segments[-1]["kind"] == kind:
            segments[-1]["text"] += text
        else:
            segments.append({"kind": kind, "text": text})

    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            add("kept", a[i1:i2])
        else:
            add("removed", a[i1:i2])
            add("added", b[j1:j2])
    return segments


def compress(text, protected_spans=()):
    spans = _merge_spans([(s["start"], s["end"]) for s in protected_spans], len(text))

    # units: {"text", "protected"}
    units, pos = [], 0
    for s, e in spans:
        units.extend(_sentence_units(text[pos:s]))
        units.append({"text": text[s:e], "protected": True})
        pos = e
    units.extend(_sentence_units(text[pos:]))

    for u in units:
        if not u["protected"]:
            u["text"] = _strip_filler(u["text"])

    kept = []  # indexes of surviving units with a signature
    sigs = {}
    for i, u in enumerate(units):
        if not u["text"].strip():
            continue
        sig = _signature(u["text"])
        dropped_self = False
        for k in list(kept):
            if not _near_duplicate(sigs[k], sig):
                continue
            other = units[k]
            if u["protected"] and other["protected"]:
                continue
            if u["protected"]:
                drop = k
            elif other["protected"]:
                drop = i
            else:
                drop = i if sig["n"] <= sigs[k]["n"] else k
            if drop == i:
                dropped_self = True
                break
            kept.remove(k)
            units[k]["text"] = _whitespace_only(units[k]["text"])
        if dropped_self:
            u["text"] = _whitespace_only(u["text"])
        else:
            kept.append(i)
            sigs[i] = sig

    # merge consecutive unprotected units, normalize whitespace per run
    pieces, run = [], ""
    for u in units:
        if u["protected"]:
            pieces.append(_normalize_whitespace(run))
            pieces.append(u["text"])
            run = ""
        else:
            run += u["text"]
    pieces.append(_normalize_whitespace(run).rstrip(" \t"))
    compressed = "".join(pieces)

    protected_texts = [text[s:e] for s, e in spans]
    cursor, intact = 0, 0
    for ptxt in protected_texts:
        idx = compressed.find(ptxt, cursor)
        if idx >= 0:
            intact += 1
            cursor = idx + len(ptxt)

    segments = _diff_segments(text, compressed)
    removed = [s["text"].strip() for s in segments if s["kind"] == "removed" and s["text"].strip()]
    return {
        "compressed": compressed,
        "segments": segments,
        "removed": removed,
        "protected": {"total": len(protected_texts), "intact": intact},
    }


def _sentence_units(chunk):
    return [{"text": m, "protected": False} for m in _SENTENCE.findall(chunk) if m]


def _whitespace_only(s):
    # keep a paragraph break if the dropped sentence ended one
    return "\n" if s.endswith("\n") else ""
