"""Meaning-risk check: which risky tokens did compression remove that appear
nowhere in the compressed text?

A token is only reported as lost if it is absent from the output entirely, so
removing one of two duplicate "never reveal" sentences is not flagged, but
dropping the only "never" or the only "8842091" is.
"""
import re

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-./']*")
_PHRASES = [re.compile(p, re.I) for p in (r"\bat most\b", r"\bat least\b")]

NEGATIONS = {"not", "no", "never", "cannot", "without", "none", "nor", "neither"}
CONSTRAINTS = {
    "must", "shall", "always", "only", "exactly", "required", "mandatory",
    "forbidden", "prohibited", "except", "unless", "maximum", "minimum",
}


def _classify(word, sentence_start):
    low = word.lower().rstrip(".'")
    if low in NEGATIONS or low.endswith("n't"):
        return "negation"
    if low in CONSTRAINTS:
        return "constraint"
    if any(c.isdigit() for c in word):
        return "number or id"
    if word.isupper() and len(word) >= 2:
        return "name or acronym"
    if word[0].isupper() and not sentence_start:
        return "name"
    return None


def risky_tokens(text):
    """Return {token: reason} for risky tokens in text."""
    found = {}
    for m in _WORD.finditer(text):
        before = text[: m.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?:\n"
        reason = _classify(m.group(), sentence_start)
        if reason:
            found.setdefault(m.group().rstrip(".'"), reason)
    for pat in _PHRASES:
        for m in pat.finditer(text):
            found.setdefault(m.group().lower(), "constraint")
    return found


def _present(token, text):
    pattern = r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text, re.I) is not None


def find_lost(compressed, segments):
    """Mark removed segments that contain lost risky tokens; return the unique lost items."""
    lost_all = {}
    for seg in segments:
        if seg["kind"] != "removed":
            continue
        lost = [
            {"token": tok, "reason": reason}
            for tok, reason in risky_tokens(seg["text"]).items()
            if not _present(tok, compressed)
        ]
        if lost:
            seg["risky"] = True
            seg["lost"] = lost
            for item in lost:
                lost_all.setdefault((item["token"].lower(), item["reason"]), item)
    return list(lost_all.values())
