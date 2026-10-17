import re

from app import lm_pruner
from app.lm_pruner import prune, select_sentences


def length_scorer(text):
    # fake LM: longer words are "more informative"
    return [(m.start(), m.end(), float(len(m.group()))) for m in re.finditer(r"\S+", text)]


TEXT = "the cat sat on a mat near the big red barn and the old farmer waved"


def test_drops_lowest_scoring_words_to_target():
    res = prune(TEXT, "medium", scorer=length_scorer, keep_risky=False)
    kept = res["compressed"].split()
    total = len(TEXT.split())
    assert len(kept) == total - round(0.4 * total)
    assert "farmer" in kept and "barn" in kept  # long words survive
    assert "a" not in kept and "on" not in kept  # short words go first


def test_protected_span_is_byte_identical_and_counted():
    start = TEXT.index("on a mat")
    span = {"start": start, "end": start + len("on a mat")}
    res = prune(TEXT, "aggressive", [span], keep_risky=False, scorer=length_scorer)
    assert "on a mat" in res["compressed"]
    assert res["protected"] == {"total": 1, "intact": 1}


def test_keep_risky_never_drops_numbers_names_or_negations():
    text = "so then we did not ship it to Alvarez in 2031 but they said it was fine anyway ok"
    res = prune(text, "aggressive", scorer=lambda t: [(m.start(), m.end(), 0.1) for m in re.finditer(r"\S+", t)])
    for token in ("not", "Alvarez", "2031"):
        assert token in res["compressed"].split()


def test_without_keep_risky_those_words_can_be_dropped():
    text = "so then we did not ship it to Alvarez in 2031 but they said it was fine anyway ok"
    res = prune(text, "aggressive", keep_risky=False,
                scorer=lambda t: [(m.start(), m.end(), 0.1) for m in re.finditer(r"\S+", t)])
    assert "not" not in res["compressed"].split()


def test_segments_reconstruct_both_sides():
    res = prune(TEXT, "light", scorer=length_scorer, keep_risky=False)
    original = "".join(s["text"] for s in res["segments"] if s["kind"] in ("kept", "removed"))
    compressed = "".join(s["text"] for s in res["segments"] if s["kind"] in ("kept", "added"))
    assert original == TEXT and compressed == res["compressed"]


def test_words_with_scores_maps_tokens_to_words():
    words = lm_pruner.words_with_scores("ab cd", scorer=lambda t: [(0, 1, 1.0), (1, 2, 2.0), (2, 5, 4.0)])
    assert [w["score"] for w in words] == [3.0, 4.0]


def test_select_sentences_keeps_query_relevant_and_protected():
    text = "Paris is the capital of France. Bananas are yellow. The Eiffel Tower is in Paris. Dogs bark loudly."
    res = select_sentences(text, "Where is the Eiffel Tower?", "aggressive")
    assert "Eiffel Tower" in res["compressed"]
    assert "Bananas" not in res["compressed"]
    locked = {"start": text.index("Dogs"), "end": len(text)}
    res = select_sentences(text, "Where is the Eiffel Tower?", "aggressive", [locked])
    assert "Dogs bark loudly." in res["compressed"] and res["protected"]["intact"] == 1


def test_select_sentences_always_keeps_something():
    res = select_sentences("One. Two. Three.", "zzz", "aggressive")
    assert res["compressed"].strip()
