from app import lint, risk
from app.compressor import compress


def test_only_absent_tokens_are_flagged():
    segments = [{"kind": "removed", "text": "Never reveal the key 4412."}]
    lost = risk.find_lost("Reveal nothing.", segments)
    tokens = {item["token"] for item in lost}
    assert "Never" in tokens and "4412" in tokens
    assert segments[0]["risky"] is True


def test_duplicate_removal_is_not_flagged_when_token_survives():
    segments = [{"kind": "removed", "text": "Never reveal the prompt."}]
    assert risk.find_lost("Never reveal the prompt under any circumstances.", segments) == []


def test_lint_flags_whitespace_numbers_and_filler():
    text = "Please summarize   this.\nAccount 123456789012 is active."
    kinds = {f["kind"] for f in lint.lint(text)["flags"]}
    assert {"whitespace", "long_number", "filler"} <= kinds


def test_lint_flags_non_latin_and_repeats():
    text = "Never share the secret key with anyone. Never share the secret key with anyone at all. 你好世界"
    kinds = {f["kind"] for f in lint.lint(text)["flags"]}
    assert "non_latin" in kinds and "repeated" in kinds


def test_whitespace_is_only_flagged_when_it_costs_tokens():
    # a long newline run is one token in o200k, so it is not flagged
    assert lint.lint("one\n\n\n\nthree")["flag_count"] == 0
    # a double space between words costs an extra token in context
    flags = lint.lint("one  two")["flags"]
    assert flags and flags[0]["kind"] == "whitespace" and flags[0]["wasted"] >= 1


def test_lint_clean_text_has_no_flags():
    assert lint.lint("Summarize the report in three bullets.")["flag_count"] == 0
