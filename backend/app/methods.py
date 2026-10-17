"""Compression methods under test, behind one signature:

    fn(text, query, idx, seed, spans) -> compressed text

`query` is the question for query-aware methods (may be empty), `spans` are
protected spans as [{"start", "end"}]. Every method honours `spans`, including
the random control, so protected-span experiments compare like with like.
"""
import random
import re
from functools import lru_cache

from . import compressor, lm_pruner

LEVELS = list(lm_pruner.LEVELS)

cached_surprisals = lru_cache(maxsize=8192)(lm_pruner.token_surprisals)


def _random_scorer(seed, idx, level):
    rng = random.Random(f"{seed}-{idx}-{level}")

    def scorer(text):
        return [(m.start(), m.end(), rng.random()) for m in re.finditer(r"\S+", text)]

    return scorer


def build_methods(with_query=True):
    methods = {
        "original": lambda t, q, i, s, sp: t,
        "rule_based": lambda t, q, i, s, sp: compressor.compress(t, sp)["compressed"],
    }
    for lvl in LEVELS:
        if with_query:
            methods[f"query_sentence_{lvl}"] = (
                lambda t, q, i, s, sp, lvl=lvl: lm_pruner.select_sentences(t, q, lvl, sp)["compressed"]
            )
        methods[f"lm_{lvl}"] = (
            lambda t, q, i, s, sp, lvl=lvl: lm_pruner.prune(
                t, lvl, sp, keep_risky=False, scorer=cached_surprisals)["compressed"]
        )
        methods[f"lm_safe_{lvl}"] = (
            lambda t, q, i, s, sp, lvl=lvl: lm_pruner.prune(
                t, lvl, sp, keep_risky=True, scorer=cached_surprisals)["compressed"]
        )
        methods[f"random_drop_{lvl}"] = (
            lambda t, q, i, s, sp, lvl=lvl: lm_pruner.prune(
                t, lvl, sp, keep_risky=False, scorer=_random_scorer(s, i, lvl))["compressed"]
        )
    return methods


def family(name):
    for prefix in ("random_drop", "lm_safe", "lm", "query_sentence"):
        if name.startswith(prefix + "_"):
            return prefix
    return name
