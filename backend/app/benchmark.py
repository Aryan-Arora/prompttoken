"""Savings-vs-quality benchmark on SQuAD v1.1 (validation).

Method: compress the *context* passage, keep the question, answer with an
extractive QA model, score against the gold answers (SQuAD EM / token F1).
Quality retained = mean F1 of a method / mean F1 on the original context, with
a paired bootstrap 95% interval. Model outputs are cached per (question, text).

Run:  python -m app.benchmark --n 200 --seed 0
Needs: pip install -r requirements-benchmark.txt
"""
import argparse
import json
import random
import re
import string
import time
from collections import Counter
from datetime import date
from pathlib import Path

from . import compressor, tokenizers

QA_MODEL = "distilbert-base-cased-distilled-squad"
DATASET = "rajpurkar/squad"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CONTROL_RATES = [0.2, 0.4, 0.6]


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def best_over_golds(metric, pred, golds):
    return max(metric(pred, g) for g in golds)


def rule_based(context, idx, seed):
    return compressor.compress(context)["compressed"]


def random_word_drop(rate):
    def method(context, idx, seed):
        rng = random.Random(f"{seed}-{idx}-{rate}")
        words = context.split()
        kept = [w for w in words if rng.random() >= rate] or words[:1]
        return " ".join(kept)
    return method


def build_methods():
    methods = {"original": lambda c, i, s: c, "rule_based": rule_based}
    for rate in CONTROL_RATES:
        methods[f"random_drop_{int(rate * 100)}"] = random_word_drop(rate)
    return methods


def prompt_tokens(question, context):
    return tokenizers.tokenize(f"{question}\n{context}", "gpt-4o")["count"]


def bootstrap_ratio(f1_method, f1_orig, seed, rounds=1000):
    rng = random.Random(seed)
    n = len(f1_orig)
    ratios = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        denom = sum(f1_orig[i] for i in idx)
        if denom > 0:
            ratios.append(sum(f1_method[i] for i in idx) / denom)
    ratios.sort()
    if not ratios:
        return None
    return [round(ratios[int(0.025 * len(ratios))], 4), round(ratios[int(0.975 * len(ratios)) - 1], 4)]


def make_qa():
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL).eval()

    def answer(question, context):
        enc = tok(
            question, context, truncation="only_second", max_length=384, stride=128,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping")
        with torch.no_grad():
            out = model(**enc)
        best_score, best_text = float("-inf"), ""
        for w in range(out.start_logits.shape[0]):
            ctx = torch.tensor([sid == 1 for sid in enc.sequence_ids(w)])
            start = out.start_logits[w].masked_fill(~ctx, -1e9)
            end = out.end_logits[w].masked_fill(~ctx, -1e9)
            for si in torch.topk(start, 20).indices.tolist():
                for ei in torch.topk(end, 20).indices.tolist():
                    if not ctx[si] or not ctx[ei] or ei < si or ei - si >= 30:
                        continue
                    score = start[si].item() + end[ei].item()
                    if score > best_score:
                        best_score = score
                        best_text = context[offsets[w][si][0]:offsets[w][ei][1]]
        return best_text

    return answer


def run(n, seed, out):
    from datasets import load_dataset

    data = load_dataset(DATASET, split="validation").shuffle(seed=seed).select(range(n))
    qa_answer = make_qa()
    cache = {}

    def answer(question, context):
        key = (question, context)
        if key not in cache:
            cache[key] = qa_answer(question, context)
        return cache[key]

    methods = build_methods()
    scores = {m: {"f1": [], "em": [], "tokens": [], "pred": []} for m in methods}
    started = time.time()

    for idx, ex in enumerate(data):
        golds = ex["answers"]["text"]
        for name, fn in methods.items():
            ctx = fn(ex["context"], idx, seed)
            pred = answer(ex["question"], ctx)
            scores[name]["f1"].append(best_over_golds(f1_score, pred, golds))
            scores[name]["em"].append(best_over_golds(exact_match, pred, golds))
            scores[name]["tokens"].append(prompt_tokens(ex["question"], ctx))
            scores[name]["pred"].append(pred)
        if (idx + 1) % 25 == 0:
            print(f"  {idx + 1}/{n} examples, {time.time() - started:.0f}s", flush=True)

    base = scores["original"]
    base_tokens = sum(base["tokens"])
    base_f1 = sum(base["f1"]) / n
    report = {}
    for name, s in scores.items():
        mean_f1 = sum(s["f1"]) / n
        drops = sorted(
            ((base["f1"][i] - s["f1"][i], i) for i in range(n) if base["f1"][i] - s["f1"][i] > 0),
            reverse=True,
        )
        report[name] = {
            "tokens_saved_pct": round(100 * (1 - sum(s["tokens"]) / base_tokens), 2),
            "f1": round(mean_f1, 4),
            "exact_match": round(sum(s["em"]) / n, 4),
            "quality_retained": round(mean_f1 / base_f1, 4) if base_f1 else None,
            "quality_retained_ci95": bootstrap_ratio(s["f1"], base["f1"], seed) if name != "original" else None,
            "examples_hurt": len(drops),
            "worst_examples": [
                {
                    "question": data[i]["question"],
                    "gold": data[i]["answers"]["text"][0],
                    "original_answer": base["pred"][i],
                    "method_answer": s["pred"][i],
                    "f1_drop": round(d, 3),
                }
                for d, i in drops[:3]
            ],
        }

    result = {
        "meta": {
            "dataset": DATASET, "split": "validation", "n": n, "seed": seed,
            "qa_model": QA_MODEL, "token_counter": "tiktoken o200k_base, question + context",
            "date": date.today().isoformat(), "runtime_s": round(time.time() - started, 1),
            "notes": "random_drop_* are blind controls, not proposed methods.",
        },
        "methods": report,
    }
    out = Path(out) if out else RESULTS_DIR / f"squad_n{n}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print_table(result)
    print(f"\nSaved {out}")
    return result


def print_table(result):
    print(f"\n{'method':<16}{'saved %':>9}{'F1':>8}{'EM':>8}{'retained':>10}  95% CI")
    for name, r in result["methods"].items():
        ci = r["quality_retained_ci95"]
        print(
            f"{name:<16}{r['tokens_saved_pct']:>9}{r['f1']:>8}{r['exact_match']:>8}"
            f"{r['quality_retained']:>10}  {ci if ci else '-'}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.n, args.seed, args.out)
