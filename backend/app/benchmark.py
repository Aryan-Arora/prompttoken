"""Savings-vs-quality benchmark.

Tasks
  squad  SQuAD v1.1 validation. Compress the context passage, keep the question,
         answer with an extractive QA model, score SQuAD F1 / exact match.
  gsm8k  GSM8K test. Compress the whole problem statement, solve with a small
         instruct model (greedy), score final-answer accuracy.

Quality retained = method score / original score, with a paired bootstrap 95%
interval. Model outputs are cached per unique input.

Run:  python -m app.benchmark --task squad --n 200 --seed 0 [--model NAME]
      python -m app.benchmark --task gsm8k --n 100 --seed 0
Needs: pip install -r requirements-ml.txt
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

from . import methods as methods_mod
from . import tokenizers

DEFAULT_MODELS = {
    "squad": "distilbert-base-cased-distilled-squad",
    "gsm8k": "Qwen/Qwen2.5-0.5B-Instruct",
}
DATASETS = {"squad": "rajpurkar/squad", "gsm8k": "openai/gsm8k"}
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
GSM8K_SYSTEM = (
    "Solve the math problem. Reason step by step, then end with a final line "
    "exactly like: The answer is <number>."
)


# ---------- metrics ----------

def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
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


def extract_number(text):
    found = re.findall(r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)", text, re.I)
    if not found:
        found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def numeric_correct(pred_text, gold):
    value = extract_number(pred_text)
    return float(value is not None and abs(value - float(gold)) < 1e-4)


def bootstrap_ratio(scores, base, seed, rounds=1000):
    rng = random.Random(seed)
    n = len(base)
    ratios = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        denom = sum(base[i] for i in idx)
        if denom > 0:
            ratios.append(sum(scores[i] for i in idx) / denom)
    ratios.sort()
    if not ratios:
        return None
    return [round(ratios[int(0.025 * len(ratios))], 4), round(ratios[int(0.975 * len(ratios)) - 1], 4)]


# ---------- tasks ----------

def load_examples(task, n, seed):
    from datasets import load_dataset

    if task == "squad":
        data = load_dataset(DATASETS[task], split="validation").shuffle(seed=seed).select(range(n))
        return [
            {"text": ex["context"], "question": ex["question"], "golds": ex["answers"]["text"]}
            for ex in data
        ]
    data = load_dataset(DATASETS[task], "main", split="test").shuffle(seed=seed).select(range(n))
    return [
        {
            "text": ex["question"],
            "question": "",
            "golds": [ex["answer"].split("####")[-1].strip().replace(",", "")],
        }
        for ex in data
    ]


def make_extractive_qa(model_name):
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()

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

    return lambda pairs, progress=None: [answer(q, c) for q, c in pairs]


def make_generator(model_name, batch_size=16, max_new_tokens=320):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()

    def solve(pairs, progress=None):
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": GSM8K_SYSTEM}, {"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True,
            )
            for _, text in pairs
        ]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outputs = [None] * len(prompts)
        for lo in range(0, len(order), batch_size):
            idx = order[lo:lo + batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, texts):
                outputs[i] = t
            if progress:
                progress(min(lo + batch_size, len(order)), len(order), "generating")
        return outputs

    return solve


# ---------- run ----------

def prompt_tokens(question, text):
    return tokenizers.tokenize(f"{question}\n{text}" if question else text, "gpt-4o")["count"]


def score_one(task, pred, golds):
    if task == "squad":
        return best_over_golds(f1_score, pred, golds), best_over_golds(exact_match, pred, golds)
    correct = numeric_correct(pred, golds[0])
    return correct, correct


def run(task="squad", n=200, seed=0, out=None, model=None, progress=None, answerer=None):
    progress = progress or (lambda done, total, stage: None)
    model = model or DEFAULT_MODELS[task]
    started = time.time()
    examples = load_examples(task, n, seed)
    methods = methods_mod.build_methods(with_query=(task == "squad"))

    compressed = {}
    for idx, ex in enumerate(examples):
        for name, fn in methods.items():
            compressed[(idx, name)] = fn(ex["text"], ex["question"], idx, seed, [])
        progress(idx + 1, len(examples), "compressing")

    pairs = sorted({(examples[i]["question"], text) for (i, _), text in compressed.items()})
    if answerer is None:
        answerer = make_extractive_qa(model) if task == "squad" else make_generator(model)
    answers = dict(zip(pairs, answerer(pairs, progress)))

    results = {m: {"score": [], "em": [], "tokens": [], "pred": []} for m in methods}
    for idx, ex in enumerate(examples):
        for name in methods:
            text = compressed[(idx, name)]
            pred = answers[(ex["question"], text)]
            s, em = score_one(task, pred, ex["golds"])
            r = results[name]
            r["score"].append(s)
            r["em"].append(em)
            r["tokens"].append(prompt_tokens(ex["question"], text))
            r["pred"].append(pred)

    base = results["original"]
    base_tokens, base_score = sum(base["tokens"]), sum(base["score"]) / n
    report = {}
    for name, r in results.items():
        mean = sum(r["score"]) / n
        drops = sorted(
            ((base["score"][i] - r["score"][i], i) for i in range(n) if base["score"][i] > r["score"][i]),
            reverse=True,
        )
        report[name] = {
            "family": methods_mod.family(name),
            "tokens_saved_pct": round(100 * (1 - sum(r["tokens"]) / base_tokens), 2),
            "score": round(mean, 4),
            "exact_match": round(sum(r["em"]) / n, 4) if task == "squad" else None,
            "quality_retained": round(mean / base_score, 4) if base_score else None,
            "quality_retained_ci95": bootstrap_ratio(r["score"], base["score"], seed) if name != "original" else None,
            "examples_hurt": len(drops),
            "worst_examples": [
                {
                    "question": examples[i]["question"] or examples[i]["text"][:200],
                    "gold": examples[i]["golds"][0],
                    "original_answer": base["pred"][i][-200:],
                    "method_answer": r["pred"][i][-200:],
                    "compressed_text": compressed[(i, name)][:600],
                    "score_drop": round(d, 3),
                }
                for d, i in drops[:3]
            ],
        }

    result = {
        "meta": {
            "task": task, "dataset": DATASETS[task], "n": n, "seed": seed,
            "metric": "F1" if task == "squad" else "accuracy",
            "answer_model": model, "scorer_lm": "distilgpt2 (lm_* methods)",
            "token_counter": "tiktoken o200k_base"
            + (", question + context" if task == "squad" else ", problem text"),
            "date": date.today().isoformat(), "runtime_s": round(time.time() - started, 1),
            "notes": "random_drop_* are blind controls. lm_safe_* never drop numbers, names, "
                     "negations, or constraint words. query_sentence_* uses the question.",
        },
        "methods": report,
    }
    out = Path(out) if out else RESULTS_DIR / f"{task}_n{n}_seed{seed}_{model.split('/')[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result, out


def print_table(result):
    meta = result["meta"]
    print(f"\n{meta['task']} n={meta['n']} seed={meta['seed']} model={meta['answer_model']} metric={meta['metric']}")
    print(f"{'method':<24}{'saved %':>9}{'score':>8}{'retained':>10}  95% CI")
    for name, r in result["methods"].items():
        ci = r["quality_retained_ci95"]
        print(f"{name:<24}{r['tokens_saved_pct']:>9}{r['score']:>8}{r['quality_retained']:>10}  {ci if ci else '-'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(DATASETS), default="squad")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def show(done, total, stage):
        if done == total or done % 25 == 0:
            print(f"  {stage}: {done}/{total}", flush=True)

    res, path = run(args.task, args.n, args.seed, args.out, args.model, progress=show)
    print_table(res)
    print(f"\nSaved {path}")
