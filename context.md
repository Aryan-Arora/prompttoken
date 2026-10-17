# PromptToken — context for collaborators (GPT/Codex/Claude)

Read this first. It says what exists, what is fixed, what is open, and what not to break.

## What this is

A tool for developers: paste a prompt, see how it tokenizes across tokenizer families, compress it, and see exactly what was saved. The goal of v1 is evidence, not a demo: show how many tokens can be removed before answer quality measurably drops, including cases where compression hurts. Full PRD plan: 4 weeks (Analyzer, Baseline compressor, Model-based pruning, Evaluation).

Non-goals for v1: hosted SaaS/accounts/billing, training a compression model, real-time API proxying, image/audio tokens.

## Status

| Week | Scope | State |
| --- | --- | --- |
| 1 | Tokenizer analyzer (`/tokenize`, Analyze tab) | Done |
| 2 | Rule-based compressor, protected spans, diff, savings report (`/compress`, Compress tab) | Done, 10 backend tests pass |
| 2.5 | Early benchmark (SQuAD, rule-based + controls), meaning-risk check, prompt linter | Done |
| 3 | Model-based pruning (distilgpt2 word surprisal), light/medium/aggressive levels, query-aware sentence selection | Done |
| 4 | Benchmark runner (SQuAD done on 2 models; GSM8K code done, not run), Benchmark tab, security experiment | Mostly done; GSM8K run and write-up outstanding |

Git: Week 2 is committed locally; the GitHub remote may only have Week 1 until it is pushed.

## Layout

```
backend/
  app/main.py          FastAPI: GET /tokenizers, POST /tokenize, POST /compress
  app/tokenizers.py    registry: tiktoken (o200k_base, cl100k_base) + HF Llama-style (TinyLlama)
  app/compressor.py    rule-based compressor + diff segments + protected-span handling
  app/risk.py          meaning-risk check: risky tokens removed and absent from the output
  app/lint.py          prompt linter: whitespace, long numbers/strings, emoji, non-Latin, filler, repeats
  app/lm_pruner.py     model-based pruning (lm), keep_risky, query-aware sentence selection
  app/methods.py       every method under test behind one signature; all honour protected spans
  app/benchmark.py     savings-vs-quality harness (squad, gsm8k), CLI, results in backend/results/
  app/security.py      guardrail-survival and injection-persistence experiments
  app/pricing.py       PLACEHOLDER cost rate ($3 / 1M input tokens). Not a real price.
  tests/test_compressor.py
frontend/              Vite + React, plain CSS
  src/App.jsx          Analyze tab, tab switcher, shared prompt text state
  src/Compress.jsx     Compress tab
```

Run: backend `uvicorn app.main:app --port 8000` (venv in `backend/.venv`), frontend `npm run dev` (port 5173). CORS only allows `http://localhost:5173`.

## API contract (do not break without updating both sides)

`POST /lint` body `{text}` -> `{flags: [{kind, start, end, text, message, tokens, wasted}], flag_count, truncated, tokens_wasted_estimate, tokenizer}`. `wasted` is measured in context (null when not applicable).

`GET /benchmark/results`, `GET /benchmark/results/{name}` (saved result JSON), `POST /benchmark` `{task: "squad"|"gsm8k"|"security", n, seed, model?}` (starts a background run, one at a time, models allowlisted) -> `{id}`, `GET /benchmark/runs/{id}` -> `{status, stage, done, total, result_name, error}`.

`/compress` also returns `lost: [{token, reason}]`, and removed segments may carry `risky: true` and `lost`.

`GET /tokenizers` -> `{tokenizers: ["gpt-4o","gpt-3.5-4","llama"]}`

`POST /tokenize` body `{text, tokenizers: [id]}` -> `{results: {id: {tokens: [{id, text}], count}}}`

`POST /compress` body `{text, protected_spans: [{start, end}], tokenizers?: [id], method?: "rules"|"lm"|"query", level?: "light"|"medium"|"aggressive", keep_risky?: bool (default true), query?: str}` (offsets are character indexes into `text`; default tokenizers `["gpt-4o","llama"]`) ->
```
{compressed, segments: [{kind: "kept"|"removed"|"added", text}], removed: [str],
 protected: {total, intact}, tokens: {id: {before, after, saved, saved_pct, cost_saved_per_1k_calls_usd}},
 latency_ms, price_note, usd_per_mtok}
```
`segments` concatenated over kept+removed reproduces the original; over kept+added reproduces `compressed`.

## Benchmark results (seed 0, n=200 SQuAD v1.1 validation examples)

Setup: compress the context passage only, keep the question, answer with an extractive QA model, score SQuAD F1, tokens counted with o200k_base over question+context. Quality retained = method F1 / original F1 (95% paired-bootstrap intervals are in the result files and the Benchmark tab). Reproduce: `cd backend && pip install -r requirements-ml.txt && python -m app.benchmark --task squad --n 200 --seed 0 [--model deepset/roberta-base-squad2]`.

Quality retained at about 30% tokens saved ("medium"), two QA models:

| Method | Saved | distilbert | roberta-squad2 |
| --- | --- | --- | --- |
| rule_based | 0.03% | 99.8% | 100% |
| query_sentence (uses the question) | 30.8% | 92.7% | 91.2% |
| lm_safe (risky words kept) | 30.5% | 80.4% | 84.5% |
| lm (distilgpt2 word surprisal) | 31.5% | 78.7% | 81.5% |
| random_drop (blind control) | 36.9% | 55.5% | 56.7% |

What this says:
- Model-based pruning clearly beats blind dropping, and keeping risky words helps a little (intervals overlap on distilbert, clearer on roberta).
- It does NOT beat question-aware sentence selection. If you have the question (RAG), selecting sentences by relevance is the better method here. Gate 3 as the PRD wrote it (beat the rule baseline) is trivially met because the rule baseline saves ~0% on Wikipedia text; the meaningful comparison is the query-aware baseline, and the LM method loses to it.
- Findings agree across both QA models. One dataset family (SQuAD), small models, 200 examples: do not generalize.
- GSM8K is wired up (`--task gsm8k`, Qwen2.5-0.5B-Instruct, greedy, batched) and its plumbing was checked with a stand-in answerer, but it has NOT been run on the real model: the machine had ~600 MB free disk and the model needs ~1 GB. No GSM8K numbers exist yet.

## Security results (synthetic, n=100 prompts, seed 0; `python -m app.security`)

Prompts: support-bot role line, polite filler, 4 sampled safety rules, a SQuAD passage, a question. "Rules kept" = all key terms of a rule still present.

| Method | Saved | Rules kept | Rules kept, rules locked | Injection working |
| --- | --- | --- | --- | --- |
| rule_based | 3.8% | 100% | 100% | 100% |
| query_sentence_medium | 37.9% | 30% | 100% | 23% |
| lm_medium | 34.4% | 56% | 100% | 77% |
| lm_safe_medium | 33.3% | 86.5% | 100% | 61% |
| random_drop_medium | 39.1% | 33.8% | 100% | 17% |
| lm_aggressive | 52.9% | 26% | 100% | 32% |
| lm_safe_aggressive | 51.0% | 57.8% | 100% | 15% |

What this says:
- Unprotected, every pruning method deletes safety rules. The best-quality method on SQuAD (query-aware) is the worst for guardrails (30% kept at medium, 7% at aggressive) because rules share no vocabulary with the question. Quality benchmarks alone would not reveal this.
- Locking the rules as protected spans kept 100% of them in every method, with no loss in token savings (the pruners drop a fixed share of words, so the rest absorbs the cuts; the effect on task quality of that was not measured).
- Keeping numbers/names/negations (lm_safe) rescues many rules but not all: key nouns still get dropped.
- Injection: LM pruning keeps the injected payload word in ~100% of cases (it is surprising text, so it scores as informative) while mangling the sentence; "working" (payload present and at least 75% of the injection's words matching in order) is higher for LM pruning than for random dropping at similar savings (lm_medium 77% vs random_drop_medium 17%). Query-aware selection removes most injections as a side effect. Text survival only: no model was asked to obey anything. "Working" is a heuristic. The payload words are plain words absent from the passage, but still odd in encyclopedic text, which likely helps LM pruning keep them.
- A first version of the injection test used an invented canary string and a keyword heuristic and reported ~100% survival for LM pruning; verbatim survival showed that was an artifact, so the test was redesigned (see `app/security.py` docstring).

## Invariants (these are the point of the project)

1. Protected spans are copied through byte-for-byte. `protected.intact` must equal `protected.total`.
2. The deduper must never merge sentences that differ in negation words, numbers, or names ("Do not share X" vs "Do share X"; "50 words" vs "500 words"). Tests cover this.
3. Never present "quality retained" as measured until the Week 4 benchmark exists. The UI says it is not measured yet.
4. The dollar figures use a placeholder rate and say so. Do not quote them as real until `pricing.py` holds a published price with a recorded date.
5. Report negative results honestly (PRD requirement). If model pruning does not beat the rule-based baseline at equal quality, that is a finding, not a failure.

## Known gaps and rough edges

- Frontend tokenizer list and labels are hardcoded in `App.jsx` and `Compress.jsx` instead of read from `/tokenizers`.
- Only 3 tokenizers (the compress/benchmark token counts use gpt-4o and, in /compress, llama); adding an ungated Hugging Face tokenizer is one dict entry in `tokenizers.py`. Closed models (Claude, Gemini) have no offline tokenizer; they would need provider token-counting APIs and keys.
- UI is functional but visually plain. Redesign is welcome as long as the API contract and the interactions above are kept. Three screens are planned: Analyze, Compress, Benchmark.
- Protected-span selection uses textarea offsets; editing the text clears the spans.
- The meaning-risk check and linter are heuristics (word-shape rules), not understanding. Lint token counts use o200k_base only.
- GSM8K has not been run (disk space). LongBench is not wired up. The LM scorer is distilgpt2, small; a larger scorer may prune better. Benchmarks use word-level pruning, which hurts readability at aggressive levels.
- Dedupe threshold (0.8 word-overlap containment, min 4 words) is a guess, not tuned on data.
- Model-based methods need `pip install -r requirements-ml.txt` (torch, datasets); the API returns 503 for them otherwise.
- Python 3.13 needs recent package versions (`requirements.txt` uses `>=`).
- HF downloads are unauthenticated (rate-limited); the first Llama-style tokenize call downloads files.

## Suggested next work

1. Free disk space (>= 1.5 GB), then `python -m app.benchmark --task gsm8k --n 100 --seed 0` to get the honest negative result the PRD asks for.
2. Try a better scorer (a larger LM, or sentence-level surprisal) and a hybrid: query-aware selection plus risky-word-safe pruning inside kept sentences.
3. Add a guardrail-aware default: auto-detect imperative safety sentences and lock them.
4. Behavioural injection test with a generative model (does the compressed prompt still get obeyed?).
5. Write-up, UI redesign, CI, deploy.
