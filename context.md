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
| 3 | Model-based pruning (small LM importance scoring), compression levels | Not started |
| 4 | Benchmark runner (SQuAD/HotpotQA, GSM8K), security tests, write-up | Not started |

Git: Week 2 is committed locally; the GitHub remote may only have Week 1 until it is pushed.

## Layout

```
backend/
  app/main.py          FastAPI: GET /tokenizers, POST /tokenize, POST /compress
  app/tokenizers.py    registry: tiktoken (o200k_base, cl100k_base) + HF Llama-style (TinyLlama)
  app/compressor.py    rule-based compressor + diff segments + protected-span handling
  app/risk.py          meaning-risk check: risky tokens removed and absent from the output
  app/lint.py          prompt linter: whitespace, long numbers/strings, emoji, non-Latin, filler, repeats
  app/benchmark.py     SQuAD savings-vs-quality harness (CLI), results in backend/results/
  app/pricing.py       PLACEHOLDER cost rate ($3 / 1M input tokens). Not a real price.
  tests/test_compressor.py
frontend/              Vite + React, plain CSS
  src/App.jsx          Analyze tab, tab switcher, shared prompt text state
  src/Compress.jsx     Compress tab
```

Run: backend `uvicorn app.main:app --port 8000` (venv in `backend/.venv`), frontend `npm run dev` (port 5173). CORS only allows `http://localhost:5173`.

## API contract (do not break without updating both sides)

`POST /lint` body `{text}` -> `{flags: [{kind, start, end, text, message, tokens, wasted}], flag_count, truncated, tokens_wasted_estimate, tokenizer}`. `wasted` is measured in context (null when not applicable).

`/compress` also returns `lost: [{token, reason}]`, and removed segments may carry `risky: true` and `lost`.

`GET /tokenizers` -> `{tokenizers: ["gpt-4o","gpt-3.5-4","llama"]}`

`POST /tokenize` body `{text, tokenizers: [id]}` -> `{results: {id: {tokens: [{id, text}], count}}}`

`POST /compress` body `{text, protected_spans: [{start, end}], tokenizers?: [id]}` (offsets are character indexes into `text`; default tokenizers `["gpt-4o","llama"]`) ->
```
{compressed, segments: [{kind: "kept"|"removed"|"added", text}], removed: [str],
 protected: {total, intact}, tokens: {id: {before, after, saved, saved_pct, cost_saved_per_1k_calls_usd}},
 latency_ms, price_note, usd_per_mtok}
```
`segments` concatenated over kept+removed reproduces the original; over kept+added reproduces `compressed`.

## Benchmark: first real numbers (SQuAD v1.1 validation, n=200, seed 0)

Setup: compress the context passage only, keep the question, answer with `distilbert-base-cased-distilled-squad`, score SQuAD F1/EM, tokens counted with o200k_base over question+context. Quality retained = method F1 / original F1, paired bootstrap 95% interval. Reproduce: `cd backend && pip install -r requirements-benchmark.txt && python -m app.benchmark --n 200 --seed 0`.

| Method | Tokens saved | F1 | Quality retained (95% CI) |
| --- | --- | --- | --- |
| original | 0% | 0.888 | 1.000 |
| rule_based | 0.03% | 0.886 | 0.998 [0.993, 1.000] |
| random_drop_20 (blind control) | 18.5% | 0.715 | 0.805 [0.745, 0.872] |
| random_drop_40 (blind control) | 37.1% | 0.516 | 0.581 [0.520, 0.646] |
| random_drop_60 (blind control) | 55.7% | 0.326 | 0.367 [0.311, 0.427] |

What this says: the rule-based compressor does essentially nothing on Wikipedia-style context (it targets filler and repeated instructions, which this data lacks), so it is not evidence that rule-based compression helps QA context. Blind word dropping saves tokens but loses about 20% of quality at 18% savings, which is the bar a Week 3 model-based method must clear. random_drop_* are controls, not proposed methods. Single dataset, single QA model, 200 examples: do not generalize.

## Invariants (these are the point of the project)

1. Protected spans are copied through byte-for-byte. `protected.intact` must equal `protected.total`.
2. The deduper must never merge sentences that differ in negation words, numbers, or names ("Do not share X" vs "Do share X"; "50 words" vs "500 words"). Tests cover this.
3. Never present "quality retained" as measured until the Week 4 benchmark exists. The UI says it is not measured yet.
4. The dollar figures use a placeholder rate and say so. Do not quote them as real until `pricing.py` holds a published price with a recorded date.
5. Report negative results honestly (PRD requirement). If model pruning does not beat the rule-based baseline at equal quality, that is a finding, not a failure.

## Known gaps and rough edges

- Frontend tokenizer list and labels are hardcoded in `App.jsx` and `Compress.jsx` instead of read from `/tokenizers`.
- Only 3 tokenizers; adding an ungated Hugging Face tokenizer is one dict entry in `tokenizers.py`. Closed models (Claude, Gemini) have no offline tokenizer; they would need provider token-counting APIs and keys.
- UI is functional but visually plain. Redesign is welcome as long as the API contract and the interactions above are kept. Three screens are planned: Analyze, Compress, Benchmark.
- Protected-span selection uses textarea offsets; editing the text clears the spans.
- The meaning-risk check and linter are heuristics (word-shape rules), not understanding. Lint token counts use o200k_base only.
- Benchmark covers one dataset (SQuAD) and one extractive QA model; GSM8K and LongBench are not wired up. No Benchmark tab in the UI yet.
- Dedupe threshold (0.8 word-overlap containment, min 4 words) is a guess, not tuned on data.
- PyTorch is not installed; Week 3 will need it (or an ONNX/llama.cpp alternative) for LM scoring.
- Python 3.13 needs recent package versions (`requirements.txt` uses `>=`).
- HF downloads are unauthenticated (rate-limited); the first Llama-style tokenize call downloads files.

## Suggested next work (Week 3)

Unit-level importance scoring with a small open LM (surprisal per token or sentence, LLMLingua-style), drop lowest-scoring units to hit light/medium/aggressive targets, never touch protected spans, re-verify they are byte-identical, expose as `level` and `method` on `/compress`, and show rule-based vs model-based side by side.
