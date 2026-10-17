# PromptToken Analyzer & Compressor

Status: tokenizer analyzer, rule-based and model-based compression with protected spans, diff and savings report, prompt linter, meaning-risk check, SQuAD benchmark on two QA models, security experiment, and a Benchmark tab are built. GSM8K is wired up but not yet run. Details and results: [context.md](context.md).

## Run the backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

First request downloads the TinyLlama tokenizer files (~a few MB, cached after that).

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Paste a prompt, pick tokenizers, hit Analyze.

## What's here

- `backend/app/tokenizers.py` — wraps tiktoken (`o200k_base`, `cl100k_base`) and a Hugging Face Llama-style tokenizer behind one interface
- `backend/app/main.py` — `POST /tokenize` endpoint
- `frontend/src/App.jsx` — prompt box, tokenizer picker, token highlighter

- `backend/app/compressor.py` — rule-based compressor (filler removal, near-duplicate sentence removal, whitespace), protected spans, diff segments
- `backend/app/pricing.py` — illustrative cost rate (placeholder, not a published price)
- `frontend/src/Compress.jsx` — compress screen: lock spans, diff, savings report
- `backend/app/risk.py`, `backend/app/lint.py` — meaning-risk check and prompt linter (`POST /lint`)
- `backend/app/lm_pruner.py`, `backend/app/methods.py` — model-based pruning (distilgpt2), query-aware sentence selection, and all methods under test
- `backend/app/benchmark.py` — `pip install -r requirements-ml.txt && python -m app.benchmark --task squad --n 200 --seed 0` (also `--task gsm8k`, `--model`); results in `backend/results/`
- `backend/app/security.py` — `python -m app.security --n 100 --seed 0`
- `frontend/src/Benchmark.jsx` — Benchmark tab: chart, results table, failure examples, security tables, run button
- Tests: `cd backend && pip install pytest && python -m pytest`

## Next

Run GSM8K, try a stronger scorer or a hybrid method, write up the results.
