# PromptToken Analyzer & Compressor

Status: Week 1 (tokenizer analyzer) and Week 2 (rule-based compressor with protected spans, diff, and savings report) are built. Model-based pruning and the benchmark are not.

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
- Tests: `cd backend && pip install pytest && python -m pytest`

## Next (Week 3)

Model-based importance pruning and compression levels, compared against this baseline.
