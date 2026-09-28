# PromptToken Analyzer & Compressor

Week 1 slice: the tokenizer analyzer only (no compression yet).

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

## Next (Week 2)

Rule-based compressor + savings report + protected spans.
