# SemanticEye — Plagiarism & AI Text Detection Suite

Checks a document for **copied / patch-written text** (against a built-in reference library, your own private
documents, and optionally live Wikipedia lookups) and for **AI-generated writing** (statistical multi-signal
analysis), with sentence-level evidence and a printable report.

## Privacy & security model

| Area | How it works |
|------|--------------|
| Accounts | Username/email + password. Passwords are hashed with **scrypt** and a per-user salt. |
| Sessions | Random token in an **HttpOnly, SameSite=Lax, Secure** cookie. Only a hash of the token is stored server-side. Logging out invalidates it immediately. |
| History | Every scan belongs to the account that ran it. History, reports and delete are all scoped by the logged-in user on the server — user IDs are never accepted from the browser. |
| Private library | Documents you add under *Corpus Library* are visible only to you and only used for your checks. The built-in library is shared and read-only. |
| Sign-in required | Running a check, uploading files and viewing history all require login. |
| Abuse protection | Login lockout (5 failures / 15 min), sign-up and per-user analysis rate limits, upload size / page / text-length caps, CSRF origin check, security headers (CSP, HSTS, no-framing), API docs disabled in production. |
| Web lookups | If `ENABLE_WEB_SEARCH=true`, the first few words of checked sentences are sent to en.wikipedia.org. Set it to `false` to keep all text on your server. |

## Run locally (development)

```bat
run.bat
```
Creates `.venv`, installs dependencies and starts http://127.0.0.1:8000 with auto-reload. Needs Python 3.9+.
Create your own account with **Sign In → Create Account** (there are no built-in/demo accounts).

Manual:
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
set APP_ENV=development                              # export APP_ENV=development on Linux/macOS
python -m uvicorn server:app --reload
```

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```
The suite covers login-required endpoints, per-user isolation of history / reports / private documents,
brute-force lockout, CSRF, upload validation and XSS escaping.

## Deploy to production

The app is a single container + one persistent folder for the SQLite database.

```bash
docker build -t semanticeye .
docker run -d --name semanticeye -p 8000:8000 -v semanticeye_data:/data --restart unless-stopped semanticeye
```

Checklist:

1. **Serve over HTTPS** (Render, Railway, Fly.io, or nginx/Caddy in front of the container). Cookies are `Secure` in
   production, so login will not work over plain `http://` except on `localhost`.
2. **Persist `/data`** (volume / persistent disk). It holds `semanticeye.db` — without a volume, accounts and history
   are lost on every redeploy. Back this file up regularly.
3. **Keep one worker/instance.** Rate limits are held in memory. To scale out, move them to Redis first.
4. Set environment variables from `.env.example` in your host's dashboard (`APP_ENV=production`, `DATA_DIR=/data`).
5. Health check URL: `/healthz`.
6. Never commit `.env` or any `*.db` file (already in `.gitignore`).

Without Docker: `pip install -r requirements.txt` then
`APP_ENV=production DATA_DIR=/var/lib/semanticeye uvicorn server:app --host 0.0.0.0 --port 8000 --proxy-headers`.

## ML engines

```
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements-ml.txt
python scripts/download_models.py
```

| Feature | Model / source | Fallback if unavailable |
|---|---|---|
| Paraphrase detection | Sentence-BERT `all-MiniLM-L6-v2` cosine similarity | TF-IDF + difflib only |
| AI-text classifier | `Hello-SimpleAI/chatgpt-detector-roberta` | statistical signals |
| Perplexity | GPT-2 token log-probs (per sentence + document) | statistical signals |
| Web-wide search | Brave / Google CSE / SerpAPI (keys) or DuckDuckGo + Wikipedia (no key); result pages are downloaded and compared sentence by sentence | local corpus only |

Every model is optional and loads lazily; `GET /api/engines` shows what is active. Set `PRELOAD_MODELS=true` in production.

## Evaluation

```
python -m evaluation.eval_ai --per-class 300            # HC3: human vs ChatGPT (downloaded from the HF Hub)
python -m evaluation.eval_ai --per-class 500 --fit      # also fits ensemble weights -> app/core/calibration.json
python -m evaluation.eval_ai --data my.csv              # your own text,label (1=AI) set
python -m evaluation.eval_plagiarism                    # lexical vs SBERT vs combined (paraphrase recall)
```

Reports (AUC, accuracy, precision/recall/F1, FPR, TPR at 1% FPR, per-subset) land in `evaluation/results/`.
The RoBERTa detector was trained on HC3, so HC3 scores are optimistic - also test on text it has not seen.
Use the reported best-F1 threshold and `--fit` to tune `PPL_MIDPOINT` and the ensemble.

## Configuration

All settings are environment variables; see `.env.example` (limits, session length, web search, allowed origins).

## Project layout

```
server.py                 FastAPI app: routes, auth, security middleware
app/config.py             Environment-driven settings
app/core/db.py            SQLite: users, sessions, scan history, private corpus (all owner-scoped)
app/core/security.py      Password hashing, session tokens, rate limiter
app/core/plagiarism_engine.py   TF-IDF + SequenceMatcher + Sentence-BERT + web-wide source matching
app/core/semantic.py      Sentence-BERT encoder with embedding cache
app/core/websearch.py     Brave/Google/SerpAPI/DuckDuckGo/Wikipedia search + SSRF-safe page fetching
app/core/ai_engine.py     RoBERTa + GPT-2 perplexity + statistical signals, optional calibrated ensemble
app/core/perplexity.py    Real token-level perplexity from a causal LM
app/core/models.py        Lazy model loader with graceful fallback
evaluation/               Benchmarks: HC3 (AI text) and paraphrase/plagiarism; metrics + reports
scripts/download_models.py  Pre-download model weights
app/core/report_generator.py    Audit report assembly
app/core/utils.py         Sentence splitting, safe file extraction, word diff
app/templates, app/static Single-page frontend (vanilla HTML/CSS/JS)
sample_docs/              Built-in reference library (.txt)
tests/                    Security & isolation tests
```

## Disclaimer

Scores are **statistical estimates** to assist a human reviewer, not proof of misconduct. AI-text detection in
particular has known false positives; always apply professional judgment before any disciplinary decision.
