"""Central, environment-driven configuration.

Everything that differs between your laptop and the production server lives
here and is read from environment variables (see .env.example).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# "production" (default, secure) or "development" (local testing over http)
APP_ENV = os.getenv("APP_ENV", "production").strip().lower()
IS_PROD = APP_ENV == "production"

# Where the SQLite database lives. On a host, point this at a persistent disk.
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DB_PATH = DATA_DIR / "semanticeye.db"

# Read-only built-in reference corpus shipped with the app.
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

# Sessions / cookies
COOKIE_NAME = "se_session"
COOKIE_SECURE = _bool("COOKIE_SECURE", IS_PROD)  # must be true when served over HTTPS
SESSION_DAYS = _int("SESSION_DAYS", 7)

# Extra origins allowed to call the API (comma separated). Normally empty:
# the app only accepts state-changing requests from its own origin.
ALLOWED_ORIGINS = {o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()}

# Limits (protect the server from huge / abusive input)
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_MB", 10) * 1024 * 1024
MAX_TEXT_CHARS = _int("MAX_TEXT_CHARS", 100_000)
MAX_PDF_PAGES = _int("MAX_PDF_PAGES", 200)
MAX_USER_CORPUS_DOCS = _int("MAX_USER_CORPUS_DOCS", 20)
MAX_USER_CORPUS_CHARS = _int("MAX_USER_CORPUS_CHARS", 1_000_000)
ANALYZE_PER_MINUTE = _int("ANALYZE_PER_MINUTE", 10)

# Live Wikipedia lookups send the first few words of each sentence to
# en.wikipedia.org. Set to false for a fully self-contained deployment.
ENABLE_WEB_SEARCH = _bool("ENABLE_WEB_SEARCH", True)
MAX_WEB_LOOKUPS = _int("MAX_WEB_LOOKUPS", 25)
WEB_SEARCH_DEADLINE_SEC = _int("WEB_SEARCH_DEADLINE_SEC", 20)


# ── ML models (all optional: every feature falls back gracefully if a model can't load) ──
ENABLE_SBERT = _bool("ENABLE_SBERT", True)            # Sentence-BERT paraphrase detection
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SBERT_SEM_FLOOR = float(os.getenv("SBERT_SEM_FLOOR", "0.55"))   # cosine mapped to score 0
SBERT_SEM_CEIL = float(os.getenv("SBERT_SEM_CEIL", "0.95"))     # cosine mapped to score 1

ENABLE_ROBERTA = _bool("ENABLE_ROBERTA", True)        # RoBERTa AI-text classifier
ROBERTA_MODEL = os.getenv("ROBERTA_MODEL", "Hello-SimpleAI/chatgpt-detector-roberta")

ENABLE_PERPLEXITY = _bool("ENABLE_PERPLEXITY", True)  # real perplexity from a causal LM
LM_MODEL = os.getenv("LM_MODEL", "gpt2")
MAX_LM_TOKENS = _int("MAX_LM_TOKENS", 2048)
PPL_MIDPOINT = float(os.getenv("PPL_MIDPOINT", "35"))  # perplexity where P(AI)=0.5 (tune with evaluation/)
PPL_STEEPNESS = float(os.getenv("PPL_STEEPNESS", "2.5"))

# Load models at start-up instead of on first request (recommended in production)
PRELOAD_MODELS = _bool("PRELOAD_MODELS", False)

# ── Web-wide source search ──
# SEARCH_PROVIDER: auto | brave | google | serpapi | duckduckgo | wikipedia
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "auto").strip().lower()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY", "")
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
WEB_RESULTS_PER_QUERY = _int("WEB_RESULTS_PER_QUERY", 5)
MAX_WEB_PAGES = _int("MAX_WEB_PAGES", 15)             # pages downloaded per scan
WEB_PAGE_MAX_BYTES = _int("WEB_PAGE_MAX_KB", 1500) * 1024
WEB_FETCH_TIMEOUT = float(os.getenv("WEB_FETCH_TIMEOUT", "5"))

# Calibrated ensemble weights written by `python -m evaluation.eval_ai --fit`
CALIBRATION_PATH = Path(os.getenv("CALIBRATION_PATH", BASE_DIR / "app" / "core" / "calibration.json"))
