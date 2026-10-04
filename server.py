"""SemanticEye - Plagiarism & AI Text Detection Suite (FastAPI application)."""
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.core import db
from app.core.ai_engine import AIDetectorEngine
from app.core.plagiarism_engine import PlagiarismEngine
from app.core.report_generator import build_audit_report
from app.core.security import (
    DUMMY_HASH, RateLimiter, hash_password, new_session_token, verify_password,
)
from app.core.utils import calculate_text_stats, extract_text_from_bytes, split_sentences

logger = logging.getLogger("semanticeye")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    if config.PRELOAD_MODELS:
        import threading
        from app.core.models import hub
        threading.Thread(target=hub.preload, daemon=True).start()  # don't block start-up
    yield


app = FastAPI(
    title="SemanticEye",
    version="2.1.0",
    lifespan=lifespan,
    # API docs are only exposed while developing
    docs_url=None if config.IS_PROD else "/docs",
    redoc_url=None,
    openapi_url=None if config.IS_PROD else "/openapi.json",
)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

plagiarism_engine = PlagiarismEngine(sample_docs_dir=str(config.SAMPLE_DOCS_DIR))
ai_engine = AIDetectorEngine()

login_failures = RateLimiter(max_events=5, window_sec=15 * 60)   # per IP + account
register_limiter = RateLimiter(max_events=10, window_sec=60 * 60)  # per IP
analyze_limiter = RateLimiter(max_events=config.ANALYZE_PER_MINUTE, window_sec=60)  # per user

SAMPLE_PRESETS = [
    {"id": "direct_plagiarism", "title": "Direct Plagiarism (Reference Match)",
     "text": ("Climate change refers to long-term shifts in temperatures and weather patterns. "
              "Burning fossil fuels like coal, oil and gas is the main driver of climate change because it "
              "releases greenhouse gases into the atmosphere. "
              "Rising global temperatures cause glaciers to melt and sea levels to rise. "
              "Many countries are now investing in solar and wind power to reduce their carbon emissions. "
              "International agreements such as the Paris Accord emphasize the necessity of keeping global warming well below two degrees Celsius.")},
    {"id": "ai_generated", "title": "ChatGPT Generated Essay",
     "text": ("Furthermore, artificial intelligence stands as a transformative milestone in modern technological evolution. "
              "It is crucial to understand that machine learning architectures enable computers to parse intricate patterns from data without explicit manual heuristics. "
              "Moreover, deep learning uses neural networks with many layers to recognise images, speech and natural language with remarkable precision. "
              "In conclusion, the seamless integration of autonomous systems will continue to play a pivotal role across healthcare, finance, and industrial ecosystems.")},
    {"id": "human_original", "title": "Human Original Narrative",
     "text": ("Last summer our university team organized an early morning riverbank restoration camp, and honestly, none of us expected more than thirty people to show up. "
              "The humidity was brutal. "
              "Yet by midday, over one hundred volunteers had gathered with shovels, watering buckets, and saplings. "
              "I remember slipping into the riverbank mud while trying to secure an unruly bamboo stake, which made everyone burst into laughter. "
              "That shared sweat and humor built stronger camaraderie than any lecture hall ever could.")},
    {"id": "mixed_hybrid", "title": "Mixed Hybrid Content",
     "text": ("Cybersecurity involves protecting computer systems, networks, and sensitive digital assets from malicious unauthorized intrusions. "
              "Furthermore, navigating the complexities of modern perimeter defenses underscores the vital component of zero-trust verification. "
              "Contemporary defense strategies rely heavily on multi-factor authentication and robust end-to-end cryptographic encryption standards. "
              "Our team spent three sleepless nights attempting to reverse-engineer an intriguing sandbox malware sample we captured during the national cyber competition. "
              "In conclusion, proactive vigilance and collaborative human intuition remain the ultimate safeguards against evolving threat landscapes.")},
]

# ── Security middleware ─────────────────────────────────────────────
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "          # the UI uses inline onclick handlers
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)


def _origin_allowed(request: Request) -> bool:
    """CSRF defence: state-changing requests must come from our own origin."""
    origin = request.headers.get("origin")
    if not origin:
        return True  # non-browser clients; browsers always send Origin on cross-site POSTs
    origin = origin.rstrip("/")
    if origin in config.ALLOWED_ORIGINS:
        return True
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return urlparse(origin).netloc == host


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.method in UNSAFE_METHODS and not _origin_allowed(request):
        return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if config.COOKIE_SECURE:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ── Auth helpers ────────────────────────────────────────────────────
def current_user(request: Request) -> dict:
    token = request.cookies.get(config.COOKIE_NAME)
    user = db.get_user_by_session(token) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    return user


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _set_session_cookie(response: Response, user_id: int) -> None:
    token = new_session_token()
    db.create_session(user_id, token)
    response.set_cookie(
        config.COOKIE_NAME, token,
        max_age=config.SESSION_DAYS * 86400,
        httponly=True, secure=config.COOKIE_SECURE, samesite="lax", path="/",
    )


def _public_user(user: dict) -> dict:
    return {"username": user["username"], "email": user["email"]}


USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _safe_filename(name: Optional[str], default: str) -> str:
    name = os.path.basename((name or "").replace("\\", "/"))
    name = re.sub(r"[\x00-\x1f<>:\"|?*]", "", name).strip()
    return name[:120] or default


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    return data


def _extract(data: bytes, filename: str) -> str:
    try:
        text = extract_text_from_bytes(data, filename)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in this file.")
    if len(text) > config.MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"Document is too long (limit {config.MAX_TEXT_CHARS:,} characters).")
    return text


# ── Request models ──────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str = Field(max_length=64)
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class LoginRequest(BaseModel):
    username_or_email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class AnalyzeRequest(BaseModel):
    text: str = Field(max_length=config.MAX_TEXT_CHARS)
    plagiarism_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    ai_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    filename: Optional[str] = Field(default="Pasted_Text.txt", max_length=200)


# ── Pages & health ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open(config.TEMPLATES_DIR / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/engines")
def engines(user: dict = Depends(current_user)):
    """Which ML models / search providers are active (for the UI and debugging)."""
    from app.core import websearch
    from app.core.models import hub
    return {"models": hub.status(),
            "web_search": {"enabled": config.ENABLE_WEB_SEARCH, "providers": websearch.active_providers()}}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# ── Authentication ──────────────────────────────────────────────────
@app.post("/api/auth/register")
def register(req: RegisterRequest, request: Request, response: Response):
    if not register_limiter.hit(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many sign-up attempts. Please try again later.")
    username, email = req.username.strip(), req.email.strip().lower()
    if not USERNAME_RE.match(username):
        raise HTTPException(status_code=400, detail="Username must be 3-30 characters: letters, numbers, . _ -")
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    user = db.create_user(username, email, hash_password(req.password))
    if not user:
        raise HTTPException(status_code=400, detail="Username or email already in use.")
    _set_session_cookie(response, user["id"])
    return {"success": True, "user": _public_user(user)}


@app.post("/api/auth/login")
def login(req: LoginRequest, request: Request, response: Response):
    key = f"{_client_ip(request)}|{req.username_or_email.strip().lower()}"
    if login_failures.blocked(key):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in 15 minutes.")
    user = db.find_user_for_login(req.username_or_email)
    # Always run one hash verification so timing doesn't reveal whether the account exists.
    valid = verify_password(req.password, user["password_hash"] if user else DUMMY_HASH)
    if not (user and valid):
        login_failures.record(key)
        raise HTTPException(status_code=401, detail="Invalid username/email or password.")
    login_failures.reset(key)
    _set_session_cookie(response, user["id"])
    return {"success": True, "user": _public_user(user)}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(config.COOKIE_NAME)
    if token:
        db.delete_session(token)
    response.delete_cookie(config.COOKIE_NAME, path="/")
    return {"success": True}


@app.get("/api/auth/me")
def me(request: Request):
    token = request.cookies.get(config.COOKIE_NAME)
    user = db.get_user_by_session(token) if token else None
    return {"user": _public_user(user) if user else None}


# ── Public ──────────────────────────────────────────────────────────
@app.get("/api/samples")
def get_samples():
    return {"samples": SAMPLE_PRESETS}


# ── Reference corpus: built-in (shared, read-only) + private per user ─
@app.get("/api/reference-docs")
def get_reference_docs(user: dict = Depends(current_user)):
    builtin = [{
        "filename": d["filename"], "title": d["title"],
        "sentence_count": len(d["sentences"]), "word_count": len(d["content"].split()),
        "snippet": d["content"][:200] + ("..." if len(d["content"]) > 200 else ""),
    } for d in plagiarism_engine.reference_documents]
    mine = []
    for d in db.list_user_corpus(user["id"], with_content=True):
        mine.append({
            "id": d["id"], "filename": d["filename"], "title": d["title"],
            "sentence_count": len(split_sentences(d["content"])),
            "word_count": len(d["content"].split()),
            "snippet": d["content"][:200] + ("..." if len(d["content"]) > 200 else ""),
        })
    return {"builtin": builtin, "mine": mine}


@app.post("/api/reference-docs")
async def add_reference_doc(file: UploadFile = File(...), user: dict = Depends(current_user)):
    filename = _safe_filename(file.filename, "document.txt")
    text = _extract(await _read_upload(file), filename)
    usage = db.corpus_usage(user["id"])
    if usage["docs"] >= config.MAX_USER_CORPUS_DOCS:
        raise HTTPException(status_code=400, detail=f"You can store up to {config.MAX_USER_CORPUS_DOCS} private reference documents.")
    if usage["chars"] + len(text) > config.MAX_USER_CORPUS_CHARS:
        raise HTTPException(status_code=400, detail="Your private reference library is full. Delete a document first.")
    title = re.sub(r"\.(txt|pdf|docx)$", "", filename, flags=re.I).replace("_", " ").title()
    doc_id = db.add_user_corpus_doc(user["id"], filename, title, text)
    return {"success": True, "id": doc_id, "filename": filename, "sentence_count": len(split_sentences(text))}


@app.delete("/api/reference-docs/{doc_id}")
def delete_reference_doc(doc_id: int, user: dict = Depends(current_user)):
    if not db.delete_user_corpus_doc(doc_id, user["id"]):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"success": True}


# ── Detection ───────────────────────────────────────────────────────
@app.post("/api/upload-file")
async def upload_file_preview(file: UploadFile = File(...), user: dict = Depends(current_user)):
    filename = _safe_filename(file.filename, "document.txt")
    text = _extract(await _read_upload(file), filename)
    sentences = split_sentences(text)
    return {"filename": filename, "text": text, "stats": calculate_text_stats(text, sentences)}


@app.post("/api/analyze")
def analyze_document(req: AnalyzeRequest, user: dict = Depends(current_user)):
    if not analyze_limiter.hit(str(user["id"])):
        raise HTTPException(status_code=429, detail="You're running checks too quickly. Please wait a minute.")
    text = req.text.strip()
    sentences = split_sentences(text)
    if not sentences or len(text.split()) < 8:
        raise HTTPException(status_code=400, detail="Text is too short. Please provide at least 15-20 words for meaningful detection.")

    stats = calculate_text_stats(text, sentences)
    private_docs = db.list_user_corpus(user["id"], with_content=True)
    plag_res = plagiarism_engine.analyze(sentences, req.plagiarism_threshold, extra_docs=private_docs)
    ai_res = ai_engine.analyze(sentences, req.ai_threshold)
    report = build_audit_report(
        text=text, stats=stats, plag_result=plag_res, ai_result=ai_res,
        filename=_safe_filename(req.filename, "Submitted_Document.txt"),
    )
    try:
        db.save_scan_report(user["id"], report)
    except Exception:
        logger.exception("Failed to save scan report")
        raise HTTPException(status_code=500, detail="Analysis finished but could not be saved. Please try again.")
    return report


# ── History (owner only) ────────────────────────────────────────────
@app.get("/api/report/{report_id}")
def get_report(report_id: str, user: dict = Depends(current_user)):
    report = db.get_saved_report(report_id, user["id"])
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.get("/api/history")
def get_history(user: dict = Depends(current_user)):
    history = db.get_user_scan_history(user["id"], limit=50)
    return {"history": history, "count": len(history)}


@app.delete("/api/history/{report_id}")
def delete_history_item(report_id: str, user: dict = Depends(current_user)):
    if not db.delete_scan_report(report_id, user["id"]):
        raise HTTPException(status_code=404, detail="Scan record not found.")
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
