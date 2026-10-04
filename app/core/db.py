"""SQLite storage: users, login sessions, per-user scan history and per-user corpus.

Every query that touches user data is scoped by user_id, so one account can
never read, change or delete another account's rows.
"""
import datetime
import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from app import config
from app.core.security import token_digest


@contextmanager
def get_db():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now(delta: datetime.timedelta = datetime.timedelta(0)) -> str:
    """UTC timestamp as a sortable ISO string (avoids sqlite's deprecated datetime adapter)."""
    return (datetime.datetime.now(datetime.timezone.utc) + delta).strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                report_id TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                plagiarism_score REAL NOT NULL,
                ai_score REAL NOT NULL,
                total_words INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                report_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scan_user_date ON scan_history(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS user_corpus (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_corpus_user ON user_corpus(user_id);
            """
        )
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_now(),))


# ── Users ───────────────────────────────────────────────────────────
def create_user(username: str, email: str, password_hash: str) -> Optional[Dict[str, Any]]:
    try:
        with get_db() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username.strip(), email.strip().lower(), password_hash),
            )
            return {"id": cur.lastrowid, "username": username.strip(), "email": email.strip().lower()}
    except sqlite3.IntegrityError:
        return None


def find_user_for_login(identifier: str) -> Optional[Dict[str, Any]]:
    ident = identifier.strip()
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ? OR email = ?",
            (ident, ident.lower()),
        ).fetchone()
    return dict(row) if row else None


# ── Sessions ────────────────────────────────────────────────────────
def create_session(user_id: int, token: str) -> None:
    expires = _now(datetime.timedelta(days=config.SESSION_DAYS))
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_digest(token), user_id, expires),
        )


def get_user_by_session(token: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            """SELECT u.id, u.username, u.email FROM sessions s
               JOIN users u ON u.id = s.user_id
               WHERE s.token_hash = ? AND s.expires_at > ?""",
            (token_digest(token), _now()),
        ).fetchone()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_digest(token),))


# ── Scan history (always scoped to the owner) ───────────────────────
def save_scan_report(user_id: int, report: dict) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO scan_history
               (user_id, report_id, filename, plagiarism_score, ai_score, total_words, report_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                report["report_id"],
                report["filename"],
                report["summary"]["plagiarism_score"],
                report["summary"]["ai_score"],
                report["stats"]["word_count"],
                json.dumps(report),
            ),
        )


def get_user_scan_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT report_id, filename, plagiarism_score, ai_score, total_words, created_at
               FROM scan_history WHERE user_id = ?
               ORDER BY created_at DESC, id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_saved_report(report_id: str, user_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT report_json FROM scan_history WHERE report_id = ? AND user_id = ?",
            (report_id, user_id),
        ).fetchone()
    return json.loads(row["report_json"]) if row else None


def delete_scan_report(report_id: str, user_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM scan_history WHERE report_id = ? AND user_id = ?", (report_id, user_id)
        )
        return cur.rowcount > 0


# ── Private reference documents (per user) ──────────────────────────
def list_user_corpus(user_id: int, with_content: bool = False) -> List[Dict[str, Any]]:
    cols = "id, filename, title, created_at" + (", content" if with_content else "")
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT {cols} FROM user_corpus WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def corpus_usage(user_id: int) -> Dict[str, int]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(content)), 0) AS chars FROM user_corpus WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return {"docs": row["n"], "chars": row["chars"]}


def add_user_corpus_doc(user_id: int, filename: str, title: str, content: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO user_corpus (user_id, filename, title, content) VALUES (?, ?, ?, ?)",
            (user_id, filename, title, content),
        )
        return cur.lastrowid


def delete_user_corpus_doc(doc_id: int, user_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM user_corpus WHERE id = ? AND user_id = ?", (doc_id, user_id)
        )
        return cur.rowcount > 0
