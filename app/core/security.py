"""Password hashing, session tokens and simple in-memory rate limiting."""
import base64
import hashlib
import hmac
import os
import secrets
import threading
import time
from collections import defaultdict, deque

# ── Passwords: scrypt with a random per-user salt (standard library) ──
_N, _R, _P, _LEN = 2 ** 14, 8, 1, 32


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=_LEN)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=len(expected))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# Used so that "unknown user" and "wrong password" take the same time.
DUMMY_HASH = hash_password(secrets.token_hex(16))


# ── Session tokens: random value in the cookie, only its hash is stored ──
def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ── Rate limiting (per process; run a single worker or move to Redis) ──
class RateLimiter:
    def __init__(self, max_events: int, window_sec: int):
        self.max_events = max_events
        self.window = window_sec
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, q: deque, now: float) -> None:
        while q and now - q[0] > self.window:
            q.popleft()

    def blocked(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            self._prune(q, now)
            if not q:
                self._events.pop(key, None)
                return False
            return len(q) >= self.max_events

    def record(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            self._prune(q, now)
            q.append(now)
            if len(self._events) > 20_000:  # keep memory bounded
                for k in list(self._events)[:5_000]:
                    self._events.pop(k, None)

    def hit(self, key: str) -> bool:
        """Record an event; return False if the limit has been exceeded."""
        if self.blocked(key):
            return False
        self.record(key)
        return True

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)
