import os
import sys
from pathlib import Path

os.environ.setdefault("APP_ENV", "production")
os.environ.setdefault("COOKIE_SECURE", "false")      # TestClient talks plain http
os.environ.setdefault("ENABLE_WEB_SEARCH", "false")  # tests must not hit the internet
for _m in ("ENABLE_SBERT", "ENABLE_ROBERTA", "ENABLE_PERPLEXITY"):
    os.environ.setdefault(_m, "false")             # tests must not download ML models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import server
from app import config


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    for limiter in (server.login_failures, server.register_limiter, server.analyze_limiter):
        limiter._events.clear()
    with TestClient(server.app) as c:
        yield c


def make_user(app_client_factory, name):
    c = app_client_factory()
    r = c.post("/api/auth/register", json={"username": name, "email": f"{name}@example.com", "password": "correct-horse-1"})
    assert r.status_code == 200, r.text
    return c


@pytest.fixture()
def two_users(client):
    """Two separate browsers (cookie jars) against the same app."""
    a = TestClient(server.app)
    b = TestClient(server.app)
    for c, n in ((a, "alice"), (b, "bob")):
        r = c.post("/api/auth/register", json={"username": n, "email": f"{n}@example.com", "password": "correct-horse-1"})
        assert r.status_code == 200, r.text
    return a, b
