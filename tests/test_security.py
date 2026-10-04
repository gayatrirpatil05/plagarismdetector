import io

from app import config
from app.core.utils import highlight_word_diff

TEXT = (
    "Climate change refers to long-term shifts in temperatures and weather patterns. "
    "Burning fossil fuels like coal, oil and gas is the main driver of climate change because it "
    "releases greenhouse gases into the atmosphere."
)


def test_no_default_accounts(client):
    r = client.post("/api/auth/login", json={"username_or_email": "demouser", "password": "demo123"})
    assert r.status_code == 401


def test_anonymous_cannot_use_protected_endpoints(client):
    assert client.post("/api/analyze", json={"text": TEXT}).status_code == 401
    assert client.get("/api/history").status_code == 401
    assert client.get("/api/history?user_id=1").status_code == 401   # old IDOR vector
    assert client.get("/api/report/SE-ANYTHING").status_code == 401
    assert client.delete("/api/history/SE-ANYTHING").status_code == 401
    assert client.get("/api/reference-docs").status_code == 401
    assert client.post("/api/upload-file", files={"file": ("a.txt", b"hello")}).status_code == 401


def test_register_validation_and_session(client):
    bad = client.post("/api/auth/register", json={"username": "ab", "email": "x@y.com", "password": "longenough1"})
    assert bad.status_code == 400
    short = client.post("/api/auth/register", json={"username": "carol", "email": "c@y.com", "password": "short"})
    assert short.status_code == 400
    ok = client.post("/api/auth/register", json={"username": "carol", "email": "c@y.com", "password": "longenough1"})
    assert ok.status_code == 200
    assert "password" not in ok.text and "id" not in ok.json()["user"]
    assert client.get("/api/auth/me").json()["user"]["username"] == "carol"
    cookie = ok.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie


def test_duplicate_username_case_insensitive(client):
    client.post("/api/auth/register", json={"username": "Dave", "email": "d@y.com", "password": "longenough1"})
    r = client.post("/api/auth/register", json={"username": "dave", "email": "other@y.com", "password": "longenough1"})
    assert r.status_code == 400


def test_history_is_private(two_users):
    a, b = two_users
    ra = a.post("/api/analyze", json={"text": TEXT, "filename": "a.txt"})
    assert ra.status_code == 200
    rid = ra.json()["report_id"]

    assert [h["report_id"] for h in a.get("/api/history").json()["history"]] == [rid]
    assert b.get("/api/history").json()["history"] == []
    assert b.get(f"/api/report/{rid}").status_code == 404
    assert b.delete(f"/api/history/{rid}").status_code == 404
    assert a.get(f"/api/report/{rid}").status_code == 200  # still there


def test_same_text_gets_distinct_reports_per_user(two_users):
    a, b = two_users
    ra = a.post("/api/analyze", json={"text": TEXT}).json()
    rb = b.post("/api/analyze", json={"text": TEXT}).json()
    assert ra["report_id"] != rb["report_id"]
    assert len(a.get("/api/history").json()["history"]) == 1
    assert len(b.get("/api/history").json()["history"]) == 1


def test_private_corpus_isolated(two_users):
    a, b = two_users
    secret = (
        "The quarterly zebra migration protocol requires seventeen purple lanterns before dawn. "
        "Nobody outside the committee should ever read this peculiar sentence about lanterns."
    )
    up = a.post("/api/reference-docs", files={"file": ("secret notes.txt", secret.encode())})
    assert up.status_code == 200
    doc_id = up.json()["id"]

    assert len(a.get("/api/reference-docs").json()["mine"]) == 1
    assert b.get("/api/reference-docs").json()["mine"] == []
    assert b.delete(f"/api/reference-docs/{doc_id}").status_code == 404

    probe = "The quarterly zebra migration protocol requires seventeen purple lanterns before dawn."
    pa = a.post("/api/analyze", json={"text": probe + " Another filler sentence appears here for length."}).json()
    pb = b.post("/api/analyze", json={"text": probe + " Another filler sentence appears here for length."}).json()
    assert pa["summary"]["plagiarism_score"] > 0          # owner's own library is used
    assert pb["summary"]["plagiarism_score"] == 0         # nobody else's is


def test_login_logout_and_bruteforce_lockout(client):
    client.post("/api/auth/register", json={"username": "erin", "email": "e@y.com", "password": "longenough1"})
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").json()["user"] is None
    assert client.post("/api/auth/login", json={"username_or_email": "erin", "password": "longenough1"}).status_code == 200
    client.post("/api/auth/logout")
    for _ in range(5):
        assert client.post("/api/auth/login", json={"username_or_email": "erin", "password": "wrong"}).status_code == 401
    locked = client.post("/api/auth/login", json={"username_or_email": "erin", "password": "longenough1"})
    assert locked.status_code == 429


def test_logout_invalidates_old_session_cookie(client):
    client.post("/api/auth/register", json={"username": "frank", "email": "f@y.com", "password": "longenough1"})
    stolen = client.cookies.get(config.COOKIE_NAME)
    client.post("/api/auth/logout")
    client.cookies.set(config.COOKIE_NAME, stolen)
    assert client.get("/api/auth/me").json()["user"] is None


def test_cross_site_post_blocked(client):
    r = client.post("/api/auth/login", json={"username_or_email": "x", "password": "y"},
                    headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_upload_filename_cannot_escape(two_users, tmp_path):
    a, _ = two_users
    r = a.post("/api/reference-docs", files={"file": ("../../evil.txt", b"A harmless sentence about nothing in particular here.")})
    assert r.status_code == 200
    assert r.json()["filename"] == "evil.txt"
    assert not (tmp_path.parent.parent / "evil.txt").exists()


def test_bad_uploads_rejected(two_users, monkeypatch):
    a, _ = two_users
    assert a.post("/api/upload-file", files={"file": ("x.exe", b"MZ....")}).status_code == 400
    assert a.post("/api/upload-file", files={"file": ("fake.pdf", b"not a pdf")}).status_code == 400
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    assert a.post("/api/upload-file", files={"file": ("big.txt", b"a" * 500)}).status_code == 413


def test_text_length_and_threshold_limits(two_users):
    a, _ = two_users
    assert a.post("/api/analyze", json={"text": "x" * (config.MAX_TEXT_CHARS + 1)}).status_code == 422
    assert a.post("/api/analyze", json={"text": TEXT, "plagiarism_threshold": 5}).status_code == 422
    assert a.post("/api/analyze", json={"text": "too short"}).status_code == 400


def test_diff_html_is_escaped():
    h1, h2 = highlight_word_diff("hello <img src=x onerror=alert(1)> world", "hello <script>alert(2)</script> world")
    assert "<img" not in h1 and "<script" not in h2
    assert "&lt;img" in h1


def test_headers_and_docs_hidden_in_production(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/healthz").json() == {"status": "ok"}
