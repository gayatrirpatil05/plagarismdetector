"""Web-wide source search.

Providers (first available wins in "auto" mode, Wikipedia is always added):
  Brave Search API  (BRAVE_API_KEY)
  Google Programmable Search (GOOGLE_CSE_KEY + GOOGLE_CSE_CX)
  SerpAPI           (SERPAPI_KEY)
  DuckDuckGo HTML   (no key; best-effort, may be rate limited)
Result pages are downloaded (SSRF-safe, size/time capped) and reduced to text
so the plagiarism engine can compare sentences against the real page content.
"""
import ipaddress
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from app import config

UA = "Mozilla/5.0 (compatible; SemanticEye-PlagiarismChecker/3.0)"
_SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe"}
_BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "td", "th", "blockquote", "div", "br", "tr", "section", "article"}


# ── safe fetching ────────────────────────────────────────────────────
def _host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return bool(infos)


def is_safe_url(url: str) -> bool:
    try:
        p = urllib.parse.urlparse(url)
    except ValueError:
        return False
    return p.scheme in ("http", "https") and bool(p.hostname) and _host_is_public(p.hostname)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not is_safe_url(newurl):
            raise urllib.error.URLError("redirect to disallowed address")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirect)


def _http_get(url: str, timeout: float, max_bytes: int, headers: Optional[Dict[str, str]] = None) -> Optional[tuple]:
    if not is_safe_url(url):
        return None
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en", **(headers or {})})
    try:
        with _opener.open(req, timeout=timeout) as r:
            ctype = r.headers.get("Content-Type", "")
            data = r.read(max_bytes + 1)[:max_bytes]
            return data, ctype
    except Exception:
        return None


# ── HTML → text ──────────────────────────────────────────────────────
class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    ex = _TextExtractor()
    try:
        ex.feed(html)
    except Exception:
        pass
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in "".join(ex.parts).split("\n")]
    # keep prose-like lines only (drops menus, buttons, cookie banners)
    return "\n".join(ln for ln in lines if len(ln.split()) >= 6)


_page_cache: Dict[str, tuple] = {}
_cache_lock = threading.Lock()
CACHE_TTL = 3600


def fetch_page_text(url: str) -> str:
    now = time.time()
    with _cache_lock:
        hit = _page_cache.get(url)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]
    res = _http_get(url, config.WEB_FETCH_TIMEOUT, config.WEB_PAGE_MAX_BYTES)
    text = ""
    if res:
        data, ctype = res
        if "html" in ctype.lower() or "text" in ctype.lower() or not ctype:
            m = re.search(r"charset=([\w-]+)", ctype)
            try:
                text = html_to_text(data.decode(m.group(1) if m else "utf-8", errors="ignore"))
            except LookupError:
                text = html_to_text(data.decode("utf-8", errors="ignore"))
    with _cache_lock:
        if len(_page_cache) > 300:
            _page_cache.clear()
        _page_cache[url] = (now, text)
    return text


# ── providers ────────────────────────────────────────────────────────
def _json_get(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[dict]:
    res = _http_get(url, 6.0, 2_000_000, headers)
    if not res:
        return None
    try:
        return json.loads(res[0].decode("utf-8", "ignore"))
    except ValueError:
        return None


def _hit(url: str, title: str, snippet: str, provider: str) -> Dict[str, Any]:
    return {"url": url, "title": re.sub(r"<[^>]+>", "", title or url).strip(),
            "snippet": re.sub(r"<[^>]+>", "", snippet or "").strip(), "provider": provider}


def _brave(q: str, n: int) -> List[dict]:
    d = _json_get("https://api.search.brave.com/res/v1/web/search?count=%d&q=%s" % (n, urllib.parse.quote(q)),
                  {"X-Subscription-Token": config.BRAVE_API_KEY, "Accept": "application/json"})
    return [_hit(r["url"], r.get("title"), r.get("description"), "brave")
            for r in ((d or {}).get("web", {}).get("results", []) or [])[:n] if r.get("url")]


def _google(q: str, n: int) -> List[dict]:
    d = _json_get("https://www.googleapis.com/customsearch/v1?key=%s&cx=%s&num=%d&q=%s" % (
        urllib.parse.quote(config.GOOGLE_CSE_KEY), urllib.parse.quote(config.GOOGLE_CSE_CX), min(n, 10),
        urllib.parse.quote(q)))
    return [_hit(r["link"], r.get("title"), r.get("snippet"), "google") for r in (d or {}).get("items", [])[:n]]


def _serpapi(q: str, n: int) -> List[dict]:
    d = _json_get("https://serpapi.com/search.json?engine=google&num=%d&api_key=%s&q=%s" % (
        n, urllib.parse.quote(config.SERPAPI_KEY), urllib.parse.quote(q)))
    return [_hit(r["link"], r.get("title"), r.get("snippet"), "serpapi")
            for r in (d or {}).get("organic_results", [])[:n] if r.get("link")]


def _duckduckgo(q: str, n: int) -> List[dict]:
    res = _http_get("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q), 6.0, 600_000)
    if not res:
        return []
    html = res[0].decode("utf-8", "ignore")
    out = []
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        href = m.group(1)
        if "uddg=" in href:
            href = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [""])[0])
        elif href.startswith("//"):
            href = "https:" + href
        if href.startswith("http"):
            out.append(_hit(href, m.group(2), "", "duckduckgo"))
        if len(out) >= n:
            break
    return out


def _wikipedia(q: str, n: int) -> List[dict]:
    d = _json_get("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=%d&srsearch=%s"
                  % (min(n, 3), urllib.parse.quote(q)))
    return [_hit("https://en.wikipedia.org/wiki/" + urllib.parse.quote(r["title"].replace(" ", "_")),
                 "Wikipedia: " + r["title"], r.get("snippet"), "wikipedia")
            for r in (d or {}).get("query", {}).get("search", [])]


def active_providers() -> List[str]:
    p = config.SEARCH_PROVIDER
    if p != "auto":
        return [p] if p == "wikipedia" else [p, "wikipedia"]
    chain = []
    if config.BRAVE_API_KEY:
        chain.append("brave")
    if config.GOOGLE_CSE_KEY and config.GOOGLE_CSE_CX:
        chain.append("google")
    if config.SERPAPI_KEY:
        chain.append("serpapi")
    if not chain:
        chain.append("duckduckgo")
    return chain[:1] + ["wikipedia"]


_PROVIDERS = {"brave": _brave, "google": _google, "serpapi": _serpapi,
              "duckduckgo": _duckduckgo, "wikipedia": _wikipedia}


def search(query: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
    n = n or config.WEB_RESULTS_PER_QUERY
    seen, out = set(), []
    for name in active_providers():
        fn = _PROVIDERS.get(name)
        if not fn:
            continue
        try:
            for h in fn(query, n):
                if h["url"] not in seen:
                    seen.add(h["url"])
                    out.append(h)
        except Exception:
            continue
    return out
