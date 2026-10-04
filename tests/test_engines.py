"""Engine tests using stand-in models (no downloads, no network)."""
import math
import re

import numpy as np
import pytest

from app import config
from app.core import websearch
from app.core.ai_engine import AIDetectorEngine, ai_prob_from_output, chunk_sentences
from app.core.perplexity import PerplexityScorer, ppl_to_prob
from app.core.plagiarism_engine import PlagiarismEngine
from app.core.semantic import SemanticEncoder
from app.core.utils import split_sentences


# ── fakes ────────────────────────────────────────────────────────────
CONCEPTS = {"cars": ["car", "automobile", "vehicle"], "fast": ["fast", "quick", "rapid", "speedy"],
            "road": ["road", "highway", "street"], "cat": ["cat", "feline", "kitten"],
            "sleep": ["sleeps", "naps", "slumbers"], "sofa": ["sofa", "couch"]}
VOCAB = list(CONCEPTS)


class FakeSBERT:
    """Synonym-aware bag of concepts, L2 normalised: a stand-in for SBERT."""
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            v = np.zeros(len(VOCAB) + 1, dtype=np.float32)
            v[-1] = 0.05
            for w in re.findall(r"[a-z]+", t.lower()):
                for k, syns in CONCEPTS.items():
                    if w in syns:
                        v[VOCAB.index(k)] += 1
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


class FakeHub:
    def __init__(self, sbert=None, roberta=None, lm=None):
        import threading
        self._s, self._r, self._l = sbert, roberta, lm
        self.infer_lock = threading.Lock()
    def sbert(self): return self._s
    def roberta(self): return self._r
    def lm(self): return self._l


@pytest.fixture()
def sem(monkeypatch):
    enc = SemanticEncoder(FakeHub(sbert=FakeSBERT()))
    monkeypatch.setattr("app.core.plagiarism_engine.sbert", enc)
    return enc


def engine_with(sentences):
    e = PlagiarismEngine(sample_docs_dir="/nonexistent")
    e.reference_documents = [{"filename": "ref.txt", "title": "Ref", "content": " ".join(sentences), "sentences": sentences}]
    return e


# ── semantic / paraphrase ────────────────────────────────────────────
def test_semantic_flags_paraphrase_that_lexical_misses(sem):
    ref = ["The fast car drove along the road."]
    para = "A speedy automobile travelled down the highway."
    unrelated = "The kitten slumbers on the couch."
    res = engine_with(ref).analyze([para, unrelated], threshold=0.55, use_web=False)
    d = res["sentence_details"]
    assert d[0]["flagged"] and d[0]["match_type"] == "Paraphrased (semantic match)"
    assert d[0]["lexical_score"] < 0.55 <= d[0]["semantic_score"]
    assert not d[1]["flagged"]
    assert res["engine"]["semantic"] == "Sentence-BERT"


def test_without_sbert_falls_back_to_lexical():
    ref = ["The fast car drove along the road."]
    res = engine_with(ref).analyze(["The fast car drove along the road.", "A speedy automobile travelled down the highway."],
                                   use_web=False)
    d = res["sentence_details"]
    assert d[0]["flagged"] and d[0]["match_type"].startswith("Direct")
    assert not d[1]["flagged"] and d[1]["semantic_score"] == 0.0


def test_embedding_cache_reuses_vectors():
    calls = []
    class Counting(FakeSBERT):
        def encode(self, texts, **kw):
            calls.append(len(texts)); return super().encode(texts)
    enc = SemanticEncoder(FakeHub(sbert=Counting()))
    enc.encode(["a car", "a cat"]); enc.encode(["a car", "a cat", "a sofa"])
    assert calls == [2, 1]


# ── web search ───────────────────────────────────────────────────────
def test_web_search_finds_source(monkeypatch, sem):
    page = ("Intro text that is long enough to be a sentence here. "
            "The fast car drove along the road near the old harbour town. "
            "Another unrelated sentence about gardening in the spring season.")
    monkeypatch.setattr(websearch, "search", lambda q, n=None: [
        {"url": "https://example.org/a", "title": "Example page", "snippet": "", "provider": "test"}])
    monkeypatch.setattr(websearch, "fetch_page_text", lambda u: page)
    e = PlagiarismEngine(sample_docs_dir="/nonexistent")
    res = e.analyze(["The fast car drove along the road near the old harbour town."], use_web=True)
    d = res["sentence_details"][0]
    assert d["flagged"] and d["origin"] == "web" and d["url"] == "https://example.org/a"
    assert res["sources_summary"][0]["url"] == "https://example.org/a"
    assert res["engine"]["web_searched"] is True


def test_web_failure_is_not_fatal(monkeypatch):
    def boom(q, n=None): raise RuntimeError("network down")
    monkeypatch.setattr(websearch, "search", boom)
    res = PlagiarismEngine(sample_docs_dir="/nonexistent").analyze(
        ["This sentence has plenty of distinct words to search for online."], use_web=True)
    assert res["flagged_count"] == 0


def test_ssrf_guard_blocks_private_hosts():
    for u in ("http://127.0.0.1/x", "http://localhost/", "http://169.254.169.254/latest", "file:///etc/passwd",
              "http://10.0.0.5/", "ftp://example.com/"):
        assert not websearch.is_safe_url(u)


def test_html_to_text_drops_chrome():
    html = ("<html><nav>Home About Contact us today please</nav><script>var x=1;</script>"
            "<p>This paragraph has more than six words in it.</p><footer>copyright notice all rights reserved here</footer></html>")
    assert websearch.html_to_text(html) == "This paragraph has more than six words in it."


def test_provider_selection(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_PROVIDER", "auto")
    monkeypatch.setattr(config, "BRAVE_API_KEY", "")
    monkeypatch.setattr(config, "GOOGLE_CSE_KEY", ""); monkeypatch.setattr(config, "SERPAPI_KEY", "")
    assert websearch.active_providers() == ["duckduckgo", "wikipedia"]
    monkeypatch.setattr(config, "BRAVE_API_KEY", "k")
    assert websearch.active_providers() == ["brave", "wikipedia"]


# ── perplexity ───────────────────────────────────────────────────────
class FakePPL(PerplexityScorer):
    """Token = word; human-looking text gets low log-prob, AI-looking text high."""
    def __init__(self, logp): self.logp = logp
    def available(self): return True
    def _token_logprobs(self, text):
        offs = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return offs, [self.logp] * len(offs)


def test_perplexity_maps_to_probability():
    assert ppl_to_prob(10) > 0.9 and ppl_to_prob(200) < 0.1
    assert ppl_to_prob(config.PPL_MIDPOINT) == pytest.approx(0.5)
    sents = ["The cat sat on the mat today.", "It was a quiet and calm afternoon outside."]
    low = FakePPL(-math.log(10)).analyze(sents)       # perplexity 10
    high = FakePPL(-math.log(300)).analyze(sents)     # perplexity 300
    assert low["available"] and low["mean_perplexity"] == pytest.approx(10, rel=0.01)
    assert low["ai_probability"] > 0.9 > 0.1 > high["ai_probability"]
    assert len(low["sentence_perplexities"]) == 2 and all(p for p in low["sentence_perplexities"])


def test_perplexity_unavailable_without_lm():
    r = PerplexityScorer(FakeHub()).analyze(["Some text here to score."])
    assert r["available"] is False


# ── roberta + ensemble ───────────────────────────────────────────────
def test_label_mapping():
    assert ai_prob_from_output([{"label": "ChatGPT", "score": 0.9}, {"label": "Human", "score": 0.1}]) == 0.9
    assert ai_prob_from_output([{"label": "Human", "score": 0.8}, {"label": "ChatGPT", "score": 0.2}]) == 0.2
    assert ai_prob_from_output([{"label": "LABEL_1", "score": 0.7}, {"label": "LABEL_0", "score": 0.3}]) == 0.7
    assert ai_prob_from_output([{"label": "Human", "score": 0.8}]) == pytest.approx(0.2)


def test_chunking_respects_window():
    sents = ["word " * 100] * 5
    chunks = chunk_sentences(sents, max_words=180)
    assert [i for c in chunks for i in c] == list(range(5)) and all(len(c) <= 1 for c in chunks)


class FakeCLF:
    def __init__(self, p): self.p = p
    def __call__(self, texts, batch_size=8):
        return [[{"label": "ChatGPT", "score": self.p}, {"label": "Human", "score": 1 - self.p}] for _ in texts]


TEXT = ("The committee reviewed the proposal carefully. After a long discussion about the budget, they agreed. "
        "Nobody expected the vote to be unanimous, but it was. Funding begins next month.")


def test_ensemble_uses_roberta_and_perplexity():
    sents = split_sentences(TEXT)
    ai = AIDetectorEngine(hub_=FakeHub(roberta=FakeCLF(0.97)), perplexity=FakePPL(-math.log(8)))
    ai.calibration = None
    hi = ai.analyze(sents)
    human = AIDetectorEngine(hub_=FakeHub(roberta=FakeCLF(0.03)), perplexity=FakePPL(-math.log(150)))
    human.calibration = None
    lo = human.analyze(sents)
    assert hi["overall_ai_pct"] > 70 > 25 > lo["overall_ai_pct"]
    assert "RoBERTa" in hi["method"] and "perplexity" in hi["method"]
    assert hi["components"]["roberta"] == pytest.approx(0.97, abs=1e-3)
    assert all(s["flagged"] for s in hi["sentence_scores"]) and not any(s["flagged"] for s in lo["sentence_scores"])


def test_degrades_to_statistical_engine():
    ai = AIDetectorEngine(hub_=FakeHub(), perplexity=PerplexityScorer(FakeHub()))
    r = ai.analyze(split_sentences(TEXT))
    assert r["roberta"]["available"] is False and r["perplexity"]["available"] is False
    assert "Statistical" in r["method"] and 0 <= r["overall_ai_pct"] <= 100


def test_calibration_changes_final_score():
    sents = split_sentences(TEXT)
    ai = AIDetectorEngine(hub_=FakeHub(roberta=FakeCLF(0.5)), perplexity=FakePPL(-math.log(35)))
    ai.calibration = {"features": ["roberta", "perplexity", "stat"], "coef": [0.0, 0.0, 0.0], "intercept": 4.0}
    assert ai.analyze(sents)["overall_ai_pct"] > 95
