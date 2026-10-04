"""Plagiarism Detection Engine for SemanticEye.

Signals, combined per sentence (best candidate wins):
  * lexical  - TF-IDF n-gram cosine + difflib sequence ratio (copying, patchwriting)
  * semantic - Sentence-BERT cosine similarity (paraphrase, when the model is available)
Sources: built-in corpus, the user's private documents, and the live web
(search engine -> downloaded pages -> sentence-level comparison).
"""
import os
import time
import re
import json
import difflib
import urllib.parse
import concurrent.futures
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app import config
from app.core.utils import split_sentences, highlight_word_diff
from app.core import websearch
from app.core.semantic import encoder as sbert


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


class PlagiarismEngine:
    def __init__(self, sample_docs_dir: str = "sample_docs"):
        self.sample_docs_dir = sample_docs_dir
        self.reference_documents = []
        self.load_reference_documents()

    def load_reference_documents(self) -> List[Dict[str, Any]]:
        """Load and cache reference documents from sample_docs directory."""
        docs = []
        if os.path.isdir(self.sample_docs_dir):
            for fname in sorted(os.listdir(self.sample_docs_dir)):
                fpath = os.path.join(self.sample_docs_dir, fname)
                if os.path.isfile(fpath) and fname.endswith((".txt", ".md")):
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if content.strip():
                                docs.append({
                                    "filename": fname,
                                    "title": fname.replace("_", " ").replace(".txt", "").title(),
                                    "content": content,
                                    "sentences": split_sentences(content)
                                })
                    except Exception as err:
                        print(f"Error loading {fname}: {err}")
        self.reference_documents = docs
        return docs

    # ------------------------------------------------------------------
    # Core matcher: targets vs. a list of reference sentences
    # ------------------------------------------------------------------
    def match(self, targets: List[str], refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """For every target sentence find the best reference sentence.

        refs: [{"text", "source", "title", ...}]. Returns one dict per target:
        {score, lexical, semantic, ref (dict|None)}.
        """
        empty = {"score": 0.0, "lexical": 0.0, "semantic": 0.0, "ref": None}
        results = [dict(empty) for _ in targets]
        if not refs or not targets:
            return results
        ref_texts = [r["text"] for r in refs]

        tfidf_sim = None
        try:
            vec = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", min_df=1)
            mat = vec.fit_transform(ref_texts + targets)
            tfidf_sim = cosine_similarity(mat[len(ref_texts):], mat[:len(ref_texts)])
        except Exception:
            tfidf_sim = None

        sem_sim = None
        if sbert.available():
            try:
                R, T = sbert.encode(ref_texts), sbert.encode(targets)
                if R is not None and T is not None:
                    sem_sim = T @ R.T
            except Exception:
                sem_sim = None

        for i, target in enumerate(targets):
            t_low = target.lower()
            cands = set()
            if tfidf_sim is not None:
                cands.update(int(j) for j in np.argsort(tfidf_sim[i])[::-1][:5])
            if sem_sim is not None:
                cands.update(int(j) for j in np.argsort(sem_sim[i])[::-1][:3])
            if not cands and len(refs) <= 3000:
                cands = set(range(len(refs)))  # tiny corpus without vectorizer

            best = None
            for j in cands:
                tf = float(tfidf_sim[i][j]) if tfidf_sim is not None else 0.0
                seq = difflib.SequenceMatcher(None, t_low, ref_texts[j].lower()).ratio()
                lex = (0.60 * tf + 0.40 * seq) if tfidf_sim is not None else seq
                sem = sbert.to_score(float(sem_sim[i][j])) if sem_sim is not None else 0.0
                score = max(lex, sem)
                if best is None or score > best[0]:
                    best = (score, lex, sem, j)
            if best:
                score, lex, sem, j = best
                results[i] = {"score": round(min(1.0, max(0.0, score)), 3), "lexical": round(lex, 3),
                              "semantic": round(sem, 3), "ref": refs[j]}
        return results

    # ------------------------------------------------------------------
    # Web-wide search
    # ------------------------------------------------------------------
    @staticmethod
    def _query_for(sentence: str) -> str:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", sentence) if w.lower() not in STOPWORDS]
        return " ".join(words[:9])

    def web_references(self, pending: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """Search the web for the given sentences and return the sentences of the
        pages found, as a flat reference list. Bounded by count and deadline."""
        deadline = time.time() + config.WEB_SEARCH_DEADLINE_SEC
        hits: Dict[str, Dict[str, Any]] = {}
        votes: Dict[str, int] = {}

        def run_search(item):
            return websearch.search(self._query_for(item[1]))

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=6)
        try:
            futs = [ex.submit(run_search, it) for it in pending]
            try:
                for f in concurrent.futures.as_completed(futs, timeout=max(1.0, deadline - time.time())):
                    try:
                        for h in f.result():
                            hits.setdefault(h["url"], h)
                            votes[h["url"]] = votes.get(h["url"], 0) + 1
                    except Exception:
                        pass
            except concurrent.futures.TimeoutError:
                pass

            urls = sorted(hits, key=lambda u: votes[u], reverse=True)[:config.MAX_WEB_PAGES]
            pages: Dict[str, str] = {}
            pfuts = {ex.submit(websearch.fetch_page_text, u): u for u in urls}
            try:
                for f in concurrent.futures.as_completed(pfuts, timeout=max(1.0, deadline - time.time())):
                    try:
                        pages[pfuts[f]] = f.result()
                    except Exception:
                        pass
            except concurrent.futures.TimeoutError:
                pass
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        refs = []
        for u in urls:
            h = hits[u]
            domain = urllib.parse.urlparse(u).netloc.replace("www.", "")
            title = f"{h['title'][:90]} ({domain})"
            text = pages.get(u) or ""
            sents = [s for s in split_sentences(text) if len(s.split()) >= 5][:600]
            if h.get("snippet") and len(h["snippet"].split()) >= 5:
                sents.append(h["snippet"])
            for s in sents:
                refs.append({"text": s, "source": u, "title": title, "url": u})
        return refs

    def analyze(self, sentences: List[str], threshold: float = 0.55,
                extra_docs: Optional[List[Dict[str, Any]]] = None,
                use_web: Optional[bool] = None) -> Dict[str, Any]:
        """Compare input sentences against local corpus, private docs and the web."""
        if not sentences:
            return {"overall_plagiarism_pct": 0.0, "flagged_count": 0, "total_sentences": 0,
                    "sources_summary": [], "sentence_details": [], "engine": self.engine_info()}

        # 1. local references (built-in + the requesting user's private documents)
        refs = []
        for doc in list(self.reference_documents) + [
                {"filename": d["filename"], "title": d["title"], "sentences": split_sentences(d["content"])}
                for d in (extra_docs or [])]:
            for s in doc["sentences"]:
                if len(s.split()) >= 3:
                    refs.append({"text": s, "source": doc["filename"], "title": doc["title"]})
        local = self.match(sentences, refs)

        # 2. web-wide search for sentences not already explained by the local corpus
        web = [None] * len(sentences)
        web_used = False
        do_web = config.ENABLE_WEB_SEARCH if use_web is None else use_web
        pending = [(i, s) for i, s in enumerate(sentences) if local[i]["score"] < 0.60 and len(s.split()) >= 6]
        if do_web and pending:
            pending.sort(key=lambda t: len(t[1]), reverse=True)
            pending = pending[:config.MAX_WEB_LOOKUPS]
            try:
                wrefs = self.web_references(pending)
                if wrefs:
                    web_used = True
                    wres = self.match([s for _, s in pending], wrefs)
                    for (i, _), r in zip(pending, wres):
                        web[i] = r
            except Exception:
                pass

        # 3. combine
        sentence_details, source_counts, source_urls = [], {}, {}
        for i, sentence in enumerate(sentences):
            best, origin = local[i], "local"
            if web[i] and web[i]["score"] > best["score"]:
                best, origin = web[i], "web"
            score, ref = best["score"], best["ref"]
            has_src = ref is not None and score >= 0.35
            src = ref["source"] if has_src else ""
            title = ref["title"] if has_src else ""
            matched = ref["text"] if has_src else ""
            url = (ref.get("url") or "") if has_src else ""
            flagged = score >= threshold

            if best["lexical"] >= 0.85:
                match_type = "Direct Copying / Exact Match"
            elif flagged and best["semantic"] >= threshold and best["lexical"] < threshold:
                match_type = "Paraphrased (semantic match)"
            elif flagged:
                match_type = "Paraphrased / High Match"
            elif score >= 0.35:
                match_type = "Moderate Overlap"
            else:
                match_type = "Original"

            diff_in, diff_src = highlight_word_diff(sentence, matched) if matched else (sentence, "")
            if flagged and src:
                source_counts[title] = source_counts.get(title, 0) + 1
                if url:
                    source_urls[title] = url
            sentence_details.append({
                "index": i, "sentence": sentence, "score": score, "percentage": round(score * 100, 1),
                "lexical_score": best["lexical"], "semantic_score": best["semantic"],
                "origin": origin if has_src else "", "url": url,
                "source": src, "source_title": title, "matched_sentence": matched,
                "flagged": flagged, "match_type": match_type,
                "diff_input_html": diff_in, "diff_source_html": diff_src,
            })

        flagged_count = sum(1 for d in sentence_details if d["flagged"])
        sources_summary = [{
            "source": t.lower().replace(" ", "_"), "title": t, "url": source_urls.get(t, ""),
            "match_count": c, "contribution_pct": round(c / max(1, flagged_count) * 100, 1),
        } for t, c in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)]

        info = self.engine_info()
        info["web_searched"] = web_used
        return {
            "overall_plagiarism_pct": round(flagged_count / len(sentences) * 100, 1),
            "flagged_count": flagged_count, "total_sentences": len(sentences),
            "sources_summary": sources_summary, "sentence_details": sentence_details, "engine": info,
        }

    @staticmethod
    def engine_info() -> Dict[str, Any]:
        return {
            "semantic": "Sentence-BERT" if sbert.available() else "unavailable (lexical only)",
            "web_providers": websearch.active_providers() if config.ENABLE_WEB_SEARCH else [],
        }
