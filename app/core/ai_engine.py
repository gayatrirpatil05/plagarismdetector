"""AI Text Detection Engine for SemanticEye.
Multi-signal statistical and linguistic analyzer measuring:
- Sentence Length Burstiness (coefficient of variation)
- Perplexity / Transition Entropy
- Lexical Diversity (Type-Token Ratio)
- Synthetic AI Discourse Patterns (typical LLM vocabulary & structural markers)
- RoBERTa transformer classifier (fine-tuned on human vs ChatGPT text)
- Real perplexity from a causal language model (GPT-2)
- Logistic-regression calibrated ensemble (optional, see evaluation/)
"""
import math
import re
import json
import statistics
from typing import List, Dict, Any, Optional

from app import config
from app.core.models import hub
from app.core.perplexity import scorer as ppl_scorer, PerplexityScorer

# Prominent AI transitional and stylistic discourse markers
AI_DISCOURSE_MARKERS = [
    r"\bin conclusion\b", r"\bfurthermore\b", r"\bmoreover\b", r"\bit is crucial to\b",
    r"\bit is important to remember\b", r"\bit is worth noting\b", r"\bdelve into\b",
    r"\btestament to\b", r"\brich tapestry\b", r"\bpivotal role\b", r"\bbeacon of\b",
    r"\bnavigating the complexities\b", r"\bin summary\b", r"\bplay a key role\b",
    r"\brapidly evolving landscape\b", r"\bseamless integration\b", r"\bprofound impact\b",
    r"\bmultifaceted\b", r"\bcomprehensive understanding\b", r"\bharnessing the power\b",
    r"\bstands as a\b", r"\bunderscores the\b", r"\bvital component\b"
]

def ai_prob_from_output(outputs: List[Dict[str, Any]]) -> float:
    """Convert a text-classification pipeline output (all labels) to P(AI)."""
    ai_words = ("chatgpt", "ai", "fake", "generated", "machine", "gpt")
    for o in outputs:
        lab = str(o["label"]).lower()
        if lab in ("label_1", "1") or any(w == lab or w in lab for w in ai_words) and "human" not in lab:
            return float(o["score"])
    # no recognisable AI label: assume the other label is "human"
    for o in outputs:
        lab = str(o["label"]).lower()
        if "human" in lab or lab in ("label_0", "0", "real"):
            return 1.0 - float(o["score"])
    return 0.5


def chunk_sentences(sentences: List[str], max_words: int = 180) -> List[List[int]]:
    """Group consecutive sentence indices into chunks that fit the 512-token window."""
    chunks, cur, n = [], [], 0
    for i, s in enumerate(sentences):
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += w
    if cur:
        chunks.append(cur)
    return chunks


class AIDetectorEngine:
    def __init__(self, enable_hf_model: Optional[bool] = None, hub_=hub, perplexity: Optional[PerplexityScorer] = None):
        # enable_hf_model kept for backwards compatibility; None => follow config
        self.hub = hub_
        self.perplexity = perplexity or ppl_scorer
        self.calibration = self._load_calibration()

    @staticmethod
    def _load_calibration() -> Optional[Dict[str, Any]]:
        try:
            with open(config.CALIBRATION_PATH, "r", encoding="utf-8") as f:
                cal = json.load(f)
            if cal.get("features") == ["roberta", "perplexity", "stat"] and len(cal.get("coef", [])) == 3:
                return cal
        except Exception:
            pass
        return None

    def roberta_scores(self, sentences: List[str]) -> Optional[Dict[str, Any]]:
        """P(AI) per chunk from the RoBERTa classifier, mapped back onto sentences."""
        clf = self.hub.roberta()
        if clf is None or not sentences:
            return None
        chunks = chunk_sentences(sentences)
        try:
            texts = [" ".join(sentences[i] for i in c) for c in chunks]
            with self.hub.infer_lock:
                outs = clf(texts, batch_size=8)
            probs = [ai_prob_from_output(o if isinstance(o, list) else [o]) for o in outs]
        except Exception as exc:  # noqa: BLE001
            print(f"RoBERTa inference failed: {exc}")
            return None
        per_sentence = [0.5] * len(sentences)
        weights = []
        for c, p in zip(chunks, probs):
            for i in c:
                per_sentence[i] = p
            weights.append(sum(len(sentences[i].split()) for i in c))
        total = max(1, sum(weights))
        doc = sum(p * w for p, w in zip(probs, weights)) / total
        return {"available": True, "ai_probability": round(doc, 4), "chunk_probabilities": [round(p, 4) for p in probs],
                "per_sentence": per_sentence}

    def calculate_burstiness(self, sentences: List[str]) -> Dict[str, Any]:
        """Measure variation in sentence lengths. Low variation = robotic/AI, High = human."""
        lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences if s.strip()]
        if len(lengths) < 2:
            return {"burstiness_score": 0.15, "coefficient_of_variation": 0.5, "std_dev": 0.0, "mean_length": lengths[0] if lengths else 0}
        
        mean_len = statistics.mean(lengths)
        if mean_len == 0:
            return {"burstiness_score": 0.15, "coefficient_of_variation": 0.5, "std_dev": 0.0, "mean_length": 0}
            
        std_dev = statistics.pstdev(lengths)
        cv = std_dev / mean_len  # Coefficient of variation
        
        # High CV (> 0.4) means high sentence length variation (human writing).
        # Low CV (< 0.25) means monotonous, uniform sentence length (robotic/AI).
        burstiness_ai_prob = max(0.05, min(0.90, 0.85 - (cv * 1.5)))
        return {
            "burstiness_score": round(burstiness_ai_prob, 3),
            "coefficient_of_variation": round(cv, 3),
            "std_dev": round(std_dev, 2),
            "mean_length": round(mean_len, 1)
        }

    def calculate_lexical_diversity(self, text: str) -> Dict[str, float]:
        """Type-Token Ratio and hapax legomena distribution."""
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{2,}\b", text)]
        total = len(words)
        if total < 10:
            return {"ttr": 0.5, "entropy": 3.0, "vocab_size": len(set(words))}
        
        unique = set(words)
        ttr = len(unique) / total
        
        freqs = {}
        for w in words:
            freqs[w] = freqs.get(w, 0) + 1
        
        entropy = 0.0
        for count in freqs.values():
            p = count / total
            entropy -= p * math.log2(p)
            
        return {
            "ttr": round(ttr, 3),
            "entropy": round(entropy, 2),
            "vocab_size": len(unique)
        }

    def detect_discourse_patterns(self, text: str) -> Dict[str, Any]:
        """Find occurrences of stereotypical AI transition formulas."""
        lower_text = text.lower()
        found = []
        for pat in AI_DISCOURSE_MARKERS:
            matches = re.findall(pat, lower_text)
            if matches:
                found.extend(matches)
        
        words = len(text.split())
        density = (len(found) / max(1, words)) * 100
        return {
            "matches_count": len(found),
            "markers": list(set(found)),
            "marker_density": round(density, 2)
        }

    def analyze_sentence(self, sentence: str, doc_cv: float) -> Dict[str, Any]:
        """Score an individual sentence for synthetic AI characteristics."""
        words = re.findall(r"\b\w+\b", sentence)
        word_count = len(words)
        
        if word_count < 5:
            return {
                "score": 0.10,
                "percentage": 10.0,
                "flagged": False,
                "reason": "Short natural sentence",
                "markers_found": []
            }

        # Check for specific AI markers in this sentence
        lower_s = sentence.lower()
        matched_markers = [m for m in AI_DISCOURSE_MARKERS if re.search(m, lower_s)]
        marker_score = min(0.75, len(matched_markers) * 0.35)

        # Monotonous sentence cadence factor (only triggers if doc_cv is unusually low, < 0.25)
        cv_ai_factor = max(0.0, (0.30 - doc_cv) * 0.8) if doc_cv < 0.30 else 0.0
        
        # Base probability starts low (0.08) - human writing assumed innocent until proven guilty
        sentence_score = 0.08 + cv_ai_factor + marker_score
        sentence_score = max(0.05, min(0.98, sentence_score))

        # Determine explanation
        reasons = []
        if matched_markers:
            reasons.append(f"AI formulaic transition detected ({', '.join(matched_markers)})")
        if cv_ai_factor > 0.15:
            reasons.append("Monotonous/robotic sentence length cadence")
        if not reasons:
            reasons.append("Natural human sentence structure")

        return {
            "score": round(sentence_score, 3),
            "percentage": round(sentence_score * 100, 1),
            "flagged": sentence_score >= 0.65,
            "reason": "; ".join(reasons),
            "markers_found": matched_markers
        }

    @staticmethod
    def _combine(stat: float, rob: Optional[float], ppl: Optional[float]) -> float:
        if rob is not None and ppl is not None:
            return 0.50 * rob + 0.25 * ppl + 0.25 * stat
        if rob is not None:
            return 0.65 * rob + 0.35 * stat
        if ppl is not None:
            return 0.50 * ppl + 0.50 * stat
        return stat

    def _final_prob(self, stat: float, rob: Optional[float], ppl: Optional[float]) -> float:
        cal = self.calibration
        if cal and rob is not None and ppl is not None:
            z = cal["intercept"] + sum(c * x for c, x in zip(cal["coef"], (rob, ppl, stat)))
            return 1.0 / (1.0 + math.exp(-z))
        return self._combine(stat, rob, ppl)

    def stat_composite(self, sentences: List[str], burst_info, lexical, discourse, sentence_evals) -> float:
        avg_sentence_score = sum(d["score"] for d in sentence_evals) / len(sentence_evals)
        discourse_factor = min(1.0, discourse.get("matches_count", 0) * 0.25)
        burst_factor = burst_info.get("burstiness_score", 0.15)
        ttr_term = 0.3 if lexical.get("ttr", 0.5) < 0.35 else 0.05
        if discourse.get("matches_count", 0) == 0:
            return 0.55 * avg_sentence_score + 0.35 * burst_factor + 0.10 * ttr_term
        return 0.35 * avg_sentence_score + 0.20 * burst_factor + 0.40 * discourse_factor + 0.05 * ttr_term

    def analyze(self, sentences: List[str], ai_threshold: float = 0.65) -> Dict[str, Any]:
        """Multi-signal AI analysis: RoBERTa + LM perplexity + statistical/linguistic signals."""
        if not sentences:
            return {
                "overall_ai_pct": 0.0, "ai_confidence_level": "None",
                "classification": "Likely Human Written", "badge_color": "#7daa85",
                "method": "SemanticEye Multi-Signal Statistical Engine",
                "burstiness": {}, "lexical": {}, "discourse": {}, "sentence_scores": [],
                "perplexity": {"available": False}, "roberta": {"available": False}, "models": {}, "components": {},
            }

        full_text = " ".join(sentences)
        burst_info = self.calculate_burstiness(sentences)
        cv = burst_info.get("coefficient_of_variation", 0.5)
        lexical = self.calculate_lexical_diversity(full_text)
        discourse = self.detect_discourse_patterns(full_text)

        sentence_evals = []
        for i, s in enumerate(sentences):
            res = self.analyze_sentence(s, cv)
            res["index"] = i
            res["sentence"] = s
            sentence_evals.append(res)
        stat = self.stat_composite(sentences, burst_info, lexical, discourse, sentence_evals)

        # ML signals (each may be None when its model isn't available)
        rob = self.roberta_scores(sentences)
        ppl = self.perplexity.analyze(sentences)
        rob_doc = rob["ai_probability"] if rob else None
        ppl_doc = ppl["ai_probability"] if ppl.get("available") else None

        # per-sentence fusion
        for i, ev in enumerate(sentence_evals):
            r_i = rob["per_sentence"][i] if rob else None
            p_i = ppl["sentence_probs"][i] if ppl.get("available") else None
            if p_i is None and ppl_doc is not None:
                p_i = None  # too short to score: leave perplexity out for this sentence
            ev["stat_score"] = ev["score"]
            if r_i is not None or p_i is not None:
                ev["score"] = round(max(0.02, min(0.99, self._combine(ev["score"], r_i, p_i))), 3)
                ev["percentage"] = round(ev["score"] * 100, 1)
                reasons = [] if ev["reason"] in ("Natural human sentence structure", "Short natural sentence") else [ev["reason"]]
                if r_i is not None and r_i >= 0.65:
                    reasons.append(f"RoBERTa classifier: {round(r_i * 100)}% AI")
                if p_i is not None and p_i >= 0.65:
                    reasons.append(f"Low perplexity ({ppl['sentence_perplexities'][i]}) - highly predictable text")
                ev["reason"] = "; ".join(reasons) if reasons else "Natural human sentence structure"
            if p_i is not None:
                ev["perplexity"] = ppl["sentence_perplexities"][i]
            ev["flagged"] = ev["score"] >= ai_threshold

        final_prob = self._final_prob(stat, rob_doc, ppl_doc)
        overall_pct = round(max(0.0, min(100.0, final_prob * 100)), 1)

        used = ["Statistical"]
        if ppl_doc is not None:
            used.insert(0, f"{config.LM_MODEL} perplexity")
        if rob_doc is not None:
            used.insert(0, "RoBERTa")
        method_used = " + ".join(used) + (" ensemble" if len(used) > 1 else " Multi-Signal Engine")
        if self.calibration and rob_doc is not None and ppl_doc is not None:
            method_used += " (calibrated)"

        if overall_pct >= 70:
            classification, badge_color = "Highly Likely AI-Generated", "#ef4444"
        elif overall_pct >= 45:
            classification, badge_color = "Mixed / AI-Assisted", "#d4956a"
        elif overall_pct >= 25:
            classification, badge_color = "Low AI Probability", "#3b82f6"
        else:
            classification, badge_color = "Likely Human Written", "#7daa85"

        return {
            "overall_ai_pct": overall_pct,
            "classification": classification,
            "badge_color": badge_color,
            "method": method_used,
            "burstiness": burst_info,
            "lexical": lexical,
            "discourse": discourse,
            "perplexity": {k: v for k, v in ppl.items() if k not in ("sentence_probs",)},
            "roberta": {"available": bool(rob), "ai_probability": rob_doc,
                        "chunk_probabilities": rob["chunk_probabilities"] if rob else []},
            "components": {"roberta": rob_doc, "perplexity": ppl_doc, "stat": round(stat, 4)},
            "models": {"roberta": config.ROBERTA_MODEL if rob else "unavailable",
                       "lm": config.LM_MODEL if ppl.get("available") else "unavailable"},
            "sentence_scores": sentence_evals,
        }
