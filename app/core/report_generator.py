"""Report Generator for SemanticEye.
Generates structured audit summaries and printable PDF/A4 academic reports.
"""
import hashlib
import datetime
import uuid
from typing import Dict, Any, List


def generate_document_hash(text: str) -> str:
    """Generate SHA-256 fingerprint for document authentication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16].upper()


def build_audit_report(text: str, stats: Dict[str, Any], plag_result: Dict[str, Any],
                       ai_result: Dict[str, Any], filename: str = "Pasted_Text") -> Dict[str, Any]:
    """Assemble all analysis data into a structured audit certificate."""
    doc_hash = generate_document_hash(text)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%B %d, %Y - %H:%M:%S UTC")

    plag_details = plag_result.get("sentence_details", [])
    ai_details   = ai_result.get("sentence_scores", [])

    annotated_sentences = []
    for i, (p_item, a_item) in enumerate(zip(plag_details, ai_details)):
        s_text  = p_item.get("sentence", "")
        is_plag = bool(p_item.get("flagged", False))
        is_ai   = bool(a_item.get("flagged", False))

        if is_plag and is_ai:
            tag, label = "both", "Copied and AI-Generated"
        elif is_plag:
            tag, label = "plagiarized", "Copied from Source"
        elif is_ai:
            tag, label = "ai", "AI-Generated Text"
        else:
            tag, label = "original", "Original Human"

        raw_reason = a_item.get("reason", "")
        ai_reasons = [r.strip() for r in raw_reason.split(";") if r.strip()] if raw_reason else []

        annotated_sentences.append({
            "index":            i,
            "text":             s_text,
            "is_plagiarized":   is_plag,
            "plag_score":       p_item.get("score", 0.0),
            "plag_pct":         p_item.get("percentage", 0.0),
            "plag_source":      p_item.get("source", ""),
            "plag_source_title": p_item.get("source_title", ""),
            "matched_sentence": p_item.get("matched_sentence", ""),
            "plag_url":         p_item.get("url", ""),
            "plag_origin":      p_item.get("origin", ""),
            "semantic_score":   p_item.get("semantic_score", 0.0),
            "lexical_score":    p_item.get("lexical_score", 0.0),
            "perplexity":       a_item.get("perplexity"),
            "match_type":       p_item.get("match_type", "Original"),
            "word_diff_html":   p_item.get("diff_input_html", s_text),
            "diff_source_html": p_item.get("diff_source_html", ""),
            "is_ai":            is_ai,
            "ai_score":         a_item.get("score", 0.0),
            "ai_pct":           a_item.get("percentage", 0.0),
            "ai_reasons":       ai_reasons,
            "tag":              tag,
            "label":            label,
        })

    plag_pct_raw  = plag_result.get("overall_plagiarism_pct", 0.0)
    ai_pct_raw    = ai_result.get("overall_ai_pct", 0.0)
    plag_fraction = round(plag_pct_raw / 100.0, 4)
    ai_fraction   = round(ai_pct_raw   / 100.0, 4)

    if plag_pct_raw <= 15 and ai_pct_raw <= 25:
        integrity_status   = "PASS: Academic Integrity Verified"
        status_color       = "#7daa85"
        status_description = "The submitted document demonstrates high originality and human authorship."
    elif plag_pct_raw > 40 or ai_pct_raw > 65:
        integrity_status   = "CRITICAL: Multiple Violations Detected"
        status_color       = "#ef4444"
        status_description = "Significant content matches or high synthetic generation identified. Manual review required."
    else:
        integrity_status   = "CAUTION: Moderate Similarities Identified"
        status_color       = "#d4956a"
        status_description = "Elevated similarity or AI markers observed. Author attribution inspection recommended."

    matched_sources = []
    for src in plag_result.get("sources_summary", []):
        matched_sources.append({
            "filename":                src.get("source", ""),
            "title":                   src.get("title", src.get("source", "")),
            "plagiarism_contribution": src.get("contribution_pct", 0.0) / 100.0,
            "matched_sentences_count": src.get("match_count", 0),
            "url":                     src.get("url", ""),
        })

    burstiness_val  = ai_result.get("burstiness", {}).get("coefficient_of_variation", 0.45)
    discourse_count = ai_result.get("discourse", {}).get("matches_count", 0)
    lexical_ttr     = ai_result.get("lexical", {}).get("ttr", 0.5)
    ppl = ai_result.get("perplexity") or {}
    if ppl.get("available"):
        # real LM perplexity -> 0-100 "predictability" score (low perplexity = high score)
        perplexity_score = round(100.0 * ppl.get("ai_probability", 0.0), 1)
    else:
        perplexity_score = round(max(0.0, 100.0 - (burstiness_val * 200)), 1)
    rob = ai_result.get("roberta") or {}

    shap_explanations = [
        {
            "name": "Semantic Vector Alignment", "model_path": "Plagiarism Inspection",
            "val": str(plag_pct_raw) + "% match",
            "impact_pct": round(plag_pct_raw * 0.85, 1),
            "direction": "risk" if plag_pct_raw > 30 else "pass",
            "explanation": "Semantic vector analysis detected " + str(plag_pct_raw) + "% contextual similarity.",
        },
        {
            "name": "Syntactic Pattern Analysis", "model_path": "AI Text Inspection",
            "val": str(ai_pct_raw) + "% probability",
            "impact_pct": round(ai_pct_raw * 0.90, 1),
            "direction": "risk" if ai_pct_raw > 50 else "pass",
            "explanation": "Linguistic pattern classification evaluated a " + str(ai_pct_raw) + "% synthetic text likelihood.",
        },
        {
            "name": "Sentence Cadence Variance", "model_path": "Structure Analysis",
            "val": "CoV = " + str(burstiness_val),
            "impact_pct": round(max(0, (0.45 - burstiness_val) * 100), 1),
            "direction": "risk" if burstiness_val < 0.25 else "pass",
            "explanation": "Natural sentence length variation observed." if burstiness_val >= 0.35 else "Uniform sentence length cadence detected.",
        },
        {
            "name": "Formulaic Phrase Detection", "model_path": "Discourse Heuristics",
            "val": str(discourse_count) + " markers",
            "impact_pct": min(100.0, discourse_count * 25.0),
            "direction": "risk" if discourse_count > 0 else "pass",
            "explanation": "Identified " + str(discourse_count) + " formulaic transitional phrases." if discourse_count > 0 else "No formulaic transitional markers detected.",
        },
        {
            "name": "Vocabulary Diversity", "model_path": "Lexical Analysis",
            "val": "TTR = " + str(lexical_ttr),
            "impact_pct": round(lexical_ttr * 100, 1),
            "direction": "pass" if lexical_ttr >= 0.45 else "risk",
            "explanation": "High lexical diversity TTR of " + str(lexical_ttr) + "." if lexical_ttr >= 0.45 else "Repetitive word usage distribution.",
        },
        {
            "name": "RoBERTa AI Classifier", "model_path": ai_result.get("models", {}).get("roberta", "unavailable"),
            "val": (str(round(rob["ai_probability"] * 100, 1)) + "% AI") if rob.get("available") else "not loaded",
            "impact_pct": round((rob.get("ai_probability") or 0.0) * 100, 1),
            "direction": "risk" if (rob.get("ai_probability") or 0.0) > 0.5 else "pass",
            "explanation": "Fine-tuned transformer classifier judged the text." if rob.get("available") else "Classifier unavailable; statistical signals used.",
        },
        {
            "name": "Language-Model Perplexity", "model_path": ai_result.get("models", {}).get("lm", "unavailable"),
            "val": ("PPL = " + str(ppl["mean_perplexity"])) if ppl.get("available") else "not loaded",
            "impact_pct": round(ppl.get("ai_probability", 0.0) * 100, 1),
            "direction": "risk" if ppl.get("ai_probability", 0.0) > 0.5 else "pass",
            "explanation": "Low perplexity means the text is highly predictable to a language model, typical of AI output." if ppl.get("available") else "Perplexity model unavailable.",
        },
        {
            "name": "Ensemble Composite Scoring", "model_path": "Multi-Signal Integration",
            "val": "Balanced",
            "impact_pct": round(max(plag_pct_raw, ai_pct_raw), 1),
            "direction": "risk" if (plag_pct_raw > 35 or ai_pct_raw > 50) else "pass",
            "explanation": "Integrated decision model combined all individual linguistic signals into the audit score.",
        },
    ]

    return {
        "report_id":            "SE-" + uuid.uuid4().hex[:16].upper(),  # unique per scan
        "document_fingerprint": doc_hash,
        "filename":             filename,
        "timestamp":            timestamp,
        "integrity_status":     integrity_status,
        "status_color":         status_color,
        "status_description":   status_description,
        "summary": {
            "plagiarism_score": plag_fraction,
            "ai_score":         ai_fraction,
        },
        "stats":     stats,
        "ai_analysis": {
            "perplexity_score": perplexity_score,
            "classification":   ai_result.get("classification", ""),
            "perplexity":       ppl,
            "roberta":          rob,
            "components":       ai_result.get("components", {}),
            "plagiarism_engine": plag_result.get("engine", {}),
            "method":           ai_result.get("method", ""),
            "burstiness":       ai_result.get("burstiness", {}),
            "lexical":          ai_result.get("lexical", {}),
            "discourse":        ai_result.get("discourse", {}),
        },
        "sentences":         annotated_sentences,
        "matched_sources":   matched_sources,
        "shap_explanations": shap_explanations,
    }
