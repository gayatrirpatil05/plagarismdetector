"""Evaluate plagiarism detection: lexical-only vs Sentence-BERT vs combined.

Builds a labelled benchmark from sample_docs/ (no network needed):
  - the corpus is built from half of the documents,
  - positives: verbatim, light-edit (patchwriting) and heavily-paraphrased
    sentences taken from the corpus documents,
  - negatives: sentences from held-out documents (same domain, so they are
    *hard* negatives - topical overlap but no copying).
The paraphrases are rule-based (synonym swaps + clause reordering). For a
real-world paraphrase test pass --pairs file.jsonl with {"text", "source", "label"}
(e.g. converted from PAWS / MRPC / ETPC).

    python -m evaluation.eval_plagiarism
    python -m evaluation.eval_plagiarism --pairs paws_pairs.jsonl
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

from app import config
from app.core.plagiarism_engine import PlagiarismEngine
from app.core.utils import split_sentences
from evaluation.metrics import binary_report, md_table

OUT_DIR = Path(__file__).parent / "results"
SYN = {"important": "crucial", "use": "employ", "used": "utilized", "help": "assist", "helps": "assists",
       "show": "demonstrate", "shows": "demonstrates", "big": "large", "many": "numerous", "make": "produce",
       "makes": "produces", "main": "primary", "need": "require", "needs": "requires", "get": "obtain",
       "increase": "rise", "reduce": "lower", "cause": "lead to", "causes": "leads to", "also": "additionally",
       "because": "since", "however": "nevertheless", "often": "frequently", "people": "individuals",
       "using": "employing", "change": "alteration", "changes": "alterations", "growth": "expansion",
       "problem": "issue", "problems": "issues", "ability": "capacity", "allow": "permit", "allows": "permits",
       "different": "distinct", "large": "substantial", "improve": "enhance", "provide": "supply",
       "provides": "supplies", "develop": "build", "include": "comprise", "includes": "comprises"}


def swap_synonyms(s: str, rate: float, rng: random.Random) -> str:
    def rep(m):
        w = m.group(0)
        syn = SYN.get(w.lower())
        if syn and rng.random() < rate:
            return syn.capitalize() if w[0].isupper() else syn
        return w
    return re.sub(r"[A-Za-z]+", rep, s)


def reorder_clauses(s: str) -> str:
    parts = re.split(r",\s+|\s+(?:because|while|which|and)\s+", s, maxsplit=1)
    m = re.search(r",\s+|\s+(?:because|while|which|and)\s+", s)
    if len(parts) == 2 and m and len(parts[0].split()) > 3 and len(parts[1].split()) > 3:
        a, b = parts[0].rstrip(".,"), parts[1].rstrip(".")
        return b[0].upper() + b[1:] + ", and " + a[0].lower() + a[1:] + "."
    return s


def build_benchmark(seed: int, per_type: int):
    rng = random.Random(seed)
    docs = []
    for f in sorted(os.listdir(config.SAMPLE_DOCS_DIR)):
        if f.endswith(".txt"):
            text = (config.SAMPLE_DOCS_DIR / f).read_text(encoding="utf-8", errors="ignore")
            sents = [s for s in split_sentences(text) if 10 <= len(s.split()) <= 45]
            if len(sents) >= 3:
                docs.append((f, sents))
    corpus_docs, held = docs[::2], docs[1::2]
    pos_pool = [(f, s) for f, ss in corpus_docs for s in ss]
    neg_pool = [s for _, ss in held for s in ss]
    rng.shuffle(pos_pool); rng.shuffle(neg_pool)
    per_type = max(1, min(per_type, len(pos_pool) // 3))
    cases = []
    for f, s in pos_pool[:per_type]:
        cases.append({"text": s, "label": 1, "type": "verbatim", "source": f})
    for f, s in pos_pool[per_type:2 * per_type]:
        cases.append({"text": swap_synonyms(s, 0.25, rng), "label": 1, "type": "light_edit", "source": f})
    for f, s in pos_pool[2 * per_type:3 * per_type]:
        cases.append({"text": reorder_clauses(swap_synonyms(s, 0.7, rng)), "label": 1, "type": "paraphrase", "source": f})
    for s in neg_pool[:per_type * 2]:
        cases.append({"text": s, "label": 0, "type": "original", "source": ""})
    return corpus_docs, cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-type", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--pairs", help="JSONL of {text, source, label}; source text is added to the corpus")
    args = ap.parse_args()

    engine = PlagiarismEngine(sample_docs_dir=str(config.SAMPLE_DOCS_DIR))
    if args.pairs:
        rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
        engine.reference_documents = [{"filename": f"pair{i}", "title": f"pair{i}", "content": r["source"],
                                       "sentences": [r["source"]]} for i, r in enumerate(rows) if r.get("source")]
        cases = [{"text": r["text"], "label": int(r["label"]), "type": r.get("type", "pair")} for r in rows]
    else:
        corpus_docs, cases = build_benchmark(args.seed, args.per_type)
        engine.reference_documents = [{"filename": f, "title": f, "content": " ".join(ss), "sentences": ss}
                                      for f, ss in corpus_docs]
        # genuinely hand-written paraphrases (low word overlap) + topical hard negatives
        hw = Path(__file__).parent / "data" / "handwritten_pairs.jsonl"
        if hw.exists():
            rows = [json.loads(l) for l in open(hw, encoding="utf-8") if l.strip()]
            seen = set()
            for r in rows:
                if r["source"] not in seen:
                    seen.add(r["source"])
                    engine.reference_documents.append({"filename": f"hw_{len(seen)}", "title": f"hw_{len(seen)}",
                                                       "content": r["source"], "sentences": [r["source"]]})
            cases += [{"text": r["text"], "label": r["label"], "type": r["type"]} for r in rows]

    # one engine pass with the whole case list (sentences are independent)
    res = engine.analyze([c["text"] for c in cases], args.threshold, use_web=False)
    det = res["sentence_details"]
    y = [c["label"] for c in cases]
    modes = {"lexical (TF-IDF+difflib)": [d["lexical_score"] for d in det],
             "semantic (SBERT)": [d["semantic_score"] for d in det],
             "combined": [d["score"] for d in det]}
    reports = {k: binary_report(y, v, args.threshold) for k, v in modes.items()}

    recall_by_type = {}
    for t in sorted({c["type"] for c in cases if c["label"] == 1}):
        idx = [i for i, c in enumerate(cases) if c["type"] == t]
        recall_by_type[t] = {k: round(sum(v[i] >= args.threshold for i in idx) / len(idx), 3) for k, v in modes.items()}

    out = {"n": len(cases), "semantic_model_active": res["engine"]["semantic"], "threshold": args.threshold,
           "modes": reports, "recall_by_type": recall_by_type}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "plagiarism_eval.json").write_text(json.dumps(out, indent=2))
    md = [f"# Plagiarism evaluation (n={len(cases)}, semantic: {res['engine']['semantic']})", "", md_table(reports), "",
          "## Recall by plagiarism type", "",
          "| type | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for t, r in recall_by_type.items():
        md.append(f"| {t} | " + " | ".join(str(r[k]) for k in modes) + " |")
    (OUT_DIR / "plagiarism_eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
