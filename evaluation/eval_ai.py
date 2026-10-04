"""Evaluate the AI-text detector on a labelled dataset (HC3 by default).

    python -m evaluation.eval_ai --per-class 300                 # HC3 from the HF Hub
    python -m evaluation.eval_ai --hc3-dir ./HC3 --subsets open_qa finance
    python -m evaluation.eval_ai --data my_set.csv               # text,label (1 = AI)
    python -m evaluation.eval_ai --per-class 500 --fit           # also learn ensemble weights

Reports AUC / accuracy / P / R / F1 / FPR / TPR@1%FPR for every signal
(RoBERTa, perplexity, statistical, final ensemble). --fit trains a logistic
regression over the three signals (5-fold CV reported) and writes
app/core/calibration.json, which the detector then uses automatically.

NOTE: Hello-SimpleAI/chatgpt-detector-roberta was fine-tuned on HC3, so HC3
numbers for RoBERTa are optimistic. For an honest estimate also run on a set
the model has not seen (--data) and compare.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.core.ai_engine import AIDetectorEngine
from app.core.utils import split_sentences
from evaluation.datasets import load_hc3, load_labelled
from evaluation.metrics import binary_report, md_table

OUT_DIR = Path(__file__).parent / "results"


def collect(data, engine, progress=True):
    rows = []
    t0 = time.time()
    for i, r in enumerate(data):
        sents = split_sentences(r["text"])
        res = engine.analyze(sents)
        c = res["components"]
        rows.append({"label": r["label"], "subset": r["subset"], "final": res["overall_ai_pct"] / 100.0,
                     "roberta": c.get("roberta"), "perplexity": c.get("perplexity"), "stat": c.get("stat"),
                     "mean_ppl": (res.get("perplexity") or {}).get("mean_perplexity")})
        if progress and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(data)}  ({(time.time() - t0) / (i + 1):.2f}s/doc)", flush=True)
    return rows


def evaluate(rows):
    y = [r["label"] for r in rows]
    out = {}
    for name in ("final", "roberta", "perplexity", "stat"):
        pairs = [(r["label"], r[name]) for r in rows if r.get(name) is not None]
        if len(pairs) >= 10:
            out[name] = binary_report([p[0] for p in pairs], [p[1] for p in pairs], 0.5)
    per_subset = {}
    by = defaultdict(list)
    for r in rows:
        by[r["subset"]].append(r)
    for sub, rs in by.items():
        if len({r["label"] for r in rs}) == 2:
            per_subset[sub] = binary_report([r["label"] for r in rs], [r["final"] for r in rs], 0.5)
    return out, per_subset


def fit_calibration(rows, write_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    full = [r for r in rows if None not in (r["roberta"], r["perplexity"], r["stat"])]
    if len(full) < 40:
        print("Not enough rows with all three signals to fit calibration (is RoBERTa/GPT-2 loaded?).")
        return None
    X = np.array([[r["roberta"], r["perplexity"], r["stat"]] for r in full])
    y = np.array([r["label"] for r in full])
    clf = LogisticRegression(C=1.0, max_iter=1000)
    cv = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    cv_report = binary_report(y.tolist(), cv.tolist(), 0.5)
    clf.fit(X, y)
    cal = {"features": ["roberta", "perplexity", "stat"], "coef": [round(float(c), 5) for c in clf.coef_[0]],
           "intercept": round(float(clf.intercept_[0]), 5), "n_fit": int(len(y)), "cv": cv_report}
    Path(write_path).write_text(json.dumps(cal, indent=2))
    print(f"Wrote calibration -> {write_path}")
    return cal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV/JSONL with text,label (1=AI)")
    ap.add_argument("--hc3-dir", help="local folder with HC3 *.jsonl files")
    ap.add_argument("--subsets", nargs="+", default=["open_qa", "wiki_csai", "finance"])
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--min-words", type=int, default=25)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--fit", action="store_true", help="fit ensemble weights and write calibration.json")
    ap.add_argument("--tag", default="hc3")
    args = ap.parse_args()

    from app import config
    data = load_labelled(args.data) if args.data else load_hc3(args.subsets, args.per_class, args.min_words,
                                                                args.seed, args.hc3_dir)
    print(f"Loaded {len(data)} texts ({sum(d['label'] for d in data)} AI / {sum(1 - d['label'] for d in data)} human)")

    engine = AIDetectorEngine()
    if args.fit:
        engine.calibration = None  # evaluate raw signals, not a previous fit
    rows = collect(data, engine)
    signals, per_subset = evaluate(rows)

    cal = fit_calibration(rows, config.CALIBRATION_PATH) if args.fit else None
    report = {"dataset": args.data or f"HC3:{','.join(args.subsets)}", "n": len(rows), "signals": signals,
              "per_subset_final": per_subset, "calibration": cal,
              "mean_perplexity": {
                  "human": round(float(np.mean([r["mean_ppl"] for r in rows if r["label"] == 0 and r["mean_ppl"]] or [0])), 2),
                  "ai": round(float(np.mean([r["mean_ppl"] for r in rows if r["label"] == 1 and r["mean_ppl"]] or [0])), 2)}}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"ai_eval_{args.tag}.json").write_text(json.dumps(report, indent=2))
    md = [f"# AI detector evaluation ({report['dataset']}, n={report['n']})", "", md_table(signals), "",
          "## Per subset (final ensemble)", "", md_table(per_subset),
          "", f"Mean GPT-2 perplexity - human: {report['mean_perplexity']['human']}, AI: {report['mean_perplexity']['ai']}"]
    if cal:
        md += ["", f"Calibrated ensemble (5-fold CV): AUC {cal['cv']['auc']}, F1 {cal['cv']['f1']}, acc {cal['cv']['accuracy']}"]
    (OUT_DIR / f"ai_eval_{args.tag}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
