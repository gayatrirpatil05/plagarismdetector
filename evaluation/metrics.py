"""Shared metric helpers (thin wrappers over scikit-learn)."""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve


def binary_report(y_true: List[int], scores: List[float], threshold: float = 0.5) -> Dict:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)), "positives": int(y.sum()), "threshold": round(float(threshold), 4),
        "accuracy": round(float((pred == y).mean()), 4), "precision": round(float(p), 4),
        "recall": round(float(r), 4), "f1": round(float(f1), 4),
        "fpr": round(float(fp / max(1, fp + tn)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "auc": None, "tpr_at_1pct_fpr": None, "best_f1_threshold": None, "best_f1": None,
    }
    if len(set(y.tolist())) == 2:
        out["auc"] = round(float(roc_auc_score(y, s)), 4)
        fpr, tpr, _ = roc_curve(y, s)
        out["tpr_at_1pct_fpr"] = round(float(np.interp(0.01, fpr, tpr)), 4)
        best = (0.0, threshold)
        for t in np.unique(np.round(s, 3)):
            pr = (s >= t).astype(int)
            _, _, f, _ = precision_recall_fscore_support(y, pr, average="binary", zero_division=0)
            if f > best[0]:
                best = (float(f), float(t))
        out["best_f1"], out["best_f1_threshold"] = round(best[0], 4), round(best[1], 3)
    return out


def md_table(rows: Dict[str, Dict], cols: Optional[List[str]] = None) -> str:
    cols = cols or ["auc", "accuracy", "precision", "recall", "f1", "fpr", "tpr_at_1pct_fpr", "best_f1_threshold"]
    lines = ["| signal | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join("-" if r.get(c) is None else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)
