"""Dataset loaders for the AI-text benchmark.

HC3 (Hello-SimpleAI/HC3): human vs ChatGPT answers to the same questions.
Files: open_qa, finance, medicine, reddit_eli5, wiki_csai (JSONL, fields
question / human_answers / chatgpt_answers). Downloaded from the Hugging Face
Hub on first use, or read from a local copy via --hc3-dir.

Any other labelled set can be supplied as CSV/JSONL with columns text,label
(label: 1 = AI, 0 = human).
"""
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

HC3_SUBSETS = ["open_qa", "finance", "medicine", "reddit_eli5", "wiki_csai"]


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_hc3(subsets: Optional[List[str]] = None, per_class: int = 300, min_words: int = 25,
             seed: int = 13, local_dir: Optional[str] = None) -> List[Dict]:
    subsets = subsets or ["open_qa", "wiki_csai", "finance"]
    rng = random.Random(seed)
    humans, ais = [], []
    for sub in subsets:
        if local_dir:
            path = Path(local_dir) / f"{sub}.jsonl"
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id="Hello-SimpleAI/HC3", filename=f"{sub}.jsonl", repo_type="dataset"))
        for row in _read_jsonl(path):
            for t in row.get("human_answers", []):
                humans.append({"text": t, "label": 0, "subset": sub})
            for t in row.get("chatgpt_answers", []):
                ais.append({"text": t, "label": 1, "subset": sub})
    def ok(r):
        return len(r["text"].split()) >= min_words
    humans, ais = [r for r in humans if ok(r)], [r for r in ais if ok(r)]
    rng.shuffle(humans); rng.shuffle(ais)
    n = min(per_class, len(humans), len(ais))
    data = humans[:n] + ais[:n]
    rng.shuffle(data)
    return data


def load_labelled(path: str) -> List[Dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = _read_jsonl(p)
    else:
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return [{"text": r["text"], "label": int(r["label"]), "subset": r.get("subset", p.stem)} for r in rows]
