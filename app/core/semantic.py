"""Sentence-BERT embeddings for paraphrase detection."""
import hashlib
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from app import config
from app.core.models import hub


class SemanticEncoder:
    """Encodes sentences to unit vectors, with an LRU cache so the (static)
    reference corpus is only embedded once per process."""

    def __init__(self, hub_=hub, cache_size: int = 60_000):
        self.hub = hub_
        self.cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.cache_size = cache_size

    @staticmethod
    def _key(t: str) -> str:
        return hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()

    def available(self) -> bool:
        return self.hub.sbert() is not None

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        model = self.hub.sbert()
        if model is None or not texts:
            return None
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.cache]
        if missing:
            with self.hub.infer_lock:
                vecs = model.encode([t for _, t in missing], batch_size=64,
                                    normalize_embeddings=True, show_progress_bar=False)
            for (k, _), v in zip(missing, np.asarray(vecs, dtype=np.float32)):
                self.cache[k] = v
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return np.vstack([self.cache[k] for k in keys])

    @staticmethod
    def to_score(cos: float) -> float:
        """Map cosine similarity to a 0-1 plagiarism-likelihood score."""
        lo, hi = config.SBERT_SEM_FLOOR, config.SBERT_SEM_CEIL
        return float(min(1.0, max(0.0, (cos - lo) / (hi - lo))))


encoder = SemanticEncoder()
