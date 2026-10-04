"""Lazy, thread-safe loaders for the optional ML models.

Nothing here is imported at module load: torch / transformers /
sentence-transformers are only touched on first use, and any failure
(not installed, no network for the first download, out of memory) is
remembered so the app simply falls back to its statistical/lexical engines.
"""
import logging
import threading
from typing import Any, Callable, Dict, Optional

from app import config

logger = logging.getLogger("semanticeye.models")


class ModelHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}
        # torch inference is not re-entrant friendly on small hosts: serialise it
        self.infer_lock = threading.Lock()

    def _get(self, key: str, enabled: bool, loader: Callable[[], Any]) -> Optional[Any]:
        if not enabled:
            return None
        if key in self._models:
            return self._models[key]
        if key in self._errors:
            return None
        with self._lock:
            if key in self._models:
                return self._models[key]
            if key in self._errors:
                return None
            try:
                logger.info("Loading model: %s", key)
                self._models[key] = loader()
                return self._models[key]
            except Exception as exc:  # noqa: BLE001 - any failure => graceful fallback
                self._errors[key] = f"{type(exc).__name__}: {exc}"
                logger.warning("Model %s unavailable (%s); using fallback", key, self._errors[key])
                return None

    # ── individual models ──
    def sbert(self):
        def load():
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(config.SBERT_MODEL)
        return self._get("sbert", config.ENABLE_SBERT, load)

    def roberta(self):
        def load():
            from transformers import pipeline
            return pipeline("text-classification", model=config.ROBERTA_MODEL,
                            truncation=True, max_length=512, top_k=None)
        return self._get("roberta", config.ENABLE_ROBERTA, load)

    def lm(self):
        """Returns (tokenizer, model) of the causal LM used for perplexity."""
        def load():
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(config.LM_MODEL)
            model = AutoModelForCausalLM.from_pretrained(config.LM_MODEL)
            model.eval()
            torch.set_grad_enabled(False)
            return tok, model
        return self._get("lm", config.ENABLE_PERPLEXITY, load)

    def preload(self) -> None:
        self.sbert(); self.roberta(); self.lm()

    def status(self) -> Dict[str, Any]:
        def st(key: str, enabled: bool, name: str) -> Dict[str, Any]:
            return {"enabled": enabled, "model": name, "loaded": key in self._models,
                    "error": self._errors.get(key)}
        return {
            "sbert": st("sbert", config.ENABLE_SBERT, config.SBERT_MODEL),
            "roberta": st("roberta", config.ENABLE_ROBERTA, config.ROBERTA_MODEL),
            "perplexity_lm": st("lm", config.ENABLE_PERPLEXITY, config.LM_MODEL),
        }


hub = ModelHub()
