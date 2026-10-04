"""Real perplexity from a causal language model (GPT-2 by default).

AI-written text is, on average, far more predictable to a language model than
human writing, so low perplexity is evidence of machine generation. We compute
token log-probabilities once for the whole document and attribute them to
sentences via character offsets.
"""
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from app import config
from app.core.models import hub

WINDOW = 512


def ppl_to_prob(ppl: Optional[float]) -> float:
    """Logistic map: perplexity == PPL_MIDPOINT -> 0.5, lower perplexity -> more AI-like."""
    if ppl is None or ppl <= 0:
        return 0.5
    z = (math.log(config.PPL_MIDPOINT) - math.log(ppl)) * config.PPL_STEEPNESS
    return 1.0 / (1.0 + math.exp(-z))


class PerplexityScorer:
    def __init__(self, hub_=hub):
        self.hub = hub_

    def available(self) -> bool:
        return self.hub.lm() is not None

    def _token_logprobs(self, text: str) -> Optional[Tuple[List[Tuple[int, int]], List[float]]]:
        """Returns (char offsets per token, log-prob per token) or None."""
        loaded = self.hub.lm()
        if loaded is None:
            return None
        import torch
        tok, model = loaded
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False,
                  truncation=True, max_length=config.MAX_LM_TOKENS)
        ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
        if len(ids) < 2:
            return None
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        logps: List[float] = []
        with self.hub.infer_lock, torch.no_grad():
            for start in range(0, len(ids), WINDOW):
                chunk = ids[start:start + WINDOW]
                inp = torch.tensor([[bos] + chunk])
                logits = model(inp).logits[0, :-1].float()
                lp = torch.log_softmax(logits, dim=-1)
                tgt = torch.tensor(chunk)
                logps.extend(lp[torch.arange(len(chunk)), tgt].tolist())
        return offsets, logps

    def analyze(self, sentences: List[str]) -> Dict[str, Any]:
        unavailable = {"available": False, "mean_perplexity": None, "ai_probability": 0.0,
                       "sentence_perplexities": [None] * len(sentences),
                       "sentence_probs": [None] * len(sentences)}
        if not sentences:
            return unavailable
        try:
            text = " ".join(sentences)
            res = self._token_logprobs(text)
        except Exception:
            res = None
        if not res:
            return unavailable
        offsets, logps = res

        spans, pos = [], 0
        for s in sentences:
            spans.append((pos, pos + len(s)))
            pos += len(s) + 1
        buckets: List[List[float]] = [[] for _ in sentences]
        k = 0
        for (a, _), lp in zip(offsets, logps):
            while k < len(spans) - 1 and a >= spans[k][1] + 1:
                k += 1
            buckets[k].append(lp)

        sent_ppl: List[Optional[float]] = []
        for b in buckets:
            sent_ppl.append(round(math.exp(-sum(b) / len(b)), 2) if len(b) >= 3 else None)

        mean_ppl = math.exp(-sum(logps) / len(logps))
        valid = [p for p in sent_ppl if p]
        log_std = statistics.pstdev([math.log(p) for p in valid]) if len(valid) > 1 else 0.0
        return {
            "available": True,
            "mean_perplexity": round(mean_ppl, 2),
            "median_sentence_perplexity": round(statistics.median(valid), 2) if valid else None,
            "perplexity_burstiness": round(log_std, 3),  # std of log-ppl across sentences; low = uniform = AI-like
            "tokens_scored": len(logps),
            "ai_probability": round(ppl_to_prob(mean_ppl), 4),
            "sentence_perplexities": sent_ppl,
            "sentence_probs": [round(ppl_to_prob(p), 4) if p else None for p in sent_ppl],
        }


scorer = PerplexityScorer()
