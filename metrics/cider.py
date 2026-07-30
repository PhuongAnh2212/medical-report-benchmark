"""
metrics/cider.py

Computes the CIDEr metric. Uses `pycocoevalcap`'s CIDEr implementation
(the standard reference implementation used by image-captioning and
report-generation benchmarks) when available, and falls back to a
lightweight pure-Python TF-IDF n-gram implementation otherwise so the
benchmark still runs on environments where installing pycocoevalcap
(which requires a Java runtime for some sibling metrics) is inconvenient.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List

from utils.logging import get_logger

logger = get_logger(__name__)


def _compute_with_pycocoevalcap(predictions: List[str], references: List[str]) -> float:
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return float(score)


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _compute_fallback(predictions: List[str], references: List[str], n_max: int = 4) -> float:
    """A simplified TF-IDF cosine-similarity CIDEr approximation.

    This is not bit-identical to the official CIDEr implementation but
    follows the same core idea (TF-IDF weighted n-gram cosine similarity,
    averaged over n=1..4) and is dependency-free.
    """
    tokenized_preds = [p.lower().split() for p in predictions]
    tokenized_refs = [r.lower().split() for r in references]

    doc_freq: Dict[int, Counter] = {n: Counter() for n in range(1, n_max + 1)}
    num_docs = len(tokenized_refs)

    for n in range(1, n_max + 1):
        for ref_tokens in tokenized_refs:
            for ngram in set(_ngrams(ref_tokens, n)):
                doc_freq[n][ngram] += 1

    def tfidf_vector(tokens: List[str], n: int) -> Dict[tuple, float]:
        counts = _ngrams(tokens, n)
        total = sum(counts.values()) or 1
        vec = {}
        for ngram, count in counts.items():
            tf = count / total
            idf = math.log(max(num_docs, 1) / (1.0 + doc_freq[n].get(ngram, 0)))
            vec[ngram] = tf * idf
        return vec

    def cosine_sim(vec_a: Dict[tuple, float], vec_b: Dict[tuple, float]) -> float:
        common = set(vec_a) & set(vec_b)
        numerator = sum(vec_a[k] * vec_b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values())) or 1e-12
        norm_b = math.sqrt(sum(v * v for v in vec_b.values())) or 1e-12
        return numerator / (norm_a * norm_b)

    total_score = 0.0
    for pred_tokens, ref_tokens in zip(tokenized_preds, tokenized_refs):
        n_scores = []
        for n in range(1, n_max + 1):
            vec_p = tfidf_vector(pred_tokens, n)
            vec_r = tfidf_vector(ref_tokens, n)
            n_scores.append(cosine_sim(vec_p, vec_r))
        total_score += sum(n_scores) / n_max

    n_samples = max(len(predictions), 1)
    # Scale to roughly match CIDEr's typical [0, 10] range.
    return (total_score / n_samples) * 10.0


def compute(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute CIDEr score.

    Args:
        predictions: Generated report strings.
        references: Ground-truth report strings.

    Returns:
        Dict with key "CIDEr".
    """
    try:
        score = _compute_with_pycocoevalcap(predictions, references)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pycocoevalcap CIDEr unavailable (%s); using pure-Python fallback approximation.", exc
        )
        score = _compute_fallback(predictions, references)

    return {"CIDEr": round(score, 4)}
