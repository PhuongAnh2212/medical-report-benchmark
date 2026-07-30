"""
metrics/bleu.py

Computes BLEU-1 through BLEU-4 using NLTK's sentence-level BLEU with
smoothing, averaged (corpus-level) over all samples. This mirrors the
standard evaluation protocol used in radiology report generation papers
(e.g. R2Gen, CvT2DistilGPT2).
"""

from __future__ import annotations

from typing import Dict, List

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from utils.logging import get_logger

logger = get_logger(__name__)

_SMOOTHING = SmoothingFunction().method1


def _ensure_nltk_punkt() -> None:
    """Download the NLTK tokenizer data if not already present."""
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not download NLTK resource '%s': %s", resource, exc)


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenization (robust fallback if NLTK data is unavailable)."""
    return text.lower().split()


def compute(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute corpus-level BLEU-1..4.

    Args:
        predictions: Generated report strings.
        references: Ground-truth report strings.

    Returns:
        Dict with keys "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4".
    """
    _ensure_nltk_punkt()

    hyps = [_tokenize(p) for p in predictions]
    refs = [[_tokenize(r)] for r in references]

    scores: Dict[str, float] = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        score = corpus_bleu(refs, hyps, weights=weights, smoothing_function=_SMOOTHING)
        scores[f"BLEU-{n}"] = round(score, 4)
    return scores
