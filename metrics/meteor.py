"""
metrics/meteor.py

Computes the METEOR metric, averaged across all samples, using NLTK's
implementation (nltk.translate.meteor_score). METEOR accounts for
stemming, synonymy, and word order, and is a standard part of the
report-generation evaluation suite alongside BLEU/ROUGE/CIDEr.
"""

from __future__ import annotations

from typing import Dict, List

import nltk
from nltk.translate.meteor_score import meteor_score

from utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_nltk_resources() -> None:
    """Download the NLTK resources required for METEOR, if not already present."""
    resources = {
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "tokenizers/punkt": "punkt",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not download NLTK resource '%s': %s", name, exc)


def compute(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute mean METEOR score across all samples.

    Args:
        predictions: Generated report strings.
        references: Ground-truth report strings.

    Returns:
        Dict with key "METEOR".
    """
    _ensure_nltk_resources()

    total = 0.0
    n = max(len(predictions), 1)
    for pred, ref in zip(predictions, references):
        score = meteor_score([ref.lower().split()], pred.lower().split())
        total += score

    return {"METEOR": round(total / n, 4)}
