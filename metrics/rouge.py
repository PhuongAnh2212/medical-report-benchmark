"""
metrics/rouge.py

Computes ROUGE-L (F-measure), averaged across all samples, using the
`rouge-score` package (the same implementation used by HuggingFace's
`evaluate` library under the hood).
"""

from __future__ import annotations

from typing import Dict, List

from rouge_score import rouge_scorer


def compute(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute mean ROUGE-L F1 across all samples.

    Args:
        predictions: Generated report strings.
        references: Ground-truth report strings.

    Returns:
        Dict with key "ROUGE-L".
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    total = 0.0
    n = max(len(predictions), 1)
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        total += result["rougeL"].fmeasure

    return {"ROUGE-L": round(total / n, 4)}
