"""
metrics/chexbert.py

Placeholder for CheXbert-based clinical accuracy metrics (label-based F1
over the 14 CheXpert observation classes, extracted from generated vs.
reference reports via the CheXbert labeler).

Not enabled by default (see configs/default.yaml -> evaluation.metrics).
To enable:
    1. Download the CheXbert checkpoint (see https://github.com/stanfordmlgroup/CheXbert).
    2. Implement `compute()` below to run the labeler on predictions and
       references, then compute micro/macro F1 over the 14 classes.
    3. Add "chexbert" to config["evaluation"]["metrics"] in default.yaml.

Because this module already exposes the same `compute(predictions, references)
-> dict` signature as every other metric, and is already registered in
metrics/__init__.py's METRIC_REGISTRY, no changes to evaluation/evaluate.py
are required once implemented.
"""

from __future__ import annotations

from typing import Dict, List


def compute(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute CheXbert-based clinical accuracy F1 scores.

    Args:
        predictions: Generated report strings.
        references: Ground-truth report strings.

    Returns:
        Dict of CheXbert-derived scores (e.g. "CheXbert-F1-micro").

    Raises:
        NotImplementedError: Always, until the CheXbert labeler integration
            is implemented.
    """
    raise NotImplementedError(
        "CheXbert metric is not yet implemented. See module docstring in "
        "metrics/chexbert.py for integration steps."
    )
