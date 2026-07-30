"""
metrics/radgraph.py

Placeholder for the RadGraph F1 metric, which compares clinical entities
and relations extracted from generated vs. reference reports using the
RadGraph information-extraction model.

Not enabled by default (see configs/default.yaml -> evaluation.metrics).
To enable:
    1. `pip install radgraph` (or vendor the RadGraph model per
       https://github.com/jbdel/radgraph).
    2. Implement `compute()` below to extract graphs for predictions and
       references and compute RadGraph F1.
    3. Add "radgraph" to config["evaluation"]["metrics"] in default.yaml.

Same `compute(predictions, references) -> dict` contract as every other
metric module, already registered in metrics/__init__.py's METRIC_REGISTRY.
"""

from __future__ import annotations

from typing import Dict, List


def compute(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute RadGraph F1 score.

    Args:
        predictions: Generated report strings.
        references: Ground-truth report strings.

    Returns:
        Dict with key "RadGraph-F1".

    Raises:
        NotImplementedError: Always, until the RadGraph integration is implemented.
    """
    raise NotImplementedError(
        "RadGraph metric is not yet implemented. See module docstring in "
        "metrics/radgraph.py for integration steps."
    )
