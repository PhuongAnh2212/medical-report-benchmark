"""
metrics package

Each module exposes a `compute(predictions: list[str], references: list[str]) -> dict`
function returning one or more named scores. `evaluation/evaluate.py` calls
every metric listed in config["evaluation"]["metrics"] and merges the
resulting dicts into one row of the leaderboard.

This uniform `compute(predictions, references) -> dict` contract is what
lets CheXbert and RadGraph (clinical-accuracy metrics) be dropped in later
without touching evaluate.py: implement metrics/chexbert.py and
metrics/radgraph.py with the same signature, register them in
METRIC_REGISTRY, and add their names to config["evaluation"]["metrics"].
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from metrics import bleu, chexbert, cider, meteor, radgraph, rouge

MetricFn = Callable[[List[str], List[str]], Dict[str, float]]

METRIC_REGISTRY: Dict[str, MetricFn] = {
    "bleu": bleu.compute,
    "rouge": rouge.compute,
    "meteor": meteor.compute,
    "cider": cider.compute,
    # Not enabled by default in configs/default.yaml -- require extra
    # model downloads (CheXbert checkpoint) / installs (RadGraph). Included
    # in the registry so they can be enabled with zero pipeline changes.
    "chexbert": chexbert.compute,
    "radgraph": radgraph.compute,
}


def compute_metrics(
    metric_names: List[str], predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute all requested metrics and merge into a single flat dict.

    Args:
        metric_names: Subset of METRIC_REGISTRY keys to compute.
        predictions: List of generated report strings.
        references: List of ground-truth report strings (same order/length).

    Returns:
        Flat dict mapping metric column name (e.g. "BLEU-1") -> score.

    Raises:
        ValueError: If an unknown metric name is requested.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must be the same length"
        )

    results: Dict[str, float] = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}")
        results.update(METRIC_REGISTRY[name](predictions, references))
    return results
