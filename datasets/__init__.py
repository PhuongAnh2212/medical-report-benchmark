"""
datasets package

Each module in this package exposes a `load(config: dict) -> pd.DataFrame`
function that returns a DataFrame with, at minimum, the columns:

    image_id        : str, unique identifier for the sample
    image_path      : str, absolute or resolvable path to the image file
    ground_truth_report : str, the reference radiology report

Adding a new dataset means adding a new module here with the same
`load()` contract, then adding it to DATASET_REGISTRY below.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import pandas as pd

from datasets import iu_xray, mimic_cxr

DATASET_REGISTRY: Dict[str, Callable[[Dict[str, Any]], pd.DataFrame]] = {
    "iu_xray": iu_xray.load,
    "mimic_cxr": mimic_cxr.load,
}


def load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """Dispatch to the correct dataset loader based on config["dataset"]["name"].

    Args:
        config: The full benchmark configuration (as loaded from default.yaml).

    Returns:
        A DataFrame with columns [image_id, image_path, ground_truth_report].

    Raises:
        ValueError: If the requested dataset name is not registered.
    """
    dataset_name = config["dataset"]["name"]
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name](config)
