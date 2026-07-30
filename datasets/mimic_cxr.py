"""
datasets/mimic_cxr.py

Loader for MIMIC-CXR (or the commonly used derivative, MIMIC-CXR-JPG).

MIMIC-CXR requires credentialed PhysioNet access and cannot be redistributed,
so this loader expects the user to have already placed a flattened metadata
CSV and JPG images on disk (e.g. via a private Kaggle dataset upload after
completing PhysioNet's data use agreement).

Expected on-disk layout (configurable via configs/default.yaml -> dataset):

    <root_dir>/
        images/
            p10/p10000032/s50414267/xxxx.jpg
            ...
        reports.csv     # columns: image_id, image_path, report, split, [subject_id, study_id]

This mirrors the iu_xray.py contract so the two are interchangeable from
the caller's perspective.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from utils.io import resolve_path
from utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"image_id", "image_path", "report"}


def load(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the MIMIC-CXR dataset into a standardized DataFrame.

    Args:
        config: Full benchmark config; uses the `dataset` section.

    Returns:
        DataFrame with columns [image_id, image_path, ground_truth_report].

    Raises:
        FileNotFoundError: If reports_csv does not exist.
        ValueError: If required columns are missing from reports_csv.
    """
    ds_cfg = config["dataset"]
    reports_csv = resolve_path(ds_cfg["reports_csv"])

    if not reports_csv.exists():
        raise FileNotFoundError(
            f"MIMIC-CXR reports CSV not found at {reports_csv}. "
            "MIMIC-CXR requires a PhysioNet data use agreement; this framework "
            "does not download or redistribute it. Provide your own flattened "
            "reports.csv with columns [image_id, image_path, report, split]."
        )

    df = pd.read_csv(reports_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"reports.csv is missing required columns: {missing}")

    if "split" in df.columns and ds_cfg.get("split"):
        df = df[df["split"] == ds_cfg["split"]].reset_index(drop=True)

    df = df.rename(columns={"report": "ground_truth_report"})
    df = df[["image_id", "image_path", "ground_truth_report"]].dropna(
        subset=["ground_truth_report"]
    )

    max_samples = ds_cfg.get("max_samples")
    if max_samples:
        df = df.head(int(max_samples)).reset_index(drop=True)

    logger.info("Loaded MIMIC-CXR dataset: %d samples (split=%s)", len(df), ds_cfg.get("split"))
    return df.reset_index(drop=True)
