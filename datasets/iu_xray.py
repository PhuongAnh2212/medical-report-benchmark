"""
datasets/iu_xray.py

Loader for the IU X-Ray (Indiana University Chest X-Ray) dataset.

Expected on-disk layout (configurable via configs/default.yaml -> dataset):

    <root_dir>/
        images/
            CXR1_1_IM-0001-1001.png
            CXR1_1_IM-0001-2001.png
            ...
        reports.csv          # columns: image_id, image_path (optional), report, split

If `image_path` is not present in reports.csv, it is constructed as
`<images_dir>/<image_id>`.

On Kaggle, a typical setup is to add the "IU X-Ray Chest X-rays" dataset
and point `dataset.root_dir` / `dataset.images_dir` / `dataset.reports_csv`
at the mounted /kaggle/input/... paths (see notebooks/01_download_dataset.ipynb).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils.io import resolve_path
from utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"image_id", "report"}


def load(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the IU X-Ray dataset into a standardized DataFrame.

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
    images_dir = resolve_path(ds_cfg["images_dir"])

    if not reports_csv.exists():
        raise FileNotFoundError(
            f"IU X-Ray reports CSV not found at {reports_csv}. "
            "See notebooks/01_download_dataset.ipynb for setup instructions."
        )

    df = pd.read_csv(reports_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"reports.csv is missing required columns: {missing}")

    if "split" in df.columns and ds_cfg.get("split"):
        df = df[df["split"] == ds_cfg["split"]].reset_index(drop=True)

    if "image_path" not in df.columns:
        df["image_path"] = df["image_id"].apply(lambda fname: str(images_dir / fname))

    df = df.rename(columns={"report": "ground_truth_report"})
    df = df[["image_id", "image_path", "ground_truth_report"]].dropna(
        subset=["ground_truth_report"]
    )

    max_samples = ds_cfg.get("max_samples")
    if max_samples:
        df = df.head(int(max_samples)).reset_index(drop=True)

    logger.info("Loaded IU X-Ray dataset: %d samples (split=%s)", len(df), ds_cfg.get("split"))
    return df.reset_index(drop=True)
