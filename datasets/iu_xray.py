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
    """Load the Kaggle Indiana University Chest X-Ray dataset."""

    ds_cfg = config["dataset"]

    root = resolve_path(ds_cfg["root_dir"])
    images_dir = root / "images"

    reports_csv = root / "indiana_reports.csv"
    projections_csv = root / "indiana_projections.csv"

    if not reports_csv.exists():
        raise FileNotFoundError(f"Cannot find {reports_csv}")

    if not projections_csv.exists():
        raise FileNotFoundError(f"Cannot find {projections_csv}")

    reports = pd.read_csv(reports_csv)
    projections = pd.read_csv(projections_csv)

    # Build the ground-truth report
    reports["ground_truth_report"] = (
        reports["findings"].fillna("").str.strip()
        + "\n"
        + reports["impression"].fillna("").str.strip()
    ).str.strip()

    # Merge reports with image filenames
    df = projections.merge(
        reports[["uid", "ground_truth_report"]],
        on="uid",
        how="inner",
    )

    # Optional: keep only frontal images
    df = df[df["projection"] == "Frontal"].copy()

    df["image_id"] = df["filename"]

    df["image_path"] = df["filename"].apply(
        lambda x: str(images_dir / x)
    )

    # Remove rows where the image doesn't exist
    df = df[df["image_path"].apply(lambda p: Path(p).exists())]

    df = df[
        [
            "image_id",
            "image_path",
            "ground_truth_report",
        ]
    ]

    max_samples = ds_cfg.get("max_samples")
    if max_samples:
        df = df.head(int(max_samples))

    logger.info("Loaded %d IU X-Ray samples", len(df))

    return df.reset_index(drop=True)
