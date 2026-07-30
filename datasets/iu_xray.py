"""
datasets/iu_xray.py

Loader for the IU X-Ray (Indiana University Chest X-Ray) dataset, backed by
the public Kaggle dataset:

    https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

Expected on-disk layout of the raw Kaggle dataset (auto-detected, see
`_locate_dataset_root`):

    <root>/
        images/
            images_normalized/
                1_IM-0001-3001.dcm.png
                1_IM-0001-4001.dcm.png
                ...
        indiana_reports.csv       # one row per study (uid): findings, impression, ...
        indiana_projections.csv   # one row per image: uid, filename, projection

`indiana_reports.csv` (study-level) and `indiana_projections.csv`
(image-level) are merged on `uid` to produce one row per image, with the
study's report text duplicated across every image belonging to that study.
This mirrors how prior IU X-Ray preprocessing pipelines (e.g. R2Gen) build
their per-image annotations from these same two files.

Dataset root resolution order:
    1. `dataset.root_dir` from configs/default.yaml, if set.
    2. Well-known Kaggle mount points:
       - /kaggle/input/chest-xrays-indiana-university
       - /kaggle/input/datasets/raddar/chest-xrays-indiana-university

Backwards compatibility: if a pre-processed `reports.csv` (the format this
loader originally expected) is found -- either at `dataset.reports_csv` or
as `<root>/reports.csv` -- it takes priority over the raw Kaggle CSVs, since
it may already encode a train/val/test split that the raw data does not.

Resolution priority is therefore:
    1. Legacy `reports.csv` (pre-processed, possibly split-aware).
    2. Raw Kaggle dataset (`indiana_reports.csv` + `indiana_projections.csv`).
    3. FileNotFoundError with setup instructions (see README.md).
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.io import resolve_path
from utils.logging import get_logger

logger = get_logger(__name__)

# Directories Kaggle is known to mount this dataset under, checked (in this
# order) whenever `dataset.root_dir` is not explicitly configured.
KAGGLE_ROOT_CANDIDATES: tuple[str, ...] = (
    "/kaggle/input/chest-xrays-indiana-university",
    "/kaggle/input/datasets/raddar/chest-xrays-indiana-university",
)

# Sub-paths (relative to the dataset root) where the normalized PNGs live.
# The official Kaggle export nests them under images/images_normalized/; a
# flat images/ directory is also accepted in case a user re-packaged it.
IMAGE_SUBDIR_CANDIDATES: tuple[str, ...] = (
    "images/images_normalized",
    "images",
)

REPORTS_CSV_NAME = "indiana_reports.csv"
PROJECTIONS_CSV_NAME = "indiana_projections.csv"
LEGACY_REPORTS_CSV_NAME = "reports.csv"

REQUIRED_LEGACY_COLUMNS = {"image_id", "report"}
REQUIRED_REPORT_COLUMNS = {"uid", "findings", "impression"}
REQUIRED_PROJECTION_COLUMNS = {"uid", "filename"}


def _candidate_roots(ds_cfg: Dict[str, Any]) -> List[Path]:
    """Build the ordered list of directories to search for the dataset."""
    roots: List[Path] = []
    configured_root = ds_cfg.get("root_dir")
    if configured_root:
        roots.append(resolve_path(configured_root))
    roots.extend(Path(p) for p in KAGGLE_ROOT_CANDIDATES)
    return roots


def _find_legacy_reports_csv(ds_cfg: Dict[str, Any], roots: List[Path]) -> Optional[Path]:
    """Locate a pre-processed reports.csv, if one exists.

    Args:
        ds_cfg: The `dataset` section of the config.
        roots: Candidate dataset root directories, in priority order.

    Returns:
        Path to the legacy reports.csv, or None if not found.
    """
    configured_reports_csv = ds_cfg.get("reports_csv")
    if configured_reports_csv:
        path = resolve_path(configured_reports_csv)
        if path.exists():
            return path

    for root in roots:
        candidate = root / LEGACY_REPORTS_CSV_NAME
        if candidate.exists():
            return candidate
    return None


def _locate_dataset_root(roots: List[Path]) -> Path:
    """Find the root directory of the raw Kaggle IU X-Ray dataset.

    Args:
        roots: Candidate dataset root directories, in priority order.

    Returns:
        The first candidate directory that contains indiana_reports.csv.

    Raises:
        FileNotFoundError: If no candidate directory contains the dataset.
    """
    for root in roots:
        if (root / REPORTS_CSV_NAME).exists():
            return root

    raise FileNotFoundError(
        "Could not locate the IU X-Ray dataset. Checked for a legacy "
        f"{LEGACY_REPORTS_CSV_NAME} and for the raw Kaggle dataset "
        f"({REPORTS_CSV_NAME}) in: {', '.join(str(r) for r in roots)}. "
        "Attach the 'raddar/chest-xrays-indiana-university' dataset on "
        "Kaggle, or set dataset.root_dir in configs/default.yaml to point "
        "at it. See README.md for full setup instructions."
    )


def _find_image_dir_recursive(root: Path, sample_filename: str) -> Optional[Path]:
    """Recursively search under `root` for the directory containing a file.

    Used as a last-resort fallback when neither of the known Kaggle
    sub-paths (`images/images_normalized`, `images`) exists, e.g. because a
    dataset mount nests the PNGs under a different or extra path segment.

    Args:
        root: Dataset root directory to search under.
        sample_filename: An exact filename (e.g. from indiana_projections.csv)
            to locate.

    Returns:
        The parent directory of the first match, or None if not found.
    """
    matches = list(root.rglob(glob.escape(sample_filename)))
    return matches[0].parent if matches else None


def _locate_images_dir(
    ds_cfg: Dict[str, Any], root: Path, sample_filename: Optional[str] = None
) -> Path:
    """Find the directory holding the normalized PNGs under `root`.

    Search order:
        1. `dataset.images_dir` from config, if it exists on disk.
        2. `<root>/images/images_normalized`
        3. `<root>/images`
        4. Recursive fallback: any directory under `root` containing
           `sample_filename` (only attempted if `sample_filename` is given).

    Args:
        ds_cfg: The `dataset` section of the config.
        root: Dataset root directory.
        sample_filename: An exact image filename known to exist in the
            dataset (e.g. the first row of indiana_projections.csv), used
            only for the recursive fallback.

    Returns:
        The resolved image directory.

    Raises:
        FileNotFoundError: If no candidate directory can be located.
    """
    configured_images_dir = ds_cfg.get("images_dir")
    if configured_images_dir:
        configured = resolve_path(configured_images_dir)
        if configured.is_dir():
            logger.info("Using image directory: %s", configured)
            return configured

    for sub in IMAGE_SUBDIR_CANDIDATES:
        candidate = root / sub
        if candidate.is_dir():
            logger.info("Using image directory: %s", candidate)
            return candidate

    if sample_filename:
        found = _find_image_dir_recursive(root, sample_filename)
        if found is not None:
            logger.info("Using image directory (recursive fallback): %s", found)
            return found

    checked = ", ".join(str(root / sub) for sub in IMAGE_SUBDIR_CANDIDATES)
    raise FileNotFoundError(
        f"Could not locate an image directory under {root}. Checked: "
        f"{checked}"
        + (f", and recursively searched for '{sample_filename}'" if sample_filename else "")
        + ". Set dataset.images_dir in configs/default.yaml to override."
    )


def _build_ground_truth_report(row: pd.Series) -> Optional[str]:
    """Combine a study's `findings` and `impression` fields into one report.

    Args:
        row: A row from indiana_reports.csv.

    Returns:
        The combined report text, or None if both fields are empty/NaN.
    """
    findings = str(row["findings"]).strip() if pd.notna(row.get("findings")) else ""
    impression = str(row["impression"]).strip() if pd.notna(row.get("impression")) else ""
    parts = [part for part in (findings, impression) if part]
    if not parts:
        return None
    return "\n".join(parts)


def _load_legacy_reports_csv(
    reports_csv: Path, ds_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Load a pre-processed reports.csv (the loader's original format).

    Args:
        reports_csv: Path to the legacy reports.csv.
        ds_cfg: The `dataset` section of the config.

    Returns:
        DataFrame with columns [image_id, image_path, ground_truth_report].

    Raises:
        ValueError: If required columns are missing.
    """
    logger.info("Found legacy %s; using it instead of the raw Kaggle CSVs.", reports_csv)
    df = pd.read_csv(reports_csv)
    missing = REQUIRED_LEGACY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{reports_csv} is missing required columns: {missing}")

    if "split" in df.columns and ds_cfg.get("split"):
        df = df[df["split"] == ds_cfg["split"]].reset_index(drop=True)

    if ds_cfg.get("frontal_only"):
        if "projection" in df.columns:
            before = len(df)
            df = df[df["projection"] == "Frontal"].copy()
            logger.info(
                "dataset.frontal_only=True: kept %d/%d Frontal images.", len(df), before
            )
        else:
            logger.warning(
                "dataset.frontal_only=True was requested, but %s has no "
                "'projection' column; keeping all images.", reports_csv
            )

    if "image_path" not in df.columns:
        configured_images_dir = ds_cfg.get("images_dir")
        images_dir = (
            resolve_path(configured_images_dir)
            if configured_images_dir
            else reports_csv.parent / "images"
        )
        df["image_path"] = df["image_id"].apply(lambda fname: str(images_dir / fname))

    df = df.rename(columns={"report": "ground_truth_report"})
    return df[["image_id", "image_path", "ground_truth_report"]]


def _load_kaggle_raw(root: Path, ds_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load and merge indiana_reports.csv / indiana_projections.csv.

    Args:
        root: Dataset root directory (contains both CSVs).
        ds_cfg: The `dataset` section of the config.

    Returns:
        DataFrame with columns [image_id, image_path, ground_truth_report],
        one row per image.

    Raises:
        FileNotFoundError: If indiana_projections.csv is missing, or if no
            image directory can be located under `root`.
        ValueError: If either CSV is missing required columns.
    """
    reports_path = root / REPORTS_CSV_NAME
    projections_path = root / PROJECTIONS_CSV_NAME

    if not projections_path.exists():
        raise FileNotFoundError(
            f"{PROJECTIONS_CSV_NAME} not found at {projections_path}. "
            "Both indiana_reports.csv and indiana_projections.csv are "
            "required from the Kaggle dataset."
        )

    logger.info("Loading raw Kaggle IU X-Ray dataset from %s", root)
    reports_df = pd.read_csv(reports_path)
    projections_df = pd.read_csv(projections_path)

    missing = REQUIRED_REPORT_COLUMNS - set(reports_df.columns)
    if missing:
        raise ValueError(f"{reports_path} is missing required columns: {missing}")
    missing = REQUIRED_PROJECTION_COLUMNS - set(projections_df.columns)
    if missing:
        raise ValueError(f"{projections_path} is missing required columns: {missing}")

    sample_filename = str(projections_df["filename"].iloc[0]) if len(projections_df) else None
    images_dir = _locate_images_dir(ds_cfg, root, sample_filename=sample_filename)

    reports_df = reports_df.copy()
    reports_df["ground_truth_report"] = reports_df.apply(_build_ground_truth_report, axis=1)
    reports_df = reports_df.dropna(subset=["ground_truth_report"])

    merged = projections_df.merge(
        reports_df[["uid", "ground_truth_report"]], on="uid", how="inner"
    )
    merged = merged.rename(columns={"filename": "image_id"})
    merged["image_path"] = merged["image_id"].apply(lambda fname: str(images_dir / fname))

    # `projection` (Frontal/Lateral) must still be present on `merged` at this
    # point -- it is only used here, immediately after the merge, and is
    # dropped by the final column selection below.
    if ds_cfg.get("frontal_only"):
        before = len(merged)
        merged = merged[merged["projection"] == "Frontal"].copy()
        logger.info(
            "dataset.frontal_only=True: kept %d/%d Frontal images.", len(merged), before
        )

    if ds_cfg.get("split"):
        logger.warning(
            "dataset.split=%r was requested, but the raw Kaggle IU X-Ray "
            "dataset does not provide train/val/test split information. "
            "Using all %d image-level samples; pre-split the data yourself "
            "(e.g. in notebooks/02_preprocess.ipynb) if you need "
            "reproducible splits.",
            ds_cfg["split"],
            len(merged),
        )

    # Only now, after any projection-based filtering, do we reduce to the
    # standard interface columns expected by the rest of the benchmark.
    return merged[["image_id", "image_path", "ground_truth_report"]]


def _filter_existing_images(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose image_path does not exist on disk.

    Logs the merged/existing/missing row counts before filtering, so a
    misconfigured image directory shows up as an explicit count instead of
    silently producing an empty dataset. Missing files are then logged as
    warnings (capped to avoid flooding the log) instead of raising, so a
    handful of bad paths don't crash the run.

    Args:
        df: DataFrame with an `image_path` column.

    Returns:
        The subset of `df` whose image_path exists, reindexed.
    """
    exists_mask = df["image_path"].apply(lambda p: Path(p).exists())
    total = len(df)
    n_existing = int(exists_mask.sum())
    n_missing = total - n_existing
    logger.info(
        "Merged rows: %d | Existing images: %d | Missing images: %d",
        total, n_existing, n_missing,
    )
    if n_missing:
        for path in df.loc[~exists_mask, "image_path"].head(10):
            logger.warning("Skipping missing image: %s", path)
        if n_missing > 10:
            logger.warning("... and %d more missing images.", n_missing - 10)
    return df[exists_mask].reset_index(drop=True)


def load(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the IU X-Ray dataset into a standardized DataFrame.

    Supports both a legacy pre-processed `reports.csv` and the raw public
    Kaggle dataset (`indiana_reports.csv` + `indiana_projections.csv`),
    auto-detecting the dataset root when `dataset.root_dir` is not set.
    See the module docstring for the full resolution order.

    Args:
        config: Full benchmark config; uses the `dataset` section.

    Returns:
        DataFrame with columns [image_id, image_path, ground_truth_report],
        containing only samples whose image file exists on disk.

    Raises:
        FileNotFoundError: If neither a legacy reports.csv nor the raw
            Kaggle dataset can be located.
        ValueError: If a located CSV is missing required columns.
    """
    ds_cfg = config["dataset"]
    roots = _candidate_roots(ds_cfg)

    legacy_reports_csv = _find_legacy_reports_csv(ds_cfg, roots)
    if legacy_reports_csv is not None:
        df = _load_legacy_reports_csv(legacy_reports_csv, ds_cfg)
    else:
        root = _locate_dataset_root(roots)
        df = _load_kaggle_raw(root, ds_cfg)

    n_before = len(df)
    df = df.dropna(subset=["ground_truth_report"])
    df = _filter_existing_images(df)

    max_samples = ds_cfg.get("max_samples")
    if max_samples:
        df = df.head(int(max_samples)).reset_index(drop=True)

    logger.info(
        "Loaded IU X-Ray dataset: %d/%d samples usable (split=%s)",
        len(df),
        n_before,
        ds_cfg.get("split"),
    )
    return df.reset_index(drop=True)
