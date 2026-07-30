"""
utils/io.py

Small, dependency-light helpers for loading YAML configuration files,
resolving paths relative to the repository root, and reading/writing
prediction CSVs. Centralizing this logic keeps every other module free
of hardcoded paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

# The repository root is defined as the parent of the `utils/` directory.
# This lets every module resolve relative paths (e.g. "outputs/predictions")
# consistently, regardless of the current working directory the script was
# launched from (important on Kaggle, where notebooks often `cd` around).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root.

    Absolute paths (e.g. Kaggle input paths like /kaggle/input/...) are
    returned unchanged. Relative paths (e.g. "outputs/predictions") are
    joined with REPO_ROOT.

    Args:
        path: A path string or Path, absolute or relative.

    Returns:
        An absolute, resolved Path object.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to the YAML file, relative or absolute.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(
    default_config_path: str | Path = "configs/default.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load default.yaml and apply an optional shallow dictionary of overrides.

    Overrides are merged one level deep so callers can do, e.g.,
    `overrides={"model": {"name": "internvl3"}}` without clobbering the
    rest of the `model` section.

    Args:
        default_config_path: Path to the base YAML config.
        overrides: Optional dict of overrides, merged one level deep.

    Returns:
        The merged configuration dictionary.
    """
    config = load_yaml(default_config_path)
    if overrides:
        for section, values in overrides.items():
            if isinstance(values, dict) and isinstance(config.get(section), dict):
                config[section].update(values)
            else:
                config[section] = values
    return config


def load_model_registry(models_config_path: str | Path = "configs/models.yaml") -> Dict[str, Any]:
    """Load the model registry from configs/models.yaml.

    Args:
        models_config_path: Path to the model registry YAML file.

    Returns:
        Dictionary mapping model_name -> model metadata.
    """
    return load_yaml(models_config_path)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path, relative or absolute.

    Returns:
        The resolved, now-guaranteed-to-exist Path.
    """
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_predictions_csv(path: str | Path) -> pd.DataFrame:
    """Read a predictions CSV into a DataFrame with the expected schema.

    Expected columns: image_id, ground_truth, prediction

    Args:
        path: Path to the predictions CSV.

    Returns:
        DataFrame with the predictions.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Predictions file not found: {resolved}")

    df = pd.read_csv(resolved)
    required_cols = {"image_id", "ground_truth", "prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file {resolved} is missing columns: {missing}")
    return df


def write_predictions_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a predictions DataFrame to CSV, creating parent dirs as needed.

    Args:
        df: DataFrame with columns image_id, ground_truth, prediction.
        path: Destination path, relative or absolute.

    Returns:
        The resolved path the file was written to.
    """
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)
    df.to_csv(resolved, index=False)
    return resolved


def list_prediction_files(predictions_dir: str | Path) -> list[Path]:
    """List all *_predictions.csv files in a directory.

    Args:
        predictions_dir: Directory to scan.

    Returns:
        Sorted list of matching file paths.
    """
    resolved = resolve_path(predictions_dir)
    if not resolved.exists():
        return []
    return sorted(resolved.glob("*_predictions.csv"))
