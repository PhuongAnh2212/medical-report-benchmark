"""
evaluation/evaluate.py

Evaluates every predictions.csv file in outputs/predictions/ using every
metric configured in config["evaluation"]["metrics"], and writes one
per-model result CSV to results/<model_name>_results.csv.

Usage:
    python -m evaluation.evaluate                       # evaluate all prediction files
    python -m evaluation.evaluate --model qwen25_vl      # evaluate a single model's predictions
    python -m evaluation.evaluate --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from metrics import compute_metrics
from utils.io import ensure_dir, list_prediction_files, load_config, read_predictions_csv, resolve_path
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _model_name_from_predictions_path(path: Path) -> str:
    """Recover the model name from a "<model_name>_predictions.csv" filename."""
    return path.stem.replace("_predictions", "")


def evaluate_predictions_file(path: Path, metric_names: List[str]) -> Dict[str, Any]:
    """Compute all configured metrics for a single predictions CSV.

    Args:
        path: Path to a "<model_name>_predictions.csv" file.
        metric_names: Metric names to compute (subset of metrics.METRIC_REGISTRY).

    Returns:
        Dict with a "Model" key plus one key per computed metric.
    """
    df = read_predictions_csv(path)
    df = df.dropna(subset=["prediction", "ground_truth"])
    df = df[df["prediction"].str.strip() != ""]

    if df.empty:
        logger.warning("No valid (non-empty) predictions in %s; skipping.", path)
        return {}

    predictions = df["prediction"].astype(str).tolist()
    references = df["ground_truth"].astype(str).tolist()

    model_name = _model_name_from_predictions_path(path)
    logger.info("Evaluating %s on %d samples using metrics=%s", model_name, len(df), metric_names)

    scores = compute_metrics(metric_names, predictions, references)
    scores["Model"] = model_name
    scores["NumSamples"] = len(df)
    return scores


def evaluate_all(config: Dict[str, Any], model_filter: str | None = None) -> pd.DataFrame:
    """Evaluate all (or one) prediction files and write per-model result CSVs.

    Args:
        config: Full benchmark config.
        model_filter: If provided, only evaluate "<model_filter>_predictions.csv".

    Returns:
        DataFrame with one row per evaluated model.
    """
    eval_cfg = config["evaluation"]
    predictions_files = list_prediction_files(eval_cfg["predictions_dir"])

    if model_filter:
        predictions_files = [
            p for p in predictions_files if _model_name_from_predictions_path(p) == model_filter
        ]

    if not predictions_files:
        logger.warning(
            "No prediction files found in %s. Run inference/generate_reports.py first.",
            resolve_path(eval_cfg["predictions_dir"]),
        )
        return pd.DataFrame()

    results_dir = ensure_dir(eval_cfg["results_dir"])
    all_rows = []

    for pred_file in predictions_files:
        row = evaluate_predictions_file(pred_file, eval_cfg["metrics"])
        if not row:
            continue
        all_rows.append(row)

        # Persist a per-model result file too, useful for CI / debugging.
        model_name = row["Model"]
        pd.DataFrame([row]).to_csv(results_dir / f"{model_name}_results.csv", index=False)

    results_df = pd.DataFrame(all_rows)
    logger.info("Evaluated %d model(s)", len(results_df))
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate radiology report predictions.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to base config YAML")
    parser.add_argument("--model", default=None, help="Only evaluate this model's predictions.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config["logging"]["level"],
        log_dir=config["logging"]["log_dir"] if config["logging"].get("log_to_file") else None,
        log_filename="evaluate.log",
    )

    evaluate_all(config, model_filter=args.model)


if __name__ == "__main__":
    main()
