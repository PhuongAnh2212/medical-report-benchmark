"""
inference/generate_reports.py

Runs a single model over a dataset and writes `outputs/predictions/<model>_predictions.csv`.

Usage:
    python -m inference.generate_reports --model qwen25_vl
    python -m inference.generate_reports --model internvl3 --config configs/default.yaml
    python -m inference.generate_reports --model qwen25_vl --max-samples 20   # quick debug run

This script is deliberately the ONLY place that ties together dataset
loading, model building, and prediction writing -- evaluation and
leaderboard code never import model classes directly, and model wrappers
never import dataset code. This keeps the pipeline modular: adding a
model or dataset only requires touching its own file plus a registry entry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from models import build_model
from utils.image import load_image
from utils.io import ensure_dir, load_config, resolve_path, write_predictions_csv
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def load_prompt_template(config: Dict[str, Any]) -> str:
    """Load the prompt template file referenced in config["prompt"]["template_file"]."""
    template_path = resolve_path(config["prompt"]["template_file"])
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8").strip()


def _output_path(config: Dict[str, Any], model_name: str) -> Path:
    inf_cfg = config["inference"]
    filename = inf_cfg["output_filename"].format(model_name=model_name)
    return resolve_path(inf_cfg["output_dir"]) / filename


def run_inference(config: Dict[str, Any], model_name: str) -> Path:
    """Run a model over the configured dataset split and save predictions.csv.

    Args:
        config: Full benchmark config.
        model_name: Key into configs/models.yaml.

    Returns:
        Path to the written predictions CSV.
    """
    prompt_template = load_prompt_template(config)
    dataset_df = load_dataset(config)

    output_path = _output_path(config, model_name)
    ensure_dir(output_path.parent)

    inf_cfg = config["inference"]
    already_done: set[str] = set()
    existing_rows: list[dict] = []

    if inf_cfg.get("resume", True) and output_path.exists():
        existing_df = pd.read_csv(output_path)
        already_done = set(existing_df["image_id"].astype(str))
        existing_rows = existing_df.to_dict("records")
        logger.info("Resuming: found %d existing predictions in %s", len(already_done), output_path)

    model = build_model(model_name, config, prompt_template)
    model.ensure_loaded()

    rows = list(existing_rows)
    save_every = inf_cfg.get("save_every_n", 20)

    pending = dataset_df[~dataset_df["image_id"].astype(str).isin(already_done)]
    logger.info("Generating reports for %d/%d remaining samples", len(pending), len(dataset_df))

    for i, (_, row) in enumerate(tqdm(pending.iterrows(), total=len(pending), desc=f"{model_name}")):
        image = load_image(row["image_path"])
        if image is None:
            logger.warning("Skipping sample %s: image could not be loaded", row["image_id"])
            continue

        try:
            prediction = model.generate(image)
        except Exception as exc:  # noqa: BLE001 - keep the benchmark run alive on single-sample failures
            logger.error("Generation failed for %s: %s", row["image_id"], exc)
            prediction = ""

        rows.append(
            {
                "image_id": row["image_id"],
                "ground_truth": row["ground_truth_report"],
                "prediction": prediction,
            }
        )

        if (i + 1) % save_every == 0:
            write_predictions_csv(pd.DataFrame(rows), output_path)
            logger.info("Checkpoint saved at %d/%d", i + 1, len(pending))

    final_df = pd.DataFrame(rows)
    write_predictions_csv(final_df, output_path)
    logger.info("Wrote %d predictions to %s", len(final_df), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate radiology reports for a given model.")
    parser.add_argument("--model", required=True, help="Model name as registered in configs/models.yaml")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to base config YAML")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug: limit number of dataset samples")
    parser.add_argument("--split", default=None, help="Override dataset split (train/val/test)")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config["logging"]["level"],
        log_dir=config["logging"]["log_dir"] if config["logging"].get("log_to_file") else None,
        log_filename=f"generate_{args.model}.log",
    )

    if args.max_samples is not None:
        config["dataset"]["max_samples"] = args.max_samples
    if args.split is not None:
        config["dataset"]["split"] = args.split

    run_inference(config, args.model)


if __name__ == "__main__":
    main()
