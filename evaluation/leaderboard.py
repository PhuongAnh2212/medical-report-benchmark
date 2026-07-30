"""
evaluation/leaderboard.py

Aggregates every "<model>_results.csv" file in results/ into a single
sorted leaderboard.csv.

Usage:
    python -m evaluation.leaderboard
    python -m evaluation.leaderboard --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import pandas as pd

from utils.io import load_config, resolve_path
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Preferred column ordering for the final leaderboard.csv; any extra
# columns present in the underlying result files (e.g. CheXbert/RadGraph
# scores once implemented) are appended after these.
PREFERRED_COLUMN_ORDER = [
    "Model",
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "ROUGE-L",
    "METEOR",
    "CIDEr",
    "NumSamples",
]


def build_leaderboard(config: Dict[str, Any]) -> pd.DataFrame:
    """Collect all per-model result CSVs into a single sorted leaderboard.

    Args:
        config: Full benchmark config.

    Returns:
        The leaderboard DataFrame (also written to disk as a side effect).
    """
    lb_cfg = config["leaderboard"]
    results_dir = resolve_path(lb_cfg["results_dir"])

    result_files = sorted(results_dir.glob("*_results.csv"))
    if not result_files:
        logger.warning(
            "No *_results.csv files found in %s. Run evaluation/evaluate.py first.", results_dir
        )
        return pd.DataFrame()

    rows = [pd.read_csv(f) for f in result_files]
    leaderboard_df = pd.concat(rows, ignore_index=True)

    ordered_cols = [c for c in PREFERRED_COLUMN_ORDER if c in leaderboard_df.columns]
    extra_cols = [c for c in leaderboard_df.columns if c not in ordered_cols]
    leaderboard_df = leaderboard_df[ordered_cols + extra_cols]

    sort_by = lb_cfg.get("sort_by", "BLEU-4")
    if sort_by in leaderboard_df.columns:
        leaderboard_df = leaderboard_df.sort_values(
            by=sort_by, ascending=lb_cfg.get("ascending", False)
        ).reset_index(drop=True)

    output_path = resolve_path(lb_cfg["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_df.to_csv(output_path, index=False)
    logger.info("Wrote leaderboard with %d model(s) to %s", len(leaderboard_df), output_path)

    return leaderboard_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the model comparison leaderboard.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to base config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config["logging"]["level"],
        log_dir=config["logging"]["log_dir"] if config["logging"].get("log_to_file") else None,
        log_filename="leaderboard.log",
    )

    leaderboard_df = build_leaderboard(config)
    if not leaderboard_df.empty:
        print(leaderboard_df.to_string(index=False))


if __name__ == "__main__":
    main()
