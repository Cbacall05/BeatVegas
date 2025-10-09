"""Command-line demo for Project Beat Vegas baseline pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from beat_vegas import features, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project Beat Vegas baseline models.")
    parser.add_argument("--seasons", nargs="*", type=int, default=[2021, 2022], help="NFL seasons to ingest")
    parser.add_argument(
        "--validation-season",
        type=int,
        default=2023,
        help="Season to hold out for validation",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/pbp"),
        help="Directory for caching play-by-play parquet files.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=200,
        help="Optional limit on number of games to train on (useful for quick demos).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    LOGGER.info("Loading datasets for seasons: %s", args.seasons)
    datasets = pipeline.load_core_datasets(args.seasons, cache_dir=args.cache_dir)

    feature_config = features.FeatureConfig()
    model_df = pipeline.build_model_ready_frame(datasets["weekly"], datasets["schedule"], feature_config)

    if args.max_games:
        model_df = model_df.sort_values(["season", "week"]).head(args.max_games)

    feature_cols = pipeline.default_feature_columns(model_df)
    LOGGER.info("Training with %d features", len(feature_cols))

    results = pipeline.train_baseline_models(model_df, feature_cols, validation_seasons=[args.validation_season])

    for group, group_results in results.items():
        LOGGER.info("%s results:", group.upper())
        for result in group_results:
            LOGGER.info("  %s -> %s", result.model_name, result.metrics)

    # Detect edges using schedule data from validation season
    val_schedule = datasets["schedule"][datasets["schedule"]["season"] == args.validation_season]
    edges = pipeline.detect_edges(
        results["moneyline"],
        results["totals"],
        val_schedule,
    )
    for model_name, df in edges.items():
        if df.empty:
            LOGGER.info("No edges detected for %s", model_name)
        else:
            LOGGER.info("Top edges for %s:\n%s", model_name, df.head())


if __name__ == "__main__":
    main()
