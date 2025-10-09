"""High-level orchestration for the Project Beat Vegas pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from . import data_load, features, models

LOGGER = logging.getLogger(__name__)


def load_core_datasets(seasons: Iterable[int], cache_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load schedule, weekly data, and play-by-play for the requested seasons."""

    schedule = data_load.load_schedule(seasons)
    weekly = data_load.load_weekly_data(seasons)
    pbp = data_load.load_play_by_play(seasons, cache_dir=cache_dir)
    return {"schedule": schedule, "weekly": weekly, "pbp": pbp}


def build_model_ready_frame(
    weekly_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    feature_config: features.FeatureConfig | None = None,
) -> pd.DataFrame:
    """Combine weekly team statistics with schedule metadata and engineer features."""

    schedule_long = data_load.explode_schedule(schedule_df)
    enriched_weekly = data_load.harmonize_weekly_with_schedule(weekly_df, schedule_long)
    engineered = features.prepare_weekly_features(enriched_weekly, feature_config)
    engineered = engineered.merge(
        schedule_long[["game_id", "team", "market_total", "market_implied_prob"]],
        on=["game_id", "team"],
        how="left",
    )
    game_level = features.to_game_level(engineered)
    if "home_market_implied_prob" in game_level.columns:
        game_level.rename(
            columns={
                "home_market_implied_prob": "home_market_prob",
                "away_market_implied_prob": "away_market_prob",
            },
            inplace=True,
        )
    return game_level


def train_baseline_models(
    dataset: pd.DataFrame,
    feature_cols: Sequence[str],
    validation_seasons: Sequence[int],
) -> dict[str, list[models.ModelResult]]:
    """Train LightGBM-focused baselines for moneyline and totals."""

    split_config = models.SplitConfig(validation_seasons=validation_seasons)
    moneyline_results = models.train_moneyline_models(dataset, feature_cols, config=split_config)
    totals_results = models.train_total_models(dataset, feature_cols, target_col="total_points", config=split_config)
    return {
        "moneyline": moneyline_results,
        "totals": totals_results,
    }


def default_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return a heuristic set of model features suitable for baselines."""

    cols = []
    for col in dataset.columns:
        if not (col.startswith("home_") or col.startswith("away_")):
            continue
        if col in {"home_points", "away_points"}:
            continue
        if col.endswith("_market_prob"):
            continue
        cols.append(col)
    return sorted(cols)


def detect_edges(
    moneyline_results: list[models.ModelResult],
    totals_results: list[models.ModelResult],
    schedule_df: pd.DataFrame,
    edge_threshold_moneyline: float = 0.05,
    edge_threshold_totals: float = 3.0,
) -> dict[str, pd.DataFrame]:
    """Compare model predictions against market odds and totals."""

    schedule_long = data_load.explode_schedule(schedule_df)
    home_market = schedule_long[schedule_long["home_away"] == "home"][
        ["game_id", "market_implied_prob"]
    ].rename(columns={"market_implied_prob": "market_prob"})

    totals_market = schedule_df[["game_id", "total_line"]].rename(columns={"total_line": "market_total"})

    edges = {}
    for result in moneyline_results:
        edges[result.model_name] = models.detect_moneyline_mispricing(
            result.predictions,
            home_market,
            threshold=edge_threshold_moneyline,
        )
    for result in totals_results:
        edges[result.model_name] = models.detect_total_mispricing(
            result.predictions,
            totals_market,
            threshold=edge_threshold_totals,
        )
    return edges
