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

    rest_travel_cols = [
        "home_rest_days",
        "away_rest_days",
        "rest_days_diff",
        "home_travel_miles",
        "away_travel_miles",
        "travel_miles_diff",
        "home_short_week",
        "away_short_week",
        "home_long_rest",
        "away_long_rest",
        "rest_advantage_bucket",
    ]
    available_rest_cols = [col for col in rest_travel_cols if col in schedule_df.columns]
    if available_rest_cols:
        rest_frame = schedule_df[["game_id"] + available_rest_cols].drop_duplicates("game_id")
        game_level = game_level.merge(rest_frame, on="game_id", how="left")
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
    allowed_suffixes = ("_avg", "_games_played", "_rest_days", "_travel_miles")
    for col in dataset.columns:
        if not (col.startswith("home_") or col.startswith("away_")):
            continue
        if col in {"home_points", "away_points"}:
            continue
        if col.endswith("_market_prob"):
            continue
        if not col.endswith(allowed_suffixes):
            continue
        cols.append(col)
    for extra in ("rest_days_diff", "travel_miles_diff"):
        if extra in dataset.columns:
            cols.append(extra)
    return sorted(set(cols))


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
        ["game_id", "season", "week", "team", "market_implied_prob"]
    ].rename(
        columns={
            "team": "home_team",
            "market_implied_prob": "market_prob",
        }
    )

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


def compute_bias_reports(
    moneyline_results: list[models.ModelResult],
    schedule_df: pd.DataFrame,
    *,
    use_calibrated: bool = True,
    min_games: int = 12,
    bucket_size: float = 0.1,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Generate calibration bias summaries for trained classifiers."""

    home_market = schedule_df[["game_id", "season", "week", "home_team", "home_moneyline"]].copy()
    if "home_moneyline" in home_market.columns:
        home_market["market_prob"] = data_load.convert_moneyline_to_probability(home_market["home_moneyline"])
    elif "home_market_prob" in schedule_df.columns:
        home_market["market_prob"] = schedule_df["home_market_prob"]
    home_market = home_market.drop(columns=["home_moneyline"], errors="ignore")

    reports: dict[str, dict[str, pd.DataFrame]] = {}
    for result in moneyline_results:
        reports[result.model_name] = models.compute_calibration_bias(
            result,
            home_market,
            use_calibrated=use_calibrated,
            min_games=min_games,
            bucket_size=bucket_size,
        )
    return reports
