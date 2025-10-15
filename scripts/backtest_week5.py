"""Backtest Week 5 of the 2025 NFL season using a rolling 4-game window.

This script simulates running the matchup models immediately before Week 5
kicked off, trains on seasons 2018-2024, and evaluates the predictions once
actual Week 5 results are known.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from beat_vegas import data_load, models
from matchup_predictor import (
    build_game_level_dataset,
    build_team_game_records,
    predict_upcoming_games,
    select_feature_columns,
)

TRAIN_SEASONS: tuple[int, ...] = tuple(range(2018, 2025))
TARGET_SEASON = 2025
TARGET_WEEK = 5
ROLLING_WINDOW = 4


@dataclass
class BacktestResult:
    summary: pd.DataFrame
    accuracy: float
    log_loss: float
    brier: float
    totals_mae: float
    totals_rmse: float
    market_mae: float
    market_rmse: float
    spread_mae: float
    spread_rmse: float
    market_spread_mae: float
    market_spread_rmse: float
    spread_slope: float
    spread_intercept: float


def _fit_spread_mapping(
    schedule_df: pd.DataFrame,
    *,
    validation_season: int,
    week: int,
) -> tuple[float, float]:
    """Fit a linear mapping from home win probability to spread."""

    historical = schedule_df.copy()
    mask_time = (historical["season"] < validation_season) | (
        (historical["season"] == validation_season) & (historical["week"] < week)
    )
    historical = historical.loc[mask_time]

    probabilities = data_load.convert_moneyline_to_probability(historical["home_moneyline"])
    spreads = historical["spread_line"]

    mask = probabilities.notna() & spreads.notna()
    if not mask.any():
        raise ValueError("Insufficient historical data to fit spread mapping.")

    prob = probabilities.loc[mask].astype(float).clip(1e-6, 1 - 1e-6)
    spread = spreads.loc[mask].astype(float)
    logit = np.log(prob / (1 - prob))
    slope, intercept = np.polyfit(logit, spread, 1)
    return float(slope), float(intercept)


def _probability_to_spread(probabilities: pd.Series, *, slope: float, intercept: float) -> pd.Series:
    clipped = probabilities.astype(float).clip(1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    return slope * logit + intercept


def _truncate_schedule(schedule_df: pd.DataFrame, *, cutoff_week: int, season: int) -> pd.DataFrame:
    """Remove results at or beyond the cutoff to simulate going back in time."""

    truncated = schedule_df.copy()
    mask_future = (truncated["season"] == season) & (truncated["week"] >= cutoff_week)
    for col in ("home_score", "away_score"):
        if col in truncated.columns:
            truncated.loc[mask_future, col] = np.nan
    return truncated


def _prepare_predictions(schedule_df: pd.DataFrame) -> tuple[pd.DataFrame, list[models.ModelResult], list[models.ModelResult]]:
    team_df = build_team_game_records(schedule_df, rolling_window=ROLLING_WINDOW)
    game_dataset = build_game_level_dataset(team_df)
    dataset_columns = list(game_dataset.columns)
    feature_cols = select_feature_columns(game_dataset)
    split_config = models.SplitConfig(validation_seasons=[TARGET_SEASON])

    moneyline_models = models.train_moneyline_models(game_dataset, feature_cols, config=split_config)
    totals_models = models.train_total_models(game_dataset, feature_cols, target_col="total_points", config=split_config)

    predictions = predict_upcoming_games(
        team_df=team_df,
        schedule_df=schedule_df,
        feature_cols=feature_cols,
        dataset_columns=dataset_columns,
        moneyline_results=moneyline_models,
        totals_results=totals_models,
        target_season=TARGET_SEASON,
        predict_week=TARGET_WEEK,
        home_team=None,
        away_team=None,
        include_all=True,
    )
    return predictions, moneyline_models, totals_models


def _compute_metrics(
    summary: pd.DataFrame,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    home_won = (summary["actual_home_score"] > summary["actual_away_score"]).astype(int)
    preds = summary["avg_home_win_prob"].clip(1e-6, 1 - 1e-6)

    accuracy = float((summary["predicted_winner"] == summary["actual_winner"]).mean())
    brier = float(np.mean((preds - home_won) ** 2))
    log_loss = float(np.mean(-(home_won * np.log(preds) + (1 - home_won) * np.log(1 - preds))))

    totals_error = summary["avg_total_pred"] - summary["actual_total_points"]
    totals_mae = float(np.abs(totals_error).mean())
    totals_rmse = float(np.sqrt(np.mean(totals_error**2)))
    market_error = summary["market_total"] - summary["actual_total_points"]
    market_mae = float(np.abs(market_error).mean())
    market_rmse = float(np.sqrt(np.mean(market_error**2)))

    spread_error = summary["model_spread"] - summary["actual_margin"]
    spread_mae = float(np.abs(spread_error).mean())
    spread_rmse = float(np.sqrt(np.mean(spread_error**2)))
    market_spread_error = summary["market_spread"] - summary["actual_margin"]
    market_spread_mae = float(np.abs(market_spread_error).mean())
    market_spread_rmse = float(np.sqrt(np.mean(market_spread_error**2)))

    return (
        accuracy,
        log_loss,
        brier,
        totals_mae,
        totals_rmse,
        market_mae,
        market_rmse,
        spread_mae,
        spread_rmse,
        market_spread_mae,
        market_spread_rmse,
    )


def backtest_week5(train_seasons: Iterable[int]) -> BacktestResult:
    seasons = tuple(sorted(set(int(season) for season in train_seasons) | {TARGET_SEASON}))
    schedule_full = data_load.load_schedule(seasons)
    schedule_truncated = _truncate_schedule(schedule_full, cutoff_week=TARGET_WEEK, season=TARGET_SEASON)
    slope, intercept = _fit_spread_mapping(
        schedule_full,
        validation_season=TARGET_SEASON,
        week=TARGET_WEEK,
    )

    predictions, _, _ = _prepare_predictions(schedule_truncated)
    if predictions.empty:
        raise RuntimeError("No predictions generated for Week 5.")
    predictions = predictions.copy()
    predictions["model_spread"] = _probability_to_spread(
        predictions["avg_home_win_prob"], slope=slope, intercept=intercept
    )

    actual = schedule_full[
        (schedule_full["season"] == TARGET_SEASON)
        & (schedule_full["week"] == TARGET_WEEK)
        & schedule_full["home_score"].notna()
        & schedule_full["away_score"].notna()
    ][
        [
            "game_id",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
            "total_line",
        ]
    ].copy()
    if actual.empty:
        raise RuntimeError("No actual results found for Week 5 in the schedule data.")

    actual.rename(
        columns={
            "home_score": "actual_home_score",
            "away_score": "actual_away_score",
        },
        inplace=True,
    )
    actual["actual_total_points"] = actual["actual_home_score"] + actual["actual_away_score"]
    actual["actual_margin"] = actual["actual_home_score"] - actual["actual_away_score"]
    actual["actual_winner"] = np.where(
        actual["actual_home_score"] >= actual["actual_away_score"],
        actual["home_team"],
        actual["away_team"],
    )

    merged = predictions.merge(actual, on="game_id", how="inner", suffixes=("_pred", "_actual"))
    if merged.empty:
        raise RuntimeError("Predictions did not align with actual Week 5 games.")

    if "market_total" not in merged.columns:
        for candidate in ("market_total_pred", "total_line"):
            if candidate in merged.columns:
                merged["market_total"] = merged[candidate]
                break
        else:
            merged["market_total"] = np.nan

    if "market_spread" not in merged.columns:
        for candidate in ("spread_line", "home_market_spread"):
            if candidate in merged.columns:
                merged["market_spread"] = merged[candidate]
                break
        else:
            merged["market_spread"] = np.nan

    rename_map = {
        "home_team_pred": "home_team",
        "away_team_pred": "away_team",
    }
    merged.rename(columns={key: value for key, value in rename_map.items() if key in merged.columns}, inplace=True)

    if "avg_total_pred" not in merged.columns:
        total_cols = [col for col in merged.columns if col.endswith("_total")]
        merged["avg_total_pred"] = merged[total_cols].mean(axis=1) if total_cols else np.nan

    merged["model_spread"] = merged["model_spread"].astype(float)
    merged["market_spread"] = merged["market_spread"].astype(float)
    merged["actual_margin"] = merged["actual_margin"].astype(float)
    merged.sort_values("avg_home_win_prob", ascending=False, inplace=True)
    merged["pred_total_vs_market"] = merged["avg_total_pred"] - merged["market_total"]
    merged["pred_vs_actual"] = merged["avg_total_pred"] - merged["actual_total_points"]
    merged["market_vs_actual"] = merged["market_total"] - merged["actual_total_points"]
    merged["spread_model_minus_market"] = merged["model_spread"] - merged["market_spread"]
    merged["spread_model_error"] = merged["model_spread"] - merged["actual_margin"]
    merged["spread_market_error"] = merged["market_spread"] - merged["actual_margin"]

    (
        accuracy,
        log_loss,
        brier,
        totals_mae,
        totals_rmse,
        market_mae,
        market_rmse,
        spread_mae,
        spread_rmse,
        market_spread_mae,
        market_spread_rmse,
    ) = _compute_metrics(merged)

    summary_cols = [
        "game_id",
        "home_team",
        "away_team",
        "avg_home_win_prob",
        "predicted_winner",
        "predicted_win_prob",
        "model_spread",
        "market_spread",
        "actual_margin",
        "spread_model_minus_market",
        "spread_model_error",
        "spread_market_error",
        "avg_total_pred",
        "market_total",
        "pred_total_vs_market",
        "pred_vs_actual",
        "market_vs_actual",
        "actual_winner",
        "actual_total_points",
    ]
    summary = merged[summary_cols].copy()
    summary.rename(
        columns={
            "game_id": "Game ID",
            "home_team": "Home",
            "away_team": "Away",
            "avg_home_win_prob": "Home Win Prob",
            "predicted_winner": "Pred Winner",
            "predicted_win_prob": "Pred Win Prob",
            "model_spread": "Model Spread",
            "market_spread": "Market Spread",
            "actual_margin": "Actual Margin",
            "spread_model_minus_market": "Spread Δ (Model-Market)",
            "spread_model_error": "Spread Error",
            "spread_market_error": "Market Spread Error",
            "avg_total_pred": "Pred Total",
            "market_total": "Market Total",
            "pred_total_vs_market": "Pred - Market",
            "pred_vs_actual": "Pred - Actual",
            "market_vs_actual": "Market - Actual",
            "actual_winner": "Actual Winner",
            "actual_total_points": "Actual Total",
        },
        inplace=True,
    )
    summary["Home Win Prob"] = summary["Home Win Prob"].astype(float)
    summary["Pred Win Prob"] = summary["Pred Win Prob"].astype(float)
    summary["Pred Total"] = summary["Pred Total"].astype(float)
    summary["Market Total"] = summary["Market Total"].astype(float)
    summary["Pred - Market"] = summary["Pred - Market"].astype(float)
    summary["Pred - Actual"] = summary["Pred - Actual"].astype(float)
    summary["Market - Actual"] = summary["Market - Actual"].astype(float)
    summary["Actual Total"] = summary["Actual Total"].astype(float)
    summary["Model Spread"] = summary["Model Spread"].astype(float)
    summary["Market Spread"] = summary["Market Spread"].astype(float)
    summary["Actual Margin"] = summary["Actual Margin"].astype(float)
    summary["Spread Δ (Model-Market)"] = summary["Spread Δ (Model-Market)"].astype(float)
    summary["Spread Error"] = summary["Spread Error"].astype(float)
    summary["Market Spread Error"] = summary["Market Spread Error"].astype(float)

    summary["Home Win Prob"] = summary["Home Win Prob"].map(lambda v: f"{v:.3f}")
    summary["Pred Win Prob"] = summary["Pred Win Prob"].map(lambda v: f"{v:.3f}")
    summary["Pred Total"] = summary["Pred Total"].map(lambda v: f"{v:.1f}")
    summary["Market Total"] = summary["Market Total"].map(lambda v: f"{v:.1f}")
    summary["Pred - Market"] = summary["Pred - Market"].map(lambda v: f"{v:+.1f}")
    summary["Pred - Actual"] = summary["Pred - Actual"].map(lambda v: f"{v:+.1f}")
    summary["Market - Actual"] = summary["Market - Actual"].map(lambda v: f"{v:+.1f}")
    summary["Actual Total"] = summary["Actual Total"].map(lambda v: f"{v:.1f}")
    summary["Model Spread"] = summary["Model Spread"].map(lambda v: f"{v:+.1f}")
    summary["Market Spread"] = summary["Market Spread"].map(lambda v: f"{v:+.1f}")
    summary["Actual Margin"] = summary["Actual Margin"].map(lambda v: f"{v:+.1f}")
    summary["Spread Δ (Model-Market)"] = summary["Spread Δ (Model-Market)"].map(lambda v: f"{v:+.1f}")
    summary["Spread Error"] = summary["Spread Error"].map(lambda v: f"{v:+.1f}")
    summary["Market Spread Error"] = summary["Market Spread Error"].map(lambda v: f"{v:+.1f}")

    return BacktestResult(
        summary=summary,
        accuracy=accuracy,
        log_loss=log_loss,
        brier=brier,
        totals_mae=totals_mae,
        totals_rmse=totals_rmse,
        market_mae=market_mae,
        market_rmse=market_rmse,
        spread_mae=spread_mae,
        spread_rmse=spread_rmse,
        market_spread_mae=market_spread_mae,
        market_spread_rmse=market_spread_rmse,
        spread_slope=slope,
        spread_intercept=intercept,
    )


def main() -> None:
    result = backtest_week5(TRAIN_SEASONS)
    print("Week 5 2025 Backtest (training seasons 2018-2024, rolling window=4)\n")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(result.summary.to_string(index=False))
    print()
    print(f"Winner Accuracy: {result.accuracy:.3f}")
    print(f"Log Loss:        {result.log_loss:.3f}")
    print(f"Brier Score:     {result.brier:.3f}")
    print(f"Totals MAE:      {result.totals_mae:.2f}")
    print(f"Totals RMSE:     {result.totals_rmse:.2f}")
    print(f"Market MAE:      {result.market_mae:.2f}")
    print(f"Market RMSE:     {result.market_rmse:.2f}")
    print(f"Spread MAE:      {result.spread_mae:.2f}")
    print(f"Spread RMSE:     {result.spread_rmse:.2f}")
    print(f"Market Spread MAE: {result.market_spread_mae:.2f}")
    print(f"Market Spread RMSE: {result.market_spread_rmse:.2f}")
    print(f"Spread Mapping:  spread = {result.spread_slope:.3f} * logit(p) + {result.spread_intercept:.3f}")


if __name__ == "__main__":
    main()
