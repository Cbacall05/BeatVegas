"""Lightweight smoke test for Project Beat Vegas modules."""
from __future__ import annotations

import pandas as pd  # type: ignore[import]

from beat_vegas import models, pipeline


def build_synthetic_dataset() -> pd.DataFrame:
    rows = []
    for season in [2021, 2022, 2023]:
        for week in range(1, 5):
            game_id = f"{season}_{week}"
            base_points = 40 + (season - 2020) * 2 + week
            home_points = base_points + (1 if week % 2 == 0 else -2)
            away_points = base_points + (2 if week % 3 == 0 else -1)
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "total_points": home_points + away_points,
                    "point_diff": home_points - away_points,
                    "home_win": int(home_points > away_points),
                    "home_offense_score": home_points / 10,
                    "away_offense_score": away_points / 10,
                    "home_defense_score": (50 - week) / 10,
                    "away_defense_score": (45 - week) / 10,
                    "home_market_total": base_points + 1,
                    "away_market_total": base_points - 1,
                }
            )
    return pd.DataFrame(rows)


def run_models_smoke_test() -> None:
    df = build_synthetic_dataset()
    feature_cols = [
        "home_offense_score",
        "away_offense_score",
        "home_defense_score",
        "away_defense_score",
        "home_market_total",
        "away_market_total",
    ]
    split_config = models.SplitConfig(validation_seasons=[2023])

    moneyline_results = models.train_moneyline_models(df, feature_cols, config=split_config)
    totals_results = models.train_total_models(df, feature_cols, config=split_config)

    assert len(moneyline_results) >= 2
    assert len(totals_results) >= 2

    market_prices = pd.DataFrame({"game_id": df[df["season"] == 2023]["game_id"].unique(), "market_prob": 0.5})
    totals_market = pd.DataFrame({"game_id": df[df["season"] == 2023]["game_id"].unique(), "market_total": 45})

    for result in moneyline_results:
        assert "calibrated_win_proba" in result.predictions.columns
        models.detect_moneyline_mispricing(result.predictions, market_prices)
    for result in totals_results:
        models.detect_total_mispricing(result.predictions, totals_market)


if __name__ == "__main__":
    run_models_smoke_test()
    print("Smoke test passed")
