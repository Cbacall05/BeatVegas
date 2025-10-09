"""Feature engineering utilities for Project Beat Vegas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""

    rolling_window: int = 4
    include_matchup_diffs: bool = True
    include_team_strength: bool = True


def prepare_weekly_features(weekly_df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Generate team-level features aggregated by game and team."""

    if config is None:
        config = FeatureConfig()

    df = weekly_df.copy()

    alias_map = {
        "points": ["team_points", "points_scored", "score"],
        "points_allowed": ["opp_points", "opponent_points", "points_conceded"],
        "turnover_diff": ["turnover_margin"],
    }
    for canonical, aliases in alias_map.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df.rename(columns={alias: canonical}, inplace=True)
                    break

    required_cols = {
        "season",
        "week",
        "team",
        "opponent",
        "game_id",
        "points",
        "points_allowed",
        "epa",
        "success",
        "turnover_diff",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Weekly dataframe missing required columns: {sorted(missing)}")

    df.sort_values(["team", "season", "week"], inplace=True)
    team_group = df.groupby("team", group_keys=False)

    df["points_scored_avg"] = team_group["points"].transform(
        lambda s: s.shift(1).rolling(config.rolling_window, min_periods=1).mean()
    )
    df["points_allowed_avg"] = team_group["points_allowed"].transform(
        lambda s: s.shift(1).rolling(config.rolling_window, min_periods=1).mean()
    )
    df["epa_avg"] = team_group["epa"].transform(
        lambda s: s.shift(1).rolling(config.rolling_window, min_periods=1).mean()
    )
    df["success_rate_avg"] = team_group["success"].transform(
        lambda s: s.shift(1).rolling(config.rolling_window, min_periods=1).mean()
    )
    df["turnover_diff_avg"] = team_group["turnover_diff"].transform(
        lambda s: s.shift(1).rolling(config.rolling_window, min_periods=1).mean()
    )

    if config.include_team_strength:
        df["team_strength_index"] = (
            df["epa_avg"].fillna(0).clip(-0.5, 0.5) * 0.6
            + df["success_rate_avg"].fillna(0).clip(0, 1) * 0.3
            + df["turnover_diff_avg"].fillna(0) * 0.1
        )

    if config.include_matchup_diffs:
        opponent_cols = [
            "points_scored_avg",
            "points_allowed_avg",
            "epa_avg",
            "success_rate_avg",
            "team_strength_index",
        ]
        opponent_df = df[["game_id", "team", "opponent"] + opponent_cols].rename(
            columns={col: f"opp_{col}" for col in opponent_cols}
        )
        opponent_df.rename(columns={"team": "opponent", "opponent": "team"}, inplace=True)
        df = df.merge(opponent_df, on=["game_id", "team", "opponent"], how="left")
        for col in opponent_cols:
            if f"opp_{col}" in df.columns:
                df[f"diff_{col}"] = df[col] - df[f"opp_{col}"]

    engineered_cols = [
        "season",
        "week",
        "team",
        "opponent",
        "game_id",
        "points",
        "points_allowed",
        "points_scored_avg",
        "points_allowed_avg",
        "epa_avg",
        "success_rate_avg",
        "turnover_diff_avg",
    ]
    if "team_strength_index" in df.columns:
        engineered_cols.append("team_strength_index")
    engineered_cols += [col for col in df.columns if col.startswith("diff_")]
    engineered_cols += [col for col in df.columns if col.startswith("opp_")]
    if "home_away" in df.columns:
        engineered_cols.append("home_away")

    return df[engineered_cols].copy()


def to_game_level(features_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot team-level features to game-level home/away representation."""

    required_cols = {"game_id", "team", "opponent", "home_away", "points"}
    missing = required_cols - set(features_df.columns)
    if missing:
        raise ValueError(f"Features dataframe missing required columns: {sorted(missing)}")

    home_df = features_df[features_df["home_away"] == "home"].copy()
    away_df = features_df[features_df["home_away"] == "away"].copy()

    home_df = home_df.add_prefix("home_")
    away_df = away_df.add_prefix("away_")

    home_df.rename(columns={"home_game_id": "game_id"}, inplace=True)
    away_df.rename(columns={"away_game_id": "game_id"}, inplace=True)

    merged = home_df.merge(away_df, on="game_id", how="inner", suffixes=("", ""))
    merged["season"] = merged.get("home_season")
    merged["week"] = merged.get("home_week")
    merged["total_points"] = merged["home_points"] + merged["away_points"]
    merged["point_diff"] = merged["home_points"] - merged["away_points"]
    merged["home_win"] = (merged["home_points"] > merged["away_points"]).astype(int)

    feature_cols = [col for col in merged.columns if col.startswith("home_") or col.startswith("away_")]
    drop_cols = {
        "home_season",
        "away_season",
        "home_week",
        "away_week",
        "home_team",
        "away_team",
        "home_opponent",
        "away_opponent",
    }
    feature_cols = [col for col in feature_cols if col not in drop_cols]
    ordered_cols = ["game_id", "season", "week", "total_points", "point_diff", "home_win"] + sorted(
        feature_cols
    )
    return merged[ordered_cols]
