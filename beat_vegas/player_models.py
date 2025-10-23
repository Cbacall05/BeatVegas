"""Player-level touchdown modeling utilities for Project Beat Vegas."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, Optional, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, r2_score

from . import data_load

LOGGER = logging.getLogger(__name__)

EXCLUDED_STATUSES = {"RET", "FA", "UFA", "CUT", "SUS"}


@dataclass
class PlayerTouchdownModelResult:
    """Container for the trained touchdown probability model."""

    model_name: str
    model: LogisticRegression
    feature_cols: list[str]
    lookback: int
    min_touches: float
    metrics: dict[str, float]
    training_frame: pd.DataFrame


@dataclass
class PlayerPassingYardsModelResult:
    """Container for the trained quarterback passing yards model."""

    model_name: str
    model: Ridge
    feature_cols: list[str]
    lookback: int
    min_attempts: float
    metrics: dict[str, float]
    training_frame: pd.DataFrame


def _prepare_roster_lookup(
    roster_df: Optional[pd.DataFrame],
    *,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    if roster_df is None or roster_df.empty:
        return pd.DataFrame()

    roster = roster_df.copy()
    if "player_id" not in roster.columns and "gsis_id" in roster.columns:
        roster["player_id"] = roster["gsis_id"]

    player_id_col = "player_id" if "player_id" in roster.columns else None
    if player_id_col is None:
        return pd.DataFrame()

    team_col = None
    for candidate in ("team", "recent_team", "team_abbr"):
        if candidate in roster.columns:
            team_col = candidate
            break
    if team_col is None:
        return pd.DataFrame()

    roster = roster[roster[player_id_col].notna()].copy()
    roster[team_col] = roster[team_col].astype(str).str.upper()

    if "status" in roster.columns:
        roster["status"] = roster["status"].astype(str).str.upper()
        roster = roster[~roster["status"].isin(EXCLUDED_STATUSES)]

    if "season" in roster.columns:
        season_mask = roster["season"] == target_season
        if season_mask.any():
            roster = roster[season_mask]
        else:
            latest_season = pd.to_numeric(roster["season"], errors="coerce").max()
            if not pd.isna(latest_season):
                roster = roster[roster["season"] == latest_season]

    if "week" in roster.columns:
        roster_before_week = roster.copy()
        roster = roster[roster["week"] <= target_week]
        if roster.empty:
            roster = roster_before_week.sort_values([player_id_col, "week"]).drop_duplicates(player_id_col, keep="last")
        else:
            roster = roster.sort_values([player_id_col, "week"]).drop_duplicates(player_id_col, keep="last")

    column_map = {player_id_col: "player_id", team_col: "team"}
    if "player_name" in roster.columns:
        column_map["player_name"] = "roster_player_name"
    if "football_name" in roster.columns and "player_name" not in roster.columns:
        column_map["football_name"] = "roster_player_name"
    if "position" in roster.columns:
        column_map["position"] = "roster_position"
    if "depth_chart_position" in roster.columns:
        column_map["depth_chart_position"] = "depth_chart_position"
    if "years_exp" in roster.columns:
        column_map["years_exp"] = "years_exp"

    roster = roster[list(column_map.keys())].drop_duplicates(subset=[player_id_col, team_col], keep="last")
    roster.rename(columns=column_map, inplace=True)
    return roster


def _safe_sum(series: pd.Series) -> float:
    return float(series.fillna(0).sum())


def build_player_game_stats(pbp_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-by-play records into per-player, per-game features."""

    if pbp_df.empty:
        raise ValueError("Play-by-play dataframe is empty. Cannot derive player stats.")

    rush_mask = pbp_df.get("rush_attempt", pd.Series(dtype=float)) == 1
    rush_cols = {
        "game_id": pbp_df.get("game_id"),
        "season": pbp_df.get("season"),
        "week": pbp_df.get("week"),
        "team": pbp_df.get("posteam"),
        "player_id": pbp_df.get("rusher_player_id"),
        "player_name": pbp_df.get("rusher_player_name"),
        "position": pbp_df.get("rusher_player_position", pd.Series(dtype=object)),
        "rush_attempts": pbp_df.get("rush_attempt", pd.Series(dtype=float)),
        "rush_touchdowns": pbp_df.get("rush_touchdown", pd.Series(dtype=float)),
        "yardline_100": pbp_df.get("yardline_100", pd.Series(dtype=float)),
    }
    rush_df = pd.DataFrame(rush_cols).loc[rush_mask]
    rush_df = rush_df[rush_df["player_id"].notna()].copy()
    rush_df["targets"] = 0.0
    rush_df["rec_touchdowns"] = 0.0
    rush_df["redzone_touches"] = (rush_df["yardline_100"] <= 20).astype(float)

    pass_mask = pbp_df.get("pass_attempt", pd.Series(dtype=float)) == 1
    pass_cols = {
        "game_id": pbp_df.get("game_id"),
        "season": pbp_df.get("season"),
        "week": pbp_df.get("week"),
        "team": pbp_df.get("posteam"),
        "player_id": pbp_df.get("receiver_player_id"),
        "player_name": pbp_df.get("receiver_player_name"),
        "position": pbp_df.get("receiver_player_position", pd.Series(dtype=object)),
        "targets": 1.0,
        "rec_touchdowns": pbp_df.get("pass_touchdown", pd.Series(dtype=float)),
        "yardline_100": pbp_df.get("yardline_100", pd.Series(dtype=float)),
    }
    recv_df = pd.DataFrame(pass_cols).loc[pass_mask]
    recv_df = recv_df[recv_df["player_id"].notna()].copy()
    recv_df["rush_attempts"] = 0.0
    recv_df["rush_touchdowns"] = 0.0
    recv_df["redzone_touches"] = (recv_df["yardline_100"] <= 20).astype(float)

    combined = pd.concat([rush_df, recv_df], ignore_index=True, sort=False)
    if combined.empty:
        raise ValueError("No player rushing/receiving stats available in play-by-play dataframe.")

    combined = combined[combined["player_id"].notna()].copy()
    if "team" in combined:
        combined = combined[combined["team"].str.upper() != "TOT"]
    combined = combined[combined["player_name"].notna()]
    combined = combined[combined["player_name"].str.strip() != ""]
    combined = combined[~combined["player_name"].str.contains("error", case=False, na=False)]

    aggregations = {
        "player_name": "first",
        "position": "first",
        "rush_attempts": _safe_sum,
        "targets": _safe_sum,
        "rush_touchdowns": _safe_sum,
        "rec_touchdowns": _safe_sum,
        "redzone_touches": _safe_sum,
    }
    grouped = (
        combined.groupby(["season", "week", "game_id", "team", "player_id"], as_index=False)
        .agg(aggregations)
        .rename(columns={"player_name": "player_display_name"})
    )

    grouped["rush_attempts"] = grouped["rush_attempts"].fillna(0.0)
    grouped["targets"] = grouped["targets"].fillna(0.0)
    grouped["redzone_touches"] = grouped["redzone_touches"].fillna(0.0)
    grouped["rush_touchdowns"] = grouped["rush_touchdowns"].fillna(0.0)
    grouped["rec_touchdowns"] = grouped["rec_touchdowns"].fillna(0.0)
    grouped["total_touchdowns"] = grouped["rush_touchdowns"] + grouped["rec_touchdowns"]

    scoring = (
        pbp_df.assign(
            rush_td=pbp_df.get("rush_touchdown", pd.Series(dtype=float)).fillna(0.0),
            pass_td=pbp_df.get("pass_touchdown", pd.Series(dtype=float)).fillna(0.0),
        )
        .groupby(["season", "week", "game_id", "posteam"], as_index=False)
        .agg(team_rush_td=("rush_td", "sum"), team_pass_td=("pass_td", "sum"))
    )
    scoring["team_touchdowns"] = scoring["team_rush_td"] + scoring["team_pass_td"]
    scoring.rename(columns={"posteam": "team"}, inplace=True)

    player_stats = grouped.merge(scoring[["season", "week", "game_id", "team", "team_touchdowns"]], on=["season", "week", "game_id", "team"], how="left")
    player_stats["team_touchdowns"] = player_stats["team_touchdowns"].fillna(0.0)

    team_schedule = data_load.explode_schedule(schedule_df)
    schedule_cols = [
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "market_total",
        "spread_line",
        "home_away",
    ]
    team_schedule = team_schedule[schedule_cols]

    player_stats = player_stats.merge(team_schedule, on=["season", "week", "game_id", "team"], how="left")
    player_stats["player_id"] = player_stats["player_id"].astype(str)
    return player_stats


def build_quarterback_game_stats(pbp_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-by-play into per-quarterback passing summaries."""

    if pbp_df.empty:
        raise ValueError("Play-by-play dataframe is empty. Cannot derive quarterback stats.")

    pass_mask = pbp_df.get("pass_attempt", pd.Series(dtype=float)) == 1
    qb_cols = {
        "game_id": pbp_df.get("game_id"),
        "season": pbp_df.get("season"),
        "week": pbp_df.get("week"),
        "team": pbp_df.get("posteam"),
        "player_id": pbp_df.get("passer_player_id"),
        "player_display_name": pbp_df.get("passer_player_name"),
        "position": "QB",
        "pass_attempts": pbp_df.get("pass_attempt", pd.Series(dtype=float)),
        "completions": pbp_df.get("complete_pass", pd.Series(dtype=float)),
        "passing_yards": pbp_df.get("passing_yards", pd.Series(dtype=float)),
        "interceptions": pbp_df.get("interception", pd.Series(dtype=float)),
        "passing_touchdowns": pbp_df.get("pass_touchdown", pd.Series(dtype=float)),
    }
    qb_df = pd.DataFrame(qb_cols).loc[pass_mask]
    qb_df = qb_df[qb_df["player_id"].notna()].copy()
    if qb_df.empty:
        raise ValueError("No quarterback passing stats available in play-by-play dataframe.")

    for col in ["pass_attempts", "completions", "passing_yards", "interceptions", "passing_touchdowns"]:
        qb_df[col] = qb_df[col].fillna(0.0)

    aggregations = {
        "player_display_name": "last",
        "position": "last",
        "pass_attempts": _safe_sum,
        "completions": _safe_sum,
        "passing_yards": _safe_sum,
        "interceptions": _safe_sum,
        "passing_touchdowns": _safe_sum,
    }
    grouped = (
        qb_df.groupby(["season", "week", "game_id", "team", "player_id"], as_index=False)
        .agg(aggregations)
    )

    grouped = grouped[grouped["team"].notna()].copy()
    grouped["team"] = grouped["team"].astype(str).str.upper()
    grouped = grouped[grouped["team"] != "TOT"]

    team_schedule = data_load.explode_schedule(schedule_df)
    schedule_cols = [
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "market_total",
        "spread_line",
        "home_away",
    ]
    team_schedule = team_schedule[schedule_cols]

    qb_stats = grouped.merge(team_schedule, on=["season", "week", "game_id", "team"], how="left")
    qb_stats["player_id"] = qb_stats["player_id"].astype(str)
    for col in ["market_total", "spread_line"]:
        if col in qb_stats.columns:
            qb_stats[col] = qb_stats[col].astype(float, errors="ignore")
    return qb_stats


def prepare_quarterback_passing_dataset(qb_stats: pd.DataFrame, lookback: int = 4) -> pd.DataFrame:
    """Compute rolling usage features for quarterback passing projections."""

    if qb_stats.empty:
        raise ValueError("Quarterback stats dataframe is empty.")

    stats = qb_stats.sort_values(["player_id", "season", "week"]).copy()
    rolling_cols = [
        "passing_yards",
        "pass_attempts",
        "completions",
        "interceptions",
        "passing_touchdowns",
        "market_total",
        "spread_line",
    ]

    for col in rolling_cols:
        stats[f"avg_{col}"] = (
            stats.groupby("player_id")[col]
            .apply(lambda s: s.shift(1).rolling(window=lookback, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    stats["avg_pass_attempts"] = stats["avg_pass_attempts"].fillna(0.0)
    stats["avg_passing_yards"] = stats["avg_passing_yards"].fillna(0.0)
    stats["avg_completions"] = stats["avg_completions"].fillna(0.0)
    stats["avg_market_total"] = stats["avg_market_total"].fillna(stats["market_total"].fillna(0.0))
    stats["avg_spread_line"] = stats["avg_spread_line"].fillna(stats["spread_line"].fillna(0.0))
    stats["avg_interceptions"] = stats["avg_interceptions"].fillna(0.0)
    stats["avg_passing_touchdowns"] = stats["avg_passing_touchdowns"].fillna(0.0)

    useful_cols = [
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "player_id",
        "player_display_name",
        "pass_attempts",
        "completions",
        "passing_yards",
        "interceptions",
        "passing_touchdowns",
        "market_total",
        "spread_line",
        "avg_pass_attempts",
        "avg_completions",
        "avg_passing_yards",
        "avg_interceptions",
        "avg_passing_touchdowns",
        "avg_market_total",
        "avg_spread_line",
    ]
    for col in useful_cols:
        if col not in stats.columns:
            stats[col] = np.nan

    return stats[useful_cols]


def train_passing_yards_model(
    qb_dataset: pd.DataFrame,
    lookback: int = 4,
    min_attempts: float = 15.0,
    model_name: str = "Ridge",
) -> PlayerPassingYardsModelResult:
    """Train a regression model to project quarterback passing yards."""

    df = qb_dataset.dropna(subset=["avg_passing_yards", "avg_pass_attempts", "avg_market_total"]).copy()
    df = df[df["avg_pass_attempts"].fillna(0.0) >= min_attempts]
    if df.empty:
        raise ValueError("No quarterback rows meet the minimum attempts threshold for training.")

    numeric_cols = [
        "avg_passing_yards",
        "avg_pass_attempts",
        "avg_completions",
        "avg_passing_touchdowns",
        "avg_interceptions",
        "avg_market_total",
        "avg_spread_line",
        "market_total",
        "spread_line",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    feature_cols = numeric_cols
    X = df[feature_cols]
    y = df["passing_yards"].astype(float).fillna(0.0)

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    metrics = {
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }
    df = df.copy()
    df["predicted_yards"] = preds

    return PlayerPassingYardsModelResult(
        model_name=model_name,
        model=model,
        feature_cols=feature_cols,
        lookback=lookback,
        min_attempts=min_attempts,
        metrics=metrics,
        training_frame=df,
    )


def prepare_player_touchdown_dataset(player_stats: pd.DataFrame, lookback: int = 4) -> pd.DataFrame:
    """Compute rolling features that capture recent usage before each game."""

    if player_stats.empty:
        raise ValueError("Player stats dataframe is empty.")

    stats = player_stats.sort_values(["player_id", "season", "week"]).copy()
    base_cols = ["rush_attempts", "targets", "redzone_touches", "team_touchdowns", "total_touchdowns", "market_total"]

    for col in base_cols:
        stats[f"avg_{col}"] = (
            stats.groupby("player_id")[col]
            .apply(lambda s: s.shift(1).rolling(window=lookback, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    stats["avg_total_touches"] = stats["avg_rush_attempts"].fillna(0.0) + stats["avg_targets"].fillna(0.0)
    stats["avg_td_rate"] = stats["avg_total_touchdowns"].fillna(0.0)
    stats["scored_td"] = (stats["total_touchdowns"] > 0).astype(int)

    useful_cols = [
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "player_id",
        "player_display_name",
        "position",
        "market_total",
        "avg_rush_attempts",
        "avg_targets",
        "avg_redzone_touches",
        "avg_team_touchdowns",
        "avg_td_rate",
        "avg_total_touches",
        "avg_market_total",
        "scored_td",
    ]
    return stats[useful_cols]


def train_touchdown_model(
    player_dataset: pd.DataFrame,
    lookback: int = 4,
    min_touches: float = 0.5,
    model_name: str = "LogisticRegression",
) -> PlayerTouchdownModelResult:
    """Train a logistic regression to predict touchdowns based on recent usage."""

    df = player_dataset.dropna(subset=[
        "avg_rush_attempts",
        "avg_targets",
        "avg_redzone_touches",
        "avg_team_touchdowns",
        "avg_td_rate",
        "avg_market_total",
    ]).copy()
    df = df[df["avg_total_touches"] >= min_touches]
    if df.empty:
        raise ValueError("No player rows meet the minimum touches threshold for training.")

    positions = df["position"].fillna("UNK").str.upper()
    for pos in ["RB", "WR", "TE", "QB"]:
        df[f"pos_{pos.lower()}"] = (positions == pos).astype(int)
    df["pos_other"] = (~positions.isin(["RB", "WR", "TE", "QB"])).astype(int)

    feature_cols = [
        "avg_rush_attempts",
        "avg_targets",
        "avg_redzone_touches",
        "avg_team_touchdowns",
        "avg_td_rate",
        "avg_market_total",
        "market_total",
        "pos_rb",
        "pos_wr",
        "pos_te",
        "pos_qb",
        "pos_other",
    ]

    X = df[feature_cols].fillna(0.0)
    y = df["scored_td"].astype(int)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)

    preds = model.predict_proba(X)[:, 1]
    metrics = {
        "auc": float(roc_auc_score(y, preds)) if len(np.unique(y)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y, preds)),
        "base_rate": float(y.mean()),
    }
    df["pred_prob"] = preds

    return PlayerTouchdownModelResult(
        model_name=model_name,
        model=model,
        feature_cols=feature_cols,
        lookback=lookback,
        min_touches=min_touches,
        metrics=metrics,
        training_frame=df,
    )


def _aggregate_recent_usage(
    player_stats: pd.DataFrame,
    target_season: int,
    target_week: int,
    lookback: int,
    min_touches: float,
) -> pd.DataFrame:
    history_mask = (player_stats["season"] < target_season) | (
        (player_stats["season"] == target_season) & (player_stats["week"] < target_week)
    )
    history = player_stats.loc[history_mask].copy()
    if history.empty:
        return pd.DataFrame()

    history = history.sort_values(["player_id", "season", "week"])
    history = history[history["team"].fillna("").str.upper() != "TOT"]
    recent = history.groupby("player_id", group_keys=False).tail(lookback)
    grouped = recent.groupby("player_id").agg(
        {
            "player_display_name": "last",
            "position": "last",
            "team": "last",
            "opponent": "last",
            "rush_attempts": "mean",
            "targets": "mean",
            "redzone_touches": "mean",
            "team_touchdowns": "mean",
            "total_touchdowns": "mean",
            "market_total": "mean",
        }
    )
    grouped.rename(columns={
        "player_display_name": "player_display_name",
        "rush_attempts": "avg_rush_attempts",
        "targets": "avg_targets",
        "redzone_touches": "avg_redzone_touches",
        "team_touchdowns": "avg_team_touchdowns",
        "total_touchdowns": "avg_td_rate",
        "market_total": "avg_market_total",
    }, inplace=True)
    grouped["avg_total_touches"] = grouped["avg_rush_attempts"].fillna(0.0) + grouped["avg_targets"].fillna(0.0)
    grouped = grouped[grouped["avg_total_touches"] >= min_touches]
    grouped = grouped.reset_index()
    grouped["player_id"] = grouped["player_id"].astype(str)
    return grouped


def _aggregate_recent_passing_usage(
    qb_stats: pd.DataFrame,
    target_season: int,
    target_week: int,
    lookback: int,
    min_attempts: float,
) -> pd.DataFrame:
    history_mask = (qb_stats["season"] < target_season) | (
        (qb_stats["season"] == target_season) & (qb_stats["week"] < target_week)
    )
    history = qb_stats.loc[history_mask].copy()
    if history.empty:
        return pd.DataFrame()

    history = history.sort_values(["player_id", "season", "week"])
    history = history[history["team"].fillna("").str.upper() != "TOT"]
    recent = history.groupby("player_id", group_keys=False).tail(lookback)
    grouped = recent.groupby("player_id").agg(
        {
            "player_display_name": "last",
            "team": "last",
            "opponent": "last",
            "pass_attempts": "mean",
            "completions": "mean",
            "passing_yards": "mean",
            "interceptions": "mean",
            "passing_touchdowns": "mean",
            "market_total": "mean",
            "spread_line": "mean",
        }
    )

    grouped.rename(
        columns={
            "player_display_name": "player_display_name",
            "pass_attempts": "avg_pass_attempts",
            "completions": "avg_completions",
            "passing_yards": "avg_passing_yards",
            "interceptions": "avg_interceptions",
            "passing_touchdowns": "avg_passing_touchdowns",
            "market_total": "avg_market_total",
            "spread_line": "avg_spread_line",
        },
        inplace=True,
    )

    grouped = grouped[grouped["avg_pass_attempts"].fillna(0.0) >= min_attempts]
    grouped = grouped.reset_index()
    grouped["player_id"] = grouped["player_id"].astype(str)
    grouped["team"] = grouped["team"].astype(str).str.upper()
    grouped["avg_passing_yards"] = grouped["avg_passing_yards"].fillna(0.0)
    grouped["avg_market_total"] = grouped["avg_market_total"].fillna(0.0)
    grouped["avg_spread_line"] = grouped["avg_spread_line"].fillna(0.0)
    grouped["is_fallback"] = False
    return grouped


def _compute_team_passing_baselines(
    qb_stats: pd.DataFrame,
    target_season: int,
    target_week: int,
    lookback: int,
) -> pd.DataFrame:
    history_mask = (qb_stats["season"] < target_season) | (
        (qb_stats["season"] == target_season) & (qb_stats["week"] < target_week)
    )
    history = qb_stats.loc[history_mask].copy()
    if history.empty:
        return pd.DataFrame()

    history = history[history["team"].fillna("").str.upper() != "TOT"].copy()
    history["team"] = history["team"].astype(str).str.upper()

    aggregations = {
        "pass_attempts": _safe_sum,
        "completions": _safe_sum,
        "passing_yards": _safe_sum,
        "interceptions": _safe_sum,
        "passing_touchdowns": _safe_sum,
        "market_total": "mean",
        "spread_line": "mean",
    }
    team_games = (
        history.groupby(["team", "season", "week"], as_index=False)
        .agg(aggregations)
        .sort_values(["team", "season", "week"])
    )

    for col in [
        "pass_attempts",
        "completions",
        "passing_yards",
        "interceptions",
        "passing_touchdowns",
        "market_total",
        "spread_line",
    ]:
        team_games[f"avg_{col}"] = (
            team_games.groupby("team")[col]
            .apply(lambda s: s.shift(1).rolling(window=lookback, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    latest = team_games.groupby("team", as_index=False).tail(1).copy()
    latest.rename(
        columns={
            "avg_pass_attempts": "avg_pass_attempts",
            "avg_completions": "avg_completions",
            "avg_passing_yards": "avg_passing_yards",
            "avg_interceptions": "avg_interceptions",
            "avg_passing_touchdowns": "avg_passing_touchdowns",
            "avg_market_total": "avg_market_total",
            "avg_spread_line": "avg_spread_line",
        },
        inplace=True,
    )

    latest = latest[[
        "team",
        "avg_pass_attempts",
        "avg_completions",
        "avg_passing_yards",
        "avg_interceptions",
        "avg_passing_touchdowns",
        "avg_market_total",
        "avg_spread_line",
    ]].copy()
    return latest


def _build_qb_fallback_usage(
    upcoming: pd.DataFrame,
    usage: pd.DataFrame,
    roster_lookup: pd.DataFrame,
    team_baselines: pd.DataFrame,
    preferred_qbs: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    if upcoming.empty or roster_lookup.empty:
        return pd.DataFrame()

    upcoming_teams = set(upcoming["team"].unique())
    existing_teams = set(usage["team"].unique()) if not usage.empty else set()
    missing_teams = upcoming_teams - existing_teams

    preferred_map = {team.upper(): name.strip().lower() for team, name in (preferred_qbs or {}).items()}

    if usage is not None and not usage.empty and preferred_map:
        usage_names = usage.assign(
            _display=usage["player_display_name"].astype(str).str.lower()
        )
        for team, name in preferred_map.items():
            if team not in upcoming_teams:
                continue
            team_mask = usage_names["team"].astype(str).str.upper() == team
            if not team_mask.any():
                missing_teams.add(team)
                continue
            if name not in set(usage_names.loc[team_mask, "_display"]):
                missing_teams.add(team)
    if not missing_teams:
        return pd.DataFrame()

    baseline_cols = [
        "avg_pass_attempts",
        "avg_completions",
        "avg_passing_yards",
        "avg_interceptions",
        "avg_passing_touchdowns",
        "avg_market_total",
        "avg_spread_line",
    ]

    default_baseline = team_baselines[baseline_cols].mean().to_dict() if not team_baselines.empty else {}
    default_baseline = {col: float(default_baseline.get(col, 0.0)) for col in baseline_cols}

    rows: list[dict[str, object]] = []
    roster_qbs = roster_lookup[
        (roster_lookup.get("roster_position").fillna("").str.upper() == "QB")
        | (roster_lookup.get("depth_chart_position").fillna("").str.upper() == "QB")
    ].copy()
    roster_qbs["team"] = roster_qbs["team"].astype(str).str.upper()
    roster_qbs["roster_player_name"] = roster_qbs.get("roster_player_name").astype(str)

    for team in sorted(missing_teams):
        team_baseline_row = team_baselines[team_baselines["team"] == team]
        if not team_baseline_row.empty:
            baseline = team_baseline_row.iloc[0].to_dict()
        else:
            baseline = default_baseline

        candidates = roster_qbs[roster_qbs["team"] == team].copy()
        if candidates.empty:
            continue

        candidates["years_exp"] = pd.to_numeric(candidates.get("years_exp"), errors="coerce")
        candidates["years_exp"] = candidates["years_exp"].fillna(10.0)
        pref_name = preferred_map.get(team)
        chosen: pd.Series
        if pref_name:
            pref_match = candidates[candidates["roster_player_name"].str.lower() == pref_name]
            if not pref_match.empty:
                chosen = pref_match.iloc[0]
            else:
                chosen = candidates.sort_values(["years_exp", "roster_player_name"]).iloc[0]
        else:
            chosen = candidates.sort_values(["years_exp", "roster_player_name"]).iloc[0]

        display_name = chosen.get("roster_player_name") or chosen.get("player_id") or "Unknown QB"

        row = {
            "player_id": str(chosen["player_id"]),
            "player_display_name": str(display_name),
            "team": team,
            "opponent": "UNK",
            "is_fallback": True,
        }
        for col in baseline_cols:
            value = baseline.get(col, default_baseline.get(col, 0.0))
            if pd.isna(value):
                value = default_baseline.get(col, 0.0)
            row[col] = float(value)
        row["years_exp"] = float(chosen.get("years_exp", 10.0))
        row["roster_player_name"] = chosen.get("roster_player_name")
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    fallback_df = pd.DataFrame(rows)
    fallback_df["player_id"] = fallback_df["player_id"].astype(str)
    return fallback_df


def predict_upcoming_touchdowns(
    model_result: PlayerTouchdownModelResult,
    player_stats: pd.DataFrame,
    schedule_df: pd.DataFrame,
    *,
    target_season: int,
    target_week: int,
    roster_lookup: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate touchdown probabilities for upcoming games using the trained model."""

    if player_stats.empty:
        return pd.DataFrame()

    upcoming_schedule = data_load.explode_schedule(schedule_df)
    upcoming = upcoming_schedule[
        (upcoming_schedule["season"] == target_season)
        & (upcoming_schedule["week"] == target_week)
        & upcoming_schedule["market_total"].notna()
    ][["game_id", "team", "opponent", "market_total", "home_away"]]
    if upcoming.empty:
        LOGGER.warning(
            "No upcoming schedule entries with market totals for season %s week %s.",
            target_season,
            target_week,
        )
        return pd.DataFrame()

    usage = _aggregate_recent_usage(
        player_stats,
        target_season=target_season,
        target_week=target_week,
        lookback=model_result.lookback,
        min_touches=model_result.min_touches,
    )
    if usage.empty:
        LOGGER.warning("No sufficient player usage history to project upcoming touchdowns.")
        return pd.DataFrame()

    if roster_lookup is not None and not roster_lookup.empty:
        roster_map = roster_lookup[["player_id", "team"]].drop_duplicates()
        roster_map["team"] = roster_map["team"].astype(str).str.upper()
        usage = usage.merge(roster_map, on="player_id", how="left", suffixes=("", "_roster"))
        usage["team"] = usage["team_roster"].where(usage["team_roster"].notna(), usage["team"])
        usage.drop(columns=[col for col in usage.columns if col.endswith("_roster")], inplace=True)

    merged = usage.merge(upcoming, on="team", how="inner", suffixes=("", "_upcoming"))
    if merged.empty:
        return pd.DataFrame()

    merged["player_id"] = merged["player_id"].astype(str)
    merged["team"] = merged["team"].astype(str).str.upper()
    if roster_lookup is not None and not roster_lookup.empty:
        roster_lookup = roster_lookup.copy()
        roster_lookup["team"] = roster_lookup["team"].astype(str).str.upper()
        roster_lookup = roster_lookup.drop_duplicates(subset=["player_id", "team"])
        merged = merged.merge(roster_lookup, on=["player_id", "team"], how="inner")
        if merged.empty:
            LOGGER.warning("No roster overlap found when filtering touchdown projections.")
            return pd.DataFrame()

    positions = merged["position"].fillna("UNK").str.upper()
    for pos in ["RB", "WR", "TE", "QB"]:
        merged[f"pos_{pos.lower()}"] = (positions == pos).astype(int)
    merged["pos_other"] = (~positions.isin(["RB", "WR", "TE", "QB"])).astype(int)
    if "opponent_upcoming" in merged.columns:
        merged["opponent"] = merged["opponent_upcoming"]
        merged.drop(columns=["opponent_upcoming"], inplace=True)
    merged["market_total"] = merged.get("market_total_upcoming", merged.get("market_total"))
    if "market_total_upcoming" in merged.columns:
        merged.drop(columns=["market_total_upcoming"], inplace=True)

    missing_cols = [col for col in model_result.feature_cols if col not in merged.columns]
    if missing_cols:
        for col in missing_cols:
            merged[col] = 0.0

    X_pred = merged[model_result.feature_cols].fillna(0.0)
    merged["touchdown_prob"] = model_result.model.predict_proba(X_pred)[:, 1]

    output_cols = [
        "game_id",
        "team",
        "opponent",
        "home_away",
        "player_id",
        "player_display_name",
        "position",
        "touchdown_prob",
        "avg_total_touches",
        "avg_rush_attempts",
        "avg_targets",
        "avg_redzone_touches",
        "avg_team_touchdowns",
        "avg_td_rate",
        "market_total",
    ]
    result = merged[output_cols].copy()
    result = result[result["touchdown_prob"].notna()]
    result = result[result["touchdown_prob"] > 0]
    result.sort_values(["game_id", "touchdown_prob"], ascending=[True, False], inplace=True)
    return result.reset_index(drop=True)


def predict_upcoming_passing_yards(
    model_result: PlayerPassingYardsModelResult,
    qb_stats: pd.DataFrame,
    schedule_df: pd.DataFrame,
    *,
    target_season: int,
    target_week: int,
    roster_lookup: Optional[pd.DataFrame] = None,
    preferred_qbs: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    if qb_stats.empty:
        return pd.DataFrame()

    upcoming_schedule = data_load.explode_schedule(schedule_df)
    upcoming = upcoming_schedule[
        (upcoming_schedule["season"] == target_season)
        & (upcoming_schedule["week"] == target_week)
    ][["game_id", "team", "opponent", "market_total", "spread_line", "home_away"]]
    if upcoming.empty:
        LOGGER.warning(
            "No upcoming schedule entries for season %s week %s.",
            target_season,
            target_week,
        )
        return pd.DataFrame()

    usage = _aggregate_recent_passing_usage(
        qb_stats,
        target_season=target_season,
        target_week=target_week,
        lookback=model_result.lookback,
        min_attempts=model_result.min_attempts,
    )

    team_baselines = _compute_team_passing_baselines(
        qb_stats,
        target_season=target_season,
        target_week=target_week,
        lookback=model_result.lookback,
    )

    if roster_lookup is not None and not roster_lookup.empty:
        roster_map = roster_lookup[["player_id", "team"]].drop_duplicates()
        roster_map["team"] = roster_map["team"].astype(str).str.upper()
        usage = usage.merge(roster_map, on="player_id", how="left", suffixes=("", "_roster"))
        usage["team"] = usage["team_roster"].where(usage["team_roster"].notna(), usage["team"])
        usage.drop(columns=[col for col in usage.columns if col.endswith("_roster")], inplace=True)

    if roster_lookup is not None and not roster_lookup.empty:
        fallback_rows = _build_qb_fallback_usage(upcoming, usage, roster_lookup, team_baselines, preferred_qbs)
        if usage.empty:
            usage = fallback_rows
        elif not fallback_rows.empty:
            usage = pd.concat([usage, fallback_rows], ignore_index=True)

    if usage.empty:
        LOGGER.warning("No quarterback usage history available; fallback projections unavailable.")
        return pd.DataFrame()

    usage["team"] = usage["team"].astype(str).str.upper()
    upcoming["team"] = upcoming["team"].astype(str).str.upper()
    merged = usage.merge(upcoming, on="team", how="inner", suffixes=("", "_upcoming"))
    if merged.empty:
        return pd.DataFrame()

    merged["player_id"] = merged["player_id"].astype(str)
    if roster_lookup is not None and not roster_lookup.empty:
        roster_lookup = roster_lookup.copy()
        roster_lookup["team"] = roster_lookup["team"].astype(str).str.upper()
        roster_lookup = roster_lookup.drop_duplicates(subset=["player_id", "team"])
        roster_lookup = roster_lookup.drop_duplicates(subset=["player_id", "team"], keep="last")
        merged = merged.merge(roster_lookup, on=["player_id", "team"], how="inner")
        if merged.empty:
            LOGGER.warning("No roster overlap found when filtering passing projections.")
            return pd.DataFrame()

    merged["market_total"] = merged.get("market_total_upcoming", merged.get("market_total")).fillna(merged["avg_market_total"].fillna(0.0))
    merged["spread_line"] = merged.get("spread_line_upcoming", merged.get("spread_line")).fillna(merged["avg_spread_line"].fillna(0.0))
    if "opponent_upcoming" in merged.columns:
        merged["opponent"] = merged["opponent_upcoming"]
        merged.drop(columns=["opponent_upcoming"], inplace=True)
    if "home_away_upcoming" in merged.columns:
        merged["home_away"] = merged["home_away_upcoming"]
        merged.drop(columns=["home_away_upcoming"], inplace=True)
    if "market_total_upcoming" in merged.columns:
        merged.drop(columns=["market_total_upcoming"], inplace=True)
    if "spread_line_upcoming" in merged.columns:
        merged.drop(columns=["spread_line_upcoming"], inplace=True)

    missing_cols = [col for col in model_result.feature_cols if col not in merged.columns]
    for col in missing_cols:
        merged[col] = 0.0

    X_pred = merged[model_result.feature_cols].fillna(0.0)
    merged["expected_passing_yards"] = model_result.model.predict(X_pred)
    merged["expected_passing_yards"] = merged["expected_passing_yards"].clip(lower=0.0)

    output_cols = [
        "game_id",
        "team",
        "opponent",
        "home_away",
        "player_id",
        "player_display_name",
        "avg_pass_attempts",
        "avg_completions",
        "avg_passing_yards",
        "avg_passing_touchdowns",
        "avg_interceptions",
        "avg_market_total",
        "avg_spread_line",
        "market_total",
        "spread_line",
        "expected_passing_yards",
    ]
    optional_cols = ["roster_player_name", "years_exp", "is_fallback"]
    for col in optional_cols:
        if col in merged.columns and col not in output_cols:
            output_cols.append(col)

    result = merged[output_cols].copy()
    result["selection_rank"] = 1.0
    if "is_fallback" in result.columns:
        result.loc[result["is_fallback"].astype(bool), "selection_rank"] = 0.5

    if preferred_qbs:
        normalized = {team.upper(): name.strip().lower() for team, name in preferred_qbs.items()}
        player_names = result["player_display_name"].astype(str).str.lower()
        roster_names = result.get("roster_player_name")
        roster_names = roster_names.astype(str).str.lower() if roster_names is not None else pd.Series(index=result.index, dtype=str)
        for team, name in normalized.items():
            mask_team = result["team"].str.upper() == team
            if not mask_team.any():
                continue
            mask_name = player_names.eq(name)
            if roster_names is not None:
                mask_name |= roster_names.eq(name)
            result.loc[mask_team & mask_name, "selection_rank"] = -1.0

    # Prefer manually prioritized quarterbacks, then highest recent usage
    result.sort_values(["game_id", "selection_rank", "avg_pass_attempts"], ascending=[True, True, False], inplace=True)
    result = result.groupby(["game_id", "team"], as_index=False, sort=False).head(1)
    result.drop(columns=[col for col in ["selection_rank"] if col in result.columns], inplace=True)
    return result.reset_index(drop=True)


def train_and_predict_touchdowns(
    schedule_df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    *,
    seasons: Iterable[int],
    target_season: int,
    target_week: int,
    lookback: int = 4,
    min_touches: float = 0.5,
    roster_df: Optional[pd.DataFrame] = None,
) -> tuple[PlayerTouchdownModelResult, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper to train and generate upcoming touchdown probabilities."""

    relevant_seasons = sorted(set(int(season) for season in seasons))
    pbp_filtered = pbp_df[pbp_df["season"].isin(relevant_seasons)].copy()
    if pbp_filtered.empty:
        raise ValueError("Filtered play-by-play dataset is empty for the requested seasons.")

    player_stats = build_player_game_stats(pbp_filtered, schedule_df)
    dataset = prepare_player_touchdown_dataset(player_stats, lookback=lookback)
    model_result = train_touchdown_model(
        dataset,
        lookback=lookback,
        min_touches=min_touches,
    )

    roster_lookup = _prepare_roster_lookup(
        roster_df,
        target_season=target_season,
        target_week=target_week,
    )

    upcoming = predict_upcoming_touchdowns(
        model_result,
        player_stats,
        schedule_df,
        target_season=target_season,
        target_week=target_week,
        roster_lookup=roster_lookup,
    )
    return model_result, dataset, upcoming


def train_and_predict_passing_yards(
    schedule_df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    *,
    seasons: Iterable[int],
    target_season: int,
    target_week: int,
    lookback: int = 4,
    min_attempts: float = 15.0,
    roster_df: Optional[pd.DataFrame] = None,
    preferred_qbs: Optional[Mapping[str, str]] = None,
) -> tuple[PlayerPassingYardsModelResult, pd.DataFrame, pd.DataFrame]:
    """Wrapper to train and project quarterback passing yards for upcoming games."""

    relevant_seasons = sorted(set(int(season) for season in seasons))
    pbp_filtered = pbp_df[pbp_df["season"].isin(relevant_seasons)].copy()
    if pbp_filtered.empty:
        raise ValueError("Filtered play-by-play dataset is empty for the requested seasons.")

    qb_stats = build_quarterback_game_stats(pbp_filtered, schedule_df)
    dataset = prepare_quarterback_passing_dataset(qb_stats, lookback=lookback)
    model_result = train_passing_yards_model(
        dataset,
        lookback=lookback,
        min_attempts=min_attempts,
    )

    roster_lookup = _prepare_roster_lookup(
        roster_df,
        target_season=target_season,
        target_week=target_week,
    )

    upcoming = predict_upcoming_passing_yards(
        model_result,
        qb_stats,
        schedule_df,
        target_season=target_season,
        target_week=target_week,
        roster_lookup=roster_lookup,
        preferred_qbs=preferred_qbs,
    )
    return model_result, dataset, upcoming