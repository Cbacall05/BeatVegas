"""Player-level touchdown modeling utilities for Project Beat Vegas."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

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
        roster = roster[roster["season"] == target_season]

    if "week" in roster.columns:
        roster = roster[roster["week"] <= target_week]
        roster = roster.sort_values([player_id_col, "week"]).drop_duplicates(player_id_col, keep="last")

    roster = roster[[player_id_col, team_col]].drop_duplicates()
    roster.rename(columns={player_id_col: "player_id", team_col: "team"}, inplace=True)
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