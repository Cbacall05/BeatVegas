"""Utilities for quantifying superstar injury impact on team projections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

SUPERSTAR_POSITIONS = {"QB", "WR", "TE", "RB"}
IMPORTANCE_LOOKBACK_GAMES = 4
STATUS_WEIGHTS = {
    "OUT": 1.0,
    "DOUBTFUL": 0.85,
    "QUESTIONABLE": 0.6,
    "DID NOT PARTICIPATE": 0.75,
    "LIMITED": 0.3,
    "SUSPENDED": 1.0,
    "INJURED RESERVE": 1.0,
    "PUP": 1.0,
}


@dataclass(frozen=True)
class InjuryAdjustment:
    team: str
    player_id: str
    player_name: str
    position: str
    status: str
    impact_score: float
    penalty: float


def _standardize_team(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper()


def _standardize_status(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.upper().str.strip()
    return cleaned.replace({
        "INJURED RESERVE - DESIGNATED FOR RETURN": "INJURED RESERVE",
        "INJURED RESERVE (DESIGNATED FOR RETURN)": "INJURED RESERVE",
        "PHYSICALLY UNABLE TO PERFORM": "PUP",
        "DID NOT PARTICIPATE IN PRACTICE": "DID NOT PARTICIPATE",
    })


def compute_usage_profile(
    weekly_df: pd.DataFrame,
    *,
    lookback_games: int = IMPORTANCE_LOOKBACK_GAMES,
    minimum_games: int = 2,
) -> pd.DataFrame:
    """Return a dataframe with usage-based impact scores for skill players."""

    if weekly_df.empty:
        return pd.DataFrame(columns=["player_id", "team", "position", "player_name", "impact_score"])

    working = weekly_df.copy()
    working["player_id"] = working["player_id"].astype(str)
    team_col = next((col for col in ("recent_team", "team", "team_abbr") if col in working.columns), None)
    if team_col is None:
        raise ValueError("Weekly dataframe missing team column required for injury adjustments.")
    working["team"] = _standardize_team(working[team_col])
    working = working[working["team"] != "TOT"]
    working["position"] = working.get("position", "").astype(str).str.upper()
    working = working[working["position"].isin(SUPERSTAR_POSITIONS)].copy()

    if working.empty:
        return pd.DataFrame(columns=["player_id", "team", "position", "player_name", "impact_score"])

    working["player_name"] = working.get("player_name", "").fillna("Unknown")
    working["season"] = working.get("season").fillna(0).astype(int)
    working["week"] = working.get("week").fillna(0).astype(int)
    working.sort_values(["player_id", "season", "week"], inplace=True)
    working["appearance_idx"] = working.groupby("player_id").cumcount()
    working["max_idx"] = working.groupby("player_id")['appearance_idx'].transform('max')
    working = working[working["appearance_idx"] >= working["max_idx"] - (lookback_games - 1)]

    usage_components = {
        "passing_attempts": working.get("attempts", 0).fillna(0),
        "rushing_attempts": working.get("carries", working.get("rushing_attempts", 0)).fillna(0),
        "targets": working.get("targets", working.get("receptions", 0)).fillna(0),
    }

    working["usage_raw"] = (
        np.where(working["position"] == "QB", usage_components["passing_attempts"] * 1.6, 0)
        + usage_components["rushing_attempts"]
        + usage_components["targets"]
    )

    if "total_epa" in working.columns:
        working["total_epa"] = working["total_epa"].fillna(0)
    else:
        working["total_epa"] = 0.0

    working["team_usage"] = working.groupby(["season", "week", "team"])["usage_raw"].transform("sum")
    working["team_epa"] = working.groupby(["season", "week", "team"])["total_epa"].transform("sum")
    working = working[working["team_usage"] > 0]

    working["usage_share"] = working["usage_raw"] / working["team_usage"]
    working["epa_share"] = np.divide(
        working["total_epa"],
        working["team_epa"].replace(0, np.nan),
    ).fillna(0)

    aggregates = (
        working.groupby(["player_id", "team", "position", "player_name"], as_index=False)
        .agg(
            games_played=("week", "nunique"),
            avg_usage_share=("usage_share", "mean"),
            avg_epa_share=("epa_share", "mean"),
        )
        .query("games_played >= @minimum_games")
    )

    if aggregates.empty:
        return pd.DataFrame(columns=["player_id", "team", "position", "player_name", "impact_score"])

    qb_boost = np.where(aggregates["position"] == "QB", 0.15, 0.0)
    usage_component = aggregates["avg_usage_share"].fillna(0)
    epa_component = aggregates["avg_epa_share"].fillna(0)
    impact = np.clip(0.7 * usage_component + 0.3 * epa_component + qb_boost, 0, 1)

    return aggregates.assign(impact_score=impact)[
        ["player_id", "team", "position", "player_name", "impact_score"]
    ].sort_values("impact_score", ascending=False)


def select_superstars(usage_df: pd.DataFrame, *, threshold: float = 0.18) -> pd.DataFrame:
    """Return superstar-caliber players filtered by impact threshold."""

    if usage_df.empty:
        return usage_df

    result = usage_df[usage_df["impact_score"] >= threshold].copy()
    return result.sort_values("impact_score", ascending=False)


def summarize_injuries(
    injury_df: pd.DataFrame,
    superstar_df: pd.DataFrame,
    *,
    include_statuses: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Join injury reports with superstar impact scores and compute penalties."""

    if injury_df.empty or superstar_df.empty:
        return pd.DataFrame(columns=[field for field in InjuryAdjustment.__annotations__.keys()])

    injuries = injury_df.copy()
    injuries["status"] = _standardize_status(
        injuries.get("injury_status", injuries.get("practice", injuries.get("report_status", "")))
    )
    if include_statuses:
        allowed = {status.upper() for status in include_statuses}
        injuries = injuries[injuries["status"].isin(allowed)]
    else:
        injuries = injuries[injuries["status"].isin(STATUS_WEIGHTS)]

    if injuries.empty:
        return pd.DataFrame(columns=[field for field in InjuryAdjustment.__annotations__.keys()])

    injuries["player_id"] = injuries.get("player_id", injuries.get("gsis_id", "")).astype(str)
    team_col = next((col for col in ("team", "recent_team", "team_abbr") if col in injuries.columns), None)
    if team_col is None:
        injuries["team"] = ""
    else:
        injuries["team"] = _standardize_team(injuries[team_col])

    merged = injuries.merge(superstar_df, on="player_id", how="inner", suffixes=("_inj", ""))
    if merged.empty:
        return pd.DataFrame(columns=[field for field in InjuryAdjustment.__annotations__.keys()])

    merged["team"] = np.where(merged["team"].eq(""), merged.get("team_inj", merged["team"]), merged["team"])
    if "team_inj" in merged.columns:
        merged.drop(columns="team_inj", inplace=True)
    merged["team"] = _standardize_team(merged["team"])
    merged["status_weight"] = merged["status"].map(STATUS_WEIGHTS).fillna(0)
    merged["penalty"] = np.clip(merged["impact_score"] * merged["status_weight"], 0, 1)

    return merged.rename(columns={"player_name": "player_name"})[
        ["team", "player_id", "player_name", "position", "status", "impact_score", "penalty"]
    ].sort_values("penalty", ascending=False)


def team_penalties(adjustments: pd.DataFrame, *, cap: float = 0.35) -> pd.DataFrame:
    """Aggregate superstar penalties by team for downstream model adjustments."""

    if adjustments.empty:
        return pd.DataFrame(columns=["team", "penalty"])

    team_df = (
        adjustments.groupby("team", as_index=False)["penalty"].sum()
        .assign(penalty=lambda df: df["penalty"].clip(upper=cap))
        .sort_values("penalty", ascending=False)
    )
    return team_df


def build_alert_messages(adjustments: pd.DataFrame) -> list[str]:
    """Convert superstar injury adjustments into human-readable alerts."""

    if adjustments.empty:
        return []

    messages = []
    for row in adjustments.itertuples(index=False):
        penalty_pct = row.penalty * 100
        messages.append(
            f"{row.team}: {row.player_name} ({row.position}) listed as {row.status.title()} — impact penalty {penalty_pct:.1f}%"
        )
    return messages
