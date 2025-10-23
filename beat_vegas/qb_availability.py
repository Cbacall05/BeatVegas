"""Quarterback availability and replacement value modeling utilities."""
from __future__ import annotations

import logging
import math
import re
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

SEASON_WEIGHT_BY_OFFSET = {
    0: 1.0,
    1: 0.6,
    2: 0.35,
    3: 0.2,
}
MIN_WEIGHT = 0.05
PRACTICE_MULTIPLIER = {
    "FULL": 1.0,
    "FULL PARTICIPATION": 1.0,
    "LIMITED": 0.75,
    "LTD": 0.75,
    "LTD PARTICIPATION": 0.75,
    "DID NOT PARTICIPATE": 0.5,
    "DNP": 0.5,
}
STATUS_AVAILABILITY = {
    "": 1.0,
    "NONE": 1.0,
    "QUESTIONABLE": 0.65,
    "PROBABLE": 0.8,
    "DOUBTFUL": 0.2,
    "OUT": 0.0,
    "INJURED RESERVE": 0.0,
    "IR": 0.0,
    "PUP": 0.15,
    "RESERVE/COVID-19": 0.1,
    "SUSPENDED": 0.0,
    "NFI": 0.1,
}
ROSTER_STATUS_AVAILABILITY = {
    "ACTIVE": 1.0,
    "QUESTIONABLE": 0.65,
    "DOUBTFUL": 0.2,
    "OUT": 0.0,
    "INJURED RESERVE": 0.0,
    "IR": 0.0,
    "SUSPENDED": 0.0,
    "PUP": 0.15,
}
QB_PENALTY_PER_POINT = 0.07  # Empirically: 3-point expected drop -> ~0.21 penalty.
QB_PENALTY_CAP = 0.4


def _normalize_name(value: str | float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = text.replace(" jr", "").replace(" sr", "").replace(" iii", "").replace(" ii", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _season_weight(season: int, target_season: int) -> float:
    offset = target_season - season
    base = SEASON_WEIGHT_BY_OFFSET.get(offset, None)
    if base is None:
        if offset < 0:
            base = 0.0
        else:
            base = SEASON_WEIGHT_BY_OFFSET.get(max(SEASON_WEIGHT_BY_OFFSET.keys()), MIN_WEIGHT)
            decay = 0.5 ** max(0, offset - max(SEASON_WEIGHT_BY_OFFSET.keys()))
            base *= decay
    return max(base, MIN_WEIGHT)


def _prepare_qb_weekly(weekly_df: pd.DataFrame) -> pd.DataFrame:
    working = weekly_df.copy()
    team_col = next((col for col in ("recent_team", "team", "team_abbr") if col in working.columns), None)
    if team_col is None:
        raise ValueError("Weekly player dataframe missing team column required for QB availability.")
    position_col = next(
        (col for col in ("position", "player_position", "pos") if col in working.columns),
        None,
    )
    if position_col is None:
        raise ValueError("Weekly player dataframe missing position column required for QB availability.")

    working["position"] = working[position_col].astype(str).str.upper()
    working = working[working["position"] == "QB"].copy()
    if working.empty:
        return working

    working["team"] = working[team_col].astype(str).str.upper()
    working = working[working["team"] != "TOT"].copy()
    working["player_id"] = working.get("player_id", working.get("gsis_id", "")).astype(str)
    working["player_name"] = working.get("player_name", working.get("name", "Unknown")).astype(str)

    working["season"] = working.get("season").astype(int)
    working["week"] = working.get("week").fillna(0).astype(int)

    dropbacks = working.get("dropbacks")
    if dropbacks is None:
        attempts = working.get("attempts", working.get("passing_attempts", 0)).fillna(0)
        sacks = working.get("sacks", working.get("sack", 0)).fillna(0)
        dropbacks = attempts + sacks
    working["dropbacks_calc"] = dropbacks.astype(float)

    epa_source = working.get("passing_epa")
    if epa_source is None:
        epa_source = working.get("total_epa", working.get("epa", 0))
    working["epa_total"] = epa_source.fillna(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        working["epa_per_play"] = np.where(
            working["dropbacks_calc"] > 0,
            working["epa_total"] / working["dropbacks_calc"],
            np.nan,
        )
    working["epa_per_play"].replace([np.inf, -np.inf], np.nan, inplace=True)
    working["player_key"] = working["player_name"].map(_normalize_name)
    return working


def _merge_status(signals: pd.DataFrame, sleeper_df: pd.DataFrame) -> pd.DataFrame:
    if sleeper_df is None or sleeper_df.empty:
        signals["injury_status"] = ""
        signals["practice_status"] = ""
        signals["roster_status"] = ""
        signals["depth_chart_order"] = np.nan
        return signals

    sleeper_working = sleeper_df.copy()
    sleeper_working["player_id"] = sleeper_working["player_id"].astype(str)
    sleeper_working["team"] = sleeper_working["team"].astype(str).str.upper()
    sleeper_working["name_key"] = sleeper_working["full_name"].map(_normalize_name)

    subset_cols = [
        "player_id",
        "team",
        "full_name",
        "name_key",
        "injury_status",
        "practice_status",
        "status",
        "depth_chart_order",
        "depth_chart_position",
    ]
    sleeper_subset = sleeper_working[subset_cols].copy()
    merged = signals.merge(
        sleeper_subset,
        on="player_id",
        how="left",
        suffixes=("", "_sl"),
    )

    missing_mask = merged["injury_status"].isna()
    if missing_mask.any():
        fallback = sleeper_subset.copy()
        fallback["depth_chart_order"] = pd.to_numeric(fallback["depth_chart_order"], errors="coerce")
        fallback.sort_values(["depth_chart_order"], inplace=True, na_position="last")
        fallback = (
            fallback.groupby(["name_key", "team"], as_index=False)
            .first()
            .set_index(["name_key", "team"])
        )
        lookup_keys = list(zip(merged.loc[missing_mask, "player_key"], merged.loc[missing_mask, "team"]))
        try:
            fallback_rows = fallback.reindex(lookup_keys)
        except ValueError:
            fallback_rows = fallback.reset_index().drop_duplicates(subset=["name_key", "team"]).set_index(["name_key", "team"]).reindex(lookup_keys)
        target_index = merged.index[missing_mask]
        for column in ("injury_status", "practice_status", "status", "depth_chart_order", "depth_chart_position"):
            if column not in fallback_rows.columns:
                continue
            values = fallback_rows[column]
            if values.isna().all():
                continue
            fill_series = pd.Series(values.to_numpy(), index=target_index)
            existing = merged.loc[target_index, column]
            merged.loc[target_index, column] = existing.where(existing.notna(), fill_series)

    merged.rename(columns={"status": "roster_status"}, inplace=True)
    merged["injury_status"] = merged["injury_status"].fillna("").astype(str).str.upper()
    merged["practice_status"] = merged["practice_status"].fillna("").astype(str).str.upper()
    merged["roster_status"] = merged["roster_status"].fillna("").astype(str).str.upper()
    merged["depth_chart_order"] = pd.to_numeric(merged.get("depth_chart_order"), errors="coerce")
    return merged


def _availability_probability(row: pd.Series) -> float:
    injury_status = row.get("injury_status", "")
    roster_status = row.get("roster_status", "")
    practice_status = row.get("practice_status", "")

    base = STATUS_AVAILABILITY.get(injury_status, None)
    if base is None:
        base = 0.9 if injury_status else 1.0
    roster_adj = ROSTER_STATUS_AVAILABILITY.get(roster_status, None)
    if roster_adj is not None:
        base = min(base, roster_adj)
    practice_multiplier = PRACTICE_MULTIPLIER.get(practice_status, 1.0)
    availability = float(np.clip(base * practice_multiplier, 0.0, 1.0))
    return availability


def compute_team_qb_availability(
    weekly_df: pd.DataFrame,
    sleeper_qbs: Optional[pd.DataFrame],
    *,
    target_season: int,
    week: int,
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    """Compute QB availability context for each team.

    Returns a dataframe with one row per team and columns describing the starter,
    backup, expected drop expressed in points, and a normalized penalty suitable
    for downstream probability adjustments.
    """

    qb_weekly = _prepare_qb_weekly(weekly_df)
    if qb_weekly.empty:
        return pd.DataFrame(columns=[
            "team",
            "starter_player_id",
            "starter_player_name",
            "starter_availability",
            "starter_status",
            "starter_practice",
            "backup_player_id",
            "backup_player_name",
            "epa_diff_per_play",
            "team_dropbacks_avg",
            "expected_points_drop",
            "qb_penalty",
        ])

    qb_weekly = qb_weekly[qb_weekly["season"] <= target_season]
    if qb_weekly.empty:
        return pd.DataFrame()

    # Usage window: recent games in the current season.
    usage_window = qb_weekly[(qb_weekly["season"] == target_season) & (qb_weekly["week"] < week)]
    if usage_window.empty:
        usage_window = qb_weekly[qb_weekly["season"] == target_season]
    if usage_window.empty:
        usage_window = qb_weekly[qb_weekly["season"] == target_season - 1]
    if usage_window.empty:  # pragma: no cover - defensive fallback
        usage_window = qb_weekly.copy()

    usage_window = usage_window.copy()
    usage_window["player_key"] = usage_window["player_name"].map(_normalize_name)

    usage_summary = (
        usage_window.groupby(["team", "player_id", "player_name", "player_key"], as_index=False)
        .agg(
            total_dropbacks=("dropbacks_calc", "sum"),
            games_active=("week", "nunique"),
            last_week=("week", "max"),
        )
    )
    if usage_summary.empty:
        return pd.DataFrame()

    usage_summary = _merge_status(usage_summary, sleeper_qbs)
    usage_summary["availability_prob"] = usage_summary.apply(_availability_probability, axis=1)

    # Historical EPA weighted average per player.
    history_window = qb_weekly[(qb_weekly["season"] >= target_season - 3) & (qb_weekly["season"] <= target_season)].copy()
    if history_window.empty:
        history_window = qb_weekly.copy()
    history_window["year_weight"] = history_window["season"].map(lambda s: _season_weight(int(s), target_season))
    history_window["weight"] = history_window["year_weight"] * history_window["dropbacks_calc"].clip(lower=1)
    history_window["weighted_epa"] = history_window["epa_per_play"].fillna(0) * history_window["weight"]

    epa_agg = (
        history_window.groupby(["player_id", "player_name"], as_index=False)
        .agg(weighted_epa_sum=("weighted_epa", "sum"), weight_sum=("weight", "sum"))
    )
    epa_agg["epa_per_play_weighted"] = np.where(
        epa_agg["weight_sum"] > 0,
        epa_agg["weighted_epa_sum"] / epa_agg["weight_sum"],
        np.nan,
    )
    league_weight_sum = epa_agg["weight_sum"].sum()
    if league_weight_sum > 0:
        league_epa = epa_agg["weighted_epa_sum"].sum() / league_weight_sum
    else:
        league_epa = 0.0

    usage_summary = usage_summary.merge(
        epa_agg[["player_id", "epa_per_play_weighted"]],
        on="player_id",
        how="left",
    )
    usage_summary["epa_per_play_weighted"] = usage_summary["epa_per_play_weighted"].fillna(league_epa)

    team_context_rows: list[dict[str, float | str]] = []
    for team, team_rows in usage_summary.groupby("team"):
        team_rows = team_rows.sort_values(
            ["total_dropbacks", "availability_prob", "depth_chart_order"],
            ascending=[False, False, True],
        )
        team_dropbacks = team_rows["total_dropbacks"].sum()
        if team_dropbacks <= 0:
            continue
        team_weeks = max(1, int(usage_window[usage_window["team"] == team]["week"].nunique()))
        team_dropbacks_avg = float(team_dropbacks / team_weeks)

        starter = team_rows.iloc[0]
        backup = team_rows.iloc[1] if len(team_rows) > 1 else None

        starter_epa = float(starter["epa_per_play_weighted"])
        backup_epa = float(backup["epa_per_play_weighted"]) if backup is not None else league_epa
        epa_diff = max(starter_epa - backup_epa, 0.0)
        availability = float(np.clip(starter["availability_prob"], 0.0, 1.0))
        expected_points_drop = epa_diff * team_dropbacks_avg * (1 - availability)
        qb_penalty = min(expected_points_drop * QB_PENALTY_PER_POINT, QB_PENALTY_CAP)

        team_context_rows.append(
            {
                "team": team,
                "starter_player_id": starter["player_id"],
                "starter_player_name": starter["player_name"],
                "starter_availability": availability,
                "starter_status": starter.get("injury_status", ""),
                "starter_practice": starter.get("practice_status", ""),
                "backup_player_id": backup["player_id"] if backup is not None else pd.NA,
                "backup_player_name": backup["player_name"] if backup is not None else pd.NA,
                "epa_diff_per_play": epa_diff,
                "team_dropbacks_avg": team_dropbacks_avg,
                "expected_points_drop": expected_points_drop,
                "qb_penalty": qb_penalty,
            }
        )

    if not team_context_rows:
        return pd.DataFrame()

    context_df = pd.DataFrame(team_context_rows)
    context_df.sort_values("expected_points_drop", ascending=False, inplace=True)
    return context_df


def build_alert_messages(context_df: pd.DataFrame) -> list[str]:
    """Translate QB availability context into human-readable alerts."""

    if context_df is None or context_df.empty:
        return []

    messages: list[str] = []
    for row in context_df.itertuples(index=False):
        penalty_pct = float(row.qb_penalty) * 100
        if penalty_pct < 1.0:
            continue
        availability_pct = float(row.starter_availability) * 100
        status = str(row.starter_status or "").title() or "Likely"
        messages.append(
            f"{row.team}: {row.starter_player_name} availability {availability_pct:.0f}% ({status}) — QB penalty {penalty_pct:.1f}%"
        )
    return messages


def as_injury_adjustments(context_df: pd.DataFrame) -> pd.DataFrame:
    """Project QB availability penalties onto the general injury adjustment schema."""

    if context_df is None or context_df.empty:
        return pd.DataFrame(columns=[
            "team",
            "player_id",
            "player_name",
            "position",
            "status",
            "impact_score",
            "penalty",
        ])

    adjustments = context_df.copy()
    adjustments = adjustments.assign(
        player_id=adjustments["starter_player_id"].astype(str),
        player_name=adjustments["starter_player_name"],
        position="QB",
        status=adjustments["starter_status"].astype(str).str.title().replace({"": "Availability"}),
        impact_score=np.clip(adjustments["expected_points_drop"] / 6.0, 0.0, 1.0),
        penalty=adjustments["qb_penalty"],
    )
    return adjustments[[
        "team",
        "player_id",
        "player_name",
        "position",
        "status",
        "impact_score",
        "penalty",
    ]].sort_values("penalty", ascending=False)