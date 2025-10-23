"""Schedule enrichment utilities for Project Beat Vegas.

This module augments NFL schedule data with rest and travel metrics derived
from historical kickoff dates and stadium locations. Rest deltas quantify the
number of days between contests for each team, while travel deltas estimate the
miles covered between consecutive game sites using a simplified Haversine
calculation.

Outputs are provided at both the team-game and matchup level so downstream
pipelines can persist nightly snapshots and join features during model
training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.8
TEAM_STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.5277, -112.2626),  # State Farm Stadium
    "ATL": (33.7554, -84.4009),  # Mercedes-Benz Stadium
    "BAL": (39.2779, -76.6229),  # M&T Bank Stadium
    "BUF": (42.7738, -78.7870),  # Highmark Stadium
    "CAR": (35.2251, -80.8520),  # Bank of America Stadium
    "CHI": (41.8623, -87.6167),  # Soldier Field
    "CIN": (39.0954, -84.5160),  # Paycor Stadium
    "CLE": (41.5061, -81.6995),  # Cleveland Browns Stadium
    "DAL": (32.7473, -97.0945),  # AT&T Stadium
    "DEN": (39.7439, -105.0201),  # Empower Field at Mile High
    "DET": (42.3400, -83.0456),  # Ford Field
    "GB": (44.5013, -88.0622),  # Lambeau Field
    "HOU": (29.6847, -95.4107),  # NRG Stadium
    "IND": (39.7601, -86.1639),  # Lucas Oil Stadium
    "JAX": (30.3240, -81.6373),  # EverBank Stadium
    "KC": (39.0489, -94.4842),  # GEHA Field at Arrowhead Stadium
    "LAC": (33.9535, -118.3391),  # SoFi Stadium (shared with LA Rams)
    "LA": (33.9535, -118.3391),  # SoFi Stadium
    "LV": (36.0909, -115.1830),  # Allegiant Stadium
    "MIA": (25.9580, -80.2389),  # Hard Rock Stadium
    "MIN": (44.9730, -93.2570),  # U.S. Bank Stadium
    "NE": (42.0909, -71.2643),  # Gillette Stadium
    "NO": (29.9509, -90.0815),  # Caesars Superdome
    "NYG": (40.8136, -74.0744),  # MetLife Stadium (shared)
    "NYJ": (40.8136, -74.0744),  # MetLife Stadium (shared)
    "PHI": (39.9008, -75.1675),  # Lincoln Financial Field
    "PIT": (40.4468, -80.0158),  # Acrisure Stadium
    "SEA": (47.5952, -122.3316),  # Lumen Field
    "SF": (37.4030, -121.9700),  # Levi's Stadium
    "TB": (27.9759, -82.5033),  # Raymond James Stadium
    "TEN": (36.1665, -86.7713),  # Nissan Stadium
    "WAS": (38.9077, -76.8645),  # Commanders Field (FedEx Field)
}

NEUTRAL_SITE_COORDS: dict[str, tuple[float, float]] = {
    "TOTTENHAM HOTSPUR STADIUM": (51.6043, -0.0657),
    "WEMBLEY STADIUM": (51.5560, -0.2796),
    "ALLIANZ ARENA": (48.2188, 11.6247),
    "DEUTSCHE BANK PARK": (50.0680, 8.6452),
    "MERCEDES-BENZ ARENA": (48.7920, 9.2319),
    "ESTADIO AZTECA": (19.3030, -99.1500),
    "ALAMODOME": (29.4246, -98.4938),
    "AVIVA STADIUM": (53.3351, -6.2283),
}


@dataclass
class RestTravelResult:
    """Container for rest/travel outputs."""

    game_level: pd.DataFrame
    team_level: pd.DataFrame


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two latitude/longitude pairs."""

    if any(map(lambda v: pd.isna(v) or v is None, (lat1, lon1, lat2, lon2))):
        return float("nan")

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return float(EARTH_RADIUS_MILES * c)


def _normalize_site_name(site: str | None) -> str | None:
    if not site or not isinstance(site, str):
        return None
    cleaned = site.replace("(London)", "").replace("(Munich)", "").replace("(Frankfurt)", "")
    cleaned = cleaned.replace("(Mexico City)", "").replace("International Series", "")
    return cleaned.strip() or None


def _lookup_site_coordinates(site: str | None) -> tuple[float, float] | None:
    normalized = _normalize_site_name(site)
    if not normalized:
        return None
    key = normalized.upper()
    if key in NEUTRAL_SITE_COORDS:
        return NEUTRAL_SITE_COORDS[key]
    for candidate, coords in NEUTRAL_SITE_COORDS.items():
        if candidate in key:
            return coords
    return None


def _resolve_coordinates(home_team: str, *, site: str | None, stadium: str | None, neutral_site: bool) -> tuple[float, float] | None:
    if neutral_site:
        coords = _lookup_site_coordinates(site) or _lookup_site_coordinates(stadium)
        if coords:
            return coords
    return TEAM_STADIUM_COORDS.get(home_team)


def _ensure_datetime(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce", utc=False)
    if converted.notna().any():
        return converted
    return pd.Series(pd.NaT, index=series.index)


def compute_rest_travel(schedule_df: pd.DataFrame) -> RestTravelResult:
    """Compute rest days and travel miles for each matchup and team.

    Parameters
    ----------
    schedule_df
        DataFrame containing at least ``game_id``, ``season``, ``week``,
        ``home_team``, ``away_team``, and a kickoff date column (``start_time``
        or ``gameday``).

    Returns
    -------
    RestTravelResult
        Dataclass exposing both matchup-level and team-level enriched frames.
    """

    required_cols = {"game_id", "season", "week", "home_team", "away_team"}
    missing = required_cols - set(schedule_df.columns)
    if missing:
        raise ValueError(f"Schedule dataframe missing required columns: {sorted(missing)}")

    schedule = schedule_df.copy()
    kickoff = pd.Series(pd.NaT, index=schedule.index, dtype="datetime64[ns]")
    for candidate in ("start_time", "kickoff", "game_time", "gameday"):
        if candidate in schedule.columns:
            converted = _ensure_datetime(schedule[candidate])
            kickoff = kickoff.fillna(converted)
    schedule["kickoff_dt"] = kickoff

    if schedule["kickoff_dt"].isna().all():
        # Fall back to ordering by season/week when timestamps are unavailable.
        schedule.sort_values(["season", "week", "game_id"], inplace=True)
        schedule["kickoff_dt"] = pd.to_datetime(
            schedule["season"].astype(str) + "-" + schedule["week"].astype(str) + "-01",
            errors="coerce",
        )

    team_rows: list[dict[str, object]] = []
    for row in schedule.itertuples(index=False):
        site_value = getattr(row, "site", None)
        stadium_value = getattr(row, "stadium", None)
        neutral = bool(getattr(row, "neutral_site", False))
        coords = _resolve_coordinates(row.home_team, site=site_value, stadium=stadium_value, neutral_site=neutral)
        lat, lon = coords if coords else (float("nan"), float("nan"))

        kickoff_dt = getattr(row, "kickoff_dt", pd.NaT)
        for team, opponent, home_away in (
            (row.home_team, row.away_team, "home"),
            (row.away_team, row.home_team, "away"),
        ):
            location_team = row.home_team
            location_coords = (lat, lon)
            team_rows.append(
                {
                    "game_id": row.game_id,
                    "season": row.season,
                    "week": row.week,
                    "team": team,
                    "opponent": opponent,
                    "home_away": home_away,
                    "kickoff_dt": kickoff_dt,
                    "location_lat": location_coords[0],
                    "location_lon": location_coords[1],
                    "site": site_value,
                    "stadium": stadium_value,
                    "neutral_site": neutral,
                }
            )

    team_df = pd.DataFrame(team_rows)
    team_df.sort_values(["team", "kickoff_dt", "week", "game_id"], inplace=True)

    rest_days = team_df.groupby("team")["kickoff_dt"].diff().dt.total_seconds() / (60 * 60 * 24)
    team_df["rest_days"] = rest_days.round(1)
    prev_lat = team_df.groupby("team")["location_lat"].shift(1)
    prev_lon = team_df.groupby("team")["location_lon"].shift(1)
    team_df["travel_miles"] = [
        _haversine_miles(lat_prev, lon_prev, lat_cur, lon_cur)
        if not (pd.isna(lat_prev) or pd.isna(lon_prev) or pd.isna(lat_cur) or pd.isna(lon_cur))
        else float("nan")
        for lat_prev, lon_prev, lat_cur, lon_cur in zip(prev_lat, prev_lon, team_df["location_lat"], team_df["location_lon"])
    ]
    team_df.loc[team_df["home_away"] == "home", "travel_miles"] = team_df.loc[
        team_df["home_away"] == "home", "travel_miles"
    ].fillna(0.0)
    team_df["travel_miles"] = team_df["travel_miles"].round(1)

    team_df["short_week"] = team_df["rest_days"].lt(6.0)
    team_df["long_rest"] = team_df["rest_days"].ge(10.0)

    home_df = team_df[team_df["home_away"] == "home"][
        [
            "game_id",
            "rest_days",
            "travel_miles",
            "short_week",
            "long_rest",
        ]
    ].rename(
        columns={
            "rest_days": "home_rest_days",
            "travel_miles": "home_travel_miles",
            "short_week": "home_short_week",
            "long_rest": "home_long_rest",
        }
    )

    away_df = team_df[team_df["home_away"] == "away"][
        [
            "game_id",
            "rest_days",
            "travel_miles",
            "short_week",
            "long_rest",
        ]
    ].rename(
        columns={
            "rest_days": "away_rest_days",
            "travel_miles": "away_travel_miles",
            "short_week": "away_short_week",
            "long_rest": "away_long_rest",
        }
    )

    game_level = home_df.merge(away_df, on="game_id", how="inner")
    game_level["rest_days_diff"] = game_level["home_rest_days"] - game_level["away_rest_days"]
    game_level["travel_miles_diff"] = game_level["home_travel_miles"] - game_level["away_travel_miles"]

    buckets = [-float("inf"), -3.5, -1.5, -0.5, 0.5, 1.5, 3.5, float("inf")]
    labels = [
        "<= -3.5 DAYS",
        "-3.5 TO -1.5",
        "-1.5 TO -0.5",
        "EVEN",
        "0.5 TO 1.5",
        "1.5 TO 3.5",
        ">= 3.5 DAYS",
    ]
    game_level["rest_advantage_bucket"] = pd.cut(
        game_level["rest_days_diff"], bins=buckets, labels=labels, include_lowest=True
    )
    game_level["rest_advantage_bucket"] = game_level["rest_advantage_bucket"].astype(str)

    return RestTravelResult(game_level=game_level, team_level=team_df)


def attach_rest_travel(schedule_df: pd.DataFrame, *, persist_path: Path | None = None) -> pd.DataFrame:
    """Attach rest/travel features to a raw schedule dataframe.

    Parameters
    ----------
    schedule_df
        Schedule dataframe from ``nfl_data_py`` or another provider.
    persist_path
        Optional path where the matchup-level parquet should be written. The
        directory will be created if necessary.

    Returns
    -------
    pd.DataFrame
        Original schedule dataframe augmented with rest and travel columns.
    """

    cols_to_remove = {
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
    }

    sanitized = schedule_df.drop(columns=list(cols_to_remove & set(schedule_df.columns)), errors="ignore")

    result = compute_rest_travel(sanitized)
    enriched = sanitized.merge(result.game_level, on="game_id", how="left")

    if persist_path is not None:
        persist_path = Path(persist_path)
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        result.game_level.to_parquet(persist_path, index=False)

    return enriched


def persist_team_travel(team_df: pd.DataFrame, path: Path) -> None:
    """Persist team-level travel ledger to parquet for auditing."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    team_df.to_parquet(path, index=False)


def load_rest_travel_cache(path: Path) -> pd.DataFrame:
    """Load a previously cached rest/travel parquet if available."""

    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(path)
