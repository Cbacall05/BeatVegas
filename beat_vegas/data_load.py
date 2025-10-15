"""Data ingestion utilities for Project Beat Vegas.

This module wraps nfl_data_py helpers to fetch schedules, play-by-play (pbp),
and weekly team data. Functions are designed to be composable, cache-friendly,
and production-ready with simple logging hooks.
"""
from __future__ import annotations

import logging
from urllib.error import HTTPError
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

_NFL_IMPORT_ERROR: Optional[ImportError]

try:  # Lazy import to keep module importable without dependency installed.
    import nfl_data_py as _nfl_data_py
    _NFL_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - handled by runtime checks in getters below.
    _nfl_data_py = None
    _NFL_IMPORT_ERROR = exc

if _nfl_data_py is not None:  # pragma: no branch - simple attribute lookups.
    import_pbp_data = getattr(_nfl_data_py, "import_pbp_data", None)
    import_schedules = getattr(_nfl_data_py, "import_schedules", None)
    import_weekly_data = getattr(_nfl_data_py, "import_weekly_data", None)
    import_injuries = getattr(_nfl_data_py, "import_injuries", None)
    import_rosters = getattr(_nfl_data_py, "import_rosters", None) or getattr(
        _nfl_data_py, "import_seasonal_rosters", None
    )
else:
    import_pbp_data = import_schedules = import_weekly_data = import_rosters = import_injuries = None


def _ensure_nfl_data_py_available() -> None:
    """Raise a descriptive error if nfl_data_py is missing."""

    if any(fn is None for fn in (import_pbp_data, import_schedules, import_weekly_data)):
        raise ImportError(
            "nfl_data_py is required for Project Beat Vegas. Install it via 'pip install nfl_data_py'."
        ) from _NFL_IMPORT_ERROR


def _ensure_rosters_available() -> None:
    if import_rosters is None:
        raise ImportError(
            "Roster data requires nfl_data_py>=0.3.2. Update the dependency to enable player props."
        ) from _NFL_IMPORT_ERROR


LOGGER = logging.getLogger(__name__)


def _validate_seasons(seasons: Iterable[int]) -> list[int]:
    """Ensure seasons are sorted, unique, and within plausible NFL history bounds."""
    unique_seasons = sorted({int(season) for season in seasons})
    if not unique_seasons:
        raise ValueError("At least one season must be provided.")

    first_season, last_season = unique_seasons[0], unique_seasons[-1]
    if first_season < 1999 or last_season > pd.Timestamp.now().year:
        raise ValueError(
            "Seasons must fall between 1999 and the current year to align with nfl_data_py coverage."
        )
    return unique_seasons


def load_schedule(seasons: Iterable[int]) -> pd.DataFrame:
    """Fetch game schedules for the requested seasons."""
    _ensure_nfl_data_py_available()
    seasons_list = _validate_seasons(seasons)
    LOGGER.info("Loading schedules for seasons: %s", seasons_list)
    schedule_df = import_schedules(seasons_list)
    schedule_df["game_id"] = schedule_df["game_id"].astype(str)
    return schedule_df


def load_weekly_data(seasons: Iterable[int], columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Fetch weekly team-level stats for the requested seasons."""
    _ensure_nfl_data_py_available()
    seasons_list = _validate_seasons(seasons)
    LOGGER.info("Loading weekly data for seasons: %s", seasons_list)
    weekly_df = import_weekly_data(seasons_list)
    if columns:
        missing_columns = [col for col in columns if col not in weekly_df.columns]
        if missing_columns:
            LOGGER.warning("Requested columns missing from weekly data: %s", missing_columns)
        weekly_df = weekly_df[[col for col in columns if col in weekly_df.columns]]
    return weekly_df


def load_play_by_play(
    seasons: Iterable[int],
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch play-by-play data with optional local caching to disk."""

    _ensure_nfl_data_py_available()
    seasons_list = _validate_seasons(seasons)
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for season in seasons_list:
        cache_path: Optional[Path] = None
        if cache_dir:
            cache_path = cache_dir / f"pbp_{season}.parquet"

        if cache_path and cache_path.exists() and not force_refresh:
            LOGGER.info("Loading cached PBP data for season %s", season)
            frames.append(pd.read_parquet(cache_path))
            continue

        LOGGER.info("Downloading PBP data for season %s", season)
        pbp_df = import_pbp_data([season])
        if cache_path:
            LOGGER.info("Caching PBP data to %s", cache_path)
            pbp_df.to_parquet(cache_path, index=False)
        frames.append(pbp_df)

    combined = pd.concat(frames, ignore_index=True)
    combined["game_id"] = combined["game_id"].astype(str)
    return combined


def load_injuries(seasons: Iterable[int]) -> pd.DataFrame:
    """Fetch weekly injury reports for the requested seasons."""

    if import_injuries is None:
        raise ImportError(
            "Injury data requires nfl_data_py. Install or upgrade the dependency to enable injury adjustments."
        ) from _NFL_IMPORT_ERROR

    seasons_list = _validate_seasons(seasons)
    LOGGER.info("Loading injuries for seasons: %s", seasons_list)
    try:
        injury_df = import_injuries(seasons_list)
    except HTTPError as exc:  # pragma: no cover - network dependent
        LOGGER.warning("Injury endpoint returned HTTP error %s; continuing without injuries.", exc)
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001 - propagate unexpected issues
        message = str(exc)
        if "404" in message or "Not Found" in message:
            LOGGER.warning("Injury endpoint returned 404 for seasons %s; continuing without injuries.", seasons_list)
            return pd.DataFrame()
        raise

    if injury_df.empty:
        LOGGER.warning("Injury endpoint returned no rows for the requested seasons: %s", seasons_list)
        return injury_df

    injury_df = injury_df.copy()
    if "gsis_id" in injury_df.columns and "player_id" not in injury_df.columns:
        injury_df["player_id"] = injury_df["gsis_id"]

    for col in ("team", "team_abbr", "recent_team"):
        if col in injury_df.columns:
            injury_df[col] = injury_df[col].astype(str).str.upper()

    if "reported_date" in injury_df.columns:
        injury_df["reported_date"] = pd.to_datetime(injury_df["reported_date"], errors="coerce")

    return injury_df


def load_rosters(seasons: Iterable[int]) -> pd.DataFrame:
    """Fetch roster data for the requested seasons, normalized for downstream joins."""

    _ensure_rosters_available()
    seasons_list = _validate_seasons(seasons)
    LOGGER.info("Loading rosters for seasons: %s", seasons_list)
    roster_df = import_rosters(seasons_list)
    if roster_df.empty:
        raise ValueError("Roster endpoint returned no rows for the requested seasons.")

    if "player_id" not in roster_df.columns and "gsis_id" in roster_df.columns:
        roster_df["player_id"] = roster_df["gsis_id"]

    if "player_id" not in roster_df.columns:
        raise ValueError("Roster dataframe missing 'player_id' column required for player props.")

    team_col = None
    for candidate in ("team", "recent_team", "team_abbr"):
        if candidate in roster_df.columns:
            team_col = candidate
            break

    if team_col is None:
        raise ValueError("Roster dataframe missing any recognized team column ('team', 'recent_team').")

    roster_df[team_col] = roster_df[team_col].astype(str).str.upper()
    roster_df["player_id"] = roster_df["player_id"].astype(str)

    if "season" not in roster_df.columns:
        roster_df["season"] = seasons_list[0] if len(seasons_list) == 1 else pd.NA

    return roster_df


def convert_moneyline_to_probability(moneyline: pd.Series) -> pd.Series:
    """Convert American moneyline odds to implied probability."""

    ml = moneyline.astype(float)
    probs = pd.Series(index=ml.index, dtype=float)
    positive = ml > 0
    negative = ml < 0
    probs.loc[positive] = 100 / (ml.loc[positive] + 100)
    probs.loc[negative] = (-ml.loc[negative]) / ((-ml.loc[negative]) + 100)
    probs.loc[ml == 0] = pd.NA
    return probs


def explode_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form schedule with one row per team per game."""

    required_cols = {"game_id", "season", "week", "home_team", "away_team"}
    missing = required_cols - set(schedule_df.columns)
    if missing:
        raise ValueError(f"Schedule dataframe missing required columns: {sorted(missing)}")

    long_rows = []
    for _, row in schedule_df.iterrows():
        base = {
            "game_id": str(row["game_id"]),
            "season": row["season"],
            "week": row["week"],
            "total_line": row.get("total_line"),
            "spread_line": row.get("spread_line"),
            "spread_favorite": row.get("spread_favorite"),
        }
        home_row = base | {
            "team": row["home_team"],
            "opponent": row["away_team"],
            "home_away": "home",
            "moneyline": row.get("home_moneyline"),
            "market_total": row.get("total_line"),
        }
        away_row = base | {
            "team": row["away_team"],
            "opponent": row["home_team"],
            "home_away": "away",
            "moneyline": row.get("away_moneyline"),
            "market_total": row.get("total_line"),
        }
        long_rows.extend([home_row, away_row])

    team_schedule = pd.DataFrame(long_rows)
    if "moneyline" in team_schedule.columns:
        team_schedule["market_implied_prob"] = convert_moneyline_to_probability(team_schedule["moneyline"])
    return team_schedule


def harmonize_weekly_with_schedule(weekly_df: pd.DataFrame, schedule_long: pd.DataFrame) -> pd.DataFrame:
    """Attach schedule context (home/away, market odds) to weekly team stats."""

    merge_cols = ["game_id", "team", "opponent"]
    enriched = weekly_df.merge(schedule_long, on=merge_cols, how="left")
    if enriched["home_away"].isna().any():
        LOGGER.warning("Some weekly rows are missing schedule context. Check for season misalignment.")
    return enriched
