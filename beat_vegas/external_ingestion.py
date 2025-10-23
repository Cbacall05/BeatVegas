"""External data ingestion helpers for nightly automation.

This module downloads public datasets from the nflfastR project and
NFLGameData.com, then persists derived rest/travel features using the shared
schedule enrichment utilities. The entry point `refresh_external_data` is used
by maintenance scripts to hydrate local caches ahead of training runs or
Streamlit sessions.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from . import data_load, schedule_enrichment

LOGGER = logging.getLogger(__name__)
NFLFASTR_PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
NFLFASTR_SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedule/sched_{season}.csv"
NFLGAMEDATA_API_URL = "https://nflgamedata.com/api/schedule"
NFLGAMEDATA_HTML_URL = "https://nflgamedata.com/schedule.php"
DEFAULT_MAX_AGE_HOURS = 20


def _is_stale(path: Path, *, max_age_hours: int) -> bool:
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime > timedelta(hours=max_age_hours)


def _stream_download(url: str, dest: Path, *, chunk_size: int = 2 ** 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)


def download_nflfastr_pbp(season: int, dest: Path, *, force: bool = False, max_age_hours: int = DEFAULT_MAX_AGE_HOURS) -> Path:
    dest = Path(dest)
    if not force and not _is_stale(dest, max_age_hours=max_age_hours):
        LOGGER.debug("Skipping nflfastR pbp download for %s (cache fresh)", season)
        return dest
    url = NFLFASTR_PBP_URL.format(season=season)
    _stream_download(url, dest)
    return dest


def download_nflfastr_schedule(season: int, dest: Path, *, force: bool = False, max_age_hours: int = DEFAULT_MAX_AGE_HOURS) -> Path:
    dest = Path(dest)
    if not force and not _is_stale(dest, max_age_hours=max_age_hours):
        LOGGER.debug("Skipping nflfastR schedule download for %s (cache fresh)", season)
        return dest
    url = NFLFASTR_SCHEDULE_URL.format(season=season)
    _stream_download(url, dest)
    return dest


def _fetch_nflgamedata_json(season: int) -> pd.DataFrame | None:
    try:
        resp = requests.get(NFLGAMEDATA_API_URL, params={"season": season}, timeout=60)
        resp.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network failure
        LOGGER.debug("NFLGameData JSON API unavailable for %s: %s", season, exc)
        return None
    except requests.RequestException:  # pragma: no cover - network failure
        LOGGER.debug("NFLGameData JSON API request failed for season %s", season)
        return None

    try:
        payload = resp.json()
    except json.JSONDecodeError:  # pragma: no cover - rare
        LOGGER.debug("NFLGameData JSON decode failed for season %s", season)
        return None

    if not payload:
        return None

    if isinstance(payload, dict) and "data" in payload:
        records = payload.get("data")
    else:
        records = payload

    if not isinstance(records, list) or not records:
        return None

    frame = pd.DataFrame(records)
    # Standardise column casing when available.
    rename_map = {
        "week": "week",
        "gamedate": "game_date",
        "gameday": "game_date",
        "home": "home_team",
        "home_abbr": "home_team",
        "away": "away_team",
        "away_abbr": "away_team",
        "spread": "closing_spread",
        "total": "closing_total",
        "moneyline_home": "home_moneyline",
        "moneyline_away": "away_moneyline",
    }
    for column, alias in rename_map.items():
        if column in frame.columns:
            frame.rename(columns={column: alias}, inplace=True)

    if "game_date" in frame.columns:
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["season"] = season
    return frame


def _fetch_nflgamedata_html(season: int) -> pd.DataFrame:
    params = {"season": season, "week": "all"}
    resp = requests.get(NFLGAMEDATA_HTML_URL, params=params, timeout=60)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    if not tables:
        raise RuntimeError(f"NFLGameData HTML schedule missing for season {season}")

    frame = tables[0].copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            "_".join(str(level).strip() for level in col if str(level).strip())
            for col in frame.columns
        ]

    frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]
    frame = frame.dropna(how="all")
    keep_mask = ~(frame.iloc[:, 0].astype(str).str.contains("bye", case=False, na=False))
    frame = frame[keep_mask]

    column_map = {
        "week": "week",
        "date": "game_date",
        "time": "kickoff_time_et",
        "away_team": "away_team",
        "away": "away_team",
        "home_team": "home_team",
        "home": "home_team",
        "spread": "closing_spread",
        "total": "closing_total",
        "moneyline": "home_moneyline",
    }
    for column, alias in column_map.items():
        if column in frame.columns and alias not in frame.columns:
            frame.rename(columns={column: alias}, inplace=True)

    if "game_date" in frame.columns:
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["season"] = season
    return frame


def fetch_nflgamedata_schedule(season: int) -> pd.DataFrame:
    json_frame = _fetch_nflgamedata_json(season)
    if json_frame is not None and not json_frame.empty:
        LOGGER.debug("Fetched NFLGameData JSON schedule for %s", season)
        return json_frame
    LOGGER.debug("Falling back to NFLGameData HTML scraper for season %s", season)
    return _fetch_nflgamedata_html(season)


def refresh_external_data(
    seasons: Iterable[int],
    *,
    base_dir: Path | None = None,
    force: bool = False,
    max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
) -> dict[str, list[Path]]:
    """Download external datasets and persist rest/travel features.

    Parameters
    ----------
    seasons
        Iterable of seasons that should be available in the local cache.
    base_dir
        Optional base directory. Defaults to ``data/external``.
    force
        When True, always re-download files even if cache copies are fresh.
    max_age_hours
        Re-download files older than this many hours.

    Returns
    -------
    dict[str, list[Path]]
        Mapping of dataset category to the file paths that were touched.
    """

    base = Path(base_dir or Path("data") / "external")
    pbp_dir = base / "nflfastr"
    schedule_dir = base / "nflgamedata"
    features_dir = base / "features"

    touched: dict[str, list[Path]] = {"pbp": [], "nflgamedata": [], "rest_travel": []}

    seasons_list = sorted({int(season) for season in seasons})
    for season in seasons_list:
        pbp_path = pbp_dir / f"play_by_play_{season}.parquet"
        download_nflfastr_pbp(season, pbp_path, force=force, max_age_hours=max_age_hours)
        touched["pbp"].append(pbp_path)

        sched_path = schedule_dir / f"nflfastr_schedule_{season}.csv"
        download_nflfastr_schedule(season, sched_path, force=force, max_age_hours=max_age_hours)
        touched.setdefault("nflfastr_schedule", []).append(sched_path)

        ngd_path = schedule_dir / f"nflgamedata_schedule_{season}.csv"
        ngd_df = fetch_nflgamedata_schedule(season)
        ngd_path.parent.mkdir(parents=True, exist_ok=True)
        ngd_df.to_csv(ngd_path, index=False)
        touched["nflgamedata"].append(ngd_path)

    schedule_all = data_load.load_schedule(seasons_list)
    rest_result = schedule_enrichment.compute_rest_travel(schedule_all)

    matchup_path = features_dir / "rest_travel.parquet"
    rest_result.game_level.to_parquet(matchup_path, index=False)
    touched["rest_travel"].append(matchup_path)

    team_path = features_dir / "team_travel.parquet"
    rest_result.team_level.to_parquet(team_path, index=False)
    touched["rest_travel"].append(team_path)

    LOGGER.info("External data refresh complete for seasons %s", seasons_list)
    return touched
