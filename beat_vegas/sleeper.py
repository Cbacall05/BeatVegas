"""Utility helpers for interacting with the public Sleeper NFL API.

These helpers provide lightweight caching on disk so that downstream calls do
not hammer the public endpoint and gracefully fall back to cached payloads when
network requests fail. The module intentionally keeps the surface area small:
fetch the raw player payload, convert it to a dataframe, and expose a
position-filtered helper that focuses on quarterbacks for availability logic.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
DEFAULT_CACHE_PATH = DEFAULT_CACHE_DIR / "sleeper_players.json"
DEFAULT_TTL_SECONDS = 1800  # 30 minutes keeps data reasonably fresh.


@dataclass(frozen=True)
class SleeperFetchResult:
    payload: Dict[str, Any]
    cache_path: Path
    fetched_from_cache: bool
    age_seconds: float


def _read_cached_payload(cache_path: Path) -> SleeperFetchResult:
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        age_seconds = time.time() - cache_path.stat().st_mtime
        return SleeperFetchResult(payload=payload, cache_path=cache_path, fetched_from_cache=True, age_seconds=age_seconds)
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - ensure cache corruption does not crash callers
        LOGGER.warning("Failed to read cached Sleeper payload: %s", exc)
        raise


def _write_cache(cache_path: Path, payload: Dict[str, Any]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception as exc:  # noqa: BLE001 - best-effort caching
        LOGGER.warning("Unable to persist Sleeper payload cache: %s", exc)


def fetch_players(
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    timeout: int = 15,
) -> SleeperFetchResult:
    """Fetch the Sleeper player payload with basic TTL-based caching.

    Parameters
    ----------
    cache_path:
        Location for persisting cached payloads. The parent directory will be
        created on first use.
    ttl_seconds:
        Maximum age (in seconds) at which the cached response is considered
        fresh. If the cache is younger than this TTL it will be returned without
        issuing a network request. Pass ``0`` to force a refresh.
    timeout:
        Timeout (in seconds) to apply to the outbound HTTP request.
    """

    if cache_path.exists() and ttl_seconds > 0:
        age_seconds = time.time() - cache_path.stat().st_mtime
        if age_seconds <= ttl_seconds:
            try:
                return _read_cached_payload(cache_path)
            except FileNotFoundError:
                pass  # Cache became unavailable between existence check and read.

    try:
        response = requests.get(SLEEPER_PLAYERS_URL, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        _write_cache(cache_path, payload)
        return SleeperFetchResult(payload=payload, cache_path=cache_path, fetched_from_cache=False, age_seconds=0.0)
    except requests.RequestException as exc:
        LOGGER.warning("Sleeper player fetch failed: %s", exc)
        if cache_path.exists():
            try:
                return _read_cached_payload(cache_path)
            except FileNotFoundError:  # pragma: no cover - race condition guard
                raise RuntimeError("Sleeper API unavailable and cache missing.") from exc
        raise RuntimeError("Sleeper API unavailable and no cache present.") from exc


def _extract_primary_name(entry: Dict[str, Any]) -> str:
    name_components: Iterable[str] = entry.get("full_name"), entry.get("display_name"), entry.get("first_name")
    for value in name_components:
        if value:
            return str(value)
    return ""


def players_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert the raw Sleeper payload into a normalized dataframe."""

    records: list[Dict[str, Any]] = []
    for player_id, entry in payload.items():
        if not isinstance(entry, dict):  # Defensive: guard against unexpected payload shapes.
            continue
        record: Dict[str, Any] = {
            "player_id": str(player_id),
            "full_name": _extract_primary_name(entry),
            "first_name": entry.get("first_name"),
            "last_name": entry.get("last_name"),
            "position": entry.get("position"),
            "team": entry.get("team"),
            "status": entry.get("status"),
            "injury_status": entry.get("injury_status"),
            "injury_start_date": entry.get("injury_start_date"),
            "practice_status": entry.get("practice_participation"),
            "practice_description": entry.get("practice_description"),
            "depth_chart_order": entry.get("depth_chart_order"),
            "depth_chart_position": entry.get("depth_chart_position"),
            "last_updated": entry.get("last_updated"),
        }
        if not record["full_name"]:
            continue  # Skip anonymous placeholder entries.
        records.append(record)

    if not records:
        return pd.DataFrame(columns=[
            "player_id",
            "full_name",
            "first_name",
            "last_name",
            "position",
            "team",
            "status",
            "injury_status",
            "injury_start_date",
            "practice_status",
            "practice_description",
            "depth_chart_order",
            "depth_chart_position",
            "last_updated",
        ])

    df = pd.DataFrame.from_records(records)
    df["team"] = df["team"].astype(str).str.upper().replace({"None": pd.NA, "": pd.NA})
    df["position"] = df["position"].astype(str).str.upper().replace({"None": pd.NA, "": pd.NA})
    df["player_id"] = df["player_id"].astype(str)
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce", utc=True)
    return df


def load_players_dataframe(*, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> pd.DataFrame:
    """Fetch the Sleeper payload (with caching) and return it as a dataframe."""

    fetch_result = fetch_players(ttl_seconds=ttl_seconds)
    df = players_dataframe(fetch_result.payload)
    df.attrs["fetched_from_cache"] = fetch_result.fetched_from_cache
    df.attrs["cache_age_seconds"] = fetch_result.age_seconds
    return df


def load_qb_status(*, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> pd.DataFrame:
    """Return a dataframe focused on quarterbacks and their availability signals."""

    df = load_players_dataframe(ttl_seconds=ttl_seconds)
    if df.empty:
        return df
    qb_df = df[df["position"] == "QB"].copy()
    qb_df.reset_index(drop=True, inplace=True)
    return qb_df
