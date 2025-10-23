"""Nightly data refresh utility for Project Beat Vegas.

Downloads nflfastR play-by-play datasets, NFLGameData schedules, and persists
rest/travel features so downstream training pipelines can operate offline.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

from beat_vegas.external_ingestion import refresh_external_data


def _default_seasons(lookback: int) -> list[int]:
    current = datetime.now().year
    start = max(1999, current - lookback)
    return list(range(start, current + 1))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh external data caches for Beat Vegas")
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=None,
        help="Specific seasons to refresh. Defaults to the last N seasons (see --lookback).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=3,
        help="Number of historical seasons to include when --seasons is not provided.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for cached artifacts (defaults to data/external).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached files are still fresh.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=20,
        help="Re-download files older than this many hours (ignored when --force is set).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    seasons = args.seasons or _default_seasons(args.lookback)
    touched = refresh_external_data(
        seasons,
        base_dir=args.base_dir,
        force=args.force,
        max_age_hours=args.max_age_hours,
    )
    for category, paths in touched.items():
        if not paths:
            continue
        joined = ", ".join(str(path) for path in paths)
        print(f"[{category}] {joined}")


if __name__ == "__main__":
    main()
