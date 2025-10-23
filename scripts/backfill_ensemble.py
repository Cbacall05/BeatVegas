"""Train calibrated ensemble models across historical seasons."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from beat_vegas import pipeline


def _season_list(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values})


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Beat Vegas ensemble models with calibration")
    parser.add_argument(
        "--train-seasons",
        nargs="*",
        type=int,
        default=None,
        help="Historical seasons to use for training (defaults to last 4 completed seasons).",
    )
    parser.add_argument(
        "--validation-season",
        type=int,
        default=None,
        help="Season reserved for validation and calibration (defaults to current season).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/pbp"),
        help="Optional cache directory for play-by-play data (passed to pipeline.load_core_datasets).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/features/model_calibration_summary.json"),
        help="Where to persist calibration summaries (JSON).",
    )
    return parser.parse_args(argv)


def _default_train_seasons() -> list[int]:
    current = datetime.now().year
    start = max(1999, current - 4)
    seasons = list(range(start, current))
    return seasons[-4:] if len(seasons) >= 4 else seasons


def _build_metrics_snapshot(results: dict[str, list[pipeline.models.ModelResult]]) -> dict[str, dict[str, dict[str, float]]]:
    snapshot: dict[str, dict[str, dict[str, float]]] = {}
    for category, items in results.items():
        snapshot[category] = {}
        for result in items:
            entry: dict[str, float | int | dict[str, float]] = {
                "games": int(result.predictions.shape[0]),
            }
            entry.update({f"raw_{k}": float(v) for k, v in result.metrics.items()})
            if result.calibrated_metrics:
                entry.update({f"calibrated_{k}": float(v) for k, v in result.calibrated_metrics.items()})
            snapshot[category][result.model_name] = {str(key): value for key, value in entry.items()}
    return snapshot


def _serialize_reports(reports: dict[str, dict[str, np.ndarray | list | dict]]) -> dict[str, dict[str, list[dict[str, float]]]]:
    serialized: dict[str, dict[str, list[dict[str, float]]]] = {}
    for model_name, report in reports.items():
        serialized[model_name] = {}
        for key, frame in report.items():
            if isinstance(frame, list):
                serialized[model_name][key] = frame
            else:
                serialized[model_name][key] = [] if frame.empty else frame.to_dict(orient="records")
    return serialized


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    train_seasons = _season_list(args.train_seasons or _default_train_seasons())
    validation_season = args.validation_season or datetime.now().year
    all_seasons = sorted(set(train_seasons + [validation_season]))

    datasets = pipeline.load_core_datasets(all_seasons, cache_dir=args.cache_dir)
    model_frame = pipeline.build_model_ready_frame(datasets["weekly"], datasets["schedule"])
    feature_cols = pipeline.default_feature_columns(model_frame)

    results = pipeline.train_baseline_models(
        model_frame,
        feature_cols,
        validation_seasons=[validation_season],
    )

    bias_reports = pipeline.compute_bias_reports(results["moneyline"], datasets["schedule"], use_calibrated=True)

    summary = {
        "train_seasons": train_seasons,
        "validation_season": validation_season,
        "features": feature_cols,
        "metrics": _build_metrics_snapshot(results),
        "bias_reports": _serialize_reports(bias_reports),
    }

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Calibration summary written to {output_path}")


if __name__ == "__main__":
    main()
