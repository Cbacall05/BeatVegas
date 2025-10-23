"""Head-to-head matchup predictor for Project Beat Vegas.

This script trains baseline models on historical NFL schedule data and outputs
win probability and total-points projections for a user-selected matchup.
It avoids the player-level weekly dataset and instead derives rolling team
features directly from the schedules.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from beat_vegas import data_load, models

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project Beat Vegas head-to-head predictor")
    parser.add_argument("home_team", nargs="?", default=None, help="Home team abbreviation (e.g. CHI)")
    parser.add_argument("away_team", nargs="?", default=None, help="Away team abbreviation (e.g. DET)")
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=[2018, 2019, 2020, 2021, 2022],
        help="Seasons to use for training the models.",
    )
    parser.add_argument(
        "--validation-season",
        type=int,
    default=2023,
    help="Season reserved for validation/prediction benchmarking (also target season for upcoming predictions).",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=4,
        help="Rolling window (in games) for computing team features.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/pbp"),
        help="Directory for cached play-by-play files (ensures consistent environment).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save validation game predictions for the matchup.",
    )
    parser.add_argument(
        "--predict-week",
        type=int,
        default=None,
        help="If set, generate predictions for this week in the target season (future games only).",
    )
    parser.add_argument(
        "--all-upcoming",
        action="store_true",
        help="When predicting upcoming games, return every matchup for the chosen week instead of a single home/away pair.",
    )
    return parser.parse_args(argv)


def moneyline_to_prob(value: float | int | None) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    series = pd.Series([value], dtype="float")
    return float(data_load.convert_moneyline_to_probability(series).iloc[0])


def build_team_game_records(schedule_df: pd.DataFrame, rolling_window: int = 4) -> pd.DataFrame:
    schedule_df = schedule_df.dropna(subset=["home_score", "away_score"]).copy()
    schedule_df.sort_values(["season", "week", "game_id"], inplace=True)

    records: list[dict[str, float | int | str | None]] = []
    for row in schedule_df.itertuples(index=False):
        home_points = row.home_score
        away_points = row.away_score
        total_line = row.total_line if hasattr(row, "total_line") else np.nan
        spread_line = row.spread_line if hasattr(row, "spread_line") else np.nan
        home_prob = moneyline_to_prob(getattr(row, "home_moneyline", np.nan))
        away_prob = moneyline_to_prob(getattr(row, "away_moneyline", np.nan))

        home_rest = getattr(row, "home_rest_days", getattr(row, "home_rest", np.nan))
        away_rest = getattr(row, "away_rest_days", getattr(row, "away_rest", np.nan))
        home_travel = getattr(row, "home_travel_miles", np.nan)
        away_travel = getattr(row, "away_travel_miles", np.nan)

        home_record = {
            "game_id": row.game_id,
            "season": row.season,
            "week": row.week,
            "team": row.home_team,
            "opponent": row.away_team,
            "home_away": "home",
            "gameday": row.gameday,
            "points": home_points,
            "points_allowed": away_points,
            "point_diff": home_points - away_points,
            "market_total": total_line,
            "market_spread": spread_line,
            "market_prob": home_prob,
            "rest": home_rest,
            "travel_miles": home_travel,
            "win": int(home_points > away_points),
        }
        away_record = {
            "game_id": row.game_id,
            "season": row.season,
            "week": row.week,
            "team": row.away_team,
            "opponent": row.home_team,
            "home_away": "away",
            "gameday": row.gameday,
            "points": away_points,
            "points_allowed": home_points,
            "point_diff": away_points - home_points,
            "market_total": total_line,
            "market_spread": -spread_line if pd.notna(spread_line) else np.nan,
            "market_prob": away_prob,
            "rest": away_rest,
            "travel_miles": away_travel,
            "win": int(away_points > home_points),
        }
        records.extend([home_record, away_record])

    team_df = pd.DataFrame(records)
    team_df.sort_values(["team", "season", "week"], inplace=True)
    team_group = team_df.groupby("team", group_keys=False)

    rolling_metrics = [
        "points",
        "points_allowed",
        "point_diff",
        "market_total",
        "market_spread",
        "market_prob",
        "rest",
        "travel_miles",
    ]
    for metric in rolling_metrics:
        team_df[f"{metric}_avg"] = team_group[metric].transform(
            lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean()
        )

    team_df["win_rate_avg"] = team_group["win"].transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean()
    )
    team_df["games_played"] = team_group.cumcount()

    return team_df


DIFF_BASES = [
    "points_avg",
    "points_allowed_avg",
    "point_diff_avg",
    "market_total_avg",
    "market_spread_avg",
    "market_prob_avg",
    "win_rate_avg",
    "rest_avg",
    "travel_miles_avg",
]


def build_game_level_dataset(team_df: pd.DataFrame) -> pd.DataFrame:
    home_df = team_df[team_df["home_away"] == "home"].copy()
    away_df = team_df[team_df["home_away"] == "away"].copy()

    home_df = home_df.add_prefix("home_")
    away_df = away_df.add_prefix("away_")
    home_df.rename(columns={"home_game_id": "game_id"}, inplace=True)
    away_df.rename(columns={"away_game_id": "game_id"}, inplace=True)

    dataset = home_df.merge(away_df, on="game_id", how="inner", suffixes=("", ""))
    dataset["season"] = dataset["home_season"]
    dataset["week"] = dataset["home_week"]
    dataset["game_date"] = dataset.get("home_gameday")
    dataset["home_team"] = dataset["home_team"]
    dataset["away_team"] = dataset["away_team"]
    dataset["home_win"] = (dataset["home_points"] > dataset["away_points"]).astype(int)
    dataset["total_points"] = dataset["home_points"] + dataset["away_points"]

    drop_cols = ["home_home_away", "away_home_away"]
    for col in drop_cols:
        if col in dataset.columns:
            dataset.drop(columns=col, inplace=True)

    for base in DIFF_BASES:
        home_col = f"home_{base}"
        away_col = f"away_{base}"
        if home_col in dataset.columns and away_col in dataset.columns:
            dataset[f"diff_{base}"] = dataset[home_col] - dataset[away_col]

    return dataset


def select_feature_columns(dataset: pd.DataFrame) -> list[str]:
    feature_cols: list[str] = []
    for col in dataset.columns:
        if col.startswith(("home_", "away_")) and (
            col.endswith("_avg") or col.endswith("_games_played")
        ):
            feature_cols.append(col)
        if col.startswith("diff_"):
            feature_cols.append(col)
    return sorted(set(feature_cols))


def _apply_calibration(result: models.ModelResult, raw_prob: float) -> float:
    calibrator = result.calibrator
    if calibrator is None:
        return float(np.clip(raw_prob, 1e-6, 1 - 1e-6))

    try:
        if result.calibration_method == "isotonic" and hasattr(calibrator, "predict"):
            arr = np.asarray([raw_prob])
            return float(np.clip(calibrator.predict(arr)[0], 1e-6, 1 - 1e-6))
        if hasattr(calibrator, "predict_proba"):
            arr = np.asarray([[raw_prob]])
            return float(np.clip(calibrator.predict_proba(arr)[0][1], 1e-6, 1 - 1e-6))
        if hasattr(calibrator, "predict"):
            arr = np.asarray([[raw_prob]]) if np.ndim(raw_prob) == 0 else raw_prob
            pred = calibrator.predict(arr)
            if np.ndim(pred) == 1:
                value = pred[0]
            else:
                value = pred
            return float(np.clip(value, 1e-6, 1 - 1e-6))
    except Exception as exc:  # noqa: BLE001 - calibration is best effort
        LOGGER.warning("Calibration fallback triggered for %s: %s", result.model_name, exc)
    return float(np.clip(raw_prob, 1e-6, 1 - 1e-6))


def predict_upcoming_games(
    *,
    team_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    feature_cols: list[str],
    dataset_columns: list[str],
    moneyline_results: list[models.ModelResult],
    totals_results: list[models.ModelResult],
    target_season: int,
    predict_week: int | None,
    home_team: str | None,
    away_team: str | None,
    include_all: bool,
) -> pd.DataFrame:
    season_schedule = schedule_df[schedule_df["season"] == target_season].copy()
    if season_schedule.empty:
        LOGGER.warning("No schedule data found for season %s.", target_season)
        return pd.DataFrame()

    completed_mask = season_schedule["home_score"].notna() & season_schedule["away_score"].notna()
    last_completed_week = (season_schedule.loc[completed_mask, "week"].max() if completed_mask.any() else 0)
    target_week = predict_week if predict_week is not None else int(last_completed_week) + 1

    upcoming = season_schedule[season_schedule["week"] == target_week].copy()
    upcoming = upcoming[upcoming["home_score"].isna() & upcoming["away_score"].isna()]

    if not include_all and home_team and away_team:
        upcoming = upcoming[(upcoming["home_team"] == home_team) & (upcoming["away_team"] == away_team)]

    if upcoming.empty:
        LOGGER.warning(
            "No upcoming games matching criteria for season %s week %s. Completed weeks end at %s.",
            target_season,
            target_week,
            last_completed_week,
        )
        return pd.DataFrame()

    predictions: list[dict[str, float | int | str | None]] = []
    for row in upcoming.itertuples(index=False):
        custom_row = build_custom_matchup_row(
            team_df,
            row.home_team,
            row.away_team,
            dataset_columns,
            override_week=row.week,
            override_game_id=row.game_id,
            game_date=getattr(row, "gameday", None),
        )
        feature_frame = custom_row[feature_cols]
        base: dict[str, float | int | str | None] = {
            "game_id": row.game_id,
            "season": row.season,
            "week": row.week,
            "game_date": getattr(row, "gameday", None),
            "home_team": row.home_team,
            "away_team": row.away_team,
            "market_total": getattr(row, "total_line", None),
            "home_moneyline": getattr(row, "home_moneyline", None),
            "away_moneyline": getattr(row, "away_moneyline", None),
            "home_rest_days": getattr(row, "home_rest_days", np.nan),
            "away_rest_days": getattr(row, "away_rest_days", np.nan),
            "home_travel_miles": getattr(row, "home_travel_miles", np.nan),
            "away_travel_miles": getattr(row, "away_travel_miles", np.nan),
        }

        home_probs_raw: list[float] = []
        home_probs_calibrated: list[float] = []

        for result in moneyline_results:
            model = result.model
            try:
                raw_prob = float(model.predict_proba(feature_frame)[0][1])
            except AttributeError:
                raw_prob = float(model.predict(feature_frame))
            calibrated_prob = _apply_calibration(result, raw_prob)
            base[f"{result.model_name}_home_win_raw"] = raw_prob
            base[f"{result.model_name}_home_win"] = calibrated_prob
            base[f"{result.model_name}_away_win"] = 1 - calibrated_prob
            home_probs_raw.append(raw_prob)
            home_probs_calibrated.append(calibrated_prob)

        total_preds: list[float] = []
        total_stds: list[float] = []
        for result in totals_results:
            pred_total = float(result.model.predict(feature_frame)[0])
            base[f"{result.model_name}_total"] = pred_total
            total_preds.append(pred_total)
            residual_std = result.metrics.get("residual_std") if result.metrics else None
            if residual_std is not None:
                residual_std_f = float(residual_std)
                base[f"{result.model_name}_total_std"] = residual_std_f
                total_stds.append(residual_std_f)

        ensemble_std: float | None = None
        if total_preds:
            avg_total = float(np.mean(total_preds))
            base["avg_total_pred"] = avg_total
            if total_stds:
                ensemble_std = float(np.sqrt(np.mean(np.square(total_stds))))
            elif len(total_preds) > 1:
                ensemble_std = float(np.std(total_preds, ddof=0))

        if ensemble_std is not None:
            base["avg_total_std"] = ensemble_std
            base["ensemble_total_low"] = float(avg_total - ensemble_std)
            base["ensemble_total_high"] = float(avg_total + ensemble_std)
            quartile_offset = 0.67448975 * ensemble_std
            base["ensemble_total_p25"] = float(avg_total - quartile_offset)
            base["ensemble_total_p75"] = float(avg_total + quartile_offset)

        if home_probs_raw:
            avg_home_raw = float(np.mean(home_probs_raw))
            base["avg_home_win_prob_raw"] = avg_home_raw
            base["avg_away_win_prob_raw"] = float(1 - avg_home_raw)
        if home_probs_calibrated:
            avg_home = float(np.mean(home_probs_calibrated))
            avg_away = float(1 - avg_home)
            base["avg_home_win_prob"] = avg_home
            base["avg_away_win_prob"] = avg_away
            if avg_home >= 0.5:
                base["predicted_winner"] = row.home_team
                base["predicted_win_prob"] = avg_home
            else:
                base["predicted_winner"] = row.away_team
                base["predicted_win_prob"] = avg_away

        predictions.append(base)

    predictions_df = pd.DataFrame(predictions)
    LOGGER.info(
        "Upcoming predictions for season %s week %s:%s%s",
        target_season,
        target_week,
        "\n" if not predictions_df.empty else " ",
        predictions_df
        if not predictions_df.empty
        else "",
    )
    return predictions_df


def build_custom_matchup_row(
    team_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    dataset_columns: list[str],
    *,
    override_week: int | None = None,
    override_game_id: str | None = None,
    game_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    home_latest = team_df[team_df["team"] == home_team].sort_values(["season", "week"]).tail(1)
    away_latest = team_df[team_df["team"] == away_team].sort_values(["season", "week"]).tail(1)

    if home_latest.empty or away_latest.empty:
        raise ValueError("Insufficient history for one or both teams. Try expanding the season range.")
    home_pref = home_latest.add_prefix("home_").reset_index(drop=True)
    away_pref = away_latest.add_prefix("away_").reset_index(drop=True)
    combined = pd.concat([home_pref, away_pref], axis=1)
    combined["game_id"] = override_game_id or f"custom_{home_team}_vs_{away_team}"
    combined["season"] = max(home_latest["season"].iloc[0], away_latest["season"].iloc[0])
    combined["week"] = override_week if override_week is not None else home_latest["week"].iloc[0] + 1
    combined["home_team"] = home_team
    combined["away_team"] = away_team
    if game_date is not None:
        combined["game_date"] = game_date

    for base in DIFF_BASES:
        home_col = f"home_{base}"
        away_col = f"away_{base}"
        if home_col in combined.columns and away_col in combined.columns:
            combined[f"diff_{base}"] = combined[home_col] - combined[away_col]

    for col in ["home_home_away", "away_home_away"]:
        if col in combined.columns:
            combined.drop(columns=col, inplace=True)

    for col in dataset_columns:
        if col not in combined.columns:
            combined[col] = pd.NA

    return combined[dataset_columns]


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args(argv)

    home_team = args.home_team.upper() if args.home_team else None
    away_team = args.away_team.upper() if args.away_team else None

    all_seasons = sorted(set(args.seasons + [args.validation_season]))
    LOGGER.info("Downloading schedules for seasons: %s", all_seasons)
    schedule_df = data_load.load_schedule(all_seasons)

    LOGGER.info("Building team-level rolling features (window=%d games)", args.rolling_window)
    team_df = build_team_game_records(schedule_df, rolling_window=args.rolling_window)

    game_dataset = build_game_level_dataset(team_df)
    game_dataset = game_dataset[game_dataset["season"].isin(all_seasons)].copy()
    dataset_columns = list(game_dataset.columns)
    feature_cols = select_feature_columns(game_dataset)
    if not feature_cols:
        raise RuntimeError("No feature columns detected. Check feature engineering pipeline.")

    LOGGER.info("Using %d features for modeling", len(feature_cols))
    target_season = args.validation_season
    split_config = models.SplitConfig(validation_seasons=[target_season])

    moneyline_results = models.train_moneyline_models(game_dataset, feature_cols, config=split_config)
    totals_results = models.train_total_models(game_dataset, feature_cols, target_col="total_points", config=split_config)

    val_mask = game_dataset["season"] == target_season
    validation_games = game_dataset[val_mask][["game_id", "season", "week", "home_team", "away_team", "home_win", "total_points"]]

    LOGGER.info("Validation season games available: %d", len(validation_games))

    def attach_game_context(result_df: pd.DataFrame) -> pd.DataFrame:
        return result_df.merge(validation_games, on="game_id", how="left")

    if home_team and away_team:
        matchup_filter = (validation_games["home_team"] == home_team) & (
            validation_games["away_team"] == away_team
        )
        matchup_games = validation_games[matchup_filter]

        if matchup_games.empty:
            LOGGER.warning(
                "No completed games for %s vs %s in validation season %s. Predictions will rely on historical averages only.",
                home_team,
                away_team,
                target_season,
            )
        else:
            LOGGER.info("Found %d validation games for %s vs %s.", len(matchup_games), home_team, away_team)

        if args.output_csv and not matchup_games.empty:
            pd.DataFrame(matchup_games).to_csv(args.output_csv, index=False)
            LOGGER.info("Saved validation matchup games to %s", args.output_csv)

        LOGGER.info("Model predictions for validation games:")
        for result in moneyline_results:
            merged = attach_game_context(result.predictions)
            mask = (merged["home_team"] == home_team) & (merged["away_team"] == away_team)
            subset = merged[mask]
            if subset.empty:
                continue
            LOGGER.info(
                "%s ->\n%s",
                result.model_name,
                subset[["season", "week", "home_team", "away_team", "pred_win_proba", "home_win"]],
            )

        for result in totals_results:
            merged = attach_game_context(result.predictions)
            mask = (merged["home_team"] == home_team) & (merged["away_team"] == away_team)
            subset = merged[mask]
            if subset.empty:
                continue
            LOGGER.info(
                "%s ->\n%s",
                result.model_name,
                subset[["season", "week", "home_team", "away_team", "pred_total_points", "total_points"]],
            )

        LOGGER.info("Generating hypothetical matchup projection based on latest rolling stats.")
        custom_row = build_custom_matchup_row(
            team_df,
            home_team,
            away_team,
            dataset_columns,
        )
        feature_frame = custom_row[feature_cols]

        for result in moneyline_results:
            model = result.model
            try:
                proba = float(model.predict_proba(feature_frame)[0][1])
            except AttributeError:
                proba = float(model.predict(feature_frame))
            LOGGER.info("%s hypothetical win probability for %s: %.3f", result.model_name, home_team, proba)
            LOGGER.info("%s hypothetical win probability for %s: %.3f", result.model_name, away_team, 1 - proba)

        for result in totals_results:
            model = result.model
            pred_total = float(model.predict(feature_frame)[0])
            LOGGER.info("%s hypothetical predicted total points: %.2f", result.model_name, pred_total)

    upcoming_df = predict_upcoming_games(
        team_df=team_df,
        schedule_df=schedule_df,
        feature_cols=feature_cols,
        dataset_columns=dataset_columns,
        moneyline_results=moneyline_results,
        totals_results=totals_results,
        target_season=target_season,
        predict_week=args.predict_week,
        home_team=home_team,
        away_team=away_team,
        include_all=args.all_upcoming or not (home_team and away_team),
    )

    if upcoming_df is not None and not upcoming_df.empty:
        summary_cols = [
            "season",
            "week",
            "game_date",
            "home_team",
            "away_team",
            "avg_home_win_prob",
            "predicted_winner",
            "predicted_win_prob",
        ]
        available_cols = [col for col in summary_cols if col in upcoming_df.columns]
        if available_cols:
            LOGGER.info(
                "Predicted winners for upcoming games:\n%s",
                upcoming_df[available_cols]
                .sort_values(["week", "game_date", "home_team"])
                .to_string(index=False),
            )

    if args.output_csv and upcoming_df is not None and not upcoming_df.empty:
        out_path = args.output_csv if args.output_csv.suffix else args.output_csv.with_suffix(".csv")
        upcoming_df.to_csv(out_path, index=False)
        LOGGER.info("Saved upcoming matchup predictions to %s", out_path)


if __name__ == "__main__":
    main()
