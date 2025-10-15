from __future__ import annotations

from apps.matchup_dashboard import (
    DEFAULT_TRAIN_SEASONS,
    CURRENT_SEASON,
    build_upcoming_predictions,
    prepare_injury_context,
    train_models,
)


def main() -> None:
    train = tuple(DEFAULT_TRAIN_SEASONS)
    validation_season = CURRENT_SEASON
    week = 6
    schedule_df, team_df, dataset_cols, feature_cols, moneyline_results, totals_results = train_models(
        train,
        validation_season,
        4,
    )
    adjustments, penalties, alerts = prepare_injury_context(train, validation_season, week)
    preds_without = build_upcoming_predictions(
        schedule_df,
        team_df,
        dataset_cols,
        feature_cols,
        moneyline_results,
        totals_results,
        validation_season=validation_season,
        week=week,
        team_penalties=None,
    )
    preds_with = build_upcoming_predictions(
        schedule_df,
        team_df,
        dataset_cols,
        feature_cols,
        moneyline_results,
        totals_results,
        validation_season=validation_season,
        week=week,
        team_penalties=penalties,
    )

    merged = preds_without[[
        "game_id",
        "home_team",
        "away_team",
        "avg_home_win_prob",
    ]].merge(
        preds_with[["game_id", "avg_home_win_prob", "injury_penalty_home", "injury_penalty_away"]],
        on="game_id",
        suffixes=("_without", "_with"),
    )
    row = merged[(merged["home_team"] == "BAL") | (merged["away_team"] == "BAL")]

    print("Alerts:")
    for message in alerts:
        print(" -", message)
    print()
    print(row.to_string(index=False))


if __name__ == "__main__":
    main()
