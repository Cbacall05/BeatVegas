"""Interactive Streamlit dashboard for Beat Vegas matchup predictions."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from beat_vegas import data_load, models
from matchup_predictor import (
    build_game_level_dataset,
    build_team_game_records,
    predict_upcoming_games,
    select_feature_columns,
)

st.set_page_config(
    page_title="Beat Vegas Matchup Studio",
    page_icon="🏈",
    layout="wide",
    menu_items={
        "Get help": "https://github.com/",
        "Report a bug": "https://github.com/",
        "About": "Project Beat Vegas — predictive analytics for NFL matchups.",
    },
)

CUSTOM_CSS = """
<style>
main .block-container {
    padding: 2.5rem 3rem 4rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    color: #f3f6f9;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #f3f6f9 !important;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] input {
    color: #0d1117;
}

.metric-card {
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    background: linear-gradient(135deg, rgba(13, 17, 23, 0.9), rgba(32, 44, 68, 0.9));
    color: #f5f7fb;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 12px 20px rgba(15, 23, 42, 0.25);
}

.metric-card h3 {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: rgba(243, 246, 249, 0.76);
}

.metric-card .metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.metric-card .metric-subtext {
    font-size: 0.85rem;
    opacity: 0.65;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DEFAULT_TRAIN_SEASONS = list(range(2018, 2025))
CURRENT_SEASON = max(datetime.now().year, 2025)

TEAM_ABBREVIATIONS = [
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LAC",
    "LA",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
]


def _normalize_team_list(teams: Iterable[str]) -> list[str]:
    return sorted({team.upper() for team in teams if team})


def _season_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(set(values)))


def _format_prob(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "–"
    return f"{value:.1%}"


@st.cache_data(show_spinner=False)
def load_schedule_cached(seasons: tuple[int, ...]) -> pd.DataFrame:
    return data_load.load_schedule(list(seasons))


@st.cache_resource(show_spinner=False)
def train_models(
    train_seasons: tuple[int, ...],
    validation_season: int,
    rolling_window: int,
):
    all_seasons = sorted(set(train_seasons + (validation_season,)))
    schedule_df = load_schedule_cached(tuple(all_seasons))

    team_df = build_team_game_records(schedule_df, rolling_window=rolling_window)
    game_dataset = build_game_level_dataset(team_df)
    game_dataset = game_dataset[game_dataset["season"].isin(all_seasons)].copy()
    dataset_columns = list(game_dataset.columns)

    feature_cols = select_feature_columns(game_dataset)
    if not feature_cols:
        raise RuntimeError("No feature columns detected. Adjust season selection or rolling window.")

    split_config = models.SplitConfig(validation_seasons=[validation_season])
    moneyline_results = models.train_moneyline_models(game_dataset, feature_cols, config=split_config)
    totals_results = models.train_total_models(game_dataset, feature_cols, target_col="total_points", config=split_config)
    return schedule_df, team_df, dataset_columns, feature_cols, moneyline_results, totals_results


def build_upcoming_predictions(
    schedule_df: pd.DataFrame,
    team_df: pd.DataFrame,
    dataset_columns: list[str],
    feature_cols: list[str],
    moneyline_results: list[models.ModelResult],
    totals_results: list[models.ModelResult],
    *,
    validation_season: int,
    week: int | None,
) -> pd.DataFrame:
    predictions = predict_upcoming_games(
        team_df=team_df,
        schedule_df=schedule_df,
        feature_cols=feature_cols,
        dataset_columns=dataset_columns,
        moneyline_results=moneyline_results,
        totals_results=totals_results,
        target_season=validation_season,
        predict_week=week,
        home_team=None,
        away_team=None,
        include_all=True,
    )
    if predictions.empty:
        return predictions

    predictions = predictions.copy()
    if "game_date" in predictions.columns:
        predictions["game_date"] = pd.to_datetime(predictions["game_date"])
    return predictions


def render_header():
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:1rem;">
            <h1 style="margin:0; font-size:2.6rem;">Beat Vegas Matchup Studio</h1>
            <span style="padding:0.3rem 0.75rem; border-radius:999px; background:#111827; color:#f9fafb; font-size:0.85rem; letter-spacing:0.04em;">SLEEK · LIVE · PROBABILITIES</span>
        </div>
        <p style="color:#6b7280; font-size:1rem; margin-top:0.75rem; max-width:60ch;">
            Explore model-driven predictions for upcoming NFL matchups. Blend historical strength, rolling form, and market context to surface sharp edges in a single glance.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.title("Matchup Controls")
        st.caption(
            "Tune the training window, validation season, and upcoming week to refresh predictions."
        )

        default_train = DEFAULT_TRAIN_SEASONS
        seasons_selected = st.multiselect(
            "Training Seasons",
            options=list(range(2015, CURRENT_SEASON + 1)),
            default=default_train,
        )

        validation_season = st.selectbox(
            "Target Season",
            options=list(range(2018, CURRENT_SEASON + 1)),
            index=list(range(2018, CURRENT_SEASON + 1)).index(max(2025, DEFAULT_TRAIN_SEASONS[-1] + 1)),
        )

        rolling_window = st.slider("Rolling Window (games)", min_value=2, max_value=8, value=4, step=1)

        schedule_preview = load_schedule_cached(_season_tuple(seasons_selected + [validation_season]))
        weeks_available = sorted(
            schedule_preview.loc[schedule_preview["season"] == validation_season, "week"].unique()
        )
        current_week_suggestion = max(weeks_available[-1], 1) if weeks_available else 1

        week = st.number_input(
            "Upcoming Week",
            min_value=1,
            max_value=int(max(18, current_week_suggestion)),
            value=int(current_week_suggestion),
            step=1,
        )

        team_filter = st.multiselect(
            "Highlight Teams",
            options=TEAM_ABBREVIATIONS,
            placeholder="Select teams to spotlight",
        )

        return _season_tuple(seasons_selected), int(validation_season), int(rolling_window), int(week), _normalize_team_list(team_filter)


def render_metrics(predictions: pd.DataFrame, team_filter: list[str]):
    if predictions.empty:
        st.info("No upcoming games found for the selected filters. Try a different week or season.")
        return

    top_confidence = predictions.sort_values("predicted_win_prob", ascending=False).head(3)
    tightest_games = (
        predictions.assign(spread=lambda df: np.abs(df["avg_home_win_prob"] - 0.5))
        .sort_values("spread")
        .head(3)
    )

    cols = st.columns(3)
    for col, row in zip(cols, top_confidence.itertuples(index=False)):
        col.markdown(
            f"""
            <div class="metric-card">
                <h3>High-Confidence Pick</h3>
                <div class="metric-value">{row.predicted_winner}</div>
                <div class="metric-subtext">Win Probability: {_format_prob(row.predicted_win_prob)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### This Week's Slate")

    display_df = predictions.copy()
    if team_filter:
        mask = display_df["home_team"].isin(team_filter) | display_df["away_team"].isin(team_filter)
        display_df = display_df[mask]

    display_df["Kickoff"] = display_df.get("game_date").dt.strftime("%b %d") if "game_date" in display_df else ""  # type: ignore[arg-type]
    display_df["Home Win %"] = display_df["avg_home_win_prob"].map(_format_prob)
    display_df["Winner"] = display_df["predicted_winner"]
    display_df["Confidence"] = display_df["predicted_win_prob"].map(_format_prob)
    show_cols = [
        "Kickoff",
        "week",
        "home_team",
        "away_team",
        "Winner",
        "Confidence",
        "Home Win %",
        "market_total",
    ]
    show_cols = [col for col in show_cols if col in display_df.columns]

    st.dataframe(
        display_df[show_cols]
        .sort_values(["week", "Kickoff", "home_team"])
        .rename(
            columns={
                "week": "Week",
                "home_team": "Home",
                "away_team": "Away",
                "market_total": "Market Total",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Confidence Leaderboard")
    leaderboard = predictions.sort_values("predicted_win_prob", ascending=False).head(10)
    chart = px.bar(
        leaderboard,
        x="predicted_win_prob",
        y="predicted_winner",
        color="predicted_winner",
        orientation="h",
        labels={"predicted_win_prob": "Win Probability", "predicted_winner": "Team"},
        text=leaderboard["predicted_win_prob"].map(lambda v: f"{v:.1%}"),
    )
    chart.update_layout(
        height=480,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(chart, use_container_width=True)

    st.markdown("### Raw Prediction Details")
    probability_cols = [col for col in predictions.columns if col.endswith("_home_win")]
    detail_cols = [
        "game_id",
        "season",
        "week",
        "game_date",
        "home_team",
        "away_team",
        "avg_home_win_prob",
        "avg_away_win_prob",
        "predicted_winner",
        "predicted_win_prob",
        "market_total",
        "home_moneyline",
        "away_moneyline",
    ] + probability_cols
    detail_cols = [col for col in detail_cols if col in predictions.columns]

    formatted = predictions[detail_cols].copy()
    for col in ["avg_home_win_prob", "avg_away_win_prob", "predicted_win_prob"]:
        if col in formatted:
            formatted[col] = formatted[col].map(_format_prob)

    st.dataframe(formatted, use_container_width=True)


def main() -> None:
    render_header()
    train_seasons, validation_season, rolling_window, week, team_filter = render_sidebar()

    with st.spinner("Training models and generating predictions..."):
        (
            schedule_df,
            team_df,
            dataset_columns,
            feature_cols,
            moneyline_results,
            totals_results,
        ) = train_models(train_seasons, validation_season, rolling_window)

        predictions = build_upcoming_predictions(
            schedule_df,
            team_df,
            dataset_columns,
            feature_cols,
            moneyline_results,
            totals_results,
            validation_season=validation_season,
            week=week,
        )

    render_metrics(predictions, team_filter)

    st.caption(
        "Model suite includes logistic regression, random forest, and LightGBM (if installed). Average win probabilities reflect the ensemble mean across available classifiers."
    )


if __name__ == "__main__":
    main()
