"""Interactive Streamlit dashboard for Beat Vegas matchup predictions."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from beat_vegas import data_load, models, player_models
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


@st.cache_data(show_spinner=False)
def load_pbp_cached(seasons: tuple[int, ...]) -> pd.DataFrame:
    cache_dir = ROOT_DIR / "data" / "pbp"
    return data_load.load_play_by_play(list(seasons), cache_dir=cache_dir)


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


def build_player_prop_predictions(
    schedule_df: pd.DataFrame,
    train_seasons: tuple[int, ...],
    validation_season: int,
    week: int,
    lookback: int = 4,
) -> tuple[pd.DataFrame, dict[str, float] | None, str | None]:
    seasons_for_pbp = tuple(sorted(set(train_seasons + (validation_season,))))
    try:
        pbp_df = load_pbp_cached(seasons_for_pbp)
        model_result, _, upcoming = player_models.train_and_predict_touchdowns(
            schedule_df,
            pbp_df,
            seasons=seasons_for_pbp,
            target_season=validation_season,
            target_week=week,
            lookback=lookback,
            min_touches=0.75,
        )
        return upcoming, model_result.metrics, None
    except Exception as exc:  # noqa: BLE001 - surface to UI
        return pd.DataFrame(), None, str(exc)


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


def render_totals_section(predictions: pd.DataFrame):
    if predictions.empty:
        st.info("No totals predictions available for the current filters.")
        return

    model_total_cols = [col for col in predictions.columns if col.endswith("_total") and col != "market_total"]
    if not model_total_cols:
        st.info("Totals models are not available for this configuration.")
        return

    st.markdown("### Total Points Outlook")

    totals_cols = [
        "game_id",
        "week",
        "home_team",
        "away_team",
        "avg_total_pred",
        "avg_total_std",
        "ensemble_total_low",
        "ensemble_total_high",
        "ensemble_total_p25",
        "ensemble_total_p75",
        "market_total",
    ]
    totals_cols = [col for col in totals_cols if col in predictions.columns]

    totals_view = predictions[totals_cols].copy()
    if {"avg_total_pred", "market_total"}.issubset(totals_view.columns):
        totals_view["total_edge"] = totals_view["avg_total_pred"] - totals_view["market_total"]

    numeric_cols = [
        "avg_total_pred",
        "avg_total_std",
        "ensemble_total_low",
        "ensemble_total_high",
        "ensemble_total_p25",
        "ensemble_total_p75",
        "market_total",
        "total_edge",
    ]
    for col in numeric_cols:
        if col in totals_view.columns:
            totals_view[col] = totals_view[col].astype(float).round(2)

    st.dataframe(
        totals_view.rename(
            columns={
                "week": "Week",
                "home_team": "Home",
                "away_team": "Away",
                "avg_total_pred": "Ensemble Total",
                "avg_total_std": "Std Dev",
                "ensemble_total_low": "68% Low",
                "ensemble_total_high": "68% High",
                "ensemble_total_p25": "25th %",
                "ensemble_total_p75": "75th %",
                "market_total": "Market Total",
                "total_edge": "Edge vs Market",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    annotated = predictions.reset_index().copy()
    if annotated.empty:
        return

    def _matchup_label(row: pd.Series) -> str:
        week = int(row.get("week", 0))
        game_id = row.get("game_id", "")
        return f"Week {week}: {row['away_team']} @ {row['home_team']} ({game_id})"

    option_pairs = [(row["index"], _matchup_label(row)) for _, row in annotated.iterrows()]
    labels = [label for _, label in option_pairs]
    selection = st.selectbox("Drill into a matchup", options=labels)
    selected_index = next(idx for idx, label in option_pairs if label == selection)
    selected_row = predictions.loc[selected_index]

    mean_total = float(selected_row.get("avg_total_pred", np.nan))
    std_total = selected_row.get("avg_total_std", np.nan)
    if pd.isna(mean_total) and model_total_cols:
        model_values = selected_row[model_total_cols].dropna().astype(float)
        if not model_values.empty:
            mean_total = float(model_values.mean())
    if pd.isna(std_total) or std_total <= 0:
        model_values = selected_row[model_total_cols].dropna().astype(float)
        if len(model_values) > 1:
            std_total = float(model_values.std(ddof=0))

    market_total = selected_row.get("market_total", np.nan)

    if pd.isna(mean_total) or pd.isna(std_total) or std_total <= 0:
        st.warning("Insufficient variance data to plot a total-points distribution for this matchup.")
    else:
        std_total = float(std_total)
        spread = 3 * std_total
        xs = np.linspace(mean_total - spread, mean_total + spread, 200)
        pdf = (1 / (std_total * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mean_total) / std_total) ** 2)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=pdf,
                mode="lines",
                line=dict(color="#2563eb", width=3),
                fill="tozeroy",
                name="Ensemble PDF",
            )
        )
        fig.add_vline(x=mean_total, line_width=2, line_dash="dash", line_color="#2563eb")
        if pd.notna(market_total):
            fig.add_vline(x=float(market_total), line_width=2, line_dash="dot", line_color="#f59e0b")
        fig.update_layout(
            title="Projected Total Distribution",
            xaxis_title="Total Points",
            yaxis_title="Density",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        col_mean, col_std, col_edge = st.columns(3)
        col_mean.metric("Ensemble Total", f"{mean_total:.1f}")
        col_std.metric("Std Dev", f"{std_total:.2f}")
        if pd.notna(market_total):
            delta = mean_total - float(market_total)
            col_edge.metric("Edge vs Market", f"{delta:+.1f}")

    per_model_rows: list[dict[str, float | str | None]] = []
    for col in model_total_cols:
        model_name = col.removesuffix("_total")
        prediction_val = selected_row.get(col)
        residual_std_val = selected_row.get(f"{model_name}_total_std")
        per_model_rows.append(
            {
                "Model": model_name,
                "Prediction": float(prediction_val) if pd.notna(prediction_val) else None,
                "Residual Std": float(residual_std_val) if pd.notna(residual_std_val) else None,
                "Edge vs Market": float(prediction_val - market_total)
                if pd.notna(prediction_val) and pd.notna(market_total)
                else None,
            }
        )

    per_model_df = pd.DataFrame(per_model_rows)
    if not per_model_df.empty:
        for col in ["Prediction", "Residual Std", "Edge vs Market"]:
            if col in per_model_df:
                per_model_df[col] = per_model_df[col].astype(float).round(2)
        st.markdown("#### Model Breakdown")
        st.dataframe(per_model_df, hide_index=True, use_container_width=True)


def render_player_props(
    player_props: pd.DataFrame,
    schedule_df: pd.DataFrame,
    validation_season: int,
    week: int,
    metrics: dict[str, float] | None,
    error: str | None,
):
    st.markdown("### Player Touchdown Probabilities")

    if error is not None:
        st.warning(f"Touchdown model unavailable: {error}")
        return

    if player_props.empty:
        st.info(
            "Touchdown projections will appear here once sufficient history is available for the selected matchup window."
        )
        return

    if metrics:
        cols = st.columns(3)
        auc = metrics.get("auc")
        avg_precision = metrics.get("average_precision")
        base_rate = metrics.get("base_rate")
        cols[0].metric("Model AUC", f"{auc:.3f}" if isinstance(auc, float) and not np.isnan(auc) else "–")
        cols[1].metric(
            "Average Precision",
            f"{avg_precision:.3f}" if isinstance(avg_precision, float) and not np.isnan(avg_precision) else "–",
        )
        cols[2].metric(
            "Base TD Rate",
            f"{base_rate:.2%}" if isinstance(base_rate, float) and not np.isnan(base_rate) else "–",
        )

    upcoming_slice = schedule_df[
        (schedule_df["season"] == validation_season)
        & (schedule_df["week"] == week)
    ][["game_id", "home_team", "away_team", "gameday"]].copy()
    if upcoming_slice.empty:
        st.info("No schedule entries available for the selected week.")
        return

    upcoming_slice["label"] = upcoming_slice.apply(
        lambda row: f"Week {week}: {row['away_team']} @ {row['home_team']}", axis=1
    )
    options = upcoming_slice.sort_values("label")
    selection = st.selectbox("Choose a matchup", options["label"].tolist())
    selected_row = upcoming_slice.loc[upcoming_slice["label"] == selection].iloc[0]
    selected_game_id = selected_row["game_id"]
    home_team = selected_row["home_team"]
    away_team = selected_row["away_team"]

    game_players = player_props[player_props["game_id"] == selected_game_id].copy()
    if game_players.empty:
        st.info("No player projections found for the selected matchup.")
        return

    game_players["td_prob"] = game_players["touchdown_prob"].astype(float)
    game_players["Touches"] = game_players["avg_total_touches"].map(lambda v: f"{v:.1f}")
    game_players["Rush Att"] = game_players["avg_rush_attempts"].map(lambda v: f"{v:.1f}")
    game_players["Targets"] = game_players["avg_targets"].map(lambda v: f"{v:.1f}")
    game_players["Red Zone"] = game_players["avg_redzone_touches"].map(lambda v: f"{v:.1f}")
    game_players["TD %"] = game_players["td_prob"].map(_format_prob)

    home_df = (
        game_players[game_players["team"] == home_team]
        .sort_values("td_prob", ascending=False)
        .head(8)
    )
    away_df = (
        game_players[game_players["team"] == away_team]
        .sort_values("td_prob", ascending=False)
        .head(8)
    )

    st.markdown("#### Top Scorers")
    cols = st.columns(2)
    cols[0].markdown(f"**{home_team} (Home)**")
    cols[0].dataframe(
        home_df[["player_display_name", "position", "TD %", "Touches", "Rush Att", "Targets", "Red Zone"]]
        .rename(columns={"player_display_name": "Player", "position": "Pos"}),
        hide_index=True,
        use_container_width=True,
    )
    cols[1].markdown(f"**{away_team} (Away)**")
    cols[1].dataframe(
        away_df[["player_display_name", "position", "TD %", "Touches", "Rush Att", "Targets", "Red Zone"]]
        .rename(columns={"player_display_name": "Player", "position": "Pos"}),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Probability Landscape")
    chart_df = game_players.sort_values("td_prob", ascending=True).tail(12)
    palette = {home_team: "#1d428a", away_team: "#c8102e"}
    colors = [palette.get(team, "#4b5563") for team in chart_df["team"]]
    fig = go.Figure(
        go.Bar(
            x=chart_df["td_prob"],
            y=chart_df["player_display_name"],
            orientation="h",
            text=chart_df["td_prob"].map(lambda v: f"{v:.1%}"),
            marker_color=colors,
            customdata=np.column_stack([chart_df["team"], chart_df["position"]]),
            hovertemplate="%{y} (%{customdata[0]}) — %{customdata[1]} | %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Touchdown Probability"),
        height=460,
        margin=dict(l=90, r=20, t=30, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Probabilities reflect the chance that a player scores at least one touchdown, driven by recent usage trends and the upcoming game's market total."
    )

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

    player_props, player_metrics, player_error = build_player_prop_predictions(
        schedule_df,
        train_seasons,
        validation_season,
        week,
        lookback=4,
    )

    matchups_tab, totals_tab, props_tab = st.tabs(["Matchups", "Totals", "Player Props"])

    with matchups_tab:
        render_metrics(predictions, team_filter)
    with totals_tab:
        render_totals_section(predictions)
    with props_tab:
        render_player_props(
            player_props,
            schedule_df,
            validation_season,
            week,
            player_metrics,
            player_error,
        )

    st.caption(
        "Model suite includes logistic regression, random forest, and LightGBM (if installed). Average win probabilities reflect the ensemble mean across available classifiers."
    )


if __name__ == "__main__":
    main()
