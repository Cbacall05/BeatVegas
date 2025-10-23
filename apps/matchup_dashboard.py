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
from beat_vegas import data_load, models, player_models, injury_impact
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
LOGIT_SHIFT_SCALE = 2.1
MANUAL_INJURY_PATH = ROOT_DIR / "configs" / "manual_injuries.csv"
DEFAULT_SPREAD_SLOPE = 6.9
DEFAULT_SPREAD_INTERCEPT = -0.5
MANUAL_QB_OVERRIDES = {
    "DEN": "Bo Nix",
    "NYG": "Jaxson Dart",
    "CIN": "Joe Flacco",
    "NYJ": "Justin Fields",
}

if "_cache_initialized" not in st.session_state:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["_cache_initialized"] = True

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


def _fit_spread_mapping(
    schedule_df: pd.DataFrame,
    *,
    validation_season: int,
    week: int | None,
) -> tuple[float, float]:
    if schedule_df.empty:
        raise ValueError("Schedule dataframe is empty.")

    hist = schedule_df.copy()
    cutoff_week = week if week is not None else int(hist.loc[hist["season"] == validation_season, "week"].max())
    mask_hist = (hist["season"] < validation_season) | (
        (hist["season"] == validation_season) & (hist["week"] < cutoff_week)
    )
    hist = hist.loc[mask_hist]
    if hist.empty:
        raise ValueError("No historical games available before the target week.")

    home_prob = data_load.convert_moneyline_to_probability(hist["home_moneyline"])
    spreads = hist["spread_line"]
    valid = home_prob.notna() & spreads.notna()
    if not valid.any():
        raise ValueError("Insufficient overlap of moneyline probabilities and spreads.")

    prob = home_prob.loc[valid].astype(float).clip(1e-6, 1 - 1e-6)
    spread = spreads.loc[valid].astype(float)
    logit = np.log(prob / (1 - prob))
    slope, intercept = np.polyfit(logit, spread, 1)
    return float(slope), float(intercept)


def _probability_to_spread(probabilities: pd.Series, *, slope: float, intercept: float) -> pd.Series:
    clipped = probabilities.astype(float).clip(1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    return slope * logit + intercept


def _load_manual_injury_adjustments() -> pd.DataFrame:
    if not MANUAL_INJURY_PATH.exists():
        return pd.DataFrame(columns=[field for field in injury_impact.InjuryAdjustment.__annotations__.keys()])

    manual_df = pd.read_csv(MANUAL_INJURY_PATH)
    if manual_df.empty:
        return manual_df

    expected = {"team", "player_name", "position", "status"}
    missing = expected - set(manual_df.columns)
    if missing:
        st.warning(f"Manual injury overrides missing columns: {sorted(missing)}")
        return pd.DataFrame(columns=[field for field in injury_impact.InjuryAdjustment.__annotations__.keys()])

    manual_df = manual_df.copy()
    manual_df["team"] = manual_df["team"].astype(str).str.upper()
    manual_df["player_name"] = manual_df["player_name"].astype(str)
    manual_df["position"] = manual_df["position"].astype(str).str.upper()
    manual_df["status"] = manual_df["status"].astype(str).str.upper()
    if "player_id" not in manual_df.columns or manual_df["player_id"].isna().all():
        manual_df["player_id"] = manual_df.apply(
            lambda row: f"MANUAL_{row['team']}_{row['player_name'].upper().replace(' ', '_')}"
            , axis=1
        )

    if "impact_score" not in manual_df.columns:
        manual_df["impact_score"] = 0.25
    manual_df["impact_score"] = manual_df["impact_score"].astype(float).clip(lower=0.0, upper=1.0)

    status_weights = {
        key: float(value) for key, value in injury_impact.DEFAULT_STATUS_WEIGHTS.items()
    }
    weight_series = manual_df["status"].map(status_weights).fillna(0.5)
    if "penalty" in manual_df.columns:
        manual_df["penalty"] = pd.to_numeric(manual_df["penalty"], errors="coerce")
    else:
        manual_df["penalty"] = np.nan
    manual_df["penalty"] = manual_df["penalty"].where(manual_df["penalty"].notna(), manual_df["impact_score"] * weight_series)
    manual_df["penalty"] = manual_df["penalty"].astype(float).clip(lower=0.0, upper=1.0)
    manual_df = injury_impact.apply_position_weight(manual_df)

    return manual_df[[
        "team",
        "player_id",
        "player_name",
        "position",
        "status",
        "impact_score",
        "penalty",
        "position_weight",
    ]]


@st.cache_data(show_spinner=False)
def load_schedule_cached(seasons: tuple[int, ...]) -> pd.DataFrame:
    return data_load.load_schedule(list(seasons))


@st.cache_data(show_spinner=False)
def load_pbp_cached(seasons: tuple[int, ...]) -> pd.DataFrame:
    cache_dir = ROOT_DIR / "data" / "pbp"
    return data_load.load_play_by_play(list(seasons), cache_dir=cache_dir)


@st.cache_data(show_spinner=False)
def load_rosters_cached(seasons: tuple[int, ...]) -> pd.DataFrame:
    return data_load.load_rosters(list(seasons))


@st.cache_data(show_spinner=False)
def load_weekly_cached(seasons: tuple[int, ...]) -> pd.DataFrame:
    return data_load.load_weekly_data(list(seasons))


@st.cache_data(show_spinner=False)
def load_injuries_cached(seasons: tuple[int, ...]) -> pd.DataFrame:
    return data_load.load_injuries(list(seasons))


def _latest_injury_records(
    injury_df: pd.DataFrame,
    *,
    validation_season: int,
    week: int,
) -> pd.DataFrame:
    if injury_df.empty:
        return injury_df

    filtered = injury_df.copy()
    if "season" in filtered.columns:
        filtered["season"] = filtered["season"].fillna(0).astype(int)
        filtered = filtered[filtered["season"] == validation_season]
    if "week" in filtered.columns:
        filtered["week"] = filtered["week"].fillna(0).astype(int)
        filtered = filtered[filtered["week"] <= week]

    if filtered.empty:
        return filtered

    sort_keys: list[str] = ["player_id"]
    if "reported_date" in filtered.columns:
        sort_keys.append("reported_date")
    elif "report_date" in filtered.columns:
        sort_keys.append("report_date")
    elif "week" in filtered.columns:
        sort_keys.append("week")

    filtered.sort_values(sort_keys, inplace=True)
    latest = filtered.groupby("player_id", as_index=False).tail(1)
    return latest


def _manual_override_fallback(
    alerts: list[str],
    note: str,
    manual_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    manual_adjustments = manual_df if manual_df is not None else _load_manual_injury_adjustments()
    if manual_adjustments.empty:
        return pd.DataFrame(), pd.DataFrame(), alerts
    penalties = injury_impact.team_penalties(manual_adjustments)
    alerts.append(note)
    manual_names = sorted({name for name in manual_adjustments["player_name"].astype(str) if name})
    if manual_names:
        message = "Manual overrides applied for: {}.".format(", ".join(manual_names))
        if message not in alerts:
            alerts.append(message)
    alerts.extend(injury_impact.build_alert_messages(manual_adjustments))
    return manual_adjustments, penalties, alerts
def prepare_injury_context(
    train_seasons: tuple[int, ...],
    validation_season: int,
    week: int,
    *,
    lookback_games: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    seasons_for_data = tuple(sorted(set(train_seasons + (validation_season,))))
    alerts: list[str] = []
    manual_overrides = _load_manual_injury_adjustments()

    try:
        injury_df = load_injuries_cached(seasons_for_data)
    except Exception as exc:  # noqa: BLE001 - present to UI
        alerts.append(f"Injury data unavailable: {exc}")
        return _manual_override_fallback(
            alerts,
            "Using manual injury overrides while live data is unavailable.",
            manual_overrides,
        )

    injury_skipped = getattr(injury_df, "attrs", {}).get("skipped_seasons") if injury_df is not None else None
    if injury_skipped:
        skipped_str = ", ".join(str(season) for season in injury_skipped)
        alerts.append(
            f"Injury reports not available from nfl_data_py for seasons: {skipped_str}. Falling back to prior data where possible."
        )

    if injury_df.empty:
        adjustments, penalties, alerts = _manual_override_fallback(
            alerts,
            "Using manual injury overrides while live data is unavailable.",
            manual_overrides,
        )
        return adjustments, penalties, alerts

    try:
        weekly_df = load_weekly_cached(seasons_for_data)
    except Exception as exc:  # noqa: BLE001 - present to UI
        alerts.append(f"Weekly player data unavailable for injury adjustments: {exc}")
        return _manual_override_fallback(
            alerts,
            "Using manual injury overrides (weekly data unavailable).",
            manual_overrides,
        )

    skipped = getattr(weekly_df, "attrs", {}).get("skipped_seasons")
    if skipped:
        skipped_list = ", ".join(str(season) for season in skipped)
        alerts.append(
            f"Weekly stats not yet published for seasons: {skipped_list}. Using historical data from the remaining seasons."
        )

    usage_df = injury_impact.compute_usage_profile(weekly_df, lookback_games=lookback_games)
    superstars = injury_impact.select_superstars(usage_df)
    if superstars.empty:
        adjustments, penalties, alerts = _manual_override_fallback(
            alerts,
            "Using manual injury overrides (insufficient usage data).",
            manual_overrides,
        )
        return adjustments, penalties, alerts

    latest_injuries = _latest_injury_records(
        injury_df,
        validation_season=validation_season,
        week=week,
    )
    if latest_injuries.empty:
        adjustments, penalties, alerts = _manual_override_fallback(
            alerts,
            "Using manual injury overrides (no recent injury reports).",
            manual_overrides,
        )
        return adjustments, penalties, alerts

    adjustments = injury_impact.summarize_injuries(latest_injuries, superstars)
    if adjustments.empty:
        adjustments, penalties, alerts = _manual_override_fallback(
            alerts,
            "Using manual injury overrides (no superstar injuries detected).",
            manual_overrides,
        )
        return adjustments, penalties, alerts

    def _combine_adjustments(auto_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
        if manual_df.empty:
            return auto_df

        manual = manual_df.copy()
        manual["team"] = manual["team"].astype(str).str.upper()
        manual["player_name"] = manual["player_name"].astype(str)
        manual["_priority"] = 0
        manual["_name_key"] = manual["player_name"].str.lower().str.strip()

        if auto_df.empty:
            return manual.drop(columns=["_priority", "_name_key"], errors="ignore")

        automated = auto_df.copy()
        automated["team"] = automated["team"].astype(str).str.upper()
        automated["player_name"] = automated["player_name"].astype(str)
        automated["_priority"] = 1
        automated["_name_key"] = automated["player_name"].str.lower().str.strip()

        combined = pd.concat([manual, automated], ignore_index=True, sort=False)
        combined.sort_values(["team", "_name_key", "_priority"], inplace=True)
        combined = combined.drop_duplicates(subset=["team", "_name_key"], keep="first")
        combined.drop(columns=["_priority", "_name_key"], inplace=True, errors="ignore")
        return combined

    adjustments = _combine_adjustments(adjustments, manual_overrides)
    penalties = injury_impact.team_penalties(adjustments)
    if not manual_overrides.empty:
        manual_names = sorted({name for name in manual_overrides["player_name"].astype(str) if name})
        if manual_names:
            message = "Manual overrides applied for: {}.".format(", ".join(manual_names))
            if message not in alerts:
                alerts.append(message)
    alerts.extend(injury_impact.build_alert_messages(adjustments))
    return adjustments, penalties, alerts


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
    team_penalties: pd.DataFrame | None = None,
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
    try:
        spread_slope, spread_intercept = _fit_spread_mapping(
            schedule_df,
            validation_season=validation_season,
            week=week,
        )
    except ValueError:
        spread_slope = DEFAULT_SPREAD_SLOPE
        spread_intercept = DEFAULT_SPREAD_INTERCEPT
    if "game_date" in predictions.columns:
        predictions["game_date"] = pd.to_datetime(predictions["game_date"])

    if team_penalties is not None and not team_penalties.empty and {"team", "penalty"}.issubset(team_penalties.columns):
        penalty_map = team_penalties.set_index("team")["penalty"].astype(float)
        predictions["raw_home_win_prob"] = predictions.get("avg_home_win_prob")
        predictions["raw_away_win_prob"] = predictions.get("avg_away_win_prob")
        predictions["injury_penalty_home"] = predictions["home_team"].map(penalty_map).fillna(0.0)
        predictions["injury_penalty_away"] = predictions["away_team"].map(penalty_map).fillna(0.0)
        predictions["injury_penalty_net"] = (
            predictions["injury_penalty_away"] - predictions["injury_penalty_home"]
        )

        def _adjust_home_probability(row: pd.Series) -> float:
            base_prob = row.get("raw_home_win_prob")
            if base_prob is None or pd.isna(base_prob):
                return base_prob
            home_pen = float(row.get("injury_penalty_home", 0.0))
            away_pen = float(row.get("injury_penalty_away", 0.0))
            if home_pen == 0 and away_pen == 0:
                return float(np.clip(base_prob, 0.001, 0.999))
            logit = np.log((base_prob + 1e-6) / (1 - base_prob + 1e-6))
            logit -= LOGIT_SHIFT_SCALE * home_pen
            logit += LOGIT_SHIFT_SCALE * away_pen
            adjusted = 1 / (1 + np.exp(-logit))
            return float(np.clip(adjusted, 0.01, 0.99))

        adjusted_home = predictions.apply(_adjust_home_probability, axis=1)
        predictions["avg_home_win_prob"] = adjusted_home
        predictions["avg_away_win_prob"] = 1 - adjusted_home
        predictions["predicted_winner"] = np.where(
            predictions["avg_home_win_prob"] >= predictions["avg_away_win_prob"],
            predictions["home_team"],
            predictions["away_team"],
        )
        predictions["predicted_win_prob"] = np.where(
            predictions["predicted_winner"] == predictions["home_team"],
            predictions["avg_home_win_prob"],
            predictions["avg_away_win_prob"],
        )

    if "avg_home_win_prob" in predictions.columns:
        predictions["model_spread"] = _probability_to_spread(
            predictions["avg_home_win_prob"], slope=spread_slope, intercept=spread_intercept
        )
    else:
        predictions["model_spread"] = np.nan

    line_lookup = (
        schedule_df.loc[:, ["game_id", "spread_line"]]
        .dropna(subset=["spread_line"])
        .drop_duplicates(subset=["game_id"], keep="last")
        .rename(columns={"spread_line": "market_spread"})
    )
    predictions = predictions.merge(line_lookup, on="game_id", how="left")
    predictions["spread_mapping_slope"] = spread_slope
    predictions["spread_mapping_intercept"] = spread_intercept
    return predictions


def build_player_prop_predictions(
    schedule_df: pd.DataFrame,
    train_seasons: tuple[int, ...],
    validation_season: int,
    week: int,
    lookback: int = 4,
    injury_adjustments: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]], list[str]]:
    seasons_for_data = tuple(sorted(set(train_seasons + (validation_season,))))
    issues: list[str] = []
    props_by_type: dict[str, pd.DataFrame] = {}
    metrics_by_type: dict[str, dict[str, float]] = {}

    try:
        pbp_df = load_pbp_cached(seasons_for_data)
    except Exception as exc:  # noqa: BLE001 - surface to UI
        message = str(exc)
        if "name 'Error' is not defined" in message and train_seasons:
            fallback_seasons = tuple(sorted(set(train_seasons)))
            warning = (
                f"Play-by-play not yet available for season {validation_season}. "
                "Using historical seasons only for touchdown model features."
            )
            issues.append(warning)
            try:
                pbp_df = load_pbp_cached(fallback_seasons)
            except Exception as fallback_exc:  # noqa: BLE001 - present error to UI
                return props_by_type, metrics_by_type, [warning, f"Play-by-play retry failed: {fallback_exc}"]
        else:
            return props_by_type, metrics_by_type, [f"Play-by-play unavailable: {exc}"]

    roster_df: pd.DataFrame | None = None
    try:
        roster_df = load_rosters_cached(seasons_for_data)
    except Exception as exc:  # noqa: BLE001 - roster filtering can be skipped
        issues.append(f"Roster lookup skipped: {exc}")

    def _apply_injury_filter(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df.empty or injury_adjustments is None or injury_adjustments.empty:
            return df
        inactive_players = set(injury_adjustments["player_id"].astype(str))
        inactive_names = (
            injury_adjustments["player_name"].astype(str).str.lower()
            if "player_name" in injury_adjustments.columns
            else pd.Series(dtype=str)
        )
        inactive_name_set = {name.strip() for name in inactive_names if name}
        if not inactive_players and not inactive_name_set:
            return df
        before = len(df)
        mask_combined = ~df["player_id"].astype(str).isin(inactive_players)
        if inactive_name_set:
            name_mask = pd.Series(True, index=df.index)
            for col in ("player_display_name", "player_name", "roster_player_name"):
                if col in df.columns:
                    name_mask &= ~df[col].astype(str).str.lower().str.strip().isin(inactive_name_set)
            mask_combined &= name_mask
        filtered = df[mask_combined]
        removed = before - len(filtered)
        if removed > 0:
            issues.append(
                "Removed {removed} injured superstar players from {label} projections.".format(
                    removed=removed,
                    label=label,
                )
            )
        return filtered

    def _apply_manual_qb_overrides(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "team" not in df.columns or "player_display_name" not in df.columns:
            return df
        updated = df.copy()
        teams_upper = updated["team"].astype(str).str.upper()
        for team, override_name in MANUAL_QB_OVERRIDES.items():
            mask = teams_upper == team
            if mask.any():
                updated.loc[mask, "player_display_name"] = override_name
        return updated

    try:
        model_result, _, upcoming = player_models.train_and_predict_touchdowns(
            schedule_df,
            pbp_df,
            seasons=seasons_for_data,
            target_season=validation_season,
            target_week=week,
            lookback=lookback,
            min_touches=0.75,
            roster_df=roster_df,
        )
        upcoming = _apply_injury_filter(upcoming, "touchdown")
        props_by_type["touchdowns"] = upcoming
        metrics_by_type["touchdowns"] = model_result.metrics
    except Exception as exc:  # noqa: BLE001 - present error to UI
        issues.append(f"Touchdown model error: {exc}")

    try:
        pass_result, _, passing_upcoming = player_models.train_and_predict_passing_yards(
            schedule_df,
            pbp_df,
            seasons=seasons_for_data,
            target_season=validation_season,
            target_week=week,
            lookback=lookback,
            min_attempts=12.0,
            roster_df=roster_df,
            preferred_qbs=MANUAL_QB_OVERRIDES,
        )
        passing_upcoming = _apply_injury_filter(passing_upcoming, "passing yards")
        passing_upcoming = _apply_manual_qb_overrides(passing_upcoming)
        props_by_type["passing_yards"] = passing_upcoming
        metrics_by_type["passing_yards"] = pass_result.metrics
    except Exception as exc:  # noqa: BLE001 - present error to UI
        issues.append(f"Passing yards model error: {exc}")

    return props_by_type, metrics_by_type, issues


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
        st.caption("Tune the training window, validation season, and upcoming week to refresh predictions.")

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

        return (
            _season_tuple(seasons_selected),
            int(validation_season),
            int(rolling_window),
            int(week),
            _normalize_team_list(team_filter),
        )


def render_injury_tables(
    adjustments: pd.DataFrame,
    team_penalties: pd.DataFrame,
    alerts: list[str],
) -> None:
    if not alerts and (adjustments is None or adjustments.empty) and (team_penalties is None or team_penalties.empty):
        return

    st.markdown("### Injury Report")

    if alerts:
        st.caption(" • ".join(alerts))

    if adjustments is not None and not adjustments.empty:
        st.caption("Player-level impact overrides")
        display_cols = [
            "team",
            "player_name",
            "position",
            "status",
            "impact_score",
            "penalty",
            "position_weight",
        ]
        display_df = adjustments[display_cols].copy()
        display_df.rename(
            columns={
                "team": "Team",
                "player_name": "Player",
                "position": "Pos",
                "status": "Status",
                "impact_score": "Usage Impact",
                "penalty": "Weighted Penalty",
                "position_weight": "Pos Weight",
            },
            inplace=True,
        )
        display_df["Usage Impact"] = display_df["Usage Impact"].astype(float).map(lambda v: f"{v:.2f}")
        display_df["Weighted Penalty"] = display_df["Weighted Penalty"].astype(float).map(lambda v: f"{v * 100:.1f}%")
        display_df["Pos Weight"] = display_df["Pos Weight"].astype(float).map(lambda v: f"{v:.2f}")
        st.dataframe(display_df, hide_index=True, width="stretch")

    if team_penalties is not None and not team_penalties.empty:
        st.caption("Team penalty summary")
        penalty_df = team_penalties.copy()
        penalty_df.rename(columns={"team": "Team", "penalty": "Penalty"}, inplace=True)
        penalty_df["Penalty"] = penalty_df["Penalty"].astype(float).map(lambda v: f"{v * 100:.1f}%")
        st.dataframe(penalty_df, hide_index=True, width="stretch")


def render_debug_panel(
    team_code: str,
    team_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    validation_season: int,
    week: int,
) -> None:
    st.markdown("### Debugging Snapshot")
    team = (team_code or "").strip().upper()
    if not team:
        st.info("Enter a team abbreviation to inspect.")
        return

    st.caption(f"Inspecting team: {team}")

    recent_df = team_df[team_df.get("team", "").str.upper() == team]
    if recent_df.empty:
        st.warning("No team strength data found for this abbreviation.")
    else:
        display_cols = [col for col in [
            "season",
            "week",
            "team",
            "opponent",
            "rolling_offense",
            "rolling_defense",
            "market_spread",
            "market_total",
        ] if col in recent_df.columns]
        preview = (
            recent_df.sort_values(["season", "week"], ascending=[False, False])
            .head(8)
            .loc[:, display_cols]
        )
        st.markdown("#### Recent team metrics")
        st.dataframe(preview, hide_index=True, width="stretch")

    upcoming = schedule_df[
        (schedule_df["season"] == validation_season)
        & (schedule_df["week"] == week)
        & (
            (schedule_df["home_team"].str.upper() == team)
            | (schedule_df["away_team"].str.upper() == team)
        )
    ]
    if upcoming.empty:
        st.info("No schedule entry for the selected team/week.")
    else:
        st.markdown("#### Upcoming schedule entry")
        sched_cols = [col for col in [
            "game_id",
            "gameday",
            "home_team",
            "away_team",
            "spread_line",
            "market_total",
            "home_moneyline",
            "away_moneyline",
        ] if col in upcoming.columns]
        st.dataframe(upcoming.loc[:, sched_cols], hide_index=True, width="stretch")

    game_predictions = predictions[
        (predictions.get("home_team", "").str.upper() == team)
        | (predictions.get("away_team", "").str.upper() == team)
    ]
    if game_predictions.empty:
        st.info("No prediction row generated for this team.")
        return

    debug_cols = [col for col in [
        "game_id",
        "home_team",
        "away_team",
        "raw_home_win_prob",
        "avg_home_win_prob",
        "avg_away_win_prob",
        "injury_penalty_home",
        "injury_penalty_away",
        "injury_penalty_net",
        "model_spread",
        "market_spread",
        "predicted_winner",
        "predicted_win_prob",
    ] if col in game_predictions.columns]
    st.markdown("#### Win probability comparison (raw vs adjusted)")
    display = game_predictions.loc[:, debug_cols].copy()
    prob_cols = [
        "raw_home_win_prob",
        "avg_home_win_prob",
        "avg_away_win_prob",
        "predicted_win_prob",
    ]
    for col in prob_cols:
        if col in display.columns:
            display[col] = display[col].astype(float).map(lambda v: f"{v:.3f}")
    penalty_cols = ["injury_penalty_home", "injury_penalty_away", "injury_penalty_net"]
    for col in penalty_cols:
        if col in display.columns:
            display[col] = display[col].astype(float).map(lambda v: f"{v:.3f}")
    if "model_spread" in display.columns:
        display["model_spread"] = display["model_spread"].astype(float).map(lambda v: f"{v:+.2f}")
    if "market_spread" in display.columns:
        display["market_spread"] = display["market_spread"].astype(float).map(lambda v: f"{v:+.2f}")
    st.dataframe(display, hide_index=True, width="stretch")

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
    if "model_spread" in display_df.columns:
        display_df["Model Spread"] = display_df["model_spread"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "–")
    if "market_spread" in display_df.columns:
        display_df["Market Spread"] = display_df["market_spread"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "–")
    show_cols = [
        "Kickoff",
        "week",
        "home_team",
        "away_team",
        "Winner",
        "Confidence",
        "Home Win %",
        "Model Spread",
        "Market Spread",
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
        width="stretch",
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
    st.plotly_chart(chart, width="stretch")

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
        "model_spread",
        "market_spread",
        "spread_mapping_slope",
        "spread_mapping_intercept",
        "market_total",
        "home_moneyline",
        "away_moneyline",
    ] + probability_cols
    detail_cols = [col for col in detail_cols if col in predictions.columns]

    formatted = predictions[detail_cols].copy()
    for col in ["avg_home_win_prob", "avg_away_win_prob", "predicted_win_prob"]:
        if col in formatted:
            formatted[col] = formatted[col].map(_format_prob)
    if "model_spread" in formatted:
        formatted["model_spread"] = formatted["model_spread"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "–")
    if "market_spread" in formatted:
        formatted["market_spread"] = formatted["market_spread"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "–")
    for col in ["spread_mapping_slope", "spread_mapping_intercept"]:
        if col in formatted:
            formatted[col] = formatted[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "–")

    st.dataframe(formatted, width="stretch")


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
        width="stretch",
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
        st.plotly_chart(fig, width="stretch")

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
    st.dataframe(per_model_df, hide_index=True, width="stretch")


def render_spreads_section(predictions: pd.DataFrame):
    if predictions.empty or "model_spread" not in predictions.columns:
        st.info("Spread projections are not available for the current filters.")
        return

    st.markdown("### Spread Outlook")

    slope_values = predictions.get("spread_mapping_slope")
    intercept_values = predictions.get("spread_mapping_intercept")
    slope = slope_values.dropna().iloc[0] if slope_values is not None and not slope_values.dropna().empty else None
    intercept = (
        intercept_values.dropna().iloc[0]
        if intercept_values is not None and not intercept_values.dropna().empty
        else None
    )
    if slope is not None and intercept is not None:
        st.caption(f"Mapping: spread = {slope:.3f} × logit(p) + {intercept:.3f}")

    required_cols = {"game_id", "week", "home_team", "away_team", "model_spread", "market_spread"}
    if not required_cols.issubset(predictions.columns):
        st.info("Insufficient columns to display spread projections.")
        return

    table = predictions[list(required_cols)].copy()
    table["Model"] = table["model_spread"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "–")
    table["Market"] = table["market_spread"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "–")
    table["Edge"] = (table["model_spread"] - table["market_spread"]).map(
        lambda v: f"{v:+.1f}" if pd.notna(v) else "–"
    )

    st.dataframe(
        table[["week", "home_team", "away_team", "Model", "Market", "Edge"]]
        .rename(columns={"week": "Week", "home_team": "Home", "away_team": "Away"})
        .sort_values("Week"),
        hide_index=True,
        width="stretch",
    )

    edge_numeric = table["model_spread"] - table["market_spread"]
    if edge_numeric.notna().any():
        chart_df = table.assign(edge_numeric=edge_numeric)
        spread_chart = px.bar(
            chart_df,
            x="edge_numeric",
            y="game_id",
            orientation="h",
            labels={"edge_numeric": "Model - Market (pts)", "game_id": "Game"},
            color="edge_numeric",
            color_continuous_scale=["#ef4444", "#f97316", "#22c55e"],
        )
        spread_chart.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=40, b=40),
            coloraxis_showscale=False,
        )
        st.plotly_chart(spread_chart, width="stretch")
    else:
        st.info("No spread mispricing bars to visualize this week.")


def render_player_props(
    props_by_type: dict[str, pd.DataFrame],
    schedule_df: pd.DataFrame,
    validation_season: int,
    week: int,
    metrics_by_type: dict[str, dict[str, float]],
    issues: list[str],
):
    st.markdown("### Player Props")

    warning_keywords = ("unavailable", "failed", "error", "missing", "could not", "invalid")
    warnings = [msg for msg in issues if any(keyword in msg.lower() for keyword in warning_keywords)]
    infos = [msg for msg in issues if msg not in warnings]

    for message in infos:
        st.info(message)

    for message in warnings:
        st.warning(message)

    label_lookup = {
        "touchdowns": "Touchdowns",
        "passing_yards": "Passing Yards",
    }
    available_types = [key for key in label_lookup if key in props_by_type]
    if not available_types:
        st.info("Player prop models are not available for the selected configuration.")
        return

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
    selection = st.selectbox("Choose a matchup", options["label"].tolist(), key="props_matchup_select")
    selected_row = upcoming_slice.loc[upcoming_slice["label"] == selection].iloc[0]
    selected_game_id = selected_row["game_id"]
    home_team = selected_row["home_team"]
    away_team = selected_row["away_team"]

    option_to_key = {label_lookup[key]: key for key in available_types}
    labels = list(option_to_key.keys())
    selected_label = st.radio(
        "Prop Type",
        options=labels,
        horizontal=True,
        key="props_type_choice",
    )
    selected_type = option_to_key[selected_label]

    selected_df = props_by_type.get(selected_type, pd.DataFrame())
    if selected_df.empty:
        st.info("No player projections found for the selected configuration.")
        return

    game_df = selected_df[selected_df["game_id"] == selected_game_id].copy()
    if game_df.empty:
        st.info("No player projections found for the selected matchup.")
        return

    selected_metrics = metrics_by_type.get(selected_type) or {}

    if selected_type == "touchdowns":
        render_touchdown_prop_details(game_df, selected_metrics, home_team, away_team)
    elif selected_type == "passing_yards":
        render_passing_yards_prop_details(game_df, selected_metrics, home_team, away_team)


def render_touchdown_prop_details(
    game_players: pd.DataFrame,
    metrics: dict[str, float],
    home_team: str,
    away_team: str,
) -> None:
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

    if game_players.empty:
        st.info("No touchdown projections found for the selected matchup.")
        return

    game_players = game_players.copy()
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

    st.markdown("#### Touchdown Leaders")
    cols = st.columns(2)
    cols[0].markdown(f"**{home_team} (Home)**")
    cols[0].dataframe(
        home_df[["player_display_name", "position", "TD %", "Touches", "Rush Att", "Targets", "Red Zone"]]
        .rename(columns={"player_display_name": "Player", "position": "Pos"}),
        hide_index=True,
        width="stretch",
    )
    cols[1].markdown(f"**{away_team} (Away)**")
    cols[1].dataframe(
        away_df[["player_display_name", "position", "TD %", "Touches", "Rush Att", "Targets", "Red Zone"]]
        .rename(columns={"player_display_name": "Player", "position": "Pos"}),
        hide_index=True,
        width="stretch",
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
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Probabilities reflect the chance that a player scores at least one touchdown, based on recent usage trends and the upcoming game's context."
    )


def render_passing_yards_prop_details(
    game_qbs: pd.DataFrame,
    metrics: dict[str, float],
    home_team: str,
    away_team: str,
) -> None:
    cols = st.columns(2)
    mae = metrics.get("mae") if metrics else None
    r2 = metrics.get("r2") if metrics else None
    cols[0].metric("MAE (yds)", f"{mae:.1f}" if isinstance(mae, float) and not np.isnan(mae) else "–")
    cols[1].metric("R²", f"{r2:.3f}" if isinstance(r2, float) and not np.isnan(r2) else "–")

    if game_qbs.empty:
        st.info("No quarterback projections found for the selected matchup.")
        return

    game_qbs = game_qbs.copy()
    game_qbs["Expected Yards"] = game_qbs["expected_passing_yards"].astype(float).map(lambda v: f"{v:.1f}")
    game_qbs["Avg Attempts"] = game_qbs["avg_pass_attempts"].map(lambda v: f"{v:.1f}")
    game_qbs["Avg Completions"] = game_qbs["avg_completions"].map(lambda v: f"{v:.1f}")
    game_qbs["Avg Yds"] = game_qbs["avg_passing_yards"].map(lambda v: f"{v:.1f}")
    game_qbs["Avg TD"] = game_qbs["avg_passing_touchdowns"].map(lambda v: f"{v:.2f}")
    game_qbs["Avg INT"] = game_qbs["avg_interceptions"].map(lambda v: f"{v:.2f}")

    home_df = (
        game_qbs[game_qbs["team"] == home_team]
        .sort_values("expected_passing_yards", ascending=False)
        .head(4)
    )
    away_df = (
        game_qbs[game_qbs["team"] == away_team]
        .sort_values("expected_passing_yards", ascending=False)
        .head(4)
    )

    st.markdown("#### Passing Yardage Outlook")
    cols = st.columns(2)
    cols[0].markdown(f"**{home_team} (Home)**")
    cols[0].dataframe(
        home_df[["player_display_name", "Expected Yards", "Avg Attempts", "Avg Completions", "Avg TD", "Avg INT"]]
        .rename(columns={"player_display_name": "Quarterback"}),
        hide_index=True,
        width="stretch",
    )
    cols[1].markdown(f"**{away_team} (Away)**")
    cols[1].dataframe(
        away_df[["player_display_name", "Expected Yards", "Avg Attempts", "Avg Completions", "Avg TD", "Avg INT"]]
        .rename(columns={"player_display_name": "Quarterback"}),
        hide_index=True,
        width="stretch",
    )

    st.markdown("#### Yardage Distribution")
    chart_df = game_qbs.sort_values("expected_passing_yards", ascending=True)
    palette = {home_team: "#1d428a", away_team: "#c8102e"}
    colors = [palette.get(team, "#555555") for team in chart_df["team"]]
    fig = go.Figure(
        go.Bar(
            x=chart_df["expected_passing_yards"],
            y=chart_df["player_display_name"],
            orientation="h",
            text=chart_df["expected_passing_yards"].map(lambda v: f"{v:.1f}"),
            marker_color=colors,
            customdata=np.column_stack([chart_df["team"], chart_df["avg_pass_attempts"]]),
            hovertemplate="%{y} (%{customdata[0]}) — Attempts %{customdata[1]:.1f} | %{x:.1f} yds<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Expected Passing Yards"),
        height=420,
        margin=dict(l=90, r=20, t=30, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Projections combine recent quarterback usage with market context to estimate expected passing yards."
    )

def main() -> None:
    render_header()
    train_seasons, validation_season, rolling_window, week, team_filter = render_sidebar()
    if st.sidebar.button("Clear cached data and rerun"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state["_cache_initialized"] = True
        st.experimental_rerun()

    debug_team = st.sidebar.text_input("Debug Team (e.g., NYJ)", value="NYJ")
    injury_adjustments = pd.DataFrame()
    team_penalties = pd.DataFrame()
    injury_alerts: list[str] = []
    predictions = pd.DataFrame()
    player_props: dict[str, pd.DataFrame] = {}
    player_metrics: dict[str, dict[str, float]] = {}
    player_issues: list[str] = []
    schedule_df = pd.DataFrame()
    team_df = pd.DataFrame()

    with st.spinner("Training models and generating predictions..."):
        injury_adjustments, team_penalties, injury_alerts = prepare_injury_context(
            train_seasons,
            validation_season,
            week,
            lookback_games=rolling_window,
        )

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
            team_penalties=team_penalties,
        )

        player_props, player_metrics, player_issues = build_player_prop_predictions(
            schedule_df,
            train_seasons,
            validation_season,
            week,
            lookback=4,
            injury_adjustments=injury_adjustments,
        )

    render_injury_tables(injury_adjustments, team_penalties, injury_alerts)

    matchups_tab, spreads_tab, totals_tab, props_tab, debug_tab = st.tabs(
        ["Matchups", "Spreads", "Totals", "Player Props", "Debug"]
    )

    with matchups_tab:
        render_metrics(predictions, team_filter)
    with spreads_tab:
        render_spreads_section(predictions)
    with totals_tab:
        render_totals_section(predictions)
    with props_tab:
        render_player_props(
            player_props,
            schedule_df,
            validation_season,
            week,
            player_metrics,
            player_issues,
        )
    with debug_tab:
        render_debug_panel(
            debug_team,
            team_df,
            schedule_df,
            predictions,
            validation_season=validation_season,
            week=week,
        )

    st.caption(
        "Model suite includes logistic regression, random forest, and LightGBM (if installed). Average win probabilities reflect the ensemble mean across available classifiers."
    )


if __name__ == "__main__":
    main()
