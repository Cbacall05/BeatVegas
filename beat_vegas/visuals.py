"""Visualization utilities for Project Beat Vegas."""
from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

ND_BLUE = "#0C2340"
ND_GOLD = "#C99700"
PALETTE = [ND_BLUE, ND_GOLD]

LOGGER = logging.getLogger(__name__)

sns.set_style("whitegrid")
sns.set_palette(PALETTE)


def _finalize_plot(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> plt.Axes:
    ax.set_title(title, fontsize=14, fontweight="bold", color=ND_BLUE)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(colors=ND_BLUE)
    plt.tight_layout()
    return ax


def plot_win_probability_distribution(
    predictions: pd.DataFrame,
    proba_col: str = "pred_win_proba",
    bins: int = 25,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot distribution of predicted win probabilities."""

    if proba_col not in predictions.columns:
        raise ValueError(f"Column '{proba_col}' not found in predictions dataframe.")

    ax = ax or plt.gca()
    sns.histplot(
        predictions[proba_col],
        kde=True,
        bins=bins,
        color=ND_BLUE,
        edgecolor="white",
        ax=ax,
    )
    return _finalize_plot(ax, "Predicted Win Probability Distribution", "Win Probability", "Frequency")


def plot_total_points_distribution(
    predictions: pd.DataFrame,
    total_col: str = "pred_total_points",
    kde: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot distribution of predicted total points."""

    if total_col not in predictions.columns:
        raise ValueError(f"Column '{total_col}' not found in predictions dataframe.")

    ax = ax or plt.gca()
    sns.kdeplot(predictions[total_col], color=ND_BLUE, ax=ax, fill=True, alpha=0.6)
    sns.histplot(
        predictions[total_col],
        bins=20,
        color=ND_GOLD,
        edgecolor="white",
        alpha=0.3,
        ax=ax,
    )
    return _finalize_plot(ax, "Predicted Total Points Distribution", "Total Points", "Density")


def plot_model_vs_market(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    game_id_col: str = "game_id",
    model_col: str = "pred_total_points",
    market_col: str = "market_total",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter comparison between model predictions and market lines."""

    merged = predictions.merge(market, on=game_id_col, how="inner")
    if merged.empty:
        raise ValueError("No overlapping games found between predictions and market data.")

    ax = ax or plt.gca()
    sns.scatterplot(
        data=merged,
        x=market_col,
        y=model_col,
        ax=ax,
        color=ND_BLUE,
        s=60,
    )
    line_min = min(merged[market_col].min(), merged[model_col].min())
    line_max = max(merged[market_col].max(), merged[model_col].max())
    ax.plot([line_min, line_max], [line_min, line_max], color=ND_GOLD, linestyle="--", linewidth=1.5)
    return _finalize_plot(ax, "Model vs. Market Totals", "Market Total", "Model Total")


def plot_error_distribution(
    predictions: pd.DataFrame,
    value_col: str,
    label: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot centered error distribution with kernel density."""

    if value_col not in predictions.columns:
        raise ValueError(f"Column '{value_col}' not found in dataframe.")

    ax = ax or plt.gca()
    sns.kdeplot(predictions[value_col], ax=ax, color=ND_BLUE, fill=True, alpha=0.5)
    ax.axvline(0, color=ND_GOLD, linestyle="--", linewidth=1.5)
    return _finalize_plot(ax, f"Distribution of {label}", label, "Density")


def plot_calibration_curve(
    predictions: pd.DataFrame,
    proba_col: str = "pred_win_proba",
    actual_col: str = "actual",
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Visualize calibration of win probability predictions."""

    missing = {col for col in (proba_col, actual_col) if col not in predictions.columns}
    if missing:
        raise ValueError(f"Predictions dataframe missing columns: {sorted(missing)}")

    ax = ax or plt.gca()
    prob_true, prob_pred = calibration_curve(
        predictions[actual_col], predictions[proba_col], n_bins=n_bins, strategy="uniform"
    )
    ax.plot(prob_pred, prob_true, marker="o", color=ND_BLUE, linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color=ND_GOLD, label="Perfect")
    ax.legend(frameon=False)
    return _finalize_plot(ax, "Calibration Curve", "Predicted Probability", "Observed Frequency")
