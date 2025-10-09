"""Model training utilities for Project Beat Vegas."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb  # type: ignore[import]
except ImportError:  # pragma: no cover - handled via runtime guard
    lgb = None

LOGGER = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for train/validation split across seasons."""

    validation_seasons: Sequence[int]
    season_column: str = "season"


@dataclass
class ModelResult:
    """Container capturing trained model, predictions, and metrics."""

    model_name: str
    model: Any
    metrics: Dict[str, float]
    predictions: pd.DataFrame


def _ensure_lightgbm_available() -> None:
    if lgb is None:
        raise ImportError("lightgbm is required. Install it via 'pip install lightgbm'.")


def time_aware_split(
    df: pd.DataFrame,
    config: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/validation sets based on seasons."""

    if config.season_column not in df.columns:
        raise ValueError(f"Season column '{config.season_column}' not found in dataframe.")

    validation_mask = df[config.season_column].isin(config.validation_seasons)
    if not validation_mask.any():
        raise ValueError("Validation seasons yielded an empty validation set.")

    train_df = df[~validation_mask].copy()
    val_df = df[validation_mask].copy()
    return train_df, val_df


def _compute_classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
) -> Dict[str, float]:
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    proba = proba / proba.sum(axis=1, keepdims=True)
    metrics = {
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, proba[:, 1])),
        "accuracy": float(((proba[:, 1] >= 0.5).astype(int) == y_true).mean()),
    }
    return metrics


def _compute_regression_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        "mae": float(mean_absolute_error(y_true, preds)),
    }
    return metrics


def train_moneyline_models(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "home_win",
    config: SplitConfig | None = None,
) -> list[ModelResult]:
    """Train baseline models predicting home team win probability."""

    if config is None:
        raise ValueError("SplitConfig must be provided for time-aware validation.")

    train_df, val_df = time_aware_split(data, config)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype(int)

    results: list[ModelResult] = []

    if lgb is not None:
        lgb_params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 300,
            "verbose": -1,
        }
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="logloss")
        proba = clf.predict_proba(X_val)
        metrics = _compute_classification_metrics(y_val.to_numpy(), proba)
        results.append(
            ModelResult(
                model_name="lightgbm_moneyline",
                model=clf,
                metrics=metrics,
                predictions=pd.DataFrame(
                    {
                        "game_id": val_df["game_id"].values,
                        "pred_win_proba": proba[:, 1],
                        "actual": y_val.values,
                    }
                ),
            )
        )
    else:
        LOGGER.warning("Skipping LightGBM moneyline model; dependency not installed.")

    logistic_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )
    logistic_pipeline.fit(X_train, y_train)
    proba = logistic_pipeline.predict_proba(X_val)
    metrics = _compute_classification_metrics(y_val.to_numpy(), proba)
    results.append(
        ModelResult(
            model_name="logistic_regression_moneyline",
            model=logistic_pipeline,
            metrics=metrics,
            predictions=pd.DataFrame(
                {
                    "game_id": val_df["game_id"].values,
                    "pred_win_proba": proba[:, 1],
                    "actual": y_val.values,
                }
            ),
        )
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train.fillna(0), y_train)
    proba = rf.predict_proba(X_val.fillna(0))
    metrics = _compute_classification_metrics(y_val.to_numpy(), proba)
    results.append(
        ModelResult(
            model_name="random_forest_moneyline",
            model=rf,
            metrics=metrics,
            predictions=pd.DataFrame(
                {
                    "game_id": val_df["game_id"].values,
                    "pred_win_proba": proba[:, 1],
                    "actual": y_val.values,
                }
            ),
        )
    )

    return results


def train_total_models(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "total_points",
    config: SplitConfig | None = None,
) -> list[ModelResult]:
    """Train baseline models predicting total combined score."""

    if config is None:
        raise ValueError("SplitConfig must be provided for time-aware validation.")

    train_df, val_df = time_aware_split(data, config)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    results: list[ModelResult] = []

    if lgb is not None:
        lgb_params = {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "n_estimators": 500,
            "verbose": -1,
        }
        reg = lgb.LGBMRegressor(**lgb_params)
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse")
        preds = reg.predict(X_val)
        residuals = y_val.to_numpy() - preds
        residual_std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else float(np.std(residuals))
        metrics = _compute_regression_metrics(y_val.to_numpy(), preds)
        metrics["residual_std"] = residual_std
        results.append(
            ModelResult(
                model_name="lightgbm_totals",
                model=reg,
                metrics=metrics,
                predictions=pd.DataFrame(
                    {
                        "game_id": val_df["game_id"].values,
                        "pred_total_points": preds,
                        "actual": y_val.values,
                    }
                ),
            )
        )
    else:
        LOGGER.warning("Skipping LightGBM totals model; dependency not installed.")

    linear = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    )
    linear.fit(X_train, y_train)
    preds = linear.predict(X_val)
    residuals = y_val.to_numpy() - preds
    residual_std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else float(np.std(residuals))
    metrics = _compute_regression_metrics(y_val.to_numpy(), preds)
    metrics["residual_std"] = residual_std
    results.append(
        ModelResult(
            model_name="linear_regression_totals",
            model=linear,
            metrics=metrics,
            predictions=pd.DataFrame(
                {
                    "game_id": val_df["game_id"].values,
                    "pred_total_points": preds,
                    "actual": y_val.values,
                }
            ),
        )
    )

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train.fillna(0), y_train)
    preds = rf.predict(X_val.fillna(0))
    residuals = y_val.to_numpy() - preds
    residual_std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else float(np.std(residuals))
    metrics = _compute_regression_metrics(y_val.to_numpy(), preds)
    metrics["residual_std"] = residual_std
    results.append(
        ModelResult(
            model_name="random_forest_totals",
            model=rf,
            metrics=metrics,
            predictions=pd.DataFrame(
                {
                    "game_id": val_df["game_id"].values,
                    "pred_total_points": preds,
                    "actual": y_val.values,
                }
            ),
        )
    )

    return results


def detect_moneyline_mispricing(
    predictions: pd.DataFrame,
    market_prices: pd.DataFrame,
    game_id_col: str = "game_id",
    market_prob_col: str = "market_prob",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Compare model win probabilities to market-implied probabilities."""

    merged = predictions.merge(market_prices, on=game_id_col, how="inner", suffixes=("_model", "_market"))
    merged["edge"] = merged["pred_win_proba"] - merged[market_prob_col]
    filtered = merged[np.abs(merged["edge"]) >= threshold].copy()
    filtered.sort_values("edge", ascending=False, inplace=True)
    return filtered


def detect_total_mispricing(
    predictions: pd.DataFrame,
    market_totals: pd.DataFrame,
    game_id_col: str = "game_id",
    market_total_col: str = "market_total",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Compare model total predictions to market totals in points."""

    merged = predictions.merge(market_totals, on=game_id_col, how="inner", suffixes=("_model", "_market"))
    merged["edge"] = merged["pred_total_points"] - merged[market_total_col]
    filtered = merged[np.abs(merged["edge"]) >= threshold].copy()
    filtered.sort_values("edge", ascending=False, inplace=True)
    return filtered
