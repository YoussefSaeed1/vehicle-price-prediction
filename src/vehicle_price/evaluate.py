"""Metrics in both log space (comparable with the original project) and dollars."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_report(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lt, lp = np.log1p(y_true), np.log1p(y_pred)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "mae_usd": mean_absolute_error(y_true, y_pred),
        "rmse_usd": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": 100 * float(np.mean(ape)),
        "median_ape_pct": 100 * float(np.median(ape)),
        "within_10pct": 100 * float(np.mean(ape <= 0.10)),
        "r2_usd": r2_score(y_true, y_pred),
        "r2_log": r2_score(lt, lp),
        "rmse_log": float(np.sqrt(mean_squared_error(lt, lp))),
    }


def cv_scorers() -> dict:
    """Scorers for cross_validate (estimators predict dollars)."""
    from sklearn.metrics import make_scorer

    def r2_log(y, p):
        return r2_score(np.log1p(y), np.log1p(np.clip(p, 0, None)))

    def mape(y, p):
        return 100 * float(np.mean(np.abs(p - y) / y))

    return {
        "r2_log": make_scorer(r2_log),
        "mape_pct": make_scorer(mape, greater_is_better=False),
        "mae_usd": "neg_mean_absolute_error",
    }


def summarize_cv(results: dict) -> dict[str, float]:
    return {
        "r2_log": float(np.mean(results["test_r2_log"])),
        "r2_log_std": float(np.std(results["test_r2_log"])),
        "mape_pct": float(-np.mean(results["test_mape_pct"])),
        "mae_usd": float(-np.mean(results["test_mae_usd"])),
    }


def format_table(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows).T
