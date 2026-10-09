"""Model zoo. Every model predicts log1p(price) and is wrapped so that
`.predict()` returns prices in dollars."""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor


def _log_target(regressor) -> TransformedTargetRegressor:
    """Train on log1p(price), predict back in dollars via expm1."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def make_pipeline(name: str, include_model_name: bool = False, **params) -> TransformedTargetRegressor:
    rs = config.RANDOM_STATE
    if name == "baseline":
        reg, scale = DummyRegressor(strategy="median"), False
    elif name == "ridge":
        reg, scale = Ridge(alpha=params.get("alpha", 1.0)), True
    elif name == "random_forest":
        reg, scale = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_features=params.get("max_features", 0.5),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=rs, n_jobs=-1,
        ), False
    elif name == "gradient_boosting":
        reg, scale = HistGradientBoostingRegressor(
            learning_rate=params.get("learning_rate", 0.05),
            max_iter=params.get("max_iter", 1500),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            min_samples_leaf=params.get("min_samples_leaf", 10),
            l2_regularization=params.get("l2_regularization", 0.0),
            random_state=rs,
        ), False
    else:
        raise ValueError(f"Unknown model: {name}")

    pipe = Pipeline([
        ("prep", build_preprocessor(scale_numeric=scale, include_model_name=include_model_name)),
        ("model", reg),
    ])
    return _log_target(pipe)


MODEL_NAMES = ["baseline", "ridge", "random_forest", "gradient_boosting"]

PRETTY = {
    "baseline": "Baseline (median price)",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}
