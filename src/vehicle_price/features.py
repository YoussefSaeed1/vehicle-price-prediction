"""Leak-free preprocessing built with scikit-learn transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    """Multi-hot encode a comma-separated column.

    "luxury,high-performance" -> luxury=1, high-performance=1.
    The original notebook one-hot encoded the raw string, which turned
    every *combination* into its own column (71 of them) instead of the
    10 real market categories.
    """

    def __init__(self, sep: str = ","):
        self.sep = sep

    @staticmethod
    def _as_series(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        return pd.Series(np.asarray(X).ravel())

    def _split(self, X) -> list[set[str]]:
        s = self._as_series(X)
        return [
            set() if pd.isna(v) else {t.strip() for t in str(v).split(self.sep) if t.strip()}
            for v in s
        ]

    def fit(self, X, y=None):
        labels = set().union(*self._split(X))
        self.classes_ = np.array(sorted(labels))
        return self

    def transform(self, X):
        rows = self._split(X)
        out = np.zeros((len(rows), len(self.classes_)), dtype=float)
        index = {c: i for i, c in enumerate(self.classes_)}
        for r, tags in enumerate(rows):
            for t in tags:
                if t in index:
                    out[r, index[t]] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        prefix = "market"
        return np.array([f"{prefix}_{c}" for c in self.classes_], dtype=object)


def build_preprocessor(scale_numeric: bool, include_model_name: bool = False) -> ColumnTransformer:
    """Column-wise preprocessing.

    scale_numeric       -> standardize numeric features (linear models only).
    include_model_name  -> one-hot the free-text `model` column as well
                           (rare names grouped into an 'infrequent' bucket).
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=5, sparse_output=False)),
    ])

    transformers = [
        ("num", Pipeline(numeric_steps), config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
        ("market", MultiLabelEncoder(), [config.MULTILABEL_FEATURE]),
    ]
    if include_model_name:
        model_name = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=5, sparse_output=False)),
        ])
        transformers.append(("model_name", model_name, [config.MODEL_NAME_FEATURE]))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def feature_columns(include_model_name: bool = False) -> list[str]:
    cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [config.MULTILABEL_FEATURE]
    if include_model_name:
        cols.append(config.MODEL_NAME_FEATURE)
    return cols
