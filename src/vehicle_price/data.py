"""Loading and cleaning the raw Kaggle car dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_raw(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded from Kaggle."""
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """'Engine HP' -> 'engine_hp', 'city mpg' -> 'city_mpg', etc."""
    out = df.copy()
    out.columns = out.columns.str.strip().str.replace(" ", "_").str.lower()
    return out


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and trim every text column; keep missing values as NaN."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].str.strip().str.lower()
    # "N/A" in market_category means "no category", not "unknown".
    if "market_category" in out:
        out["market_category"] = out["market_category"].replace("n/a", np.nan)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age"] = config.REFERENCE_YEAR - out["year"]
    return out


def clean(df: pd.DataFrame, apply_scope: bool = True) -> pd.DataFrame:
    """Full cleaning pipeline used everywhere in the project.

    Steps
    -----
    1. Standardize column names.
    2. Normalize text values (case / whitespace).
    3. Drop exact duplicate rows (after normalization).
    4. Add engineered features (age).
    5. Optionally restrict to the modelling scope:
       model year >= MIN_YEAR and price < MAX_PRICE.

    Missing values are *not* imputed here: imputation lives inside the
    sklearn pipeline so it is learned from the training split only.
    """
    out = standardize_columns(df)
    out = normalize_text(out)
    out = out.drop_duplicates().reset_index(drop=True)
    out = add_features(out)
    if apply_scope:
        mask = (out["year"] >= config.MIN_YEAR) & (out[config.TARGET] < config.MAX_PRICE)
        out = out.loc[mask].reset_index(drop=True)
    return out


def load_clean(path: str | Path = config.DATA_PATH, apply_scope: bool = True) -> pd.DataFrame:
    return clean(load_raw(path), apply_scope=apply_scope)
