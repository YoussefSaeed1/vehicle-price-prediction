"""Predict the MSRP of one or more vehicles with the saved model.

Usage:
    python -m vehicle_price.predict --json '{"make": "bmw", "year": 2016, "engine_hp": 300, ...}'
    python -m vehicle_price.predict --csv new_cars.csv
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import add_features, normalize_text, standardize_columns
from .features import feature_columns

MODEL_PATH = config.MODELS_DIR / "vehicle_price_model.joblib"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same column/text cleaning as training; missing fields become NaN
    and are imputed by the pipeline."""
    out = normalize_text(standardize_columns(df))
    if "age" not in out and "year" in out:
        out = add_features(out)
    for col in feature_columns():
        if col not in out:
            out[col] = np.nan
    return out[feature_columns()]


def predict(df: pd.DataFrame, model_path=MODEL_PATH) -> np.ndarray:
    model = joblib.load(model_path)
    return model.predict(prepare(df))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="One vehicle as a JSON object")
    src.add_argument("--csv", help="CSV file with one vehicle per row")
    args = p.parse_args()

    df = pd.DataFrame([json.loads(args.json)]) if args.json else pd.read_csv(args.csv)
    for i, price in enumerate(predict(df)):
        print(f"Vehicle {i + 1}: estimated MSRP ${price:,.0f}")


if __name__ == "__main__":
    main()
