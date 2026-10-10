"""Train, compare and save models.

Usage:
    python -m vehicle_price.train
"""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config, data, evaluate, models, validate
from .features import feature_columns


def split(df: pd.DataFrame):
    return train_test_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)


def unseen_model_split(df: pd.DataFrame):
    """Every car model (e.g. 'camry') lands entirely in train OR test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    tr, te = next(gss.split(df, groups=df[config.GROUP_COLUMN]))
    return df.iloc[tr], df.iloc[te]


def fit_eval(name: str, train: pd.DataFrame, test: pd.DataFrame, include_model_name: bool = False):
    cols = feature_columns(include_model_name)
    est = models.make_pipeline(name, include_model_name)
    t0 = time.time()
    est.fit(train[cols], train[config.TARGET])
    metrics = evaluate.regression_report(test[config.TARGET], est.predict(test[cols]))
    metrics["fit_seconds"] = round(time.time() - t0, 1)
    return est, metrics


def main() -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    # 0) Validate the raw data before anything is trained on it.
    raw = data.load_raw()
    report = validate.assert_valid(raw)
    with open(config.REPORTS_DIR / "data_validation.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Data validation passed ({len(report.warnings)} warnings, see reports/data_validation.json)")

    df = data.clean(raw)
    train, test = split(df)
    print(f"Rows in scope: {len(df):,}  (train {len(train):,} / test {len(test):,})")

    # 1) Compare all models on the same random hold-out split.
    results, fitted = {}, {}
    for name in models.MODEL_NAMES:
        est, m = fit_eval(name, train, test)
        results[models.PRETTY[name]] = m
        fitted[name] = est
        print(f"{models.PRETTY[name]:<26} R2(log)={m['r2_log']:.4f}  MAE=${m['mae_usd']:,.0f}  "
              f"MAPE={m['mape_pct']:.1f}%")

    comparison = evaluate.format_table(results)
    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv", float_format="%.4f")

    best = (comparison.drop(index=models.PRETTY["baseline"])["mape_pct"].idxmin())
    best_key = {v: k for k, v in models.PRETTY.items()}[best]
    print(f"\nBest model: {best}")

    # 2) Harder test: can the best model price car models it has never seen?
    g_train, g_test = unseen_model_split(df)
    _, unseen = fit_eval(best_key, g_train, g_test)
    print(f"Unseen-model test       R2(log)={unseen['r2_log']:.4f}  MAPE={unseen['mape_pct']:.1f}%")

    # 3) Does adding the free-text model name help? (random split vs unseen split)
    _, with_name_random = fit_eval(best_key, train, test, include_model_name=True)
    _, with_name_unseen = fit_eval(best_key, g_train, g_test, include_model_name=True)

    summary = {
        "rows_in_scope": len(df),
        "train_rows": len(train),
        "test_rows": len(test),
        "best_model": best,
        "random_split": results,
        "unseen_model_split": {best: unseen},
        "with_model_name": {"random_split": with_name_random, "unseen_model_split": with_name_unseen},
    }
    with open(config.REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 4) Refit the best model on all in-scope data and save it.
    final = models.make_pipeline(best_key)
    final.fit(df[feature_columns()], df[config.TARGET])
    joblib.dump(final, config.MODELS_DIR / "vehicle_price_model.joblib")
    print(f"Saved model -> {config.MODELS_DIR / 'vehicle_price_model.joblib'}")


if __name__ == "__main__":
    main()
