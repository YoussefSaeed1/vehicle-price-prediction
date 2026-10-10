"""Data validation for training data and prediction inputs.

Two entry points:

* ``check_training_data(raw)`` runs on the raw Kaggle CSV before cleaning.
  Errors stop training; warnings are reported and kept in
  ``reports/data_validation.json``.
* ``check_inputs(df)`` runs on new cars passed to ``predict``. It never blocks a
  prediction, but warns when an input is outside what the model was trained on.

Usage:
    python -m vehicle_price.validate            # validate data/data.csv, write the report
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config
from .data import load_raw, normalize_text, standardize_columns

# ---------------------------------------------------------------------------
# Expectations
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "make", "model", "year", "engine_fuel_type", "engine_hp", "engine_cylinders",
    "transmission_type", "driven_wheels", "number_of_doors", "market_category",
    "vehicle_size", "vehicle_style", "highway_mpg", "city_mpg", "popularity", config.TARGET,
]

NUMERIC_COLUMNS = ["year", "engine_hp", "engine_cylinders", "number_of_doors",
                   "highway_mpg", "city_mpg", "popularity", config.TARGET]

# Plausible physical ranges (inclusive). Values outside are flagged, not dropped.
RANGES = {
    "year": (1990, config.REFERENCE_YEAR),
    "engine_hp": (50, 1_200),
    "engine_cylinders": (0, 16),
    "number_of_doors": (2, 4),
    "highway_mpg": (5, 150),
    "city_mpg": (5, 150),
    "popularity": (0, 10_000),
}

ALLOWED_VALUES = {
    "engine_fuel_type": {
        "diesel", "electric", "natural gas", "regular unleaded",
        "premium unleaded (recommended)", "premium unleaded (required)",
        "flex-fuel (premium unleaded recommended/e85)", "flex-fuel (premium unleaded required/e85)",
        "flex-fuel (unleaded/e85)", "flex-fuel (unleaded/natural gas)",
    },
    "transmission_type": {"automated_manual", "automatic", "direct_drive", "manual"},
    "driven_wheels": {"all wheel drive", "four wheel drive", "front wheel drive", "rear wheel drive"},
    "vehicle_size": {"compact", "midsize", "large"},
    "vehicle_style": {
        "2dr hatchback", "2dr suv", "4dr hatchback", "4dr suv", "cargo minivan", "cargo van",
        "convertible", "convertible suv", "coupe", "crew cab pickup", "extended cab pickup",
        "passenger minivan", "passenger van", "regular cab pickup", "sedan", "wagon",
    },
}

MARKET_TAGS = {
    "crossover", "diesel", "exotic", "factory tuner", "flex fuel", "hatchback",
    "high-performance", "hybrid", "luxury", "performance",
}

# Prices at or below this, on cars from before MIN_YEAR, are the dataset's placeholder value.
PLACEHOLDER_PRICE = 2_500


# ---------------------------------------------------------------------------
# Report objects
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    severity: str          # "error" | "warning" | "info"
    passed: bool
    failing_rows: int = 0
    detail: str = ""
    examples: list = field(default_factory=list)


@dataclass
class ValidationReport:
    rows: int
    checks: list[Check]

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "error" and not c.passed]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "warning" and not c.passed]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "passed": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "checks": [asdict(c) for c in self.checks],
        }

    def summary(self) -> str:
        lines = [f"Rows checked: {self.rows:,}"]
        for c in self.checks:
            mark = "PASS" if c.passed else c.severity.upper()
            rows = f" ({c.failing_rows:,} rows)" if c.failing_rows else ""
            lines.append(f"  [{mark:<7}] {c.name}{rows}" + (f" - {c.detail}" if c.detail else ""))
        lines.append("Result: " + ("PASSED" if self.ok else f"FAILED with {len(self.errors)} error(s)"))
        return "\n".join(lines)


class DataValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("Data validation failed:\n" + report.summary())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _examples(df: pd.DataFrame, mask: pd.Series, cols: list[str], n: int = 3) -> list:
    keep = [c for c in dict.fromkeys(["make", "model", "year", *cols]) if c in df]
    rows = df.loc[mask, keep].drop_duplicates().head(n)
    return json.loads(rows.to_json(orient="records"))


def _range_check(df: pd.DataFrame, col: str, severity: str) -> Check:
    lo, hi = RANGES[col]
    values = pd.to_numeric(df[col], errors="coerce")
    mask = values.notna() & ((values < lo) | (values > hi))
    return Check(f"{col} within [{lo:,} - {hi:,}]", severity, not mask.any(), int(mask.sum()),
                 examples=_examples(df, mask, [col]))


def _allowed_check(df: pd.DataFrame, col: str, severity: str) -> Check:
    values = df[col]
    mask = values.notna() & ~values.isin(ALLOWED_VALUES[col])
    found = sorted(values[mask].unique().tolist())
    return Check(f"{col} uses known values", severity, not mask.any(), int(mask.sum()),
                 detail=("unexpected: " + ", ".join(map(str, found[:5]))) if found else "")


def _unknown_market_tags(series: pd.Series) -> pd.Series:
    def bad(v):
        if pd.isna(v):
            return False
        return any(t.strip() not in MARKET_TAGS for t in str(v).split(","))
    return series.map(bad).astype(bool)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def check_training_data(raw: pd.DataFrame) -> ValidationReport:
    """Validate the raw CSV (original column names are fine)."""
    df = normalize_text(standardize_columns(raw))
    checks: list[Check] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df]
    checks.append(Check("required columns present", "error", not missing_cols,
                        detail=", ".join(missing_cols)))
    checks.append(Check("dataset is not empty", "error", len(df) > 0))
    if missing_cols or df.empty:
        return ValidationReport(len(df), checks)

    bad_types = [c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    checks.append(Check("numeric columns are numeric", "error", not bad_types, detail=", ".join(bad_types)))
    if bad_types:
        return ValidationReport(len(df), checks)

    # Target and keys: must be complete and valid, or training stops.
    target = df[config.TARGET]
    mask = target.isna() | (target <= 0)
    checks.append(Check("msrp present and > 0", "error", not mask.any(), int(mask.sum())))
    mask = df["make"].isna() | df["year"].isna()
    checks.append(Check("make and year present", "error", not mask.any(), int(mask.sum())))
    checks.append(_range_check(df, "year", "error"))

    # Duplicates are dropped by cleaning; reported so the count is visible.
    dups = int(df.duplicated().sum())
    checks.append(Check("no exact duplicate rows", "warning", dups == 0, dups, detail="dropped during cleaning"))

    # Missing values in model features (imputed inside the pipeline).
    for col in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
        if col == "age":
            continue
        n = int(df[col].isna().sum())
        checks.append(Check(f"{col} has no missing values", "warning", n == 0, n,
                            detail="imputed by the pipeline" if n else ""))

    # Plausible ranges.
    for col in RANGES:
        if col != "year":
            checks.append(_range_check(df, col, "warning"))

    # Engine consistency: only electric cars should report 0 cylinders.
    mask = (df["engine_cylinders"] == 0) & (df["engine_fuel_type"] != "electric")
    checks.append(Check("0 cylinders only on electric cars", "warning", not mask.any(), int(mask.sum()),
                        examples=_examples(df, mask, ["engine_fuel_type"])))

    # Categorical domains.
    for col in ALLOWED_VALUES:
        checks.append(_allowed_check(df, col, "warning"))
    mask = _unknown_market_tags(df["market_category"])
    checks.append(Check("market_category tags are known", "warning", not mask.any(), int(mask.sum())))

    # Popularity should be a property of the make.
    per_make = df.groupby("make")["popularity"].nunique()
    bad_makes = per_make[per_make > 1].index.tolist()
    checks.append(Check("one popularity value per make", "warning", not bad_makes, len(bad_makes),
                        detail=", ".join(bad_makes[:5])))

    # Scope facts (information only; the scope rules in config remove these rows).
    mask = (df["year"] < config.MIN_YEAR) & (target <= PLACEHOLDER_PRICE)
    checks.append(Check(f"placeholder prices (year < {config.MIN_YEAR}, msrp <= ${PLACEHOLDER_PRICE:,})",
                        "info", True, int(mask.sum()), detail="out of scope"))
    mask = target >= config.MAX_PRICE
    checks.append(Check(f"msrp >= ${config.MAX_PRICE:,}", "info", True, int(mask.sum()), detail="out of scope"))

    return ValidationReport(len(df), checks)


def assert_valid(raw: pd.DataFrame) -> ValidationReport:
    """Run the training checks and raise if any error-level check fails."""
    report = check_training_data(raw)
    if not report.ok:
        raise DataValidationError(report)
    return report


# ---------------------------------------------------------------------------
# Prediction inputs
# ---------------------------------------------------------------------------

def check_inputs(df: pd.DataFrame) -> list[str]:
    """Warnings for each input row that is outside the model's training scope.

    Never raises: missing fields are imputed and unseen categories fall into the
    'infrequent' bucket, but the user should know the estimate is less reliable.
    """
    d = normalize_text(standardize_columns(df))
    messages: list[str] = []
    for i, row in d.reset_index(drop=True).iterrows():
        tag = f"Vehicle {i + 1}"
        year = row.get("year", np.nan)
        if pd.isna(year) and pd.isna(row.get("age", np.nan)):
            messages.append(f"{tag}: no year given; age will be imputed")
        elif not pd.isna(year) and not (config.MIN_YEAR <= year <= config.REFERENCE_YEAR):
            messages.append(f"{tag}: year {int(year)} is outside the trained range "
                            f"{config.MIN_YEAR}-{config.REFERENCE_YEAR}")
        for col, (lo, hi) in RANGES.items():
            if col == "year" or col not in d:
                continue
            v = pd.to_numeric(row[col], errors="coerce")
            if not pd.isna(v) and not (lo <= v <= hi):
                messages.append(f"{tag}: {col}={v:g} is outside the plausible range {lo:,}-{hi:,}")
        for col, allowed in ALLOWED_VALUES.items():
            v = row.get(col, np.nan)
            if isinstance(v, str) and v not in allowed | {"unknown"}:
                messages.append(f"{tag}: {col}='{v}' was not seen in training")
        missing = [c for c in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
                   if c != "age" and pd.isna(row.get(c, np.nan))]
        if missing:
            messages.append(f"{tag}: {len(missing)} field(s) missing and imputed: {', '.join(missing)}")
    return messages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    report = check_training_data(load_raw())
    print(report.summary())
    config.REPORTS_DIR.mkdir(exist_ok=True)
    out = config.REPORTS_DIR / "data_validation.json"
    with open(out, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Report -> {out}")
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
