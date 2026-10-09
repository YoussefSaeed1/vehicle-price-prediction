"""Project-wide constants: paths, feature lists and scope rules."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "data.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 40
TEST_SIZE = 0.2

TARGET = "msrp"

# Dataset was scraped in 2017, so age is measured against that year.
REFERENCE_YEAR = 2017

# Scope of the model (see README > "Data quality findings").
MIN_YEAR = 2001          # pre-2001 rows carry a ~$2,000 placeholder price, not a real MSRP
MAX_PRICE = 500_000      # hypercars (Bugatti, etc.) are too few to learn from

NUMERIC_FEATURES = [
    "age",
    "engine_hp",
    "engine_cylinders",
    "number_of_doors",
    "highway_mpg",
    "city_mpg",
    "popularity",
]

CATEGORICAL_FEATURES = [
    "make",
    "engine_fuel_type",
    "transmission_type",
    "driven_wheels",
    "vehicle_size",
    "vehicle_style",
]

# Free-text model name: high cardinality, used only by the "with model name" variant.
MODEL_NAME_FEATURE = "model"

# Comma-separated multi-label column, e.g. "luxury,high-performance".
MULTILABEL_FEATURE = "market_category"

# Column used to build groups for the "unseen car model" evaluation.
GROUP_COLUMN = "model"
