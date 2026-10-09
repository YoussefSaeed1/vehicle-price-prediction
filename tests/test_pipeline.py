import numpy as np
import pandas as pd
import pytest

from vehicle_price import config, data, models
from vehicle_price.features import MultiLabelEncoder, feature_columns
from vehicle_price.predict import prepare


@pytest.fixture(scope="module")
def df():
    return data.load_clean()


def test_cleaning_removes_duplicates_and_applies_scope(df):
    assert not df.duplicated().any()
    assert df["year"].min() >= config.MIN_YEAR
    assert df["msrp"].max() < config.MAX_PRICE
    assert (df["age"] == config.REFERENCE_YEAR - df["year"]).all()


def test_multilabel_encoder_splits_tags():
    enc = MultiLabelEncoder().fit(pd.DataFrame({"m": ["luxury,performance", None, "crossover"]}))
    out = enc.transform(pd.DataFrame({"m": ["performance,luxury", "unknown-tag", np.nan]}))
    assert list(enc.classes_) == ["crossover", "luxury", "performance"]
    assert out.tolist() == [[0, 1, 1], [0, 0, 0], [0, 0, 0]]


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_every_model_trains_and_predicts_dollars(df, name):
    sample = df.sample(600, random_state=0)
    est = models.make_pipeline(name, n_estimators=20, max_iter=50)
    est.fit(sample[feature_columns()], sample["msrp"])
    pred = est.predict(sample[feature_columns()].head(20))
    assert pred.shape == (20,)
    assert (pred > 1_000).all()


def test_prepare_accepts_raw_column_names_and_missing_fields():
    raw = pd.DataFrame([{"Make": "BMW", "Year": 2016, "Engine HP": 300}])
    X = prepare(raw)
    assert list(X.columns) == feature_columns()
    assert X.loc[0, "make"] == "bmw"
    assert X.loc[0, "age"] == 1
