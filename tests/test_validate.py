import pandas as pd
import pytest

from vehicle_price import data, validate


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


def by_name(report, name):
    return next(c for c in report.checks if c.name.startswith(name))


def test_real_dataset_passes_with_known_warnings(raw):
    report = validate.check_training_data(raw)
    assert report.ok
    assert by_name(report, "no exact duplicate rows").failing_rows == 715
    assert by_name(report, "engine_hp has no missing values").failing_rows == 69
    # Audi A6 2017 lists highway_mpg = 354, almost certainly a typo for 34.
    hwy = by_name(report, "highway_mpg within")
    assert hwy.failing_rows == 1 and hwy.examples[0]["model"] == "a6"
    assert by_name(report, "transmission_type uses known values").failing_rows == 19


def test_missing_column_is_an_error(raw):
    report = validate.check_training_data(raw.drop(columns=["MSRP"]))
    assert not report.ok
    assert "msrp" in report.errors[0].detail
    with pytest.raises(validate.DataValidationError):
        validate.assert_valid(raw.drop(columns=["MSRP"]))


def test_bad_target_and_year_are_errors(raw):
    bad = raw.head(50).copy()
    bad.loc[0, "MSRP"] = 0
    bad.loc[1, "Year"] = 2030
    report = validate.check_training_data(bad)
    names = {c.name.split(" ")[0] for c in report.errors}
    assert {"msrp", "year"} <= names


def test_text_in_numeric_column_is_an_error(raw):
    bad = raw.head(20).copy()
    bad["Engine HP"] = bad["Engine HP"].astype(str)
    assert not validate.check_training_data(bad).ok


def test_unknown_category_is_a_warning_not_an_error(raw):
    bad = raw.head(20).copy()
    bad.loc[0, "Vehicle Size"] = "Huge"
    report = validate.check_training_data(bad)
    assert report.ok
    assert by_name(report, "vehicle_size uses known values").failing_rows == 1


def test_check_inputs_flags_out_of_scope_cars():
    good = pd.DataFrame([{"Make": "Toyota", "Year": 2017, "Engine HP": 178, "Engine Cylinders": 4,
                          "Transmission Type": "AUTOMATIC", "Driven_Wheels": "front wheel drive",
                          "Number of Doors": 4, "Vehicle Size": "Midsize", "Vehicle Style": "Sedan",
                          "highway MPG": 33, "city mpg": 24, "Popularity": 2031,
                          "Engine Fuel Type": "regular unleaded"}])
    assert validate.check_inputs(good) == []

    odd = pd.DataFrame([{"Make": "Toyota", "Year": 1995, "Engine HP": 5000, "Vehicle Size": "Huge"}])
    msgs = " | ".join(validate.check_inputs(odd))
    assert "outside the trained range" in msgs
    assert "engine_hp=5000" in msgs
    assert "vehicle_size='huge'" in msgs
    assert "missing and imputed" in msgs
