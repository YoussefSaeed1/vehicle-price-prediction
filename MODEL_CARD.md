# Model Card: Vehicle MSRP Estimator

| | |
|---|---|
| **Model** | Histogram Gradient Boosting regressor (scikit-learn), trained on `log(1 + price)` |
| **Version** | 1.0.0 |
| **Owner** | Youssef Saeed |
| **Code** | `src/vehicle_price/` · train with `python -m vehicle_price.train` |
| **Metrics source** | [`reports/metrics.json`](reports/metrics.json), [`reports/data_validation.json`](reports/data_validation.json) |
| **License** | MIT |

## Intended use

**Use it for:** a quick estimate of a new car's **US list price (MSRP)** from its specifications, such as checking whether a list price is in a normal range for its specs, or portfolio and teaching examples of a leak-free ML pipeline.

**Don't use it for:**
- Used-car or resale prices. The data is manufacturer list prices, not transaction prices.
- Cars from before model year 2001, after 2017, or priced at $500K or more. These are outside the training scope.
- Markets outside the US, or prices in other currencies.
- Pricing decisions that carry real financial risk without a person reviewing them.

## Training data

[Car Features and MSRP](https://www.kaggle.com/datasets/cooperunion/cardataset) (Kaggle, Cooper Union), scraped from Edmunds and Twitter in 2017. The raw file has 11,914 rows and 16 columns.

| Step | Rows |
|---|---:|
| Raw file | 11,914 |
| Exact duplicates dropped | −715 |
| Out of scope: model year before 2001 (mostly a ~$2,000 placeholder price) | −1,368 |
| Out of scope: MSRP of $500K or more | −11 |
| **Modelling dataset** | **9,820** |

The modelling dataset covers 47 makes, 737 car models and model years 2001 to 2017. The median MSRP is $33,008 and the range is $9,299 to $497,650. 66 cars are electric.

**Split:** a random 80/20 hold-out (7,856 train / 1,964 test, `random_state=40`). A second split keeps each car model entirely in train or test, to measure performance on models the system has never seen.

## Features

- **Numeric:** age (2017 − year), engine HP, cylinders, doors, highway MPG, city MPG, popularity.
- **Categorical:** make, fuel type, transmission, driven wheels, vehicle size, vehicle style. Categories with fewer than 5 cars are grouped.
- **Multi-label:** market category tags (luxury, performance, exotic, …), multi-hot encoded.
- **Not used:** the free-text model name. It helped the random split slightly but hurt on unseen car models (MAPE 10.2% → 10.7%).

All imputation and encoding live inside the scikit-learn pipeline and are fitted on training rows only.

## Performance

Test set: 1,964 cars, random split.

| Model | MAE | MAPE | Within 10% | R² (log) |
|---|---:|---:|---:|---:|
| Baseline (median price) | $20,796 | 36.8% | 19.5% | −0.02 |
| Ridge Regression | $6,481 | 11.4% | 52.4% | 0.936 |
| Random Forest | $3,316 | 7.5% | 72.4% | 0.973 |
| **Gradient Boosting (selected)** | **$3,097** | **6.9%** | **75.6%** | **0.976** |

**Unseen car models** (group split): MAE $5,668, MAPE 10.2%, 58.7% within 10%, R² (log) 0.957. This is the better guide for a car model that isn't in the dataset.

**By price band** (gradient boosting, random split):

| MSRP band | Cars | MAPE | Within 10% |
|---|---:|---:|---:|
| Under $25K | 469 | 6.5% | 77.4% |
| $25K – $40K | 850 | 7.0% | 75.2% |
| $40K – $75K | 460 | 7.3% | 72.2% |
| $75K – $150K | 108 | 7.2% | 74.1% |
| $150K – $500K | 77 | 5.5% | 90.9% |
| Electric cars only | 12 | 8.5% | 66.7% |

Error is fairly even across price bands, and the median signed error is −0.03%, so the model doesn't consistently over- or under-price. Electric cars are too few in the test set (12) to draw a firm conclusion.

## Data validation

`vehicle_price/validate.py` runs automatically before training, and also on its own with `python -m vehicle_price.validate`. **Errors** stop training; **warnings** are logged to `reports/data_validation.json`.

| Check | Level | Result on `data.csv` |
|---|---|---|
| Required columns present, numeric columns numeric | Error | Pass |
| MSRP present and > 0; make and year present | Error | Pass |
| Model year within 1990 – 2017 | Error | Pass |
| Exact duplicate rows | Warning | 715 rows (dropped) |
| Missing feature values | Warning | Engine HP 69, cylinders 30, doors 6, fuel type 3 (imputed) |
| Plausible ranges (HP, cylinders, doors, MPG, popularity) | Warning | 1 row: Audi A6 2017 with highway MPG 354, almost certainly a typo for 34 |
| 0 cylinders only on electric cars | Warning | Pass |
| Categorical values from the known set | Warning | 19 rows with transmission `unknown` |
| Market category tags known | Warning | Pass |
| One popularity value per make | Warning | Pass |

At prediction time, `check_inputs()` warns, without blocking, when a car is outside the trained year range, has implausible numbers or unseen categories, or has fields that will be imputed.

## Limitations and risks

- **List price, not market price.** Discounts, dealer markups, options and regional pricing are not in the data.
- **Trims look alike.** Trims of the same model often share specs, so the model can't see the packages that separate their prices.
- **Data is from 2017.** Prices have changed since then (inflation, new EV pricing), so estimates for today's cars will be off. Retrain on current data before real use.
- **Popularity is a make-level proxy.** It has exactly one value per make, so it adds nothing beyond `make`.
- **Known data issue left in place.** The Audi A6 highway MPG value (354) is flagged but not corrected, so the published metrics stay reproducible.
- **Small segments.** Electric cars and hypercars have few examples, so estimates for them are less reliable.

## Fairness

The model prices cars, not people, and uses no personal data. The main risk is uneven accuracy across segments (above), not harm to individuals.

## Maintenance

Retrain with `python -m vehicle_price.train`. It re-runs validation, compares all models, rewrites `reports/metrics.json`, and saves the best model to `models/`. Tests: `pytest` (pipeline and validation).
