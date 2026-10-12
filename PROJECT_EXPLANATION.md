# Taxiflow: NYC Taxi Fare Prediction Under Pricing Drift — Project Explanation

This document explains the design choices behind Taxiflow. Every result is in the [README](README.md) and is generated from [`reports/metrics.json`](reports/metrics.json) and [`reports/walkforward.json`](reports/walkforward.json); this file does not repeat numbers.

## 1. Design principles

**Only booking-time information.** The fare has to be estimated before the trip, so realised distance and duration cannot be inputs. The model sees *historical* route statistics instead: the median distance, median duration and trip count of the zone pair, computed from the training window only. A test ([`tests/test_features.py`](tests/test_features.py)) fails if a post-trip column becomes a feature.

**Out-of-time evaluation.** Training, validation, calibration and test are separate consecutive months. Training samples rows uniformly within each month; the test months are scored on every trip.

**Zones as places, not numbers.** TLC zone IDs (1–263) are labels, so they are LightGBM categorical features, and geography enters through coordinates: all 263 zone polygons from the official TLC shapefile are reduced to centroids, which give haversine and Manhattan distances and bearings ([`scripts/build_zone_table.py`](scripts/build_zone_table.py)).

**Out-of-core data processing.** DuckDB cleans each monthly parquet file as a stream, so 119.6M rows are processed with flat memory on a laptop. The served model answers a single request in under 1 ms at the 95th percentile (measured in the README).

**Uncertainty and change over time.** Every prediction comes with a conformal interval whose coverage is measured on the test months (section 4), and pricing changes are handled explicitly (section 3).

## 2. Pipeline

| Step | Command | Output |
|---|---|---|
| Download 32 months of Yellow taxi data from the TLC CDN | `taxiflow download` | `data/raw/` (~1.9 GB) |
| Clean with DuckDB (filters + known-bad records from the config) | `taxiflow clean` | `data/processed/`, `cleaning_report.json` |
| Zone centroids and congestion-zone flags | `python scripts/build_zone_table.py` | `src/taxiflow/resources/zone_table.csv` |
| Train the current-regime model, two drift models and two baselines | `taxiflow train` | `models/` |
| Evaluate on all test trips | `taxiflow evaluate` | `reports/metrics.json` |
| Compare monthly maintenance strategies | `taxiflow walkforward` | `reports/walkforward.json` |
| Apply last month's correction for serving | `taxiflow correct --month 2026-08` | `models/taxiflow_lgbm_serving.joblib` |
| Figures and README tables | `python scripts/make_report.py` | `docs/figures/`, README |

[`scripts/reproduce.sh`](scripts/reproduce.sh) runs all of them in order. The target is the total charge excluding the tip, because tips are only recorded for card payments.

## 3. Pricing changes and how the model handles them

The cleaned data contains three events, all found by analysing the data and described with figures in the README:

1. **MTA congestion fee (5 Jan 2025).** The congestion zone is derived from the data (the share of pickups per zone carrying `cbd_congestion_fee`) rather than hard-coded.
2. **Corrupt Flex Fare records (Nov 2025, one vendor).** Excluded through `data.known_bad_records` in [`configs/default.yaml`](configs/default.yaml) and counted in the cleaning report.
3. **Flex Fare price step (1 Dec 2025).** App-booked fares rose for both vendors, while metered fares stayed flat.

The response, in order:

- **Train on the current regime only** (Dec 2025 – Apr 2026), and measure the cost of not doing so with two models trained on earlier regimes (the drift study).
- **Model seasonality separately.** A window shorter than a year cannot learn calendar month, so `month` is not a model input. A route-controlled seasonal price index estimated from metered trips in Jan 2024 – Nov 2025 ([`features/seasonality.py`](src/taxiflow/features/seasonality.py)) rescales the target instead.
- **Correct monthly.** Flex Fare prices keep moving within the regime, so the served model subtracts last month's mean error per booking type and recalibrates its intervals on held-out data from the same month ([`FareModel.apply_monthly_correction`](src/taxiflow/models/train.py)). The walk-forward evaluation compares this with leaving the model unchanged and with retraining every month.

## 4. Uncertainty

Split conformal prediction with the finite-sample correction `ceil((n+1)(1-α))/n`. Two variants are evaluated:

- **Fixed width:** `prediction ± q`.
- **Adaptive:** `prediction ± q × s(x)`, where `s(x)` is a second LightGBM model of the expected absolute error, trained on the validation month, which the main model uses only for early stopping, not for fitting.

The coverage guarantee assumes the calibration month resembles the test month. Coverage is therefore reported per segment (Flex Fare, metered, airport, congestion zone), where the two variants differ most.

## 5. Serving

`POST /v1/predict-fare` takes pickup and dropoff zones (1–263), an optional pickup time (NYC local if no timezone is given) and `booking_type` (`street_hail` or `flex_fare`). It returns the point estimate, the adaptive interval, the historical route median distance and duration, and flags for airport and congestion-zone trips. `GET /health` reports the training window and the month used for the correction. The Docker image packages only the serving model.

## 6. Limitations and next steps

- Traffic and the route actually driven are unknown at booking, which bounds the achievable accuracy.
- Flex Fare repricing is followed one month late by the correction. Detecting a shift within the month (for example, from the first week's residuals) would shorten that lag.
- The walk-forward evaluation assumes last month's trips are available immediately; the public TLC release lags by about two months.
- Weather, events and live traffic are not used. Only Yellow taxis are modelled.
