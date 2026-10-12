"""Tests for seasonal adjustment, the monthly bias correction and the seasonal index fit."""

import copy

import numpy as np
import pandas as pd
import pytest

from taxiflow.data.download import month_add, month_range
from taxiflow.features.build import FLEX_COL, MONTH_COL
from taxiflow.features.seasonality import fit_seasonal_index


def test_month_add():
    assert month_add("2026-01", -1) == "2025-12"
    assert month_add("2025-11", 2) == "2026-01"
    assert month_add("2026-07", -5) == "2026-02"


def test_seasonal_index_scales_predictions(fare_model, synthetic_data):
    X = synthetic_data[0][:200]
    base = fare_model.predict(X)
    seasonal = copy.deepcopy(fare_model)
    seasonal.seasonal_index = np.ones(13)
    seasonal.seasonal_index[1:] = 1.10  # every month 10% dearer
    assert np.allclose(seasonal.predict(X), base * 1.10)


def test_monthly_correction_removes_channel_bias(fare_model, synthetic_data):
    X, fare, _ = synthetic_data
    flex = X[:, FLEX_COL] > 0
    shifted = fare - np.where(flex, 4.0, 1.0)  # prices fell: $4 for Flex Fare, $1 for metered
    before = fare_model.predict(X) - shifted

    corrected = copy.deepcopy(fare_model)
    offsets = corrected.apply_monthly_correction(X, shifted, "2026-08")
    after = corrected.predict(X) - shifted

    assert offsets["flex_fare"] == pytest.approx(before[flex].mean(), abs=0.15)
    assert offsets["metered"] == pytest.approx(before[~flex].mean(), abs=0.15)
    assert abs(after[flex].mean()) < 0.15 and abs(after[~flex].mean()) < 0.15
    assert corrected.metadata["correction_month"] == "2026-08"
    lower, upper = corrected.conformal_adaptive.predict_interval(corrected.predict(X), scale=corrected.scale(X))
    assert np.mean((shifted >= lower) & (shifted <= upper)) > 0.87


def test_month_column_is_not_a_model_input(fare_model):
    assert "month" not in fare_model.booster.feature_name()
    assert MONTH_COL >= 0


def test_fit_seasonal_index_recovers_pattern(tmp_path):
    """Two synthetic years with a known monthly price factor and a different trip mix."""
    true = {m: 1.0 + 0.05 * np.sin(2 * np.pi * m / 12) for m in range(1, 13)}
    rng = np.random.default_rng(0)
    for month in month_range("2024-01", "2025-11"):
        m = int(month[5:])
        n = 3000
        long_trips = rng.random(n) < (0.2 + 0.3 * (m in (6, 7, 8)))  # summer: more long trips
        pu = np.where(long_trips, 132, 161)
        base = np.where(long_trips, 70.0, 15.0)
        df = pd.DataFrame({
            "pickup_datetime": pd.Timestamp(f"{month}-10 12:00"),
            "pu_zone": pu.astype("int16"),
            "do_zone": np.full(n, 230, dtype="int16"),
            "is_flex": rng.random(n) < 0.2,
            "target": (base * true[m] * rng.uniform(0.98, 1.02, n)).astype("float32"),
        })
        df.to_parquet(tmp_path / f"clean_{month}.parquet")

    index = fit_seasonal_index(tmp_path, ("2024-01", "2025-11"))
    expected = np.array([true[m] for m in range(1, 13)])
    expected /= expected.mean()
    assert np.allclose(index[1:], expected, atol=0.005)  # trip mix does not leak into the index


def test_fit_seasonal_index_requires_full_year(tmp_path):
    with pytest.raises(ValueError):
        fit_seasonal_index(tmp_path, ("2025-01", "2025-06"))
