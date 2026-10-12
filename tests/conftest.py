"""Shared fixtures: a small FareModel trained on synthetic trips (no real data needed)."""

import lightgbm as lgb
import numpy as np
import pytest

from taxiflow.features.build import (
    CATEGORICAL,
    MODEL_COLS,
    MODEL_FEATURES,
    RouteStats,
    build_features,
)
from taxiflow.features.geospatial import load_zone_table, manhattan_distance_km
from taxiflow.models.conformal import ConformalPredictor
from taxiflow.models.train import FareModel


def synthetic_trips(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    zones = load_zone_table()
    pu, do = rng.integers(1, 264, n), rng.integers(1, 264, n)
    pickup = np.datetime64("2026-01-01T00:00") + rng.integers(0, 60 * 24 * 90, n).astype("timedelta64[m]")
    is_flex = rng.random(n) < 0.25
    km = manhattan_distance_km(zones.lat[pu], zones.lon[pu], zones.lat[do], zones.lon[do]) + 0.5
    minutes = km * 3.0 + rng.normal(0, 2, n).clip(-1, None) + 2
    fare = 3.0 + 1.8 * km + 0.5 * minutes + 2.5 * zones.in_cbd[pu] - 3.0 * is_flex + rng.normal(0, 2, n)
    return pu, do, pickup, is_flex, km, minutes, fare


def _train(X: np.ndarray, y: np.ndarray, rounds: int) -> lgb.Booster:
    params = {"objective": "regression", "num_leaves": 15, "verbose": -1, "seed": 0}
    ds = lgb.Dataset(X[:, MODEL_COLS], y, feature_name=MODEL_FEATURES, categorical_feature=CATEGORICAL)
    return lgb.train(params, ds, num_boost_round=rounds)


@pytest.fixture(scope="session")
def synthetic_data():
    pu, do, pickup, is_flex, km, minutes, fare = synthetic_trips(20_000)
    routes = RouteStats.fit(pu, do, km, minutes)
    return build_features(pu, do, pickup, is_flex, routes), fare, routes


@pytest.fixture
def fare_model(synthetic_data) -> FareModel:
    X, fare, routes = synthetic_data
    booster = _train(X, fare, 50)
    model = FareModel(booster, routes, None, ConformalPredictor(0.9), ConformalPredictor(0.9),
                      {"train_window": ["synthetic", "synthetic"]})
    model.width_booster = _train(X, np.abs(fare - model.predict(X)), 20)
    model.calibrate(X, fare)
    return model
