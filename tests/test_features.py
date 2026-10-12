"""Tests for the shared feature builder and historical route statistics."""

import numpy as np
import pytest

from taxiflow.evaluation.metrics import regression_metrics
from taxiflow.features.build import (
    FEATURES,
    MIN_ROUTE_TRIPS,
    RouteStats,
    build_features,
    time_parts,
)
from taxiflow.features.geospatial import load_zone_table


def test_zone_table_covers_all_zones():
    zones = load_zone_table()
    assert np.isfinite(zones.lat[1:]).all() and np.isfinite(zones.lon[1:]).all()
    assert zones.lat[1:].min() > 40.4 and zones.lat[1:].max() < 41.0
    assert zones.in_cbd[230] and zones.in_cbd[161]  # Times Sq, Midtown Center
    assert not zones.in_cbd[236] and not zones.in_cbd[132]  # Upper East Side North, JFK


def test_time_parts():
    dt = np.array(["2026-05-21T11:41:12", "2026-05-02T00:43:44", "2026-12-31T23:59:59"], dtype="datetime64[s]")
    hour, dow, month = time_parts(dt)
    assert list(hour) == [11, 0, 23]
    assert list(dow) == [3, 5, 3]  # Thursday, Saturday, Thursday
    assert list(month) == [5, 5, 12]


def test_route_stats_min_count():
    pu = np.array([161] * MIN_ROUTE_TRIPS + [1])
    do = np.array([230] * MIN_ROUTE_TRIPS + [2])
    km = np.arange(len(pu), dtype=float)
    routes = RouteStats.fit(pu, do, km, km * 2)
    assert routes.median_km[161, 230] == pytest.approx(np.median(km[:MIN_ROUTE_TRIPS]))
    assert np.isnan(routes.median_km[1, 2])  # too few trips
    assert routes.log_count[1, 2] == 0.0


def test_build_features_shape_and_values():
    routes = RouteStats.fit(np.array([161] * 20), np.array([230] * 20), np.ones(20), np.full(20, 7.0))
    pickup = np.array(["2026-07-06T17:30:00", "2026-07-11T03:00:00"], dtype="datetime64[s]")  # Mon, Sat
    X = build_features([161, 132], [230, 1], pickup, [False, True], routes)
    col = {name: X[:, i] for i, name in enumerate(FEATURES)}

    assert X.shape == (2, len(FEATURES)) and X.dtype == np.float32
    assert list(col["is_rush_hour"]) == [1, 0]
    assert list(col["is_overnight"]) == [0, 1]
    assert list(col["is_weekend"]) == [0, 1]
    assert list(col["is_flex"]) == [0, 1]
    assert list(col["pu_cbd"]) == [1, 0]
    assert list(col["pu_airport"]) == [0, 1] and list(col["do_airport"]) == [0, 1]
    assert col["route_median_min"][0] == 7.0 and np.isnan(col["route_median_min"][1])
    assert col["haversine_km"][1] > 25  # JFK -> Newark


def test_no_post_trip_inputs():
    """Realised trip outcomes must never be model inputs."""
    for leaky in ("distance_km", "duration_min", "trip_distance", "target", "cbd_fee", "tip"):
        assert leaky not in FEATURES


def test_regression_metrics():
    m = regression_metrics(np.array([10.0, 20.0]), np.array([12.0, 16.0]))
    assert m["mae"] == pytest.approx(3.0)
    assert m["rmse"] == pytest.approx(np.sqrt(10.0))
    assert m["bias"] == pytest.approx(-1.0)
