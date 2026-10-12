"""Unit tests for Taxiflow geospatial and temporal feature engineering."""

import pytest

from taxiflow.features.geospatial import (
    cyclical_time_encoding,
    get_zone_coords,
    haversine_distance_km,
    is_airport_zone,
    manhattan_distance_km,
)


def test_haversine_distance():
    # Times Square (230) to JFK Airport (132)
    jfk = get_zone_coords(132)
    times_sq = get_zone_coords(230)

    dist = haversine_distance_km(times_sq[0], times_sq[1], jfk[0], jfk[1])
    assert 20.0 < dist < 25.0, f"Expected distance between 20 and 25 km, got {dist}"


def test_manhattan_distance():
    jfk = get_zone_coords(132)
    times_sq = get_zone_coords(230)

    man_dist = manhattan_distance_km(times_sq[0], times_sq[1], jfk[0], jfk[1])
    hav_dist = haversine_distance_km(times_sq[0], times_sq[1], jfk[0], jfk[1])
    # Manhattan distance should be strictly greater than or equal to Euclidean/Haversine distance
    assert man_dist >= hav_dist


def test_airport_zone_detection():
    assert is_airport_zone(132) is True  # JFK
    assert is_airport_zone(138) is True  # LGA
    assert is_airport_zone(1) is True    # EWR
    assert is_airport_zone(230) is False # Times Square


def test_cyclical_time_encoding():
    enc_midnight = cyclical_time_encoding(0, 0)
    assert pytest.approx(enc_midnight["hour_sin"], abs=1e-5) == 0.0
    assert pytest.approx(enc_midnight["hour_cos"], abs=1e-5) == 1.0

    enc_noon = cyclical_time_encoding(12, 0)
    assert pytest.approx(enc_noon["hour_sin"], abs=1e-5) == 0.0
    assert pytest.approx(enc_noon["hour_cos"], abs=1e-5) == -1.0
