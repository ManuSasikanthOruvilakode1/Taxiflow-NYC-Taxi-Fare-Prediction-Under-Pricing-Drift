"""Leakage-free feature matrix shared by training and serving.

Every feature is known at booking time: pickup/dropoff zone, scheduled pickup time and
booking channel. Realised trip distance and duration are never inputs; the model instead
sees *historical* route statistics computed from the training window only.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxiflow.features.geospatial import (
    N_ZONES,
    bearing_degrees,
    cyclical_time_encoding,
    haversine_distance_km,
    load_zone_table,
    manhattan_distance_km,
)

FEATURES = [
    "pu_zone", "do_zone", "pu_borough", "do_borough",
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "is_overnight",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "haversine_km", "manhattan_km", "bearing",
    "pu_airport", "do_airport", "pu_cbd", "do_cbd", "same_zone",
    "is_flex",
    "route_median_km", "route_median_min", "route_log_count",
]
CATEGORICAL = ["pu_zone", "do_zone", "pu_borough", "do_borough"]
# Calendar month is handled by the seasonal index (features/seasonality.py), not learned:
# training windows are shorter than a year, so most months are never seen in training.
MODEL_FEATURES = [f for f in FEATURES if f != "month"]
MODEL_COLS = [FEATURES.index(f) for f in MODEL_FEATURES]
MONTH_COL = FEATURES.index("month")
FLEX_COL = FEATURES.index("is_flex")
MIN_ROUTE_TRIPS = 10


@dataclass
class RouteStats:
    """Historical per-(pickup, dropoff) statistics, as dense (264 x 264) lookup tables."""

    median_km: np.ndarray
    median_min: np.ndarray
    log_count: np.ndarray

    @classmethod
    def fit(cls, pu: np.ndarray, do: np.ndarray, distance_km: np.ndarray,
            duration_min: np.ndarray) -> "RouteStats":
        size = N_ZONES + 1
        df = pd.DataFrame({"pu": pu, "do": do, "km": distance_km, "min": duration_min})
        g = df.groupby(["pu", "do"]).agg(km=("km", "median"), mins=("min", "median"), n=("km", "size"))
        g = g[g["n"] >= MIN_ROUTE_TRIPS].reset_index()
        median_km = np.full((size, size), np.nan, dtype=np.float32)
        median_min = np.full((size, size), np.nan, dtype=np.float32)
        log_count = np.zeros((size, size), dtype=np.float32)
        idx = (g["pu"].to_numpy(), g["do"].to_numpy())
        median_km[idx] = g["km"].to_numpy()
        median_min[idx] = g["mins"].to_numpy()
        log_count[idx] = np.log1p(g["n"].to_numpy())
        return cls(median_km, median_min, log_count)


def time_parts(pickup: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hour, day_of_week with Monday=0, month 1-12) from a datetime64 array."""
    pickup = np.asarray(pickup, dtype="datetime64[s]")
    days = pickup.astype("datetime64[D]")
    hour = ((pickup - days).astype(np.int64) // 3600).astype(np.int8)
    dow = ((days.astype(np.int64) + 3) % 7).astype(np.int8)  # 1970-01-01 was a Thursday
    month = (pickup.astype("datetime64[M]").astype(np.int64) % 12 + 1).astype(np.int8)
    return hour, dow, month


def build_features(pu_zone, do_zone, pickup_datetime, is_flex, routes: RouteStats) -> np.ndarray:
    """Return a float32 matrix with columns in FEATURES order."""
    zones = load_zone_table()
    pu = np.asarray(pu_zone, dtype=np.int64)
    do = np.asarray(do_zone, dtype=np.int64)
    hour, dow, month = time_parts(pickup_datetime)
    weekday = dow < 5

    lat1, lon1, lat2, lon2 = zones.lat[pu], zones.lon[pu], zones.lat[do], zones.lon[do]
    cyc = cyclical_time_encoding(hour, dow)

    cols = {
        "pu_zone": pu,
        "do_zone": do,
        "pu_borough": zones.borough[pu],
        "do_borough": zones.borough[do],
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "is_weekend": ~weekday,
        "is_rush_hour": weekday & (hour >= 16) & (hour < 20),  # TLC $2.50 peak surcharge window
        "is_overnight": (hour >= 20) | (hour < 6),  # TLC $1.00 overnight surcharge window
        **cyc,
        "haversine_km": haversine_distance_km(lat1, lon1, lat2, lon2),
        "manhattan_km": manhattan_distance_km(lat1, lon1, lat2, lon2),
        "bearing": bearing_degrees(lat1, lon1, lat2, lon2),
        "pu_airport": zones.is_airport[pu],
        "do_airport": zones.is_airport[do],
        "pu_cbd": zones.in_cbd[pu],
        "do_cbd": zones.in_cbd[do],
        "same_zone": pu == do,
        "is_flex": np.asarray(is_flex, dtype=bool),
        "route_median_km": routes.median_km[pu, do],
        "route_median_min": routes.median_min[pu, do],
        "route_log_count": routes.log_count[pu, do],
    }
    n = len(pu)
    X = np.empty((n, len(FEATURES)), dtype=np.float32)
    for j, name in enumerate(FEATURES):
        X[:, j] = np.broadcast_to(cols[name], n)
    return X
