"""Geospatial and temporal feature transformations for NYC Taxi trips.

All functions accept scalars or NumPy arrays, so the same code path is used for
batch training and single-request serving.
"""

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np

EARTH_RADIUS_KM = 6371.0088
N_ZONES = 263  # TLC zones 264 ("Unknown") and 265 ("Outside of NYC") have no geometry
AIRPORT_ZONES = (1, 132, 138)  # EWR, JFK, LaGuardia
BOROUGHS = ("EWR", "Queens", "Bronx", "Manhattan", "Staten Island", "Brooklyn")


@dataclass(frozen=True)
class ZoneTable:
    """Per-zone attributes as arrays indexed directly by TLC LocationID (index 0 unused)."""

    lat: np.ndarray
    lon: np.ndarray
    borough: np.ndarray  # integer code into BOROUGHS
    in_cbd: np.ndarray  # MTA Congestion Relief Zone membership
    is_airport: np.ndarray
    name: tuple[str, ...]


@lru_cache(maxsize=1)
def load_zone_table() -> ZoneTable:
    size = N_ZONES + 1
    lat, lon = np.full(size, np.nan), np.full(size, np.nan)
    borough = np.full(size, -1, dtype=np.int8)
    in_cbd = np.zeros(size, dtype=bool)
    names = [""] * size
    text = resources.files("taxiflow.resources").joinpath("zone_table.csv").read_text(encoding="utf-8")
    for row in csv.DictReader(text.splitlines()):
        i = int(row["location_id"])
        lat[i], lon[i] = float(row["lat"]), float(row["lon"])
        borough[i] = BOROUGHS.index(row["borough"])
        in_cbd[i] = row["in_cbd"] == "1"
        names[i] = row["zone"]
    is_airport = np.zeros(size, dtype=bool)
    is_airport[list(AIRPORT_ZONES)] = True
    return ZoneTable(lat, lon, borough, in_cbd, is_airport, tuple(names))


def get_zone_coords(zone_id: int) -> tuple[float, float]:
    """Centroid (lat, lon) of a TLC zone."""
    if not 1 <= zone_id <= N_ZONES:
        raise ValueError(f"zone_id must be in 1..{N_ZONES}, got {zone_id}")
    zones = load_zone_table()
    return float(zones.lat[zone_id]), float(zones.lon[zone_id])


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(np.subtract(lon2, lon1))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def manhattan_distance_km(lat1, lon1, lat2, lon2):
    """L1 (grid) distance approximation in km."""
    km_per_deg = EARTH_RADIUS_KM * np.pi / 180.0
    avg_lat = np.radians((np.add(lat1, lat2)) / 2.0)
    d_lat = np.abs(np.subtract(lat2, lat1)) * km_per_deg
    d_lon = np.abs(np.subtract(lon2, lon1)) * km_per_deg * np.cos(avg_lat)
    return d_lat + d_lon


def bearing_degrees(lat1, lon1, lat2, lon2):
    """Initial compass bearing from point 1 to point 2, in [0, 360)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlmb = np.radians(np.subtract(lon2, lon1))
    y = np.sin(dlmb) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlmb)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def is_airport_zone(zone_id: int) -> bool:
    """Whether the zone is an airport (JFK, LGA, EWR)."""
    return zone_id in AIRPORT_ZONES


def cyclical_time_encoding(hour, day_of_week) -> dict:
    """Sine/cosine encoding of hour of day (0-23) and day of week (0=Mon..6)."""
    hour_rad = 2.0 * np.pi * np.divide(hour, 24.0)
    dow_rad = 2.0 * np.pi * np.divide(day_of_week, 7.0)
    return {
        "hour_sin": np.sin(hour_rad),
        "hour_cos": np.cos(hour_rad),
        "dow_sin": np.sin(dow_rad),
        "dow_cos": np.cos(dow_rad),
    }
