"""Tests for downloading helpers and the DuckDB cleaning step (synthetic parquet, no network)."""

from datetime import datetime, timedelta

import duckdb
import pandas as pd
import pytest

from taxiflow.config import BadRecordRule, FilterConfig
from taxiflow.data.clean import clean_month
from taxiflow.data.download import month_range, yellow_url


def test_month_range_crosses_year():
    assert month_range("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert month_range("2026-08", "2026-08") == ["2026-08"]


def test_yellow_url():
    assert yellow_url("2026-08").endswith("/yellow_tripdata_2026-08.parquet")


def _trip(**overrides):
    start = datetime(2025, 11, 10, 12, 0)
    row = {
        "VendorID": 1,
        "tpep_pickup_datetime": start,
        "tpep_dropoff_datetime": start + timedelta(minutes=15),
        "passenger_count": 1,
        "trip_distance": 3.0,  # miles
        "RatecodeID": 1,
        "PULocationID": 161,
        "DOLocationID": 75,
        "payment_type": 1,
        "fare_amount": 18.0,
        "tip_amount": 4.0,
        "total_amount": 27.25,
        "cbd_congestion_fee": 0.75,
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_month(tmp_path):
    start = datetime(2025, 11, 10, 12, 0)
    rows = {
        "valid": _trip(),
        "previous_month": _trip(tpep_pickup_datetime=datetime(2025, 10, 31, 23, 0),
                                tpep_dropoff_datetime=datetime(2025, 10, 31, 23, 20)),
        "unknown_zone": _trip(PULocationID=264),
        "voided": _trip(payment_type=6),
        "negotiated_rate": _trip(RatecodeID=5),
        "negative_total": _trip(total_amount=-27.25, tip_amount=0.0),
        "teleport": _trip(trip_distance=30.0, tpep_dropoff_datetime=start + timedelta(minutes=2)),
        "flex_vendor1": _trip(payment_type=0, RatecodeID=None, passenger_count=None),
        "flex_vendor2_bad": _trip(VendorID=2, payment_type=0, RatecodeID=None, total_amount=5.0),
    }
    path = tmp_path / "yellow_tripdata_2025-11.parquet"
    pd.DataFrame(list(rows.values())).to_parquet(path)
    return path


def test_clean_month_filters_and_target(raw_month, tmp_path):
    out = tmp_path / "clean.parquet"
    rules = [BadRecordRule(month="2025-11", vendor_id=2, payment_type=0)]
    stats = clean_month(raw_month, out, "2025-11", FilterConfig(), rules)

    df = duckdb.sql(f"SELECT * FROM '{out.as_posix()}' ORDER BY is_flex").df()
    assert stats == {"month": "2025-11", "raw_rows": 9, "clean_rows": 2, "known_bad_rows": 1}
    assert list(df["is_flex"]) == [False, True]
    # Target excludes the tip
    assert df["target"].iloc[0] == pytest.approx(23.25)
    assert df["cbd_fee"].iloc[0] == pytest.approx(0.75)
    assert df["distance_km"].iloc[0] == pytest.approx(3.0 * 1.609344, rel=1e-5)
    assert df["duration_min"].iloc[0] == pytest.approx(15.0)


def test_clean_month_handles_pre_2025_schema(tmp_path):
    """2024 files have no cbd_congestion_fee column."""
    path = tmp_path / "raw.parquet"
    row = _trip(tpep_pickup_datetime=datetime(2024, 3, 5, 9, 0), tpep_dropoff_datetime=datetime(2024, 3, 5, 9, 15))
    del row["cbd_congestion_fee"]
    pd.DataFrame([row]).to_parquet(path)
    out = tmp_path / "clean.parquet"
    stats = clean_month(path, out, "2024-03", FilterConfig())
    assert stats["clean_rows"] == 1
    assert duckdb.sql(f"SELECT cbd_fee FROM '{out.as_posix()}'").fetchone()[0] == 0.0
