"""Clean raw TLC Yellow taxi months into a compact, model-ready parquet per month (DuckDB)."""

import json
from pathlib import Path

import duckdb

from taxiflow.config import BadRecordRule, FilterConfig

KM_PER_MILE = 1.609344

# Columns written for each cleaned month.
#   target        = total_amount - tip_amount: everything the passenger is charged except the
#                   discretionary tip (tips are only recorded for card payments).
#   is_flex       = payment_type 0, TLC "Flex Fare" (app-booked, upfront-priced) trips.
#   distance_km / duration_min are realised post-trip values, used only for filtering,
#   historical route statistics and analysis, never as model inputs.
BASE_CTE = """
WITH base AS (
    SELECT
        tpep_pickup_datetime AS pickup_datetime,
        CAST(PULocationID AS SMALLINT) AS pu_zone,
        CAST(DOLocationID AS SMALLINT) AS do_zone,
        coalesce(payment_type, 0) = 0 AS is_flex,
        total_amount - tip_amount AS target,
        trip_distance * {km_per_mile} AS distance_km,
        epoch(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0 AS duration_min,
        {cbd_fee_expr} AS cbd_fee,
        coalesce(payment_type, 0) AS payment_type,
        coalesce(RatecodeID, 1) AS ratecode,
        VendorID AS vendor_id
    FROM read_parquet('{path}')
)
"""

CLEAN_SQL = BASE_CTE + """
SELECT pickup_datetime, pu_zone, do_zone, is_flex,
       CAST(target AS FLOAT) AS target,
       CAST(distance_km AS FLOAT) AS distance_km,
       CAST(duration_min AS FLOAT) AS duration_min,
       CAST(cbd_fee AS FLOAT) AS cbd_fee
FROM base
WHERE pickup_datetime >= TIMESTAMP '{month_start}'
  AND pickup_datetime < TIMESTAMP '{month_end}'
  AND pu_zone BETWEEN 1 AND 263 AND do_zone BETWEEN 1 AND 263
  AND payment_type NOT IN (3, 4, 6)         -- no charge, dispute, voided
  AND ratecode IN (1, 2, 3, 4)              -- standard, JFK, Newark, Nassau/Westchester
  AND target BETWEEN {f.min_fare} AND {f.max_fare}
  AND distance_km BETWEEN {f.min_distance_km} AND {f.max_distance_km}
  AND duration_min BETWEEN {f.min_duration_min} AND {f.max_duration_min}
  AND distance_km / (duration_min / 60.0) BETWEEN {f.min_speed_kmph} AND {f.max_speed_kmph}
  AND NOT ({bad_expr})                      -- known-corrupt records (see config)
"""


def _next_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y + 1:04d}-01" if m == 12 else f"{y:04d}-{m + 1:02d}"


def _bad_record_expr(rules: list[BadRecordRule], month: str) -> str:
    clauses = [f"(vendor_id = {r.vendor_id} AND payment_type = {r.payment_type})"
               for r in rules if r.month == month]
    return " OR ".join(clauses) if clauses else "FALSE"


def clean_month(raw_path: Path, out_path: Path, month: str, filters: FilterConfig,
                bad_records: list[BadRecordRule] = ()) -> dict:
    """Clean one raw monthly file. Returns row counts before and after filtering."""
    con = duckdb.connect()
    cols = {r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{raw_path.as_posix()}')").fetchall()}
    cbd_fee_expr = "coalesce(cbd_congestion_fee, 0)" if "cbd_congestion_fee" in cols else "0.0"
    params = dict(
        km_per_mile=KM_PER_MILE,
        cbd_fee_expr=cbd_fee_expr,
        path=raw_path.as_posix(),
        month_start=f"{month}-01",
        month_end=f"{_next_month(month)}-01",
        f=filters,
        bad_expr=_bad_record_expr(list(bad_records), month),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.sql(f"COPY ({CLEAN_SQL.format(**params)}) TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    n_raw = con.sql(f"SELECT count(*) FROM read_parquet('{raw_path.as_posix()}')").fetchone()[0]
    n_clean = con.sql(f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    n_bad = con.sql(BASE_CTE.format(**params) + f"SELECT count(*) FROM base WHERE {params['bad_expr']}").fetchone()[0]
    con.close()
    return {"month": month, "raw_rows": n_raw, "clean_rows": n_clean, "known_bad_rows": n_bad}


def clean_all(months: list[str], raw_dir: str | Path, processed_dir: str | Path,
              filters: FilterConfig, bad_records: list[BadRecordRule] = (), force: bool = False) -> list[dict]:
    """Clean every month; skips months whose cleaned file already exists unless force=True."""
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    report_path = processed_dir / "cleaning_report.json"
    previous = {}
    if report_path.exists():
        previous = {r["month"]: r for r in json.loads(report_path.read_text())}

    report = []
    for month in months:
        out_path = processed_dir / f"clean_{month}.parquet"
        if out_path.exists() and month in previous and not force:
            report.append(previous[month])
            continue
        stats = clean_month(raw_dir / f"yellow_tripdata_{month}.parquet", out_path, month, filters, bad_records)
        print(f"{month}: {stats['raw_rows']:>10,} raw -> {stats['clean_rows']:>10,} clean")
        report.append(stats)

    processed_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report
