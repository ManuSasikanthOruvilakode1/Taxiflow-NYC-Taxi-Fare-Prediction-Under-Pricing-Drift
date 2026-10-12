"""Route-controlled seasonal price index estimated from metered trips.

For each year in the fit window, every metered trip's fare is divided by the median fare of
its (pickup, dropoff) pair in that year, so changes in trip mix (longer summer trips, more
airport runs) cancel out. The monthly mean of that ratio, normalised to average 1 within the
year, is the year's index; the final index averages the years. Metered pricing is regulated
and was unchanged across 2024-2026, so what remains is the seasonal price level (traffic,
surcharge mix). Models are trained on fare / index[month] and predict model(x) * index[month].
"""

from pathlib import Path

import duckdb
import numpy as np

from taxiflow.config import MonthRange
from taxiflow.data.download import month_range

MIN_PAIR_TRIPS = 200


def fit_seasonal_index(processed_dir: str | Path, window: MonthRange) -> np.ndarray:
    """Return a length-13 array; index[m] is the multiplicative factor for calendar month m (1-12)."""
    months = month_range(*window)
    covered = {int(m[5:]) for m in months}
    if covered != set(range(1, 13)):
        raise ValueError(f"Seasonality window {window} must cover all 12 calendar months")
    paths = ", ".join(f"'{(Path(processed_dir) / f'clean_{m}.parquet').as_posix()}'" for m in months)
    rows = duckdb.sql(f"""
        WITH t AS (
            SELECT year(pickup_datetime) AS y, month(pickup_datetime) AS mo, pu_zone, do_zone, target
            FROM read_parquet([{paths}]) WHERE NOT is_flex
        ),
        base AS (
            SELECT y, pu_zone, do_zone, median(target) AS base
            FROM t GROUP BY ALL HAVING count(*) >= {MIN_PAIR_TRIPS}
        ),
        monthly AS (
            SELECT t.y, t.mo, avg(t.target / base.base) AS r
            FROM t JOIN base USING (y, pu_zone, do_zone) GROUP BY ALL
        )
        SELECT y, mo, r FROM monthly
    """).fetchall()
    by_year: dict[int, dict[int, float]] = {}
    for y, mo, r in rows:
        by_year.setdefault(int(y), {})[int(mo)] = float(r)
    # Normalise each year over the months every year shares, so a partial year is comparable.
    common = set.intersection(*(set(v) for v in by_year.values()))
    sums, counts = np.zeros(13), np.zeros(13)
    for year in by_year.values():
        norm = np.mean([year[m] for m in common])
        for mo, r in year.items():
            sums[mo] += r / norm
            counts[mo] += 1
    index = np.ones(13)
    index[1:] = sums[1:] / counts[1:]
    return index / index[1:].mean()
