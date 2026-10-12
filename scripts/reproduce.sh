#!/usr/bin/env bash
# End-to-end reproduction: download -> clean -> zone table -> train -> evaluate -> walk-forward -> correct -> report.
# Needs ~3 GB disk and ~12 GB RAM; the full run takes roughly an hour on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --python 3.11 --group geo --group viz

uv run taxiflow download                          # 32 monthly TLC files, ~1.9 GB
uv run taxiflow clean                             # -> data/processed/clean_YYYY-MM.parquet
uv run python scripts/build_zone_table.py         # zone centroids + congestion-zone flags
uv run taxiflow train                             # -> models/
uv run taxiflow evaluate                          # -> reports/metrics.json
uv run taxiflow walkforward                       # -> reports/walkforward.json
uv run taxiflow correct --month 2026-08           # -> models/taxiflow_lgbm_serving.joblib
uv run python scripts/make_report.py              # -> docs/figures/, README results table
uv run pytest -q
