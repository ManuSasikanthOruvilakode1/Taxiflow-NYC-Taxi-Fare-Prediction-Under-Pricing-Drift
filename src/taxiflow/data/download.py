"""Download monthly NYC TLC Yellow taxi trip records (parquet) from the official TLC CDN."""

import shutil
import urllib.request
from pathlib import Path

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings between start and end."""
    y, m = map(int, start.split("-"))
    end_y, end_m = map(int, end.split("-"))
    months = []
    while (y, m) <= (end_y, end_m):
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def month_add(month: str, k: int) -> str:
    """Shift a 'YYYY-MM' string by k months (k may be negative)."""
    y, m = map(int, month.split("-"))
    idx = y * 12 + (m - 1) + k
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def yellow_url(month: str) -> str:
    return f"{TLC_BASE_URL}/yellow_tripdata_{month}.parquet"


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        length = resp.headers.get("Content-Length")
    return int(length) if length else None


def download_file(url: str, dest: Path) -> bool:
    """Download url to dest. Skips if dest already exists with the remote size. Returns True if fetched."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _remote_size(url)
    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        return False

    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    if expected is not None and tmp.stat().st_size != expected:
        tmp.unlink()
        raise OSError(f"Size mismatch for {url}: expected {expected} bytes")
    tmp.replace(dest)
    return True


def download_yellow(start: str, end: str, raw_dir: str | Path) -> list[Path]:
    """Download all Yellow taxi monthly files in [start, end] into raw_dir."""
    raw_dir = Path(raw_dir)
    paths = []
    for month in month_range(start, end):
        dest = raw_dir / f"yellow_tripdata_{month}.parquet"
        fetched = download_file(yellow_url(month), dest)
        print(f"{'fetched' if fetched else 'cached '} {dest.name} ({dest.stat().st_size / 2**20:.0f} MB)")
        paths.append(dest)
    return paths
