"""Build src/taxiflow/resources/zone_table.csv: one row per TLC taxi zone (1-263).

  lat, lon   WGS84 centroid of the official TLC zone polygons (multi-part zones are merged;
             if the centroid falls outside the polygon, a guaranteed interior point is used).
  area_km2   polygon area.
  in_cbd     1 if the zone lies in the MTA Congestion Relief Zone (Manhattan below 60th St),
             derived empirically: >= 90% of 2025+ pickups in the zone carry cbd_congestion_fee.

Run after `taxiflow clean`:
    uv run --group geo python scripts/build_zone_table.py
"""

import csv
import io
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import shapefile
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union

ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "src" / "taxiflow" / "resources" / "zone_table.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CBD_SHARE_THRESHOLD = 0.90


def zone_centroids() -> dict[int, dict]:
    with urllib.request.urlopen(ZONE_SHAPEFILE_URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    with TemporaryDirectory() as tmp:
        archive.extractall(tmp)
        shp_path = next(Path(tmp).rglob("*.shp"))
        reader = shapefile.Reader(str(shp_path))
        fields = [f[0] for f in reader.fields[1:]]
        geoms, meta = defaultdict(list), {}
        for rec in reader.iterShapeRecords():
            attrs = dict(zip(fields, rec.record, strict=True))
            zid = int(attrs["LocationID"])
            geoms[zid].append(shape(rec.shape.__geo_interface__))
            meta[zid] = (attrs["borough"], attrs["zone"])
        reader.close()

    # Shapefile CRS is NAD83 / New York Long Island State Plane (US feet).
    to_wgs84 = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    zones = {}
    for zid, parts in geoms.items():
        poly = unary_union(parts)
        point = poly.centroid if poly.contains(poly.centroid) else poly.representative_point()
        lon, lat = to_wgs84.transform(point.x, point.y)
        zones[zid] = {
            "borough": meta[zid][0],
            "zone": meta[zid][1],
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "area_km2": round(poly.area * 0.3048**2 / 1e6, 4),
        }
    return zones


def cbd_zones() -> set[int]:
    files = sorted(PROCESSED_DIR.glob("clean_202[5-9]-*.parquet"))
    if not files:
        raise FileNotFoundError("No cleaned 2025+ months found; run `taxiflow clean` first.")
    paths = ", ".join(f"'{p.as_posix()}'" for p in files)
    rows = duckdb.sql(f"""
        SELECT pu_zone FROM read_parquet([{paths}])
        GROUP BY pu_zone
        HAVING count(*) >= 1000 AND avg((cbd_fee > 0)::INT) >= {CBD_SHARE_THRESHOLD}
    """).fetchall()
    return {int(r[0]) for r in rows}


def main() -> None:
    zones = zone_centroids()
    cbd = cbd_zones()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["location_id", "borough", "zone", "lat", "lon", "area_km2", "in_cbd"])
        for zid in sorted(zones):
            z = zones[zid]
            w.writerow([zid, z["borough"], z["zone"], z["lat"], z["lon"], z["area_km2"], int(zid in cbd)])
    print(f"Wrote {len(zones)} zones ({len(cbd)} in congestion zone) to {OUT_PATH}")
    print("Congestion zone:", ", ".join(zones[z]["zone"] for z in sorted(cbd)))


if __name__ == "__main__":
    main()
