"""Pre-flight check: can the system reach every data source it needs, right now?

Run before a live demo:  python -m baadhi.preflight
Checks the data sources the hackathon allows as inputs (Sentinel-1, Sentinel-2, Copernicus DEM,
pre-event OpenStreetMap) plus the software stack. Nothing here uses a published flood or damage map.
"""
from __future__ import annotations

import datetime as dt
import importlib
import shutil
import sys
import time

import requests

from .grid import make_grid
from .sources import planetary as P

TEST_BBOX = (85.33, 28.15, 85.36, 28.18)  # a small box at Syaphrubesi
EVENT = dt.date(2026, 8, 26)
results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        t = time.time()
        try:
            detail = fn() or ""
            results.append((name, True, f"{detail} ({time.time() - t:.1f}s)"))
        except Exception as e:  # noqa: BLE001 — report every failure, keep going
            results.append((name, False, f"{type(e).__name__}: {e}"[:200]))
        return fn
    return deco


@check("Python packages")
def _pkgs():
    mods = ["numpy", "scipy", "rasterio", "shapely", "pyproj", "networkx", "torch", "onnxruntime", "pystac_client", "fastapi", "osmium"]
    vers = {m: getattr(importlib.import_module(m), "__version__", "?") for m in mods}
    return f"torch {vers['torch']}, rasterio {vers['rasterio']}, onnxruntime {vers['onnxruntime']}"


@check("Disk space on the data drive")
def _disk():
    free = shutil.disk_usage(__file__).free / 1e9
    if free < 5:
        raise RuntimeError(f"only {free:.1f} GB free")
    return f"{free:.0f} GB free"


@check("Sentinel-1 same-track before/after pair")
def _s1():
    passes = P.s1_passes(TEST_BBOX, EVENT - dt.timedelta(days=25), EVENT + dt.timedelta(days=13))
    after = [p for p in passes if p.date >= EVENT]
    for a in after:
        before = [p for p in passes if p.track == a.track and p.date < EVENT]
        if before:
            b = before[-1]
            return f"track {a.track}: {b.date} → {a.date} ({(a.date - b.date).days} days apart)"
    raise RuntimeError("no same-track pair around the event")


@check("Sentinel-1 read onto grid")
def _s1read():
    grid = make_grid(TEST_BBOX, 20)
    p = [p for p in P.s1_passes(TEST_BBOX, EVENT, EVENT + dt.timedelta(days=12))][0]
    vv = P.read_s1(p, grid, pols=("vv",))["vv"]
    import numpy as np
    return f"{vv.shape}, {np.isfinite(vv).mean():.0%} valid"


@check("Sentinel-2 L2A read onto grid")
def _s2():
    grid = make_grid(TEST_BBOX, 20)
    by_date = P.group_by_date(P.s2_scenes(TEST_BBOX, EVENT - dt.timedelta(days=30), EVENT, max_cloud=60))
    d, items = list(by_date.items())[-1]
    s2 = P.read_s2(items, grid, bands=("B04",))
    import numpy as np
    return f"{d}: {np.isfinite(s2['B04']).mean():.0%} clear"


@check("Copernicus DEM (GLO-30)")
def _dem():
    import numpy as np
    dem = P.read_dem(make_grid(TEST_BBOX, 30))
    return f"{np.nanmin(dem):.0f}–{np.nanmax(dem):.0f} m"


@check("OpenStreetMap history (official Overpass instances)")
def _osm():
    from .sources import osm
    w, s_, e, n = TEST_BBOX
    q = f'[out:json][timeout:60][date:"2026-08-24T00:00:00Z"];way["building"]({s_},{w},{n},{e});out count;'
    ok, bad = [], []
    for url in osm.OVERPASS:
        host = url.split("//")[1].split("/")[0]
        t = time.time()
        try:
            r = requests.post(url, data={"data": q}, headers=osm.UA, timeout=70)
            if r.status_code == 200:
                ok.append(f"{host}: {int(r.json()['elements'][0]['tags']['ways'])} buildings in {time.time() - t:.0f}s")
            else:
                bad.append(f"{host}: HTTP {r.status_code}")
        except requests.RequestException as ex:
            bad.append(f"{host}: {type(ex).__name__}")
    if not ok:
        raise RuntimeError("no official instance answered — " + "; ".join(bad))
    return "; ".join(ok + bad)


@check("Pre-event OpenStreetMap fallback (Geofabrik dated snapshot + pyosmium)")
def _geofabrik():
    import osmium  # noqa: F401 — needed to cut an area out of a snapshot
    from .sources import geofabrik
    url, snap, region = geofabrik.choose_file(TEST_BBOX, EVENT - dt.timedelta(days=2))
    cached = (geofabrik.CACHE / url.rsplit("/", 1)[1]).exists()
    return f"{region}, snapshot {snap}" + (" (already downloaded)" if cached else " (downloaded when first needed)")


def main() -> int:
    width = max(len(n) for n, _, _ in results)
    ok = all(r[1] for r in results)
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name.ljust(width)}  {detail}")
    print("\nAll checks passed." if ok else "\nSome checks failed — see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
