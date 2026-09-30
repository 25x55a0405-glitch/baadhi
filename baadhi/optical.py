"""Sentinel-2 before/after views for any area and date.

Tile-level cloud cover says little about a small mountain valley (a scene can be 2 % cloudy and still
miss half the valley because it lies outside the orbit's strip). So each candidate date is scored by
how much of *our* area it actually shows clear, and the before/after views are built pixel by pixel:
"after" takes each pixel's first clear look after the event, "before" its last clear look before it.
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .grid import Grid, make_grid
from .sources import planetary as P

BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


def _clear_fraction(items, grid_coarse: Grid) -> float:
    scl = P.mosaic([P.read_to_grid(it.assets["SCL"].href, grid_coarse, P.Resampling.nearest, nodata=0) for it in items])
    good = np.isin(scl, [2, 4, 5, 6, 7, 11])  # dark area, vegetation, bare soil, water, unclassified, snow
    return float(good.mean())


def pick_dates(grid: Grid, start: dt.date, end: dt.date, want: int, newest_first: bool) -> list[tuple[dt.date, list, float]]:
    """Up to `want` dates in [start, end] with the most clear view of the area (≥ 15 % clear)."""
    by_date = P.group_by_date(P.s2_scenes(grid.lonlat_bounds(), start, end, max_cloud=95))
    coarse = make_grid(grid.lonlat_bounds(), 120)
    with ThreadPoolExecutor(6) as ex:
        fracs = list(ex.map(lambda kv: _clear_fraction(kv[1], coarse), by_date.items()))
    cands = [(d, items, f) for (d, items), f in zip(by_date.items(), fracs) if f >= 0.15]
    cands.sort(key=lambda c: c[0], reverse=newest_first)
    picked, cover = [], 0.0
    for c in cands:
        picked.append(c)
        cover = 1 - (1 - cover) * (1 - c[2])  # rough: assumes independent cloud gaps
        if len(picked) >= want or cover > 0.97:
            break
    return picked


def composite(dates, grid: Grid, bands=BANDS) -> dict[str, np.ndarray]:
    """First clear observation per pixel, in the given date order."""
    out = {b: np.full(grid.shape, np.nan, "float32") for b in bands}
    out["date_index"] = np.full(grid.shape, -1, "int16")
    with ThreadPoolExecutor(3) as ex:
        reads = list(ex.map(lambda d: P.read_s2(d[1], grid, bands), dates))
    for i, s2 in enumerate(reads):
        m = np.isnan(out[bands[0]]) & np.isfinite(s2[bands[0]])
        for b in bands:
            out[b][m] = s2[b][m]
        out["date_index"][m] = i
    out["dates"] = [d[0] for d in dates]
    return out


def before_after(grid: Grid, event: dt.date, pre_days: int = 60, post_days: int = 45, want: int = 3):
    pre_dates = pick_dates(grid, event - dt.timedelta(days=pre_days), event - dt.timedelta(days=1), want, newest_first=True)
    post_dates = pick_dates(grid, event, event + dt.timedelta(days=post_days), want, newest_first=False)
    pre = composite(pre_dates, grid) if pre_dates else None
    post = composite(post_dates, grid) if post_dates else None
    return pre, post


def indices(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    eps = 1e-6
    return {
        "ndvi": (s["B08"] - s["B04"]) / (s["B08"] + s["B04"] + eps),
        "mndwi": (s["B03"] - s["B11"]) / (s["B03"] + s["B11"] + eps),
        "bright": (s["B02"] + s["B03"] + s["B04"]) / 3,
        # bare-soil index: high for exposed sediment/rock, low for vegetation and water
        "bsi": ((s["B11"] + s["B04"]) - (s["B08"] + s["B02"])) / ((s["B11"] + s["B04"]) + (s["B08"] + s["B02"]) + eps),
    }
