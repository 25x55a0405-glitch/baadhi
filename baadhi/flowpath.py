"""Bonus: from any point upstream, trace the path a flood would take down the valley (Copernicus DEM)
and list the settlements along it.

Method: flow directions (D8) from the conditioned Copernicus DEM (GLO-90, fast over wide areas) in a
window around the start point, then follow the flow line downstream; when the line reaches the edge of
the window, a new window is opened there and the trace continues (up to `max_km`). Settlements count
as "on the path" when they lie within a set distance of the flow line AND not far above the river at
that point (a village 200 m above the gorge is not in a flood's way). Distances are along the river.
No arrival times are given: wave speed depends on the volume released, which a DEM cannot know.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, shape
from shapely.ops import transform as shp_transform

from .grid import make_grid
from .sources import osm
from .sources import planetary as P


@dataclass
class FlowPath:
    line: dict                              # GeoJSON LineString (lon/lat)
    length_km: float
    settlements: list = field(default_factory=list)   # dicts: name, km_downstream, height_above_river_m, exposure, lon, lat
    windows: int = 1


# D8 direction codes used by pysheds → (drow, dcol)
_D8 = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}


def _flowdir(lon: float, lat: float, radius_km: float, res: float):
    from pysheds.grid import Grid as SGrid
    from pysheds.sview import Raster, ViewFinder

    if not hasattr(np, "in1d"):
        np.in1d = lambda a, b, **kw: np.isin(np.ravel(a), b, **kw)
    d = radius_km / 111.0
    grid = make_grid((lon - d, lat - d, lon + d, lat + d), res)
    dem = P.read_dem(grid, "cop-dem-glo-90" if res >= 60 else "cop-dem-glo-30")
    nodata = -9999.0
    z = np.where(np.isfinite(dem), dem, nodata).astype("float64")
    vf = ViewFinder(affine=grid.transform, shape=z.shape, crs=grid.crs.to_string(), nodata=nodata)
    g = SGrid(viewfinder=vf)
    fdir = np.asarray(g.flowdir(g.resolve_flats(g.fill_depressions(g.fill_pits(Raster(z, viewfinder=vf))))))
    return grid, dem, fdir


def _walk(grid, fdir, lon, lat, res, max_steps):
    """Follow D8 from (lon, lat); returns cells and whether the walk ran off the window's edge."""
    x, y = grid.from_lonlat().transform(lon, lat)
    r = int((grid.transform.f - y) / res)
    c = int((x - grid.transform.c) / res)
    path, seen = [], set()
    while 0 <= r < grid.height and 0 <= c < grid.width and (r, c) not in seen and len(path) < max_steps:
        seen.add((r, c))
        path.append((r, c))
        step = _D8.get(int(fdir[r, c]))
        if step is None:                             # no direction: a pit, the sea — or the window's border
            near_edge = r <= 1 or c <= 1 or r >= grid.height - 2 or c >= grid.width - 2
            return path, near_edge
        r, c = r + step[0], c + step[1]
    off_edge = not (0 <= r < grid.height and 0 <= c < grid.width)
    return path, off_edge


def trace(lon: float, lat: float, event: dt.date, radius_km: float = 35.0, max_km: float = 70.0,
          near_m: float = 400.0, max_height_m: float = 60.0, res: float = 90.0, max_windows: int = 4) -> FlowPath:
    coords, found, seen_ids = [], [], set()
    total_km, windows = 0.0, 0
    cur = (lon, lat)
    snap = osm.snapshot_date(event)
    while windows < max_windows and total_km < max_km:
        windows += 1
        grid, dem, fdir = _flowdir(cur[0], cur[1], radius_km, res)
        path, off_edge = _walk(grid, fdir, cur[0], cur[1], res, int((max_km - total_km) * 1000 / res) + 1)
        if len(path) < 2:
            break
        to_ll = grid.to_lonlat().transform
        xy = [(grid.transform.c + (cc + 0.5) * res, grid.transform.f - (rr + 0.5) * res) for rr, cc in path]
        line_m = LineString(xy)
        seg_ll = [list(to_ll(px, py)) for px, py in xy]
        coords += seg_ll if not coords else seg_ll[1:]

        # settlements near this stretch of the flow line and not far above it
        path_z = np.array([dem[rr, cc] for rr, cc in path])
        cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff([p[0] for p in xy]), np.diff([p[1] for p in xy])))])
        places = osm.fetch("places", grid.lonlat_bounds(), snap)["features"]
        fwd = grid.from_lonlat().transform
        for f in places:
            oid = f["properties"].get("osm_id")
            if oid in seen_ids or not (f["properties"].get("name:en") or f["properties"].get("name")):
                continue                              # unnamed places only add noise to the list
            pt = shp_transform(fwd, shape(f["geometry"]))
            if line_m.distance(pt) > near_m:
                continue
            s = line_m.project(pt)
            k = min(int(np.searchsorted(cum, s)), len(path_z) - 1)
            pr = int((grid.transform.f - pt.y) / res)
            pc = int((pt.x - grid.transform.c) / res)
            if not (0 <= pr < grid.height and 0 <= pc < grid.width):
                continue
            height = float(dem[pr, pc] - path_z[k])
            if height > max_height_m:
                continue
            seen_ids.add(oid)
            p = f["properties"]
            ll = to_ll(pt.x, pt.y)
            found.append({"name": p.get("name:en") or p.get("name") or "unnamed", "place": p.get("place"),
                          "km_downstream": round(total_km + s / 1000, 1), "height_above_river_m": round(max(height, 0), 0),
                          "exposure": "directly in the path" if height <= 20 else "possibly in the path",
                          "lon": round(ll[0], 5), "lat": round(ll[1], 5)})
        total_km += line_m.length / 1000
        if not off_edge:
            break
        cur = tuple(seg_ll[-1])                      # continue from the edge in a fresh window
    return FlowPath(line={"type": "LineString", "coordinates": coords}, length_km=round(total_km, 1),
                    settlements=sorted(found, key=lambda s: s["km_downstream"]), windows=windows)
