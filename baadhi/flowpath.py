"""Bonus: from any point upstream, trace the path a flood would take down the valley (Copernicus DEM)
and list the settlements along it.

Method: flow directions (D8) from the conditioned DEM over a window around the start point, then
follow the flow line downstream. Settlements count as "on the path" when they lie within a set
distance of the flow line AND not far above the river at that point (a village 200 m above the
gorge is not in a flood's way). Distances are along the river. No arrival times are given: wave
speed depends on the volume released, which a DEM cannot know.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform as shp_transform

from .grid import make_grid
from .sources import osm
from .sources import planetary as P


@dataclass
class FlowPath:
    line: dict                              # GeoJSON LineString (lon/lat)
    length_km: float
    settlements: list = field(default_factory=list)   # dicts: name, km_downstream, height_above_river_m, lon, lat


# D8 direction codes used by pysheds → (drow, dcol)
_D8 = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}


def trace(lon: float, lat: float, event: dt.date, radius_km: float = 30.0, max_km: float = 60.0,
          near_m: float = 400.0, max_height_m: float = 40.0, res: float = 30.0) -> FlowPath:
    from pysheds.grid import Grid as SGrid
    from pysheds.sview import Raster, ViewFinder

    if not hasattr(np, "in1d"):
        np.in1d = lambda a, b, **kw: np.isin(np.ravel(a), b, **kw)
    d = radius_km / 111.0
    grid = make_grid((lon - d, lat - d, lon + d, lat + d), res)
    dem = P.read_dem(grid)
    nodata = -9999.0
    z = np.where(np.isfinite(dem), dem, nodata).astype("float64")
    vf = ViewFinder(affine=grid.transform, shape=z.shape, crs=grid.crs.to_string(), nodata=nodata)
    g = SGrid(viewfinder=vf)
    fdir = np.asarray(g.flowdir(g.resolve_flats(g.fill_depressions(g.fill_pits(Raster(z, viewfinder=vf))))))

    x, y = grid.from_lonlat().transform(lon, lat)
    r = int((grid.transform.f - y) / res)
    c = int((x - grid.transform.c) / res)
    path, seen = [], set()
    while 0 <= r < grid.height and 0 <= c < grid.width and (r, c) not in seen and len(path) * res / 1000 < max_km:
        seen.add((r, c))
        path.append((r, c))
        step = _D8.get(int(fdir[r, c]))
        if step is None:
            break
        r, c = r + step[0], c + step[1]

    to_ll = grid.to_lonlat().transform
    xy = [(grid.transform.c + (cc + 0.5) * res, grid.transform.f - (rr + 0.5) * res) for rr, cc in path]
    line_m = LineString(xy) if len(xy) > 1 else None
    coords = [list(to_ll(px, py)) for px, py in xy]
    out = FlowPath(line={"type": "LineString", "coordinates": coords}, length_km=round((line_m.length if line_m else 0) / 1000, 1))
    if not line_m:
        return out

    # settlements near the flow line and not far above it
    path_z = np.array([dem[rr, cc] for rr, cc in path])
    cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff([p[0] for p in xy]), np.diff([p[1] for p in xy])))])
    places = osm.fetch("places", grid.lonlat_bounds(), osm.snapshot_date(event))["features"]
    fwd = grid.from_lonlat().transform
    found = []
    for f in places:
        pt = shp_transform(fwd, shape(f["geometry"]))
        if line_m.distance(pt) > near_m:
            continue
        s = line_m.project(pt)
        k = int(np.searchsorted(cum, s))
        k = min(k, len(path_z) - 1)
        pr = int((grid.transform.f - pt.y) / res)
        pc = int((pt.x - grid.transform.c) / res)
        if not (0 <= pr < grid.height and 0 <= pc < grid.width):
            continue
        height = float(dem[pr, pc] - path_z[k])
        if height > max_height_m:
            continue
        p = f["properties"]
        ll = to_ll(pt.x, pt.y)
        found.append({"name": p.get("name:en") or p.get("name") or "unnamed", "place": p.get("place"),
                      "km_downstream": round(s / 1000, 1), "height_above_river_m": round(max(height, 0), 0),
                      "lon": round(ll[0], 5), "lat": round(ll[1], 5)})
    out.settlements = sorted(found, key=lambda s: s["km_downstream"])
    return out
