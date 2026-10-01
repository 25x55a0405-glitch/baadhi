"""What was damaged? Pre-event OpenStreetMap buildings, roads, bridges, health facilities and
settlements overlaid on the flood/debris footprint.

At 10 m resolution a satellite cannot see whether a roof collapsed; it can see that a building now
sits inside flood water or debris. So the categories describe exposure, honestly:
  buildings  hit           ≥ 50 % of the footprint inside the affected area
             possibly hit  touches it, or lies within ~10 m of it (image alignment error)
  roads      cut           ≥ 20 m of the road runs through the affected area
             touched       some of it does (< 20 m)
  bridges    at risk       crosses or touches the affected river reach — likely damaged in a debris
                           flow, but must be verified on the ground or in very-high-resolution images
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rasterio.features import rasterize
from scipy import ndimage as ndi
from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
from shapely.ops import transform as shp_transform

from .grid import Grid

HIT, POSSIBLE, OUTSIDE = "hit", "possibly hit", "outside"


@dataclass
class DamageResult:
    buildings: list = field(default_factory=list)   # GeoJSON features (lon/lat) with properties.status
    roads: list = field(default_factory=list)
    bridges: list = field(default_factory=list)
    health: list = field(default_factory=list)
    places: list = field(default_factory=list)
    stretches: list = field(default_factory=list)   # the road stretches actually under water/debris (for maps)
    stats: dict = field(default_factory=dict)


def _to_grid_geom(grid: Grid):
    fwd = grid.from_lonlat().transform
    return lambda g: shp_transform(fwd, g)


def _sample(mask: np.ndarray, grid: Grid, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    col = ((xs - grid.transform.c) / grid.res).astype(int)
    row = ((grid.transform.f - ys) / grid.res).astype(int)
    ok = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
    out = np.zeros(len(xs), bool)
    out[ok] = mask[row[ok], col[ok]]
    return out


def buildings_status(features: list, affected: np.ndarray, grid: Grid, near_px: int = 1):
    """Fraction of each building footprint inside the affected area (rasterised at grid resolution)."""
    if not features:
        return []
    to_g = _to_grid_geom(grid)
    geoms = [to_g(shape(f["geometry"])) for f in features]
    ids = rasterize([(g, i + 1) for i, g in enumerate(geoms)], out_shape=grid.shape, transform=grid.transform,
                    all_touched=True, dtype="int32")
    near = ndi.binary_dilation(affected, iterations=near_px)
    n = len(features) + 1
    total = np.bincount(ids.ravel(), minlength=n)
    inside = np.bincount(ids.ravel(), weights=affected.ravel().astype(float), minlength=n)
    close = np.bincount(ids.ravel(), weights=near.ravel().astype(float), minlength=n)
    out = []
    for i, (f, g) in enumerate(zip(features, geoms), start=1):
        if total[i] == 0:  # hidden under a neighbour at this resolution: use its centre
            c = g.representative_point()
            frac = float(_sample(affected, grid, np.array([c.x]), np.array([c.y]))[0])
            nearby = bool(_sample(near, grid, np.array([c.x]), np.array([c.y]))[0])
        else:
            frac = inside[i] / total[i]
            nearby = close[i] > 0
        status = HIT if frac >= 0.5 else POSSIBLE if (frac > 0 or nearby) else OUTSIDE
        out.append({"type": "Feature", "geometry": f["geometry"],
                    "properties": {**f["properties"], "status": status, "affected_frac": round(float(frac), 2)}})
    return out


def _lines(g):
    if isinstance(g, LineString):
        return [g]
    if isinstance(g, MultiLineString):
        return list(g.geoms)
    return []


def roads_status(features: list, affected: np.ndarray, grid: Grid, step_m: float = 5.0, cut_m: float = 20.0,
                 stretches: list | None = None, near_m: float = 20.0):
    """Per road: cut (a continuous stretch ≥ cut_m under water/debris), touched (some of it under), near (within
    near_m of the footprint — a bank road can be undercut, and a 10 m map cannot place the edge exactly), or outside.
    If `stretches` is a list, the affected stretches themselves are appended to it as GeoJSON lines."""
    to_g = _to_grid_geom(grid)
    to_ll = grid.to_lonlat().transform
    near_mask = ndi.binary_dilation(affected, iterations=max(1, int(round(near_m / grid.res))))
    out = []
    for f in features:
        g = to_g(shape(f["geometry"]))
        length = g.length
        in_len, longest, close = 0.0, 0.0, False   # a zero-length way stays in the list: callers rely on the order
        for line in (_lines(g) if length > 0 else []):
            n = max(int(line.length / step_m), 1)
            pts = [line.interpolate(i / n, normalized=True) for i in range(n + 1)]
            xs, ys = np.array([p.x for p in pts]), np.array([p.y for p in pts])
            hit = _sample(affected, grid, xs, ys)
            close = close or bool(_sample(near_mask, grid, xs, ys).any())
            in_len += hit.sum() * line.length / (n + 1)
            run, seg = 0, []
            for pt, h in zip(pts + [None], list(hit) + [False]):  # longest continuous stretch under water/debris
                run = run + 1 if h else 0
                longest = max(longest, run * line.length / (n + 1))
                if h:
                    seg.append(pt)
                elif seg:
                    if stretches is not None and len(seg) >= 2:
                        stretches.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [list(to_ll(q.x, q.y)) for q in seg]},
                                          "properties": {"highway": f["properties"].get("highway"), "ref": f["properties"].get("ref"),
                                                         "name": f["properties"].get("name:en") or f["properties"].get("name"),
                                                         "length_m": round(len(seg) * line.length / (n + 1)), "blocks": len(seg) * line.length / (n + 1) >= cut_m}})
                    seg = []
        status = "cut" if longest >= cut_m else "touched" if in_len > 0 else "near" if close else OUTSIDE
        out.append({"type": "Feature", "geometry": f["geometry"],
                    "properties": {**f["properties"], "status": status, "length_m": round(length), "affected_m": round(in_len)}})
    return out


def bridges_status(features: list, affected: np.ndarray, grid: Grid, near_px: int = 3):
    near = ndi.binary_dilation(affected, iterations=near_px)
    to_g = _to_grid_geom(grid)
    out = []
    for f in features:
        g = to_g(shape(f["geometry"]))
        pts = [g] if isinstance(g, Point) else [g.interpolate(t, normalized=True) for t in np.linspace(0, 1, 9)] if hasattr(g, "interpolate") else [g.centroid]
        hit = _sample(near, grid, np.array([p.x for p in pts]), np.array([p.y for p in pts])).any()
        out.append({"type": "Feature", "geometry": f["geometry"], "properties": {**f["properties"], "status": "at risk" if hit else OUTSIDE}})
    return out


def points_status(features: list, affected: np.ndarray, grid: Grid, near_px: int = 2):
    """Health facilities / settlements: inside the footprint, next to it, or outside."""
    near = ndi.binary_dilation(affected, iterations=near_px)
    to_g = _to_grid_geom(grid)
    out = []
    for f in features:
        g = to_g(shape(f["geometry"]))
        c = g if isinstance(g, Point) else g.representative_point()
        xs, ys = np.array([c.x]), np.array([c.y])
        status = HIT if _sample(affected, grid, xs, ys)[0] else POSSIBLE if _sample(near, grid, xs, ys)[0] else OUTSIDE
        out.append({"type": "Feature", "geometry": mapping(shape(f["geometry"]).centroid) if not isinstance(shape(f["geometry"]), Point) else f["geometry"],
                    "properties": {**f["properties"], "status": status}})
    return out


def assess(cls: np.ndarray, grid: Grid, osm_layers: dict) -> DamageResult:
    affected = cls > 0
    r = DamageResult()
    r.buildings = buildings_status(osm_layers.get("buildings", {}).get("features", []), affected, grid)
    r.roads = roads_status(osm_layers.get("roads", {}).get("features", []), affected, grid, stretches=r.stretches)
    r.bridges = bridges_status(osm_layers.get("bridges", {}).get("features", []), affected, grid)
    r.health = points_status(osm_layers.get("health", {}).get("features", []), affected, grid)
    r.places = points_status(osm_layers.get("places", {}).get("features", []), affected, grid)
    count = lambda feats, s: sum(1 for f in feats if f["properties"]["status"] == s)  # noqa: E731
    r.stats = {
        "buildings_hit": count(r.buildings, HIT), "buildings_possibly_hit": count(r.buildings, POSSIBLE), "buildings_total": len(r.buildings),
        # km of road actually under water/debris: where it blocks the road ("cut") / anywhere ("affected")
        "roads_cut": count(r.roads, "cut"),
        "roads_blocked_km": round(sum(f["properties"]["affected_m"] for f in r.roads if f["properties"]["status"] == "cut") / 1000, 2),
        "roads_affected_km": round(sum(f["properties"]["affected_m"] for f in r.roads) / 1000, 2),
        "bridges_at_risk": count(r.bridges, "at risk"), "bridges_total": len(r.bridges),
        "health_hit": count(r.health, HIT), "health_near": count(r.health, POSSIBLE), "health_total": len(r.health),
        "places_hit": count(r.places, HIT), "places_near": count(r.places, POSSIBLE),
        "affected_km2": round(float(affected.sum()) * grid.res * grid.res / 1e6, 3),
        "water_km2": round(float((cls == 1).sum()) * grid.res * grid.res / 1e6, 3),
        "debris_km2": round(float((cls == 2).sum()) * grid.res * grid.res / 1e6, 3),
    }
    return r
