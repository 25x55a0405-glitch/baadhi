"""Terrain from the Copernicus DEM: slope, river network, height above the nearest river (HAND),
and where each radar track cannot see the ground properly (layover / shadow).

HAND matters because floods and debris flows travel along valley floors: a pixel 150 m above the
river cannot be flooded by it, whatever the radar says.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .grid import Grid, make_grid
from .sources import planetary as P


@dataclass
class Terrain:
    grid: Grid
    dem: np.ndarray          # m
    slope: np.ndarray        # degrees
    aspect: np.ndarray       # downslope direction, degrees clockwise from north
    hand: np.ndarray         # height above nearest river with ≥ 1 km² catchment, m
    hand_major: np.ndarray   # height above nearest major river (≥ 50 km² catchment), m
    catchment_km2: np.ndarray  # upstream area draining through each pixel
    rivers: np.ndarray       # bool, river cells (≥ 1 km²)


def slope_aspect(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(np.where(np.isfinite(dem), dem, np.nanmean(dem)), res)  # gy: +south (rows go down), gx: +east
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # downslope direction: opposite of the gradient; north is -rows
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360
    return slope.astype("float32"), aspect.astype("float32")


def _to_grid(arr: np.ndarray, src: Grid, dst: Grid, resampling=Resampling.bilinear) -> np.ndarray:
    out = np.full(dst.shape, np.nan, dtype="float32")
    reproject(arr.astype("float32"), out, src_transform=src.transform, src_crs=src.crs, dst_transform=dst.transform,
              dst_crs=dst.crs, resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def rasterize_lines(features, grid: Grid, where=lambda props: True) -> np.ndarray:
    """Burn OSM line features (lon/lat GeoJSON) onto a grid; True where a line passes."""
    from rasterio.features import rasterize
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform

    fwd = grid.from_lonlat().transform
    shapes = [(shp_transform(fwd, shape(f["geometry"])), 1) for f in features if f.get("geometry") and where(f["properties"])]
    if not shapes:
        return np.zeros(grid.shape, bool)
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform, all_touched=True).astype(bool)


def hydrology(dem: np.ndarray, grid: Grid, minor_km2: float = 1.0, major_km2: float = 50.0,
              osm_minor: np.ndarray | None = None, osm_major: np.ndarray | None = None):
    """Flow routing with pysheds on the DEM grid. Returns catchment area (km²), HAND to minor and major rivers.

    Rivers entering the area from outside (e.g. from Tibet) have a tiny *local* catchment, so the DEM
    alone would call them streams. Mapped OSM rivers fix that: they are burned into the DEM (so flow
    follows them) and always count as rivers; OSM waterway=river counts as a major river."""
    from pysheds.grid import Grid as SGrid
    from pysheds.sview import Raster, ViewFinder

    if not hasattr(np, "in1d"):  # pysheds still calls np.in1d, removed in NumPy 2.4
        np.in1d = lambda a, b, **kw: np.isin(np.ravel(a), b, **kw)

    nodata = -9999.0
    d = np.where(np.isfinite(dem), dem, nodata).astype("float64")
    vf = ViewFinder(affine=grid.transform, shape=d.shape, crs=grid.crs.to_string(), nodata=nodata)
    r = Raster(d, viewfinder=vf)
    burned = d.copy()
    if osm_minor is not None:
        burned[osm_minor & (d != nodata)] -= 10
    if osm_major is not None:
        burned[osm_major & (d != nodata)] -= 20
    g = SGrid(viewfinder=vf)
    filled = g.resolve_flats(g.fill_depressions(g.fill_pits(Raster(burned, viewfinder=vf))))
    fdir = g.flowdir(filled)
    acc = np.asarray(g.accumulation(fdir), dtype="float64")
    cell_km2 = grid.res * grid.res / 1e6
    catch = (acc * cell_km2).astype("float32")
    minor, major = catch >= minor_km2, catch >= major_km2
    if osm_minor is not None:
        minor |= osm_minor
    if osm_major is not None:
        major |= osm_major
    vf_mask = ViewFinder(affine=grid.transform, shape=d.shape, crs=grid.crs.to_string(), nodata=False)
    as_raster = lambda m: Raster(m.astype(bool), viewfinder=vf_mask)  # noqa: E731 — pysheds wants its own type
    hand = np.asarray(g.compute_hand(fdir, r, as_raster(minor)), dtype="float32")
    hand_major = np.asarray(g.compute_hand(fdir, r, as_raster(major)), dtype="float32") if major.any() else np.full(d.shape, np.nan, "float32")
    for a in (hand, hand_major):
        a[(a < -1000) | ~np.isfinite(dem)] = np.nan
    return catch, hand, hand_major, minor


def prefetch_dem(grid: Grid, dem_res: float = 30.0, buffer_km: float = 3.0):
    """Start the DEM download early (it is cached on disk), so load_terrain later only computes."""
    w, s, e, n = grid.lonlat_bounds()
    pad = buffer_km / 111.0
    P.read_dem(make_grid((w - pad, s - pad, e + pad, n + pad), dem_res))


def load_terrain(grid: Grid, waterways: dict | None = None, dem_res: float = 30.0, buffer_km: float = 3.0) -> Terrain:
    """DEM + derived layers on the analysis grid. Hydrology runs at the DEM's native 30 m (faster, and
    10 m would only resample the same information) over the area plus a margin, so flow paths are not
    cut at the edge; everything is then resampled to the analysis grid.

    `waterways`: pre-event OSM waterway lines (GeoJSON), used to recognise rivers entering from outside."""
    w, s, e, n = grid.lonlat_bounds()
    pad = buffer_km / 111.0
    g30 = make_grid((w - pad, s - pad, e + pad, n + pad), dem_res)
    dem30 = P.read_dem(g30)
    osm_minor = osm_major = None
    if waterways and waterways.get("features"):
        feats = waterways["features"]
        osm_minor = rasterize_lines(feats, g30)
        osm_major = rasterize_lines(feats, g30, lambda p: p.get("waterway") == "river")
    catch, hand, hand_major, rivers = hydrology(dem30, g30, osm_minor=osm_minor, osm_major=osm_major)
    dem = _to_grid(dem30, g30, grid)
    slope, aspect = slope_aspect(dem, grid.res)
    return Terrain(
        grid=grid, dem=dem, slope=slope, aspect=aspect,
        hand=_to_grid(hand, g30, grid), hand_major=_to_grid(hand_major, g30, grid),
        catchment_km2=_to_grid(catch, g30, grid, Resampling.max),
        rivers=_to_grid(rivers.astype("float32"), g30, grid, Resampling.max) > 0.5,
    )


# ------------------------------------------------------------------------------ radar viewing geometry
# Sentinel-1 flies roughly north (ascending) or south (descending) and looks to the right.
HEADING = {"ascending": -12.5, "descending": 192.5}   # degrees from north, typical at 28°N
NEAR_INC, FAR_INC = 30.9, 46.0                         # IW swath incidence range


def incidence_from_footprint(grid: Grid, footprint, direction: str) -> np.ndarray:
    """Approximate incidence angle per pixel from its position across the swath (near → far range)."""
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform

    look = np.radians((HEADING[direction] + 90) % 360)
    ux, uy = np.sin(look), np.cos(look)  # east, north components of the look direction
    poly = shp_transform(grid.from_lonlat().transform, shape(footprint))
    xs, ys = np.asarray(poly.exterior.coords).T
    proj = xs * ux + ys * uy
    lo, hi = proj.min(), proj.max()
    l, b, r, t = grid.bounds
    cols = l + (np.arange(grid.width) + 0.5) * grid.res
    rows = t - (np.arange(grid.height) + 0.5) * grid.res
    p = cols[None, :] * ux + rows[:, None] * uy
    frac = np.clip((p - lo) / max(hi - lo, 1), 0, 1)
    return (NEAR_INC + frac * (FAR_INC - NEAR_INC)).astype("float32")


def radar_geometry(terrain: Terrain, direction: str, incidence: np.ndarray | float = 38.0):
    """Local incidence angle and masks for where this viewing direction sees the ground badly.

    cos θ_loc = cos s·cos θ − sin s·sin θ·cos(A − φ), with slope s, downslope azimuth A, look azimuth φ.
    Slopes facing the radar steeper than θ fold over (layover); slopes facing away with s + θ > 90° are
    in shadow. Returns (local_incidence_deg, good) where good excludes layover, shadow and severe
    foreshortening."""
    s = np.radians(terrain.slope)
    A = np.radians(terrain.aspect)
    phi = np.radians((HEADING[direction] + 90) % 360)
    th = np.radians(incidence)
    cos_loc = np.cos(s) * np.cos(th) - np.sin(s) * np.sin(th) * np.cos(A - phi)
    loc = np.degrees(np.arccos(np.clip(cos_loc, -1, 1)))
    facing = np.cos(A - phi) < 0  # slope faces the radar
    layover = facing & (s > th)
    shadow = loc >= 88
    foreshort = loc < 15
    good = ~(layover | shadow | foreshort) & np.isfinite(terrain.dem)
    return loc.astype("float32"), good
