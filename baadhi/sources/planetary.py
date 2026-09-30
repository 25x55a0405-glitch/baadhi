"""Sentinel-1, Sentinel-2 and Copernicus DEM from Microsoft Planetary Computer (free, no account).

These are the same Copernicus Sentinel and Copernicus DEM data the hackathon allows as inputs;
Planetary Computer serves them as cloud-optimised GeoTIFFs, which we read window-by-window
straight onto the analysis grid (only the pixels we need are downloaded).

Sentinel-1 comes as RTC (radiometrically terrain-corrected gamma0, linear power), which matters in
the Himalaya: slopes facing the radar would otherwise look bright and slopes facing away dark.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from ..grid import Grid

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
_catalog = None

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
    GDAL_CACHEMAX=512,
)


def catalog():
    global _catalog
    if _catalog is None:
        _catalog = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    return _catalog


def search(collection: str, bbox, start: dt.date, end: dt.date, **query):
    items = catalog().search(collections=[collection], bbox=list(bbox), datetime=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}", query=query or None).items()
    return sorted(items, key=lambda i: i.datetime)


def read_to_grid(href: str, grid: Grid, resampling=Resampling.bilinear, nodata=None, band: int = 1) -> np.ndarray:
    """Read one band of a (remote) GeoTIFF, warped onto the analysis grid. Missing areas → NaN."""
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            nd = src.nodata if nodata is None else nodata
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width, height=grid.height,
                           resampling=resampling, src_nodata=nd, nodata=np.nan, dtype="float32") as vrt:
                return vrt.read(band, out_dtype="float32")


def mosaic(arrays: list[np.ndarray]) -> np.ndarray:
    """First valid (non-NaN) value wins — used to join consecutive frames of the same pass."""
    out = np.full(arrays[0].shape, np.nan, dtype="float32")
    for a in arrays:
        m = np.isnan(out) & np.isfinite(a)
        out[m] = a[m]
    return out


# ---------------------------------------------------------------------------------- Sentinel-1
@dataclass
class S1Pass:
    """All frames of one satellite pass over the area (same date, same relative orbit)."""
    date: dt.date
    track: int
    direction: str
    platform: str
    items: list = field(default_factory=list)

    @property
    def key(self):
        return (self.date, self.track)


def s1_passes(bbox, start: dt.date, end: dt.date) -> list[S1Pass]:
    groups: dict[tuple, S1Pass] = {}
    for it in search("sentinel-1-rtc", bbox, start, end):
        p = it.properties
        d = it.datetime.date()
        k = (d, p["sat:relative_orbit"])
        if k not in groups:
            groups[k] = S1Pass(d, p["sat:relative_orbit"], p["sat:orbit_state"], p.get("platform", "sentinel-1"))
        groups[k].items.append(it)
    return sorted(groups.values(), key=lambda s: (s.date, s.track))


def read_s1(s1: S1Pass, grid: Grid, pols=("vv", "vh")) -> dict[str, np.ndarray]:
    """gamma0 (linear power) for each polarisation, mosaicked across the pass's frames."""
    out = {}
    for pol in pols:
        out[pol] = mosaic([read_to_grid(it.assets[pol].href, grid, Resampling.bilinear, nodata=0) for it in s1.items if pol in it.assets])
    return out


# ---------------------------------------------------------------------------------- Sentinel-2
S2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


def s2_scenes(bbox, start: dt.date, end: dt.date, max_cloud: float = 80.0):
    return search("sentinel-2-l2a", bbox, start, end, **{"eo:cloud_cover": {"lt": max_cloud}})


def read_s2(items, grid: Grid, bands=S2_BANDS) -> dict[str, np.ndarray]:
    """Surface reflectance (0–1) for one date, mosaicked across tiles, with clouds and shadows set to NaN."""
    scl = mosaic([read_to_grid(it.assets["SCL"].href, grid, Resampling.nearest, nodata=0) for it in items])
    # SCL: 3 cloud shadow, 8/9 cloud medium/high, 10 thin cirrus, 1 saturated
    bad = np.isin(scl, [1, 3, 8, 9, 10]) | np.isnan(scl)
    out = {"SCL": scl}
    for b in bands:
        arr = mosaic([read_to_grid(it.assets[b].href, grid, Resampling.bilinear, nodata=0) for it in items])
        # L2A processing baseline ≥ 04.00 adds an offset of 1000 to digital numbers
        base = float(items[0].properties.get("s2:processing_baseline", "04.00"))
        arr = (arr - (1000 if base >= 4 else 0)) / 10000.0
        arr[bad] = np.nan
        out[b] = arr
    return out


# ---------------------------------------------------------------------------------- Copernicus DEM
def read_dem(grid: Grid) -> np.ndarray:
    items = search("cop-dem-glo-30", grid.lonlat_bounds(), dt.date(2000, 1, 1), dt.date(2100, 1, 1))
    return mosaic([read_to_grid(it.assets["data"].href, grid, Resampling.bilinear) for it in items])


def group_by_date(items) -> dict[dt.date, list]:
    g = defaultdict(list)
    for it in items:
        g[it.datetime.date()].append(it)
    return dict(sorted(g.items()))
