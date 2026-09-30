"""The analysis grid: every input (radar, optical, terrain, map layers) is resampled onto one
metric grid for the area of interest, so pixels line up exactly.

The grid uses the UTM zone of the area's centre and 10 m pixels by default (Sentinel-1/2 resolution).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from affine import Affine
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class Grid:
    crs: CRS
    transform: Affine
    width: int
    height: int
    bbox_ll: tuple[float, float, float, float]  # the requested lon/lat box (west, south, east, north)

    @property
    def res(self) -> float:
        return self.transform.a

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform.c, self.transform.f
        return left, top - self.height * self.res, left + self.width * self.res, top

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def to_lonlat(self):
        return Transformer.from_crs(self.crs, 4326, always_xy=True)

    def from_lonlat(self):
        return Transformer.from_crs(4326, self.crs, always_xy=True)

    def lonlat_bounds(self) -> tuple[float, float, float, float]:
        """Exact lon/lat bounds of the grid (slightly larger than bbox_ll after snapping)."""
        t = self.to_lonlat()
        l, b, r, tp = self.bounds
        xs, ys = t.transform([l, r, l, r], [b, b, tp, tp])
        return min(xs), min(ys), max(xs), max(ys)

    def pixel_area_m2(self) -> float:
        return self.res * self.res


def utm_crs(lon: float, lat: float) -> CRS:
    zone = int(math.floor((lon + 180) / 6)) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def make_grid(bbox_ll: tuple[float, float, float, float], res: float = 10.0) -> Grid:
    """Grid covering a lon/lat box, snapped to whole pixels in the local UTM zone."""
    west, south, east, north = bbox_ll
    crs = utm_crs((west + east) / 2, (south + north) / 2)
    t = Transformer.from_crs(4326, crs, always_xy=True)
    xs, ys = t.transform([west, east, west, east], [south, south, north, north])
    left = math.floor(min(xs) / res) * res
    right = math.ceil(max(xs) / res) * res
    bottom = math.floor(min(ys) / res) * res
    top = math.ceil(max(ys) / res) * res
    width, height = int(round((right - left) / res)), int(round((top - bottom) / res))
    return Grid(crs, Affine(res, 0, left, 0, -res, top), width, height, tuple(bbox_ll))
