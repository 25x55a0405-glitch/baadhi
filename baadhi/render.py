"""Map overlays for the dashboard: results and evidence as transparent PNGs in Web Mercator.

Every overlay is reprojected from the analysis grid (UTM) to EPSG:3857 so it sits exactly on the web
map; `bounds` is [west, south, east, north] in degrees for the map's image source.

  classes      flood water / debris-mud / affected river channel
  confidence   how many independent kinds of evidence agree (0.35 … 1)
  s2_before / s2_after   Sentinel-2 true colour, clear pixels only
  radar_change Sentinel-1 before/after composite: red = darker after (water, smooth mud),
               cyan = brighter after (rough debris, collapsed structures), grey = unchanged
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

from .grid import Grid

WEB = CRS.from_epsg(3857)
CLASS_RGBA = {1: (37, 111, 217, 215), 2: (214, 120, 32, 215), 3: (120, 170, 200, 150)}


def _warp(arrays: list[np.ndarray], grid: Grid, resampling=Resampling.nearest, nodata=np.nan):
    """Reproject same-shaped arrays from the grid to Web Mercator; returns (arrays, bounds_lonlat)."""
    l, b, r, t = grid.bounds
    dst_t, w, h = calculate_default_transform(grid.crs, WEB, grid.width, grid.height, left=l, bottom=b, right=r, top=t)
    out = []
    for a in arrays:
        src = a.astype("float32")
        dst = np.full((h, w), nodata, "float32")
        reproject(src, dst, src_transform=grid.transform, src_crs=grid.crs, dst_transform=dst_t, dst_crs=WEB,
                  resampling=resampling, src_nodata=nodata, dst_nodata=nodata)
        out.append(dst)
    wb, sb, eb, nb = array_bounds(h, w, dst_t)
    from pyproj import Transformer
    to_ll = Transformer.from_crs(WEB, "EPSG:4326", always_xy=True).transform
    (w_, s_), (e_, n_) = to_ll(wb, sb), to_ll(eb, nb)
    return out, [round(w_, 7), round(s_, 7), round(e_, 7), round(n_, 7)]


def _png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def classes_png(cls: np.ndarray, grid: Grid) -> tuple[bytes, list]:
    (c,), bounds = _warp([np.where(cls == 255, np.nan, cls)], grid)
    rgba = np.zeros(c.shape + (4,), np.uint8)
    for k, col in CLASS_RGBA.items():
        rgba[c == k] = col
    return _png(rgba), bounds


def confidence_png(conf: np.ndarray, cls: np.ndarray, grid: Grid) -> tuple[bytes, list]:
    (c,), bounds = _warp([np.where(cls > 0, conf, np.nan)], grid)
    rgba = np.zeros(c.shape + (4,), np.uint8)
    ok = np.isfinite(c)
    x = np.clip((c - 0.35) / 0.65, 0, 1)
    # low → pale yellow, high → deep red
    rgba[..., 0] = np.where(ok, 250 - 60 * x, 0).astype(np.uint8)
    rgba[..., 1] = np.where(ok, 225 - 175 * x, 0).astype(np.uint8)
    rgba[..., 2] = np.where(ok, 120 - 90 * x, 0).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 210, 0).astype(np.uint8)
    return _png(rgba), bounds


def prob_png(prob: np.ndarray, grid: Grid, lo: float = 0.3) -> tuple[bytes, list]:
    """Model flood-water probability: transparent below `lo`, then light → deep violet-blue."""
    (q,), bounds = _warp([prob], grid, Resampling.bilinear)
    ok = np.isfinite(q) & (q >= lo)
    x = np.clip((np.nan_to_num(q) - lo) / (1 - lo), 0, 1)
    rgba = np.zeros(q.shape + (4,), np.uint8)
    rgba[..., 0] = (150 - 110 * x).astype(np.uint8)
    rgba[..., 1] = (170 - 130 * x).astype(np.uint8)
    rgba[..., 2] = (255 - 55 * x).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 90 + 140 * x, 0).astype(np.uint8)
    return _png(rgba), bounds


def truecolor_png(s2: dict, grid: Grid, gain: float = 3.2, gamma: float = 0.85) -> tuple[bytes, list]:
    (r, g, b), bounds = _warp([s2["B04"], s2["B03"], s2["B02"]], grid, Resampling.bilinear)
    ok = np.isfinite(r) & np.isfinite(g) & np.isfinite(b)
    rgb = np.stack([np.clip(np.nan_to_num(x) * gain, 0, 1) ** gamma for x in (r, g, b)], -1)
    rgba = np.concatenate([(rgb * 255).astype(np.uint8), np.where(ok, 255, 0).astype(np.uint8)[..., None]], -1)
    return _png(rgba), bounds


def radar_change_png(pre_db: np.ndarray, post_db: np.ndarray, grid: Grid, lo: float = -25.0, hi: float = -3.0) -> tuple[bytes, list]:
    (a, p), bounds = _warp([pre_db, post_db], grid, Resampling.bilinear)
    ok = np.isfinite(a) & np.isfinite(p)
    s = lambda x: (np.clip((np.nan_to_num(x, nan=lo) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)  # noqa: E731
    rgba = np.stack([s(a), s(p), s(p), np.where(ok, 255, 0).astype(np.uint8)], -1)
    return _png(rgba), bounds


def write_layers(run, out: Path) -> dict:
    """Write every overlay PNG of a run into `out` and return the manifest (also saved as layers.json)."""
    out.mkdir(parents=True, exist_ok=True)
    g = run.grid
    manifest = {}

    def put(name, png_bounds, label):
        png, bounds = png_bounds
        (out / f"{name}.png").write_bytes(png)
        manifest[name] = {"file": f"{name}.png", "bounds": bounds, "label": label}

    put("classes", classes_png(run.cls, g), "Flood water · debris/mud · affected channel")
    put("confidence", confidence_png(run.conf, run.cls, g), "Confidence (agreeing evidence)")
    if getattr(run, "model_prob", None) is not None:
        put("model_prob", prob_png(run.model_prob, g), "AI model: flood-water probability")
    for key, label in (("before", "Sentinel-2 before"), ("after", "Sentinel-2 after")):
        s2 = (getattr(run, "s2", None) or {}).get(key)
        if s2 is not None:
            put(f"s2_{key}", truecolor_png(s2, g), f"{label} ({', '.join(run.optical_dates.get(key, [])) or '—'})")
    radar = [t for t in (getattr(run, "radar", None) or []) if t.get("pre_vv") is not None]
    if radar:
        best = max(radar, key=lambda t: float(np.isfinite(t["post_vv"]).mean()))
        put("radar_change", radar_change_png(best["pre_vv"], best["post_vv"], g), f"Radar change, {best.get('label', 'Sentinel-1')}")
    (out / "layers.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    return manifest
