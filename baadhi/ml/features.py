"""Model inputs — defined ONCE and used identically for training (Kuro Siwo) and live inference.

Kuro Siwo radar is sigma0 with a Lee-sigma speckle filter. Our live radar (Planetary Computer RTC) is
terrain-flattened gamma0 without filtering. To give the model the same kind of picture it learned
from, live data is converted to sigma0-equivalent (gamma0 × cos θ) and filtered the same way before
these features are computed.

Channels (all float32, roughly −3…3):
  0 post VV (dB)   1 post VH (dB)   2 pre VV (dB)   3 pre VH (dB)
  4 ΔVV = post − pre (dB)           5 ΔVH (dB)
  6 slope (from the DEM)            7 height above the local valley floor (from the DEM)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter

CHANNELS = ("post_vv", "post_vh", "pre_vv", "pre_vh", "d_vv", "d_vh", "slope", "rel_elev")
N_CHANNELS = len(CHANNELS)


def to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(power, 1e-5, None))


def lee_sigma_like(power: np.ndarray, size: int = 7, looks: float = 4.4) -> np.ndarray:
    """A Lee filter (close to SNAP's Lee Sigma in effect): smooths speckle in flat areas, keeps edges."""
    p = np.where(np.isfinite(power), power, 0).astype("float64")
    valid = np.isfinite(power).astype("float64")
    n = np.maximum(uniform_filter(valid, size), 1e-6)
    mean = uniform_filter(p, size) / n
    sq = uniform_filter(p * p, size) / n
    var = np.maximum(sq - mean * mean, 0)
    cu2 = 1.0 / looks                        # speckle variance coefficient for `looks`-look intensity
    ci2 = var / np.maximum(mean * mean, 1e-12)
    w = np.clip((ci2 - cu2) / np.maximum(ci2 * (1 + cu2), 1e-12), 0, 1)
    out = mean + w * (p - mean)
    out[valid == 0] = np.nan
    return out.astype("float32")


def slope_deg(dem: np.ndarray, res: float = 10.0) -> np.ndarray:
    d = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    gy, gx = np.gradient(d, res)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")


def rel_elev(dem: np.ndarray, window_px: int = 151) -> np.ndarray:
    """Height above the lowest ground within ~1.5 km (at 10 m) — a DEM-only stand-in for 'height above
    the river' that can be computed the same way on a 224-pixel training tile and on a live scene."""
    d = np.where(np.isfinite(dem), dem, np.nanmax(dem) if np.isfinite(dem).any() else 0)
    return (d - minimum_filter(d, size=window_px)).astype("float32")


def build(post_vv, post_vh, pre_vv, pre_vh, dem, res: float = 10.0) -> np.ndarray:
    """Stack the 8 normalised channels from linear sigma0 (already speckle-filtered) and a DEM."""
    pv, ph, qv, qh = (to_db(a) for a in (post_vv, post_vh, pre_vv, pre_vh))
    x = np.stack([
        (pv + 15) / 6, (ph + 22) / 6, (qv + 15) / 6, (qh + 22) / 6,
        (pv - qv) / 3, (ph - qh) / 3,
        slope_deg(dem, res) / 20,
        np.clip(rel_elev(dem) / 40, 0, 4),
    ]).astype("float32")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
