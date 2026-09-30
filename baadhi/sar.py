"""Sentinel-1 before/after stacks for any area and date, speckle filtering and change measures.

Change detection compares images from the SAME track (same orbit, same viewing angle, same time of
day, 12 days apart). Each track gets up to three pre-event images, averaged to beat radar speckle and
to measure how much the ground normally varies between passes.
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter

from .grid import Grid
from .sources import planetary as P


@dataclass
class TrackStack:
    track: int
    direction: str
    pre: list = field(default_factory=list)   # S1Pass, oldest first
    post: object = None                       # S1Pass (first pass on/after the event)

    @property
    def lag_days(self) -> int:
        return (self.post.date - self._event).days

    def describe(self) -> str:
        return f"track {self.track} ({self.direction}): before {', '.join(str(p.date) for p in self.pre)} → after {self.post.date}"


def select_stacks(bbox, event: dt.date, max_pre: int = 3, pre_days: int = 40, post_days: int = 13) -> list[TrackStack]:
    """All tracks with a post-event pass within `post_days` and at least one earlier pass on the same track."""
    passes = P.s1_passes(bbox, event - dt.timedelta(days=pre_days), event + dt.timedelta(days=post_days))
    by_track: dict[int, list] = {}
    for p in passes:
        by_track.setdefault(p.track, []).append(p)
    stacks = []
    for track, ps in by_track.items():
        post = next((p for p in ps if p.date >= event), None)
        pre = [p for p in ps if p.date < event][-max_pre:]
        if post and pre:
            st = TrackStack(track, post.direction, pre, post)
            st._event = event
            stacks.append(st)
    return sorted(stacks, key=lambda s: (s.post.date, -len(s.pre)))


@dataclass
class StackData:
    stack: TrackStack
    pre_vv: list
    pre_vh: list
    post_vv: np.ndarray
    post_vh: np.ndarray


def load_stack(stack: TrackStack, grid: Grid, workers: int = 6) -> StackData:
    """Read all passes of a stack in parallel (each read is a handful of HTTP range requests)."""
    passes = stack.pre + [stack.post]
    with ThreadPoolExecutor(workers) as ex:
        arrays = list(ex.map(lambda p: P.read_s1(p, grid), passes))
    pre, post = arrays[:-1], arrays[-1]
    return StackData(stack, [a["vv"] for a in pre], [a["vh"] for a in pre], post["vv"], post["vh"])


def boxcar(power: np.ndarray, size: int = 3) -> np.ndarray:
    """Mean filter on linear power (NaN-aware). 3×3 on 4.4-look data ≈ 40 looks."""
    valid = np.isfinite(power)
    num = uniform_filter(np.where(valid, power, 0), size)
    den = uniform_filter(valid.astype("float32"), size)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.5] = np.nan
    return out.astype("float32")


def db(power: np.ndarray) -> np.ndarray:
    return (10 * np.log10(np.clip(power, 1e-5, None))).astype("float32")


def change_features(sd: StackData, size: int = 3) -> dict[str, np.ndarray]:
    """Per-pixel change measures for one track (all in dB).

    d_vv/d_vh:  post minus mean of pre-event (log-ratio)
    z_vv/z_vh:  that change divided by the natural pass-to-pass variability (needs ≥ 2 pre images)
    post_vv/post_vh, pre_vv/pre_vh: filtered backscatter levels
    """
    f = {}
    for pol, pre, post in (("vv", sd.pre_vv, sd.post_vv), ("vh", sd.pre_vh, sd.post_vh)):
        pre_f = [boxcar(p, size) for p in pre]
        pre_mean = np.nanmean(pre_f, axis=0) if len(pre_f) > 1 else pre_f[0]
        post_f = boxcar(post, size)
        f[f"pre_{pol}"], f[f"post_{pol}"] = db(pre_mean), db(post_f)
        f[f"d_{pol}"] = f[f"post_{pol}"] - f[f"pre_{pol}"]
        if len(pre_f) > 1:
            pre_db = np.stack([db(p) for p in pre_f])
            noise = np.nanstd(pre_db, axis=0)
            # speckle floor so a lucky stable pixel doesn't blow up the ratio
            f[f"z_{pol}"] = f[f"d_{pol}"] / np.maximum(noise, 0.8)
        else:
            f[f"z_{pol}"] = f[f"d_{pol}"] / 1.5
    return f
