"""End to end: an area + a flood date → flood/debris map, damage, cut-off settlements, report numbers.

Only allowed inputs are used: Sentinel-1, Sentinel-2, Copernicus DEM and pre-event OpenStreetMap.
Independent downloads run in parallel (they wait on the network, not the CPU).

    from baadhi.pipeline import analyse
    run = analyse((85.30, 28.13, 85.40, 28.29), date(2026, 8, 26), out_dir="runs/trishuli")
"""
from __future__ import annotations

import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import access, damage, optical, sar
from . import terrain as T
from .detect import detect
from .grid import Grid, make_grid
from .sources import osm

Progress = Callable[[str, float], None]


@dataclass
class Run:
    bbox: tuple
    event: dt.date
    grid: Grid
    log: list = field(default_factory=list)
    tracks: list = field(default_factory=list)       # descriptions of radar stacks used
    optical_dates: dict = field(default_factory=dict)
    cls: np.ndarray | None = None
    conf: np.ndarray | None = None
    evidence: dict = field(default_factory=dict)
    terrain: T.Terrain | None = None
    damage: damage.DamageResult | None = None
    access: access.AccessResult | None = None
    osm_snapshot: str = ""
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0


def _say(run: Run, progress: Progress | None, msg: str, frac: float):
    run.log.append((round(time.time(), 1), msg))
    if progress:
        progress(msg, frac)


def analyse(bbox, event: dt.date, res: float = 10.0, out_dir: str | Path | None = None,
            progress: Progress | None = None, ml_model=None) -> Run:
    t0 = time.time()
    grid = make_grid(bbox, res)
    run = Run(tuple(bbox), event, grid)
    pad = 0.03
    obox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    _say(run, progress, f"Area {grid.width * res / 1000:.1f} × {grid.height * res / 1000:.1f} km at {res:.0f} m; fetching data", 0.02)

    with ThreadPoolExecutor(8) as ex:
        f_osm = ex.submit(osm.fetch_all, obox, event)
        f_ctx = ex.submit(osm.fetch_context, obox, event)
        f_stacks = ex.submit(sar.select_stacks, bbox, event)
        f_s2 = ex.submit(optical.before_after, grid, event)
        layers = f_osm.result()
        run.osm_snapshot = osm.snapshot_date(event).isoformat()
        _say(run, progress, f"OpenStreetMap as of {run.osm_snapshot}: {len(layers['buildings']['features'])} buildings, {len(layers['roads']['features'])} roads", 0.2)
        f_ter = ex.submit(T.load_terrain, grid, layers["waterways"])
        stacks = f_stacks.result()
        f_loads = {st.track: ex.submit(sar.load_stack, st, grid) for st in stacks}
        ter = f_ter.result()
        run.terrain = ter
        _say(run, progress, "Terrain: rivers, height above river, radar blind spots", 0.35)
        tracks = []
        for st in stacks:
            try:
                sd = f_loads[st.track].result()
            except Exception as e:  # noqa: BLE001 — one bad track must not sink the run
                run.tracks.append(f"{st.describe()} — skipped ({e})")
                continue
            cover = float(np.isfinite(sd.post_vv).mean())
            if cover < 0.3:
                run.tracks.append(f"{st.describe()} — skipped (covers {cover:.0%} of the area)")
                continue
            feats = sar.change_features(sd)
            inc = T.incidence_from_footprint(grid, st.post.items[0].geometry, st.direction)
            _, good = T.radar_geometry(ter, st.direction, inc)
            tracks.append({**{k: feats[k] for k in ("z_vv", "z_vh", "post_vv", "post_vh", "d_vv", "d_vh")}, "good": good, "stack": sd})
            run.tracks.append(st.describe())
        _say(run, progress, f"Radar: {len(tracks)} usable track(s)", 0.6)
        pre, post = f_s2.result()
        run.optical_dates = {"before": [str(d) for d in (pre or {}).get("dates", [])], "after": [str(d) for d in (post or {}).get("dates", [])]}
        _say(run, progress, "Optical: before/after views built", 0.7)
        ctx = f_ctx.result()

    s2pre = optical.indices(pre) if pre else None
    s2post = optical.indices(post) if post else None
    ml_prob = ml_model(tracks, ter, grid) if (ml_model and tracks) else None
    r = detect({"hand_major": ter.hand_major, "slope": ter.slope}, tracks, s2pre, s2post, ml_flood_prob=ml_prob, res=res)
    run.cls, run.conf, run.evidence = r["cls"], r["conf"], r["evidence"]
    _say(run, progress, f"Flood/debris map: {(run.cls > 0).sum() * res * res / 1e6:.2f} km² affected", 0.8)
    run.damage = damage.assess(run.cls, grid, layers)
    _say(run, progress, f"Damage: {run.damage.stats['buildings_hit']} buildings hit, {run.damage.stats['roads_cut']} roads cut", 0.88)
    run.access = access.analyse(layers, run.damage.roads, run.damage.bridges, run.cls > 0, grid, context=ctx)
    _say(run, progress, f"Access: {run.access.stats['cut_off']} settlements cut off", 0.95)
    run.seconds = round(time.time() - t0, 1)
    run.stats = {**run.damage.stats, **{f"access_{k}": v for k, v in run.access.stats.items()}, "seconds": run.seconds,
                 "tracks": run.tracks, "optical_dates": run.optical_dates, "osm_snapshot": run.osm_snapshot,
                 "bbox": list(bbox), "event": event.isoformat()}
    if out_dir:
        save(run, Path(out_dir))
    _say(run, progress, f"Done in {run.seconds:.0f} s", 1.0)
    return run


def save(run: Run, out: Path):
    """Write the run bundle: class/confidence GeoTIFFs, GeoJSON layers, stats."""
    import rasterio

    out.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", width=run.grid.width, height=run.grid.height, count=1, crs=run.grid.crs,
                transform=run.grid.transform, compress="deflate")
    with rasterio.open(out / "classes.tif", "w", dtype="uint8", nodata=255, **prof) as dst:
        dst.write(run.cls, 1)
    with rasterio.open(out / "confidence.tif", "w", dtype="float32", **prof) as dst:
        dst.write(run.conf, 1)
    fc = lambda feats: {"type": "FeatureCollection", "features": feats}  # noqa: E731
    for name, feats in (("buildings", run.damage.buildings), ("roads", run.damage.roads), ("bridges", run.damage.bridges),
                        ("health", run.damage.health), ("settlements", run.access.settlements), ("cut_roads", run.access.cut_edges)):
        (out / f"{name}.geojson").write_text(json.dumps(fc(feats)), encoding="utf8")
    (out / "stats.json").write_text(json.dumps(run.stats, indent=1, default=str), encoding="utf8")
    (out / "log.json").write_text(json.dumps(run.log), encoding="utf8")
