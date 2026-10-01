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
from concurrent.futures import TimeoutError as FuturesTimeout
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
    s2: dict = field(default_factory=dict)            # "before"/"after" composites, for the map
    radar: list = field(default_factory=list)         # per track: pre/post VV (dB) and a label, for the map
    model: str = ""                                   # which flood model ran ("" = none)
    osm_layers: dict = field(default_factory=dict)    # pre-event OSM layers used (for maps and exports)
    model_prob: np.ndarray | None = None              # flood-water probability from the model (NaN = no radar)
    osm_note: str = ""                                # set when OpenStreetMap could not be used
    osm_source: str = ""                              # "Overpass history" or "Geofabrik snapshot (<region>)"


def _say(run: Run, progress: Progress | None, msg: str, frac: float):
    run.log.append((round(time.time(), 1), msg))
    frac = max(frac, getattr(run, "_frac", 0.0))        # downloads finish in any order; the bar never goes back
    run._frac = frac
    if progress:
        progress(msg, frac)


# OpenStreetMap downloads run on their own daemon threads: if the public server is slow, the flood map is
# delivered anyway (after OSM_WAIT_S), while the download finishes in the background and is cached for a re-run.
OSM_WAIT_S = 600.0


class NoDataError(RuntimeError):
    """The area/date has no usable satellite image after the event: say so, don't draw an empty "no flood" map."""

    @classmethod
    def after(cls, event: dt.date):
        return cls(f"No usable satellite image of this area was found after {event:%d %b %Y}. Sentinel-1 and Sentinel-2 "
                   f"pass every few days and new images take a few days to appear — try a later flood date, and check "
                   f"that the area is on land.")


def _empty_layers() -> dict:
    return {name: {"type": "FeatureCollection", "features": []} for name in osm.LAYERS}


def analyse(bbox, event: dt.date, res: float = 10.0, out_dir: str | Path | None = None,
            progress: Progress | None = None, ml_model=None, osm_wait_s: float = OSM_WAIT_S) -> Run:
    t0 = time.time()
    grid = make_grid(bbox, res)
    run = Run(tuple(bbox), event, grid)
    pad = 0.03
    obox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    _say(run, progress, f"Area {grid.width * res / 1000:.1f} × {grid.height * res / 1000:.1f} km at {res:.0f} m; fetching data", 0.02)

    with ThreadPoolExecutor(10) as ex:
        # everything that only waits on the network starts at once; nothing waits for the map download
        f_pre = osm.run_in_background(osm.fetch_pre_event, obox, event, deadline=max(30.0, osm_wait_s - 20.0),
                                      say=lambda m: _say(run, progress, m, 0.1))
        f_s2 = ex.submit(optical.before_after, grid, event)
        f_dem = ex.submit(T.prefetch_dem, grid)
        try:
            stacks = sar.select_stacks(bbox, event)          # a quick catalogue search
        except Exception as e:  # noqa: BLE001 — optical and terrain can still map the area
            stacks = []
            run.tracks.append(f"Sentinel-1 catalogue unavailable ({e})")
        f_loads = {st.track: ex.submit(sar.load_stack, st, grid) for st in stacks}
        if not stacks:                                        # no radar at all: is there any optical image?
            try:
                _pre, _post = f_s2.result()
            except Exception:  # noqa: BLE001
                _post = None
            if not _post:
                raise NoDataError.after(event)
        _say(run, progress, f"Radar: {len(stacks)} Sentinel-1 track(s) with images before and after — downloading; "
                            f"OpenStreetMap before the event — one query", 0.05)
        finished = [0]

        def announce(label):
            def cb(fut):
                if fut.exception() is None:
                    finished[0] += 1
                    _say(run, progress, f"Downloaded: {label} ({time.time() - t0:.0f} s)", min(0.19, 0.05 + 0.025 * finished[0]))
            return cb

        f_s2.add_done_callback(announce("Sentinel-2 optical composites"))
        f_dem.add_done_callback(announce("Copernicus DEM"))
        for st in stacks:
            f_loads[st.track].add_done_callback(announce(f"Sentinel-1 track {st.track} ({st.direction}, {len(st.pre) + 1} images)"))
        run.osm_snapshot = osm.snapshot_date(event).isoformat()
        ctx = None
        try:
            pre_event = f_pre.result(timeout=max(1.0, osm_wait_s - (time.time() - t0)))
            layers, ctx = pre_event["layers"], pre_event["context"]
            run.osm_snapshot, run.osm_source = pre_event["snapshot"], pre_event["source"]
            _say(run, progress, f"OpenStreetMap as of {run.osm_snapshot} ({run.osm_source}): {len(layers['buildings']['features'])} buildings, "
                                f"{len(layers['roads']['features'])} roads", 0.2)
        except Exception as e:  # noqa: BLE001 — slow/failed map server: still map the flood
            layers = _empty_layers()
            run.osm_note = ("OpenStreetMap did not answer in time — buildings, roads and cut-off settlements are not counted. "
                            "Run the same area again in a few minutes: the map download continues and is cached."
                            if isinstance(e, FuturesTimeout) else f"OpenStreetMap unavailable ({e}) — damage and access not computed.")
            _say(run, progress, "OpenStreetMap: no answer yet — continuing with the flood map", 0.2)
        run.osm_layers = layers
        try:
            f_dem.result()
        except Exception:  # noqa: BLE001 — load_terrain retries the read itself
            pass
        f_ter = ex.submit(T.load_terrain, grid, layers["waterways"] if layers["waterways"]["features"] else None)
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
            run.radar.append({"pre_vv": feats["pre_vv"], "post_vv": feats["post_vv"],
                              "label": f"track {st.track} ({st.direction}), {st.pre[-1].date} → {st.post.date}"})
        _say(run, progress, f"Radar: {len(tracks)} usable track(s)", 0.6)
        pre, post = f_s2.result()
        run.s2 = {k: v for k, v in (("before", pre), ("after", post)) if v}
        run.optical_dates = {"before": [str(d) for d in (pre or {}).get("dates", [])], "after": [str(d) for d in (post or {}).get("dates", [])]}
        _say(run, progress, "Optical: before/after views built", 0.7)
        if not tracks and not post:
            raise NoDataError.after(event)


    s2pre = optical.indices(pre) if pre else None
    s2post = optical.indices(post) if post else None
    ml_prob = None
    if ml_model is not None and tracks:
        _say(run, progress, "AI flood model: reading the radar images", 0.72)
        try:
            ml_prob = ml_model(tracks, ter, grid)
            run.model = getattr(ml_model, "name", "flood model")
        except Exception as e:  # noqa: BLE001 — the physical rules still produce a map without the model
            run.log.append((round(time.time(), 1), f"flood model failed: {e}"))
    run.model_prob = ml_prob
    r = detect({"hand_major": ter.hand_major, "slope": ter.slope}, tracks, s2pre, s2post, ml_flood_prob=ml_prob, res=res)
    run.cls, run.conf, run.evidence = r["cls"], r["conf"], r["evidence"]
    _say(run, progress, f"Flood/debris map: {(run.cls > 0).sum() * res * res / 1e6:.2f} km² affected", 0.8)
    run.damage = damage.assess(run.cls, grid, layers)
    _say(run, progress, f"Damage: {run.damage.stats['buildings_hit']} buildings hit, {run.damage.stats['roads_cut']} roads cut", 0.88)
    run.access = access.analyse(layers, run.damage.roads, run.damage.bridges, run.cls > 0, grid, context=ctx)
    _say(run, progress, f"Access: {run.access.stats['cut_off']} settlements cut off", 0.95)
    run.seconds = round(time.time() - t0, 1)
    run.stats = {**run.damage.stats, **{f"access_{k}": v for k, v in run.access.stats.items()}, "seconds": run.seconds,
                 "tracks": run.tracks, "optical_dates": run.optical_dates, "osm_snapshot": run.osm_snapshot, "model": run.model,
                 "bbox": list(bbox), "event": event.isoformat(), "evidence_km2": evidence_areas(run), "osm_note": run.osm_note, "osm_source": run.osm_source}
    if out_dir:
        save(run, Path(out_dir))
    _say(run, progress, f"Done in {run.seconds:.0f} s", 1.0)
    return run


def evidence_areas(run: Run) -> dict:
    """How much of the affected area each kind of evidence supports (km²) — the map explains itself."""
    ev, aff = run.evidence or {}, run.cls > 0
    px = run.grid.res * run.grid.res / 1e6
    km2 = lambda m: round(float((m & aff).sum()) * px, 2)  # noqa: E731
    out = {k: km2(ev[k]) for k in ("radar_change", "radar_water", "optical_change", "optical_water", "model_water", "terrain_completed") if k in ev}
    out["affected"] = round(float(aff.sum()) * px, 2)
    return out


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
    affected_b = [f for f in run.damage.buildings if f["properties"]["status"] != damage.OUTSIDE]
    for name, feats in (("buildings", run.damage.buildings), ("buildings_affected", affected_b), ("roads", run.damage.roads),
                        ("roads_blocked", run.damage.stretches), ("bridges", run.damage.bridges),
                        ("health", run.damage.health), ("settlements", run.access.settlements), ("cut_roads", run.access.cut_edges)):
        (out / f"{name}.geojson").write_text(json.dumps(fc(feats)), encoding="utf8")
    from .render import write_layers
    write_layers(run, out)
    (out / "stats.json").write_text(json.dumps(run.stats, indent=1, default=str), encoding="utf8")
    (out / "log.json").write_text(json.dumps(run.log), encoding="utf8")
