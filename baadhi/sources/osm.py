"""OpenStreetMap as it was BEFORE the event.

The hackathon rules forbid OSM edits made after the event (mappers trace damage from the same
satellite images), so every query asks for the map as of the day before the event.

Two back-ends, same result:
  1. ohsome API (full OSM history; the rules' suggested source) — elements/geometry at a date;
  2. Overpass API "attic" query with [date:"…"] — the map as it stood at that moment.
ohsome's geometry endpoint has been answering 403 from some networks, so Overpass is the fallback.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import requests

OHSOME = "https://api.ohsome.org/v1"
# The three official instances all hold the full history ("attic") and have separate rate limits; when one is
# overloaded the next is tried. (Mirrors without history would silently return today's map — never use those.)
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://lz4.overpass-api.de/api/interpreter",
            "https://z.overpass-api.de/api/interpreter"]
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "osm"
UA = {"User-Agent": "Baadhi/0.1 (flood damage mapping research; github.com)", "Accept": "application/json"}

LAYERS = ("waterways", "water", "buildings", "roads", "bridges", "places", "health")
# Fetched over a wider area (~25 km around), so routes can reach hospitals and towns beyond the analysis area
CONTEXT_LAYERS = ("roads_major", "health", "towns")

OHSOME_FILTER = {
    "waterways": "waterway in (river, stream, canal, drain) and type:way",
    "water": "(natural=water or waterway=riverbank or landuse=reservoir) and geometry:polygon",
    "buildings": "building=* and geometry:polygon",
    "roads": "highway in (motorway, trunk, primary, secondary, tertiary, unclassified, residential, service, track, living_street, road, "
             "motorway_link, trunk_link, primary_link, secondary_link, tertiary_link, path, footway, steps, bridleway) and type:way",
    "bridges": "(bridge=* and bridge!=no and type:way) or man_made=bridge",
    "places": "place in (city, town, village, hamlet, isolated_dwelling, suburb, neighbourhood, locality) and type:node",
    "health": "(amenity in (hospital, clinic, doctors) or healthcare in (hospital, clinic, centre, doctor)) and (type:node or type:way)",
    "roads_major": "highway in (motorway, trunk, primary, secondary, tertiary, unclassified, road, motorway_link, trunk_link, "
                   "primary_link, secondary_link, tertiary_link) and type:way",
    "towns": "place in (city, town) and type:node",
}
ROAD_RE = "^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track|living_street|road|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|path|footway|steps|bridleway)$"
OVERPASS_Q = {
    "waterways": 'way["waterway"~"^(river|stream|canal|drain)$"]({b});',
    "water": 'way["natural"="water"]({b});way["waterway"="riverbank"]({b});way["landuse"="reservoir"]({b});',
    "buildings": 'way["building"]({b});',
    "roads": f'way["highway"~"{ROAD_RE}"]({{b}});',
    "bridges": 'way["bridge"]["bridge"!="no"]({b});way["man_made"="bridge"]({b});node["man_made"="bridge"]({b});',
    "places": 'node["place"~"^(city|town|village|hamlet|isolated_dwelling|suburb|neighbourhood|locality)$"]({b});',
    "health": 'nwr["amenity"~"^(hospital|clinic|doctors)$"]({b});nwr["healthcare"~"^(hospital|clinic|centre|doctor)$"]({b});',
    "roads_major": 'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|road|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link)$"]({b});',
    "towns": 'node["place"~"^(city|town)$"]({b});',
}
POLYGON_LAYERS = {"water", "buildings"}
KEEP_TAGS = {"name", "name:en", "name:ne", "highway", "bridge", "waterway", "place", "amenity", "healthcare", "building",
             "surface", "population", "layer", "tunnel", "ford", "natural", "man_made", "emergency", "landuse", "ref"}

_latest: dict[str, str] = {}
_ohsome_down = {"flag": False}


def latest_snapshot() -> dt.date:
    """Newest date the OSM history services know about (Overpass is near real-time)."""
    if "d" not in _latest:
        try:
            _latest["d"] = requests.get(f"{OHSOME}/metadata", timeout=30).json()["extractRegion"]["temporalExtent"]["toTimestamp"][:10]
        except Exception:  # noqa: BLE001
            _latest["d"] = dt.date.today().isoformat()
    return dt.date.fromisoformat(_latest["d"])


def snapshot_date(event: dt.date) -> dt.date:
    """Midnight UTC two days before the event date. A flood "on 26 Aug" in Nepal can start on 25 Aug in
    UTC (EMSR927: 25 Aug 22:00 UTC), so "the day before" is not safe; two days before always is."""
    return event - dt.timedelta(days=2)


def _tags(p: dict) -> dict:
    return {k: v for k, v in (p or {}).items() if k in KEEP_TAGS}


def _via_ohsome(layer: str, bbox, when: dt.date, timeout: int) -> dict:
    if when > latest_snapshot():
        raise RuntimeError(f"ohsome data ends {latest_snapshot()}")
    params = {"bboxes": ",".join(f"{v:.6f}" for v in bbox), "time": when.isoformat(), "filter": OHSOME_FILTER[layer],
              "properties": "tags", "clipGeometry": "false"}
    r = requests.post(f"{OHSOME}/elements/geometry", timeout=timeout, data=params, headers=UA)
    if r.status_code != 200:
        raise RuntimeError(f"ohsome HTTP {r.status_code}")
    feats = []
    for f in r.json().get("features", []):
        f["properties"] = {**_tags(f.get("properties")), "osm_id": (f.get("properties") or {}).get("@osmId")}
        feats.append(f)
    return {"type": "FeatureCollection", "features": feats}


_slots = threading.BoundedSemaphore(2)   # overpass-api.de allows 2 running queries per IP address


def _slot_wait(url: str) -> float:
    """Seconds until the server frees a slot for us (its /status page says so), 0 if one is free."""
    try:
        txt = requests.get(url.replace("/interpreter", "/status"), headers=UA, timeout=15).text
    except requests.RequestException:
        return 10.0
    free = re.search(r"(\d+) slots? available now", txt)
    if free and int(free.group(1)) > 0:
        return 0.0
    waits = [int(m) for m in re.findall(r"in (\d+) seconds", txt)]
    return float(min(waits)) + 1 if waits else 5.0


def _run_query(q: str, timeout: int) -> dict:
    """Run an Overpass query, rotating over the official instances and waiting politely for a free slot instead
    of hammering them. A reply that reports a runtime error (e.g. a timeout part-way) is rejected — partial map
    data would silently undercount — and a server that never answers is not asked the same job again."""
    last = None

    def ok(r):
        if r.status_code != 200:
            return None
        data = r.json()
        remark = str(data.get("remark") or "")
        if "error" in remark.lower():
            raise RuntimeError(f"Overpass: {remark[:160]}")
        return data

    with _slots:
        for attempt in range(6):
            url = OVERPASS[attempt % len(OVERPASS)]
            try:
                r = requests.post(url, data={"data": q}, headers=UA, timeout=timeout + 30)
                data = ok(r)
                if data is not None:
                    return data
                last = f"{url} HTTP {r.status_code}"
                if r.status_code not in (429, 503, 504):
                    break
            except requests.ReadTimeout:
                last = f"{url} gave no answer within {timeout + 30} s"
                break
            except requests.RequestException as ex:
                last = f"{url} {type(ex).__name__}"
            time.sleep(min(_slot_wait(url) or 3.0 * (attempt + 1), 30))
    raise RuntimeError(f"Overpass failed: {last}")


def _bbox_q(bbox) -> str:
    w, s, e, n = bbox
    return f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"


def _via_overpass(layer: str, bbox, when: dt.date, timeout: int) -> dict:
    q = f'[out:json][timeout:{timeout}][date:"{when.isoformat()}T00:00:00Z"];({OVERPASS_Q[layer].format(b=_bbox_q(bbox))});out tags geom;'
    return _overpass_to_geojson(_run_query(q, timeout), layer)


def _via_overpass_combined(layers, bbox, when: dt.date, timeout: int = 480) -> dict[str, dict]:
    """Every layer in ONE attic query: the public server throttles per query, so one round trip is far
    faster than one per layer and tile. Each layer's elements are announced by a marker element."""
    b = _bbox_q(bbox)
    sets = "".join(f"({OVERPASS_Q[name].format(b=b)})->.s{i};" for i, name in enumerate(layers))
    outs = "".join(f'make layer name="{name}";out;.s{i} out tags geom;' for i, name in enumerate(layers))
    q = f'[out:json][timeout:{timeout}][maxsize:1073741824][date:"{when.isoformat()}T00:00:00Z"];{sets}{outs}'
    data = _run_query(q, timeout)
    groups, cur = {name: [] for name in layers}, None
    for el in data.get("elements", []):
        if el.get("type") == "layer":
            cur = (el.get("tags") or {}).get("name")
        elif cur in groups:
            groups[cur].append(el)
    missing = [name for name in layers if not any(e.get("type") == "layer" and (e.get("tags") or {}).get("name") == name
                                                  for e in data.get("elements", []))]
    if missing:
        raise RuntimeError(f"Overpass reply is missing layers {missing}")
    return {name: _overpass_to_geojson({"elements": els}, name) for name, els in groups.items()}


def _overpass_to_geojson(data: dict, layer: str) -> dict:
    feats = []
    for el in data.get("elements", []):
        tags = _tags(el.get("tags"))
        props = {**tags, "osm_id": f"{el['type']}/{el['id']}"}
        if el["type"] == "node":
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif el["type"] == "way" and el.get("geometry"):
            coords = [[p["lon"], p["lat"]] for p in el["geometry"] if p]
            closed = len(coords) >= 4 and coords[0] == coords[-1]
            if layer in POLYGON_LAYERS or (layer == "health" and closed):
                if not closed:
                    continue
                geom = {"type": "Polygon", "coordinates": [coords]}
            else:
                geom = {"type": "LineString", "coordinates": coords}
        elif el["type"] == "relation" and el.get("bounds") and layer == "health":
            bb = el["bounds"]
            geom = {"type": "Point", "coordinates": [(bb["minlon"] + bb["maxlon"]) / 2, (bb["minlat"] + bb["maxlat"]) / 2]}
        else:
            continue
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def _cache_path(layer: str, bbox, when: dt.date) -> Path:
    key = hashlib.sha1(f"{layer}|{','.join(f'{v:.5f}' for v in bbox)}|{when}".encode()).hexdigest()[:16]
    return CACHE / f"{layer}-{when}-{key}.geojson"


def _cache_read(layer: str, bbox, when: dt.date) -> dict | None:
    path = _cache_path(layer, bbox, when)
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else None


def _cache_write(layer: str, bbox, when: dt.date, fc: dict):
    CACHE.mkdir(parents=True, exist_ok=True)
    _cache_path(layer, bbox, when).write_text(json.dumps(fc), encoding="utf8")


def fetch(layer: str, bbox, when: dt.date, timeout: int = 180) -> dict:
    """GeoJSON FeatureCollection of one layer as of `when`, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = _cache_path(layer, bbox, when)
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    errors = []
    for attempt in range(4):
        for name, fn in (("ohsome", _via_ohsome), ("overpass", _via_overpass)):
            if name == "ohsome" and _ohsome_down["flag"]:
                continue  # circuit breaker: after one failure, don't wait on ohsome again this run
            try:
                fc = fn(layer, bbox, when, 30 if name == "ohsome" else timeout)
                fc.update(snapshot=when.isoformat(), source=name)
                path.write_text(json.dumps(fc), encoding="utf8")
                return fc
            except Exception as ex:  # noqa: BLE001
                errors.append(f"{name}: {ex}")
                if name == "ohsome":
                    _ohsome_down["flag"] = True
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"OSM {layer} unavailable — " + "; ".join(errors[-4:]))


TILED = {"buildings", "roads"}   # dense layers: fetched in ~0.1° tiles so no single query is huge


def _tiles(bbox, step: float = 0.1):
    w, s, e, n = bbox
    xs = list(np.arange(w, e, step)) + [e]
    ys = list(np.arange(s, n, step)) + [n]
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(len(xs) - 1) for j in range(len(ys) - 1)]


def fetch_tiled(layer: str, bbox, when: dt.date) -> dict:
    feats, seen, src = [], set(), set()
    for tb in _tiles(bbox):
        fc = fetch(layer, tb, when)
        src.add(fc.get("source"))
        for f in fc["features"]:
            k = f["properties"].get("osm_id") or json.dumps(f["geometry"])[:80]
            if k not in seen:
                seen.add(k)
                feats.append(f)
    return {"type": "FeatureCollection", "features": feats, "snapshot": when.isoformat(), "source": "+".join(sorted(s for s in src if s))}


def _tiled_cache_read(layer: str, bbox, when: dt.date) -> dict | None:
    """An area fetched tile by tile earlier (older runs) is still reused from disk."""
    tiles = _tiles(bbox)
    if not all(_cache_path(layer, tb, when).exists() for tb in tiles):
        return None
    return fetch_tiled(layer, bbox, when)


def fetch_many(layers, bbox, when: dt.date, timeout: int = 480) -> dict[str, dict]:
    """Several layers for one box: from disk if cached, else ONE combined query, else layer by layer."""
    out = {}
    for name in layers:
        fc = _cache_read(name, bbox, when)
        if fc is None and name in TILED:
            fc = _tiled_cache_read(name, bbox, when)
        if fc is not None:
            out[name] = fc
    missing = [name for name in layers if name not in out]
    if missing:
        try:
            got = _via_overpass_combined(missing, bbox, when, timeout)
            for name, fc in got.items():
                fc.update(snapshot=when.isoformat(), source="overpass")
                _cache_write(name, bbox, when, fc)
                out[name] = fc
        except Exception:  # noqa: BLE001 — too big or the server refused: fall back to smaller queries
            for name in missing:
                out[name] = fetch_tiled(name, bbox, when) if name in TILED else fetch(name, bbox, when)
    return out


def fetch_all(bbox, event: dt.date, layers=LAYERS) -> dict[str, dict]:
    return fetch_many(layers, bbox, snapshot_date(event))


def expand(bbox, km: float):
    import math
    lat = (bbox[1] + bbox[3]) / 2
    dlat, dlon = km / 111.0, km / (111.0 * max(math.cos(math.radians(lat)), 0.2))
    return (bbox[0] - dlon, bbox[1] - dlat, bbox[2] + dlon, bbox[3] + dlat)


def fetch_context(bbox, event: dt.date, km: float = 25.0) -> dict[str, dict]:
    """Main roads, health facilities and towns in a wider box, for routing beyond the analysis area."""
    when, big = snapshot_date(event), expand(bbox, km)
    return {f"context_{name}": fc for name, fc in fetch_many(CONTEXT_LAYERS, big, when).items()}


def run_in_background(fn, *args, **kwargs):
    """Like ThreadPoolExecutor.submit, but on a daemon thread: a request stuck on a slow public server must
    never keep the process (a command-line run, the dashboard on Ctrl+C) from exiting."""
    from concurrent.futures import Future
    fut: Future = Future()

    def work():
        if not fut.set_running_or_notify_cancel():
            return
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001 — handed to whoever reads the future
            fut.set_exception(e)

    threading.Thread(target=work, daemon=True, name="osm-bg").start()
    return fut


def fetch_pre_event(bbox, event: dt.date, km: float = 25.0, patience: float = 150.0, deadline: float = 570.0, say=None) -> dict:
    """Pre-event OpenStreetMap for an area and its surroundings, from whichever source answers first.

    1. Overpass history ("attic") queries — the map exactly as it stood two days before the event.
    2. If they have not answered after `patience` seconds, Geofabrik's newest dated regional snapshot before
       the event is prepared in parallel (downloaded once, cut out locally); the first complete result wins.
    Returns {"layers", "context", "source", "snapshot"}; raises if nothing is ready within `deadline` s.
    Slow downloads keep going on daemon threads and fill the disk cache for the next run."""
    from concurrent.futures import wait

    from . import geofabrik

    when, big = snapshot_date(event), expand(bbox, km)
    t0 = time.time()
    f_area = run_in_background(fetch_many, LAYERS, bbox, when)
    f_ctx = run_in_background(fetch_many, CONTEXT_LAYERS, big, when)

    def overpass():
        if f_area.done() and f_ctx.done() and f_area.exception() is None and f_ctx.exception() is None:
            return {"layers": f_area.result(), "context": {f"context_{k}": v for k, v in f_ctx.result().items()},
                    "source": "Overpass history", "snapshot": when.isoformat()}
        return None

    wait([f_area, f_ctx], timeout=patience)
    if (got := overpass()) is not None:
        return got
    if say:
        say("OpenStreetMap history server is slow — preparing a Geofabrik snapshot in parallel")
    cancel = threading.Event()
    f_gf = run_in_background(geofabrik.fetch_layers, bbox, LAYERS, big, CONTEXT_LAYERS, when, cancel)
    try:
        while time.time() - t0 < deadline:
            if (got := overpass()) is not None:
                return got
            if f_gf.done() and f_gf.exception() is None:
                area, ctx, snap, region = f_gf.result()
                return {"layers": area, "context": {f"context_{k}": v for k, v in ctx.items()},
                        "source": f"Geofabrik snapshot ({region})", "snapshot": snap.isoformat()}
            if f_gf.done() and f_area.done() and f_ctx.done():
                errs = [str(f.exception()) for f in (f_area, f_ctx, f_gf) if f.exception() is not None]
                raise RuntimeError("no OpenStreetMap source: " + "; ".join(errs)[:300])
            time.sleep(1.0)
        raise TimeoutError("no OpenStreetMap source answered in time")
    finally:
        cancel.set()          # whichever source lost stops downloading / scanning now
