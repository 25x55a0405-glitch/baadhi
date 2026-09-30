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
import time
from pathlib import Path

import requests

OHSOME = "https://api.ohsome.org/v1"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "osm"
UA = {"User-Agent": "Baadhi/0.1 (flood damage mapping research; github.com)", "Accept": "application/json"}

LAYERS = ("waterways", "water", "buildings", "roads", "bridges", "places", "health")

OHSOME_FILTER = {
    "waterways": "waterway in (river, stream, canal, drain) and type:way",
    "water": "(natural=water or waterway=riverbank or landuse=reservoir) and geometry:polygon",
    "buildings": "building=* and geometry:polygon",
    "roads": "highway in (motorway, trunk, primary, secondary, tertiary, unclassified, residential, service, track, living_street, road, "
             "motorway_link, trunk_link, primary_link, secondary_link, tertiary_link, path, footway, steps, bridleway) and type:way",
    "bridges": "(bridge=* and bridge!=no and type:way) or man_made=bridge",
    "places": "place in (city, town, village, hamlet, isolated_dwelling, suburb, neighbourhood, locality) and type:node",
    "health": "(amenity in (hospital, clinic, doctors) or healthcare in (hospital, clinic, centre, doctor)) and (type:node or type:way)",
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
}
POLYGON_LAYERS = {"water", "buildings"}
KEEP_TAGS = {"name", "name:en", "name:ne", "highway", "bridge", "waterway", "place", "amenity", "healthcare", "building",
             "surface", "population", "layer", "tunnel", "ford", "natural", "man_made", "emergency", "landuse"}

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


def _via_overpass(layer: str, bbox, when: dt.date, timeout: int) -> dict:
    w, s, e, n = bbox
    b = f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"
    q = f'[out:json][timeout:{timeout}][date:"{when.isoformat()}T00:00:00Z"];({OVERPASS_Q[layer].format(b=b)});out tags geom;'
    last = None
    for url in OVERPASS:
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=timeout + 30)
            if r.status_code == 200:
                return _overpass_to_geojson(r.json(), layer)
            last = f"{url} HTTP {r.status_code}"
        except requests.RequestException as ex:
            last = f"{url} {type(ex).__name__}"
    raise RuntimeError(f"Overpass failed: {last}")


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


def fetch(layer: str, bbox, when: dt.date, timeout: int = 180) -> dict:
    """GeoJSON FeatureCollection of one layer as of `when`, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{layer}|{','.join(f'{v:.5f}' for v in bbox)}|{when}".encode()).hexdigest()[:16]
    path = CACHE / f"{layer}-{when}-{key}.geojson"
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    errors = []
    for attempt in range(3):
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
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"OSM {layer} unavailable — " + "; ".join(errors[-4:]))


def fetch_all(bbox, event: dt.date, layers=LAYERS) -> dict[str, dict]:
    when = snapshot_date(event)
    return {name: fetch(name, bbox, when) for name in layers}
