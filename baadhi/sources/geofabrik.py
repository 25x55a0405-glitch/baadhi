"""Pre-event OpenStreetMap from Geofabrik's dated regional snapshots — the fallback when the public
Overpass history server is slow or refuses requests.

Geofabrik keeps, for every region, a snapshot on 1 January of each year, on the 1st of recent months and
daily for the last week. The newest snapshot dated on or before our snapshot date (event − 2 days) is
"OpenStreetMap as it was before the event" — only less recent than the day-before history query. The file
is downloaded once (cached on disk) and the area is cut out locally with pyosmium; the same tag rules as the
Overpass queries decide which feature goes into which layer.

    layers, ctx, snap = fetch_layers(area_bbox, ctx_bbox, before=date(2026, 8, 24))
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "geofabrik"
INDEX_URL = "https://download.geofabrik.de/index-v1.json"
UA = {"User-Agent": "Baadhi/0.1 (flood damage mapping research)"}

ROAD = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service", "track",
        "living_street", "road", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
        "path", "footway", "steps", "bridleway"}
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "road", "motorway_link", "trunk_link",
         "primary_link", "secondary_link", "tertiary_link"}
PLACES = {"city", "town", "village", "hamlet", "isolated_dwelling", "suburb", "neighbourhood", "locality"}
WATERWAYS = {"river", "stream", "canal", "drain"}
# "natural"/"landuse" are left out of the scan (millions of land-cover polygons, and the only layer using them,
# standing-water polygons, is not used by any analysis step; river-bank polygons still come via "waterway")
KEYS = ("building", "highway", "waterway", "bridge", "man_made", "place", "amenity", "healthcare")


def _classify(kind: str, tags) -> list[str]:
    """Which layers a feature belongs to — the same rules as the Overpass queries in osm.py."""
    g = tags.get
    out = []
    if kind == "way":
        if g("waterway") in WATERWAYS:
            out.append("waterways")
        if g("natural") == "water" or g("waterway") == "riverbank" or g("landuse") == "reservoir":
            out.append("water")
        if "building" in tags:
            out.append("buildings")
        if g("highway") in ROAD:
            out.append("roads")
        if g("highway") in MAJOR:
            out.append("roads_major")
        if (g("bridge") not in (None, "no")) or g("man_made") == "bridge":
            out.append("bridges")
    if kind == "node":
        if g("place") in PLACES:
            out.append("places")
        if g("place") in ("city", "town"):
            out.append("towns")
        if g("man_made") == "bridge":
            out.append("bridges")
    if g("amenity") in ("hospital", "clinic", "doctors") or g("healthcare") in ("hospital", "clinic", "centre", "doctor"):
        out.append("health")
    return out


# ------------------------------------------------------------------ which file
def _index() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "index-v1.json"
    if not p.exists() or (dt.datetime.now().timestamp() - p.stat().st_mtime) > 7 * 86400:
        r = requests.get(INDEX_URL, headers=UA, timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
    return json.loads(p.read_text(encoding="utf8"))


def _dated_files(latest_url: str) -> dict[dt.date, str]:
    """Dated snapshots next to a region's '-latest' file: {date: url}."""
    base, name = latest_url.rsplit("/", 1)
    stem = name.replace("-latest.osm.pbf", "")
    html = requests.get(base + "/", headers=UA, timeout=60).text
    out = {}
    for m in set(re.findall(rf'"({re.escape(stem)}-(\d{{6}})\.osm\.pbf)"', html)):
        d = dt.datetime.strptime(m[1], "%y%m%d").date()
        out[d] = f"{base}/{m[0]}"
    return out


def choose_file(bbox, before: dt.date) -> tuple[str, dt.date, str]:
    """(url, snapshot date, region name) for the box: the country-level region that covers most of it
    (smallest first on a tie) and has a dated snapshot on or before `before`. A box straddling a border
    (the upper Trishuli reaches Tibet) is served by the region holding most of it; the part across the
    border simply has no data in that file. Continent-wide files are never used (13 GB)."""
    from shapely.geometry import box, shape

    target = box(*bbox)
    regions = []
    for f in _index()["features"]:
        props = f["properties"]
        pbf = (props.get("urls") or {}).get("pbf")
        if not pbf or not f.get("geometry") or not props.get("parent"):
            continue                                  # no parent = a whole continent
        geom = shape(f["geometry"])
        if not geom.intersects(target):
            continue
        cover = geom.intersection(target).area / target.area
        regions.append((-round(cover * 20) / 20, geom.area, props.get("name") or props["id"], pbf))
    for _, _, name, pbf in sorted(regions):           # most coverage first, then the smaller region
        try:
            files = _dated_files(pbf)
        except requests.RequestException:
            continue
        ok = [d for d in files if d <= before]
        if ok:
            d = max(ok)
            return files[d], d, name
    raise RuntimeError("no Geofabrik region snapshot covers this area before the event")


class Cancelled(RuntimeError):
    """The caller no longer needs this snapshot (another source answered first)."""


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled("snapshot no longer needed")


def _download(url: str, cancel=None) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / url.rsplit("/", 1)[1]
    if path.exists():
        return path
    part = path.with_suffix(".part")
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(part, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                _check(cancel)
                fh.write(chunk)
    part.replace(path)
    return path


# ------------------------------------------------------------------ cutting the area out
def _geometry(kind: str, obj, layer: str):
    if kind == "node":
        return {"type": "Point", "coordinates": [obj.location.lon, obj.location.lat]}
    coords = [[n.location.lon, n.location.lat] for n in obj.nodes if n.location.valid()]
    if len(coords) < 2:
        return None
    closed = len(coords) >= 4 and coords[0] == coords[-1]
    if layer in ("water", "buildings") or (layer == "health" and closed):
        return {"type": "Polygon", "coordinates": [coords]} if closed else None
    return {"type": "LineString", "coordinates": coords}


def extract(pbf: Path, area_bbox, area_layers, ctx_bbox=None, ctx_layers=(), cancel=None) -> tuple[dict, dict]:
    """One pass over the region file: features of `area_layers` touching `area_bbox`, and of `ctx_layers`
    touching `ctx_bbox` (the wider routing context)."""
    import osmium

    from .osm import KEEP_TAGS

    def inside(lon, lat, b):
        return b[0] <= lon <= b[2] and b[1] <= lat <= b[3]

    want_area, want_ctx = set(area_layers), set(ctx_layers)
    area = {name: [] for name in area_layers}
    ctx = {name: [] for name in ctx_layers}
    outer = ctx_bbox or area_bbox
    fp = osmium.FileProcessor(str(pbf)).with_locations().with_filter(osmium.filter.KeyFilter(*KEYS))
    seen = 0
    for obj in fp:
        seen += 1
        if not seen % 50000:
            _check(cancel)
        if obj.is_node():
            kind = "node"
            if not obj.location.valid():
                continue
            pts = [(obj.location.lon, obj.location.lat)]
        elif obj.is_way():
            kind = "way"
            nodes = obj.nodes
            if len(nodes) == 0 or not nodes[0].location.valid():
                continue
            # quick reject for the millions of buildings (small: the first node tells where it is); long
            # features such as rivers and highways can start far away and still cross the area, so they
            # are always checked node by node
            tags0 = obj.tags
            if "highway" in tags0 or "waterway" in tags0:
                # long features: skip only when BOTH ends are far (> 0.5°, ~50 km) from the box
                l0, l1 = nodes[0].location, nodes[len(nodes) - 1].location
                far = lambda loc: not (outer[0] - 0.5 <= loc.lon <= outer[2] + 0.5 and outer[1] - 0.5 <= loc.lat <= outer[3] + 0.5)  # noqa: E731
                if l1.valid() and far(l0) and far(l1):
                    continue
            else:
                # compact features (buildings, land-use and water polygons, bridges): the first node tells
                # where they are. Health facilities are wanted in the wide context box, the rest in the area.
                b = outer if ("amenity" in tags0 or "healthcare" in tags0) else area_bbox
                l0 = nodes[0].location
                if not (b[0] - 0.02 <= l0.lon <= b[2] + 0.02 and b[1] - 0.02 <= l0.lat <= b[3] + 0.02):
                    continue
            pts = [(n.location.lon, n.location.lat) for n in nodes if n.location.valid()]
        else:
            continue                          # relations are skipped (rare for the layers we use)
        tags = obj.tags
        layers = _classify(kind, tags)
        if not layers:
            continue
        in_area = any(inside(x, y, area_bbox) for x, y in pts)
        in_ctx = ctx_bbox is not None and any(inside(x, y, ctx_bbox) for x, y in pts)
        if not (in_area or in_ctx):
            continue
        props = {t.k: t.v for t in tags if t.k in KEEP_TAGS}
        props["osm_id"] = f"{kind}/{obj.id}"
        for layer in layers:
            for target, want, hit in ((area, want_area, in_area), (ctx, want_ctx, in_ctx)):
                if hit and layer in want:
                    geom = _geometry(kind, obj, layer)
                    if geom is not None:
                        target[layer].append({"type": "Feature", "geometry": geom, "properties": dict(props)})
    fc = lambda feats: {"type": "FeatureCollection", "features": feats}  # noqa: E731
    return {k: fc(v) for k, v in area.items()}, {k: fc(v) for k, v in ctx.items()}


def fetch_layers(area_bbox, area_layers, ctx_bbox, ctx_layers, before: dt.date, cancel=None):
    """Download (once) and cut out: returns (area layers, context layers, snapshot date, region name).
    The cut-out itself is cached too, so running the same area again is instant."""
    import hashlib
    url, snap, region = choose_file(area_bbox, before)
    reused = _from_bigger_cut(url, area_bbox, area_layers, ctx_bbox, ctx_layers)
    if reused is not None:
        area, ctx = reused
        return area, ctx, snap, region
    key = hashlib.sha1(json.dumps([url, [round(v, 5) for v in area_bbox], list(area_layers),
                                   [round(v, 5) for v in (ctx_bbox or ())], list(ctx_layers)]).encode()).hexdigest()[:16]
    cached = CACHE / f"cut-{key}.json"
    if cached.exists():
        d = json.loads(cached.read_text(encoding="utf8"))
        return d["area"], d["ctx"], dt.date.fromisoformat(d["snapshot"]), d["region"]
    pbf = _download(url, cancel)
    area, ctx = extract(pbf, area_bbox, area_layers, ctx_bbox, ctx_layers, cancel)
    for d in (area, ctx):
        for fc in d.values():
            fc.update(snapshot=snap.isoformat(), source=f"geofabrik:{region}")
    meta = {"snapshot": snap.isoformat(), "region": region, "url": url, "area_bbox": list(area_bbox),
            "ctx_bbox": list(ctx_bbox or area_bbox), "area_layers": list(area_layers), "ctx_layers": list(ctx_layers)}
    cached.write_text(json.dumps({"area": area, "ctx": ctx, **meta}), encoding="utf8")
    cached.with_suffix(".meta").write_text(json.dumps(meta), encoding="utf8")    # small: read to decide reuse
    return area, ctx, snap, region


def _contains(outer, inner) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _touches(feature, b) -> bool:
    g = feature["geometry"]
    pts = [g["coordinates"]] if g["type"] == "Point" else g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"]
    return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for x, y in pts)


def _from_bigger_cut(url, area_bbox, area_layers, ctx_bbox, ctx_layers):
    """A cut-out made earlier for a larger box (e.g. a whole river basin prepared before a demo) serves
    any box inside it: its features are simply filtered — no rescan of the region file."""
    for m in sorted(CACHE.glob("cut-*.meta"), key=lambda q: q.stat().st_size):
        try:
            meta = json.loads(m.read_text(encoding="utf8"))
        except Exception:  # noqa: BLE001
            continue
        if meta.get("url") != url or not _contains(meta.get("area_bbox", [0, 0, 0, 0]), area_bbox):
            continue
        if ctx_bbox is not None and not _contains(meta.get("ctx_bbox", [0, 0, 0, 0]), ctx_bbox):
            continue
        if not set(area_layers) <= set(meta.get("area_layers", [])) or not set(ctx_layers) <= set(meta.get("ctx_layers", [])):
            continue
        d = json.loads(m.with_suffix(".json").read_text(encoding="utf8"))
        fc = lambda feats: {"type": "FeatureCollection", "features": feats, "snapshot": d["snapshot"], "source": f"geofabrik:{d['region']}"}  # noqa: E731
        area = {k: fc([f for f in d["area"][k]["features"] if _touches(f, area_bbox)]) for k in area_layers}
        ctx = {k: fc([f for f in d["ctx"][k]["features"] if _touches(f, ctx_bbox or area_bbox)]) for k in ctx_layers}
        return area, ctx
    return None


def prepare(area_bbox, before: dt.date, ctx_km: float = 25.0):
    """Cut out a large box ahead of time (e.g. a whole basin before a demo), so later analyses anywhere
    inside it get their map data instantly when the Overpass server is slow."""
    from .osm import CONTEXT_LAYERS, LAYERS, expand
    return fetch_layers(area_bbox, LAYERS, expand(area_bbox, ctx_km), CONTEXT_LAYERS, before)
