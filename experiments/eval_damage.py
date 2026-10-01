"""Step 3 check: our building / road / bridge damage estimates vs the EMSR927 damage grading
(checking only — never an input). Only features inside the EMS AOIs are compared.

usage: python experiments/eval_damage.py upper|bidur|phosretar
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.ops import transform as shp_transform, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baadhi import damage  # noqa: E402
from baadhi.detect import detect  # noqa: E402
from baadhi.grid import make_grid  # noqa: E402
from baadhi.sources import osm  # noqa: E402
import eval_detect as E  # noqa: E402
import reference as R  # noqa: E402

AREAS = {"upper": ((85.30, 28.13, 85.40, 28.29), (1, 2)), "bidur": ((85.09, 27.85, 85.20, 28.02), (3,)),
         "phosretar": ((84.93, 27.78, 85.12, 27.875), (5,))}
name = sys.argv[1] if len(sys.argv) > 1 else "upper"
bbox, aois = AREAS[name]
grid = make_grid(bbox, 10)
fwd = grid.from_lonlat().transform
g = lambda geom: shp_transform(fwd, geom)  # noqa: E731
domain = unary_union([g(R.aoi_polygon(a)) for a in aois])

import system_run as S  # noqa: E402

RUN = S.run_arg()      # python eval_damage.py <area> [runs/<id>]: with a run, the DEPLOYED system's footprint is scored
if RUN is not None:
    r = {"cls": S.footprint(RUN)}
    layers, _, src = S.pre_event(bbox, dt.date(2026, 8, 26))
    print(f"scoring the deployed system: footprint of {RUN.name}; OSM {src}")
else:
    z, terrain, tracks, pre, post = E.load(name)
    r = detect(terrain, tracks, pre, post)
    pad = 0.05
    layers = osm.fetch_all((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), dt.date(2026, 8, 26))
dmg = damage.assess(r["cls"], grid, layers)
affected = r["cls"] > 0


def pts(feats, grades):
    out = []
    for f in feats:
        if f["properties"].get("damage_gra") not in grades:
            continue
        geom = g(shape(f["geometry"]))
        p = geom if geom.geom_type == "Point" else geom.representative_point()
        if domain.contains(p):
            out.append((p.x, p.y))
    return np.array(out) if out else np.zeros((0, 2))


# ---- buildings
ems_b = [f for a in aois for f in R.layer(a, "builtUpP")]
ems_hit = pts(ems_b, {"Destroyed", "Damaged"})
ems_any = pts(ems_b, {"Destroyed", "Damaged", "Possibly damaged"})
ours = [(g(shape(f["geometry"])).representative_point(), f["properties"]["status"]) for f in dmg.buildings]
ours_in = [(p, s) for p, s in ours if domain.contains(p)]
flag_xy = np.array([(p.x, p.y) for p, s in ours_in if s in (damage.HIT, damage.POSSIBLE)]) if ours_in else np.zeros((0, 2))
hit_xy = np.array([(p.x, p.y) for p, s in ours_in if s == damage.HIT]) if ours_in else np.zeros((0, 2))


def near(a, b, m):
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    return cKDTree(b).query(a)[0] <= m


def in_affected(xy):
    col = ((xy[:, 0] - grid.transform.c) / grid.res).astype(int)
    row = ((grid.transform.f - xy[:, 1]) / grid.res).astype(int)
    ok = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
    out = np.zeros(len(xy), bool)
    out[ok] = affected[row[ok], col[ok]]
    return out


rec = (near(ems_hit, flag_xy, 15) | in_affected(ems_hit)).mean() if len(ems_hit) else float("nan")
prec = near(hit_xy, ems_any, 15).mean() if len(hit_xy) else float("nan")
print(f"{name}: EMS destroyed/damaged buildings in AOI: {len(ems_hit)} | ours 'hit' in AOI: {len(hit_xy)} ('hit'+'possibly': {len(flag_xy)})")
print(f"  building recall  (EMS destroyed/damaged we flagged or map as affected): {rec:.0%}")
print(f"  building precision proxy (our 'hit' with an EMS damaged/possibly point ≤ 15 m): {prec:.0%}")

# ---- roads (length-based)
ems_rl = [f for a in aois for f in R.layer(a, "transportationL") if "Bridge" not in str(f["properties"].get("obj_type"))]
bad = [g(shape(f["geometry"])).intersection(domain) for f in ems_rl if f["properties"].get("damage_gra") in ("Destroyed", "Damaged")]
bad_len = sum(x.length for x in bad)
covered = 0.0
for line in bad:
    for seg in getattr(line, "geoms", [line]):
        if seg.is_empty or seg.length == 0 or seg.geom_type != "LineString":
            continue
        n = max(int(seg.length / 5), 1)
        ps = np.array([(seg.interpolate(i / n, normalized=True).x, seg.interpolate(i / n, normalized=True).y) for i in range(n + 1)])
        covered += in_affected(ps).mean() * seg.length
print(f"  roads: EMS destroyed/damaged length in AOI {bad_len/1000:.1f} km, of which inside our affected area {covered / max(bad_len, 1):.0%}")

# ---- bridges
ems_br = [f for a in aois for lay in ("transportationP", "transportationL") for f in R.layer(a, lay) if "Bridge" in str(f["properties"].get("obj_type"))]
br_bad = pts(ems_br, {"Destroyed", "Damaged"})
ours_br = [g(shape(f["geometry"])) for f in dmg.bridges if f["properties"]["status"] == "at risk"]
all_br = [g(shape(f["geometry"])) for f in dmg.bridges]
# distance to the bridge LINE (a long bridge's midpoint can be far from where EMS put its point)
hit = [any(b.distance(p) <= 25 for b in ours_br) for p in (Point(xy) for xy in br_bad)]
in_osm = [any(b.distance(p) <= 25 for b in all_br) for p in (Point(xy) for xy in br_bad)]
print(f"  bridges: EMS destroyed/damaged in AOI {len(br_bad)} | in pre-event OSM (≤ 25 m) {sum(in_osm)} | "
      f"flagged 'at risk' by us (bridge line ≤ 25 m) {sum(hit)} = {np.mean(hit) if hit else float('nan'):.0%} "
      f"({sum(hit)}/{max(sum(in_osm), 1)} = {sum(hit) / max(sum(in_osm), 1):.0%} of those OSM knows)")
