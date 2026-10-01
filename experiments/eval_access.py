"""Step 4 check: are the cut-off conclusions right? (EMS used for CHECKING only — never an input.)

The same routing is run twice on the same pre-event OSM network:
  ours       roads cut where OUR flood/debris footprint covers ≥ 20 m of them, bridges over affected reaches
  reference  roads cut where Copernicus EMS graded the road destroyed/damaged (≥ 20 m within 10 m of it),
             OSM bridges within 25 m of an EMS destroyed/damaged bridge
EMS only graded roads inside its AOIs, so our cuts are also limited to the AOIs for the comparison.
Then every settlement's status (cut off / not) is compared.

usage: python experiments/eval_access.py upper|bidur|phosretar
"""
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import transform as shp_transform, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baadhi import access, damage  # noqa: E402
from baadhi.detect import detect  # noqa: E402
from baadhi.grid import make_grid  # noqa: E402
from baadhi.sources import osm  # noqa: E402
import eval_detect as E  # noqa: E402
import reference as R  # noqa: E402

AREAS = {"upper": ((85.30, 28.13, 85.40, 28.29), (1, 2)), "bidur": ((85.09, 27.85, 85.20, 28.02), (3,)),
         "phosretar": ((84.93, 27.78, 85.12, 27.875), (5,))}
name = sys.argv[1] if len(sys.argv) > 1 else "upper"
bbox, aois = AREAS[name]
event = dt.date(2026, 8, 26)
grid = make_grid(bbox, 10)
fwd = grid.from_lonlat().transform
g = lambda geom: shp_transform(fwd, geom)  # noqa: E731
domain = unary_union([g(R.aoi_polygon(a)) for a in aois])

pad = 0.05
obox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
layers = osm.fetch_all(obox, event)
ctx = osm.fetch_context(obox, event)
roads = layers["roads"]["features"]

# ---- ours
z, terrain, tracks, pre, post = E.load(name)
r = detect(terrain, tracks, pre, post)
dmg = damage.assess(r["cls"], grid, layers)


def limit(feats, statuses):
    """Only cuts inside the EMS AOIs count (EMS graded nothing outside them)."""
    out = []
    for f in feats:
        p = dict(f["properties"])
        if p["status"] in statuses and not g(shape(f["geometry"])).intersects(domain):
            p["status"] = "outside"
        out.append({**f, "properties": p})
    return out


ours = access.analyse(layers, limit(dmg.roads, ("cut", "touched", "near")), limit(dmg.bridges, ("at risk",)), r["cls"] > 0, grid, context=ctx)

# ---- reference from EMS grading
lines = [f for a in aois for f in R.layer(a, "transportationL")]
bad_roads = unary_union([g(shape(f["geometry"])) for f in lines
                         if f["properties"].get("damage_gra") in ("Destroyed", "Damaged") and "Bridge" not in str(f["properties"].get("obj_type"))])
zone = bad_roads.buffer(10) if not bad_roads.is_empty else None
ref_roads = []
for f in roads:
    cut = zone is not None and g(shape(f["geometry"])).intersection(zone).length >= 20
    ref_roads.append({**f, "properties": {**f["properties"], "status": "cut" if cut else "outside"}})
ems_br = [g(shape(f["geometry"])) for a in aois for lay in ("transportationP", "transportationL") for f in R.layer(a, lay)
          if "Bridge" in str(f["properties"].get("obj_type")) and f["properties"].get("damage_gra") in ("Destroyed", "Damaged")]
br_zone = unary_union(ems_br).buffer(25) if ems_br else None
ref_br = []
for f in layers["bridges"]["features"]:
    risk = br_zone is not None and g(shape(f["geometry"])).intersects(br_zone)
    ref_br.append({**f, "properties": {**f["properties"], "status": "at risk" if risk else "outside"}})
ref = access.analyse(layers, ref_roads, ref_br, r["cls"] > 0, grid, context=ctx)

# ---- compare
print(f"{name}: roads cut — ours (in AOIs) {sum(f['properties']['status'] == 'cut' for f in limit(dmg.roads, ('cut',)))}, "
      f"EMS-derived {sum(f['properties']['status'] == 'cut' for f in ref_roads)}")
print(f"  ours stats: {ours.stats}")
print(f"  ref  stats: {ref.stats}")
pairs = Counter()
rows = []
for a, b in zip(ours.settlements, ref.settlements):
    sa, sb = a["properties"]["status"], b["properties"]["status"]
    if sa == "no_mapped_road" and sb == "no_mapped_road":
        continue
    pairs[(sa, sb)] += 1
    if (sa == "cut_off") != (sb == "cut_off"):
        rows.append((a["properties"].get("label") or "unnamed", sa, sb, a["properties"].get("hospital_min_after"), b["properties"].get("hospital_min_after")))
print("  status pairs (ours, EMS-derived):", dict(pairs))
cut_o = sum(v for (sa, _), v in pairs.items() if sa == "cut_off")
cut_r = sum(v for (_, sb), v in pairs.items() if sb == "cut_off")
both = pairs.get(("cut_off", "cut_off"), 0)
n = sum(pairs.values())
agree = sum(v for (sa, sb), v in pairs.items() if (sa == "cut_off") == (sb == "cut_off"))
print(f"  cut off: ours {cut_o}, EMS-derived {cut_r}, both {both} | agreement on cut-off yes/no: {agree}/{n} = {agree / max(n, 1):.0%}")
for row in rows[:20]:
    print("   differs:", row)
# the "possibly cut off (verify)" tier: how many EMS-based cut-offs does ours flag as cut off OR possibly cut off?
ems_cut = [a["properties"]["status"] for a, b in zip(ours.settlements, ref.settlements) if b["properties"]["status"] == "cut_off"]
if ems_cut:
    flagged = sum(s in ("cut_off", "possibly_cut_off") for s in ems_cut)
    extra = sum(1 for a, b in zip(ours.settlements, ref.settlements)
                if a["properties"]["status"] == "possibly_cut_off" and b["properties"]["status"] != "cut_off")
    print(f"  with the verify tier: {flagged}/{len(ems_cut)} EMS-based cut-offs flagged cut off or possibly cut off; "
          f"{extra} 'possibly' flags where EMS-based routing still finds a road")
# travel times after the flood, where both analyses still find a road route
import numpy as np  # noqa: E402
dt_min = [abs(a["properties"]["hospital_min_after"] - b["properties"]["hospital_min_after"])
          for a, b in zip(ours.settlements, ref.settlements)
          if a["properties"].get("hospital_min_after") is not None and b["properties"].get("hospital_min_after") is not None]
if dt_min:
    d = np.array(dt_min)
    print(f"  hospital minutes after the flood (both reachable, n={len(d)}): median |diff| {np.median(d):.0f} min, "
          f"within 5 min {np.mean(d <= 5):.0%}, within 15 min {np.mean(d <= 15):.0%}, max {d.max():.0f} min")
