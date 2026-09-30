"""Build and cache every candidate feature for a development area, for the Step 2 feature study.

usage: python experiments/prep_dev.py upper|bidur|phosretar
"""
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baadhi import optical, sar  # noqa: E402
from baadhi import terrain as T  # noqa: E402
from baadhi.grid import make_grid  # noqa: E402
from baadhi.sources import osm  # noqa: E402
import reference  # noqa: E402  (checking only)

EVENT = dt.date(2026, 8, 26)
AREAS = {
    "upper": ((85.30, 28.13, 85.40, 28.29), (1, 2)),       # Syaphrubesi + Timure (development)
    "bidur": ((85.09, 27.85, 85.20, 28.02), (3,)),         # held out
    "phosretar": ((84.93, 27.78, 85.12, 27.875), (5,)),     # held out
}
name = sys.argv[1] if len(sys.argv) > 1 else "upper"
bbox, aois = AREAS[name]
out = Path(__file__).resolve().parents[1] / "data" / "cache" / f"dev_{name}.npz"
t0 = time.time()
grid = make_grid(bbox, 10)
pad = 0.05
ob = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
ways = osm.fetch("waterways", ob, osm.snapshot_date(EVENT))
print(f"[{time.time()-t0:5.0f}s] grid {grid.width}x{grid.height}; {len(ways['features'])} OSM waterways")
ter = T.load_terrain(grid, waterways=ways)
print(f"[{time.time()-t0:5.0f}s] terrain: HAND-major p5/50 {np.nanpercentile(ter.hand_major, [5, 50]).round(1)}")
arrs = dict(dem=ter.dem, slope=ter.slope, aspect=ter.aspect, hand=ter.hand, hand_major=ter.hand_major, catch=ter.catchment_km2, rivers=ter.rivers)
tracks = []
for st in sar.select_stacks(bbox, EVENT):
    try:
        sd = sar.load_stack(st, grid)
    except RuntimeError as e:
        print(f"[{time.time()-t0:5.0f}s] {st.describe()} — skipped: {e}")
        continue
    cover = float(np.isfinite(sd.post_vv).mean())
    if cover < 0.3:
        print(f"[{time.time()-t0:5.0f}s] {st.describe()} — skipped: covers only {cover:.0%} of the area")
        continue
    f = sar.change_features(sd)
    inc = T.incidence_from_footprint(grid, st.post.items[0].geometry, st.direction)
    loc, good = T.radar_geometry(ter, st.direction, inc)
    for k, v in f.items():
        arrs[f"t{st.track}_{k}"] = v
    arrs[f"t{st.track}_good"], arrs[f"t{st.track}_loc"] = good, loc
    tracks.append(st.track)
    print(f"[{time.time()-t0:5.0f}s] {st.describe()} — good geometry {good.mean():.0%}")
pre, post = optical.before_after(grid, EVENT)
for tag, s in (("pre", pre), ("post", post)):
    if s is None:
        continue
    for k, v in optical.indices(s).items():
        arrs[f"s2{tag}_{k}"] = v
    arrs[f"s2{tag}_rgb"] = np.dstack([s["B04"], s["B03"], s["B02"]])
    print(f"[{time.time()-t0:5.0f}s] S2 {tag}: dates {s['dates']}, clear {np.isfinite(s['B04']).mean():.0%}")
label, domain = reference.rasters(grid, aois)
arrs["label"], arrs["domain"] = label, domain
np.savez_compressed(out, tracks=np.array(tracks), **arrs)
print(f"[{time.time()-t0:5.0f}s] saved {out.name}: {label.sum()*100/1e6:.2f} km² affected in {domain.sum()*100/1e6:.2f} km² of checked area")
