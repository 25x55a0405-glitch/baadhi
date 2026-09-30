"""Functional test of damage (Step 3) and access (Step 4) on a prepared development area."""
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baadhi import access, damage  # noqa: E402
from baadhi.detect import detect  # noqa: E402
from baadhi.grid import make_grid  # noqa: E402
from baadhi.sources import osm  # noqa: E402
import eval_detect as E  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "upper"
bbox = {"upper": (85.30, 28.13, 85.40, 28.29), "bidur": (85.09, 27.85, 85.20, 28.02), "phosretar": (84.93, 27.78, 85.12, 27.875)}[name]
grid = make_grid(bbox, 10)
z, terrain, tracks, pre, post = E.load(name)
t = time.time()
r = detect(terrain, tracks, pre, post)
print(f"detect {time.time()-t:.1f}s: affected {(r['cls'] > 0).sum() * 1e-4:.2f} km²")
pad = 0.05
layers = osm.fetch_all((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), dt.date(2026, 8, 26))
t = time.time()
dmg = damage.assess(r["cls"], grid, layers)
print(f"damage {time.time()-t:.1f}s:", dmg.stats)
t = time.time()
ctx = osm.fetch_context((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), dt.date(2026, 8, 26))
acc = access.analyse(layers, dmg.roads, dmg.bridges, r["cls"] > 0, grid, context=ctx)
print(f"access {time.time()-t:.1f}s:", acc.stats)
for f in sorted(acc.settlements, key=lambda f: f["properties"]["status"])[:12]:
    p = f["properties"]
    print(f"   {p['status']:15s} {str(p.get('label'))[:28]:28s} hospital {p.get('hospital_min_before')}→{p.get('hospital_min_after')} min | town {p.get('town_min_before')}→{p.get('town_min_after')} | on foot {p.get('hospital_on_foot_min')}")
