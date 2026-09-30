"""Step 1 check (optical): Sentinel-2 before/after over the upper corridor, NDVI and brightness change
inside vs outside the EMSR927 reference outlines (reference used for checking only)."""
import datetime as dt
import glob
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baadhi.grid import make_grid  # noqa: E402
from baadhi.sources import planetary as P  # noqa: E402

OUT = Path(__file__).parent / "out"
BBOX = (85.30, 28.13, 85.40, 28.29)
t0 = time.time()
grid = make_grid(BBOX, 10)
by_date = P.group_by_date(P.s2_scenes(BBOX, dt.date(2026, 8, 10), dt.date(2026, 9, 30), max_cloud=60))
pre, post = P.read_s2(by_date[dt.date(2026, 8, 12)], grid), P.read_s2(by_date[dt.date(2026, 9, 28)], grid)
print(f"loaded S2 in {time.time()-t0:.1f}s | clear pixels before {np.isfinite(pre['B04']).mean():.0%}, after {np.isfinite(post['B04']).mean():.0%}")

ndvi = lambda s: (s["B08"] - s["B04"]) / (s["B08"] + s["B04"] + 1e-6)
bright = lambda s: (s["B02"] + s["B03"] + s["B04"]) / 3
dndvi, dbright = ndvi(post) - ndvi(pre), bright(post) - bright(pre)

to_grid = grid.from_lonlat()
polys = [shp_transform(to_grid.transform, shape(ft["geometry"]))
         for f in glob.glob("data/reference/EMSR927/*AOI0[12]*/*observedEventA*.json")
         for ft in json.load(open(f, encoding="utf8"))["features"]]
ref = rasterize([(p, 1) for p in polys], out_shape=grid.shape, transform=grid.transform).astype(bool)
edge = rasterize([(p.boundary, 1) for p in polys], out_shape=grid.shape, transform=grid.transform).astype(bool)
ok = np.isfinite(dndvi)
for name, d in [("dNDVI", dndvi), ("dBrightness", dbright)]:
    a, b = d[ref & ok], d[~ref & ok]
    print(f"{name:12s} inside ref: median {np.median(a):+.3f}  | outside: median {np.median(b):+.3f}")
thr = (dndvi < -0.2) & (dbright > 0.02)
print(f"simple optical rule (dNDVI<-0.2 & brighter): catches {thr[ref & ok].mean():.0%} of reference pixels, flags {thr[~ref & ok].mean():.1%} of the rest")

rgb = lambda s: np.clip(np.dstack([s["B04"], s["B03"], s["B02"]]) * 3.2, 0, 1)
fig, ax = plt.subplots(1, 3, figsize=(15, 9), dpi=90)
for a, img, t in [(ax[0], rgb(pre), "12 Aug (before)"), (ax[1], rgb(post), "28 Sep (after)"), (ax[2], dndvi, "NDVI change")]:
    a.imshow(np.nan_to_num(img, nan=0) if img.ndim == 3 else img, **({} if img.ndim == 3 else dict(cmap="RdYlGn", vmin=-0.6, vmax=0.6)))
    ys, xs = np.nonzero(edge); a.scatter(xs, ys, s=0.3, c="cyan", alpha=0.8); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "quicklook_upper_s2.png"); print("saved", OUT / "quicklook_upper_s2.png")
