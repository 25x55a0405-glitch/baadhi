"""Step 1 check: load before/after Sentinel-1 for the upper Trishuli corridor and look at the change,
with the EMSR927 reference outlines drawn on top (reference used for checking only)."""
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
OUT.mkdir(exist_ok=True)
BBOX = (85.30, 28.13, 85.40, 28.29)  # Syaphrubesi (EMSR927 AOI1) → Timure (AOI2)
TRACK, PRE, POST = 85, [dt.date(2026, 8, 4), dt.date(2026, 8, 16)], dt.date(2026, 8, 28)

t0 = time.time()
grid = make_grid(BBOX, 10)
print(f"grid {grid.width}x{grid.height} px @10 m, {grid.crs}")
passes = {s.key: s for s in P.s1_passes(BBOX, dt.date(2026, 7, 25), dt.date(2026, 9, 1))}
pre = [P.read_s1(passes[(d, TRACK)], grid) for d in PRE]
post = P.read_s1(passes[(POST, TRACK)], grid)
dem = P.read_dem(grid)
print(f"loaded in {time.time()-t0:.1f}s")

db = lambda x: 10 * np.log10(np.clip(x, 1e-5, None))
pre_vv = db(np.nanmean([p["vv"] for p in pre], axis=0))
post_vv = db(post["vv"])
pre_vh = db(np.nanmean([p["vh"] for p in pre], axis=0))
post_vh = db(post["vh"])
dvv, dvh = post_vv - pre_vv, post_vh - pre_vh
valid = np.isfinite(dvv)
print(f"valid {valid.mean():.1%} | pre VV {np.nanpercentile(pre_vv, [5, 50, 95]).round(1)} dB | post VV {np.nanpercentile(post_vv, [5, 50, 95]).round(1)} dB | dVV p1/p99 {np.nanpercentile(dvv, [1, 99]).round(1)}")
print(f"DEM {np.nanmin(dem):.0f}–{np.nanmax(dem):.0f} m")

# EMSR927 observed-event outlines (checking only)
to_grid = grid.from_lonlat()
polys = []
for f in glob.glob("data/reference/EMSR927/*AOI0[12]*/*observedEventA*.json"):
    for ft in json.load(open(f, encoding="utf8"))["features"]:
        polys.append(shp_transform(to_grid.transform, shape(ft["geometry"])))
ref = rasterize([(p.boundary, 1) for p in polys], out_shape=grid.shape, transform=grid.transform) if polys else np.zeros(grid.shape)
ref_fill = rasterize([(p, 1) for p in polys], out_shape=grid.shape, transform=grid.transform) if polys else np.zeros(grid.shape)
print(f"reference: {len(polys)} polygons, {ref_fill.sum()*100/1e6:.2f} km² inside the grid")
if ref_fill.any():
    inside, outside = dvv[(ref_fill == 1) & valid], dvv[(ref_fill == 0) & valid]
    print(f"dVV inside reference: median {np.median(inside):+.2f} dB, |dVV|>3dB {np.mean(np.abs(inside) > 3):.0%} | outside: median {np.median(outside):+.2f} dB, |dVV|>3dB {np.mean(np.abs(outside) > 3):.0%}")

fig, ax = plt.subplots(1, 4, figsize=(20, 9), dpi=90)
for a, img, title, kw in [
    (ax[0], pre_vv, "VV before (mean 4 & 16 Aug)", dict(cmap="gray", vmin=-22, vmax=2)),
    (ax[1], post_vv, "VV after (28 Aug)", dict(cmap="gray", vmin=-22, vmax=2)),
    (ax[2], dvv, "VV change (dB)", dict(cmap="RdBu", vmin=-8, vmax=8)),
    (ax[3], dem, "Copernicus DEM (m)", dict(cmap="terrain")),
]:
    a.imshow(img, **kw)
    ys, xs = np.nonzero(ref)
    a.scatter(xs, ys, s=0.3, c="yellow", alpha=0.7)
    a.set_title(title); a.axis("off")
plt.tight_layout()
plt.savefig(OUT / "quicklook_upper.png")
print("saved", OUT / "quicklook_upper.png", f"total {time.time()-t0:.1f}s")
