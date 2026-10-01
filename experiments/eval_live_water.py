"""Does the AI model find flood water on LIVE satellite data better than the plain radar rule?

Test: EMSR838, Punjab (Pakistan) riverine floods of late August 2025 — standing water on a plain, the
kind of flood the Himalayan case study does not have. Copernicus EMS flood outlines are used ONLY to
score (never as an input). For a fair test we use the very Sentinel-1 pass EMS mapped from, and
before-images from before the flood began.

For each area: the ~14 km box inside the EMS AOI with the most mapped flooding, then three variants
  rule      the detector's radar-water rule alone (very dark after AND ≥ 3 dB darker than before)
  model     the Kuro Siwo-trained network alone (flood probability ≥ 0.5)
  combined  the full detector with the model's evidence (what the system reports)
scored against EMS "Flooded area" inside the AOI (areas EMS marked "not analysed" are left out).

usage: python experiments/eval_live_water.py [model.pt]
"""
import datetime as dt
import gc
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baadhi import sar  # noqa: E402
from baadhi import terrain as T  # noqa: E402
from baadhi.detect import WATER, detect  # noqa: E402
from baadhi.grid import make_grid  # noqa: E402
from baadhi.ml.infer import FloodModel  # noqa: E402
from baadhi.sources import osm  # noqa: E402
from baadhi.sources import planetary as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference" / "EMSR838"
# area → (EMS package, date of the Sentinel-1 image EMS mapped from)
CASES = {"AOI01 Rasoo Nagar": ("EMSR838_AOI01_DEL_PRODUCT_v1", dt.date(2025, 8, 28)),
         "AOI02 Sharaqpur": ("EMSR838_AOI02_DEL_PRODUCT_v1", dt.date(2025, 8, 28)),
         "AOI04 Najabat": ("EMSR838_AOI04_DEL_PRODUCT_v1", dt.date(2025, 8, 27)),
         "AOI05 Trimmu": ("EMSR838_AOI05_DEL_PRODUCT_v1", dt.date(2025, 8, 28))}
ONSET = dt.date(2025, 8, 24)     # before-images strictly before this date
SIZE = 0.14                      # box side, degrees (~13 × 15 km)


def layer(pkg, name):
    f = glob.glob(str(REF / pkg / f"*_{name}_v*.json"))
    return [x for x in json.loads(Path(f[0]).read_text(encoding="utf8"))["features"] if x.get("geometry")] if f else []


def pick_box(pkg):
    """The SIZE° box (on a 0.02° lattice) inside the AOI holding the most EMS-mapped flood."""
    aoi = shape(layer(pkg, "areaOfInterestA")[0]["geometry"])
    flood = unary_union([shape(f["geometry"]) for f in layer(pkg, "observedEventA")])
    w, s, e, n = aoi.bounds
    best, bb = -1.0, None
    for x in np.arange(w, max(e - SIZE, w + 1e-9), 0.02):
        for y in np.arange(s, max(n - SIZE, s + 1e-9), 0.02):
            b = box(x, y, x + SIZE, y + SIZE)
            if aoi.intersection(b).area >= 0.3 * b.area:      # AOIs can be thin strips; scoring stays inside the AOI
                a = flood.intersection(b).area
                if a > best:
                    best, bb = a, (round(x, 4), round(y, 4), round(x + SIZE, 4), round(y + SIZE, 4))
    return bb, aoi


def stacks_for(bbox, img_date):
    """Tracks with a pass on EMS's image date (±1 day) and up to 3 before-images before the onset."""
    passes = P.s1_passes(bbox, ONSET - dt.timedelta(days=40), img_date + dt.timedelta(days=1))
    out = []
    for track in {p.track for p in passes}:
        ps = [p for p in passes if p.track == track]
        post = next((p for p in ps if abs((p.date - img_date).days) <= 1), None)
        pre = [p for p in ps if p.date < ONSET][-3:]
        if post and pre:
            st = sar.TrackStack(track, post.direction, pre, post)
            st._event = ONSET
            out.append(st)
    return out


def score(pred, ref, dom):
    p, r = pred & dom, ref & dom
    tp, fp, fn = (p & r).sum(), (p & ~r).sum(), (~p & r).sum()
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1)}


def main():
    model = FloodModel.load(sys.argv[1] if len(sys.argv) > 1 else ROOT / "models" / "flood_model.pt")
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None          # e.g. AOI02,AOI04
    out_json = ROOT / "experiments" / "out" / "live_water.json"
    results = json.loads(out_json.read_text(encoding="utf8")) if (only and out_json.exists()) else {}
    for name, (pkg, img_date) in CASES.items():
        if only and name.split()[0] not in only:
            continue
        gc.collect()
        t0 = time.time()
        bbox, aoi = pick_box(pkg)
        grid = make_grid(bbox, 10)
        fwd = grid.from_lonlat().transform
        g = lambda geom: shp_transform(fwd, geom)  # noqa: E731
        kw = dict(out_shape=grid.shape, transform=grid.transform)
        ref = rasterize([(g(shape(f["geometry"])), 1) for f in layer(pkg, "observedEventA")], **kw).astype(bool)
        dom = rasterize([(g(aoi), 1)], **kw).astype(bool)
        na = [g(shape(f["geometry"])) for f in layer(pkg, "imageFootprintA") if "Not Analysed" in str(f["properties"].get("obj_type"))]
        if na:
            dom &= ~rasterize([(x, 1) for x in na], **kw).astype(bool)

        rivers = osm.fetch("waterways", bbox, osm.snapshot_date(ONSET))      # pre-event OSM rivers, as the pipeline uses
        ter = T.load_terrain(grid, rivers)
        tracks = []
        for st in stacks_for(bbox, img_date):
            try:
                sd = sar.load_stack(st, grid)
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: track {st.track} skipped ({e})")
                continue
            if np.isfinite(sd.post_vv).mean() < 0.3:
                continue
            feats = sar.change_features(sd)
            inc = T.incidence_from_footprint(grid, st.post.items[0].geometry, st.direction)
            _, good = T.radar_geometry(ter, st.direction, inc)
            tracks.append({**{k: feats[k] for k in ("z_vv", "z_vh", "post_vv", "post_vh", "d_vv", "d_vh")}, "good": good, "stack": sd})
            print(f"  {name}: {st.describe()}", flush=True)
        if not tracks:
            print(f"{name}: no Sentinel-1 pass on {img_date} — skipped")
            continue
        seen = np.zeros(grid.shape, bool)
        for tr in tracks:
            seen |= np.isfinite(tr["post_vv"]) & tr["good"]
        dom &= seen
        terrain = {"hand_major": ter.hand_major, "slope": ter.slope}
        prob = model(tracks, ter, grid)
        rule = detect(terrain, tracks, None, None, None)["cls"] == WATER
        comb = detect(terrain, tracks, None, None, prob)["cls"] == WATER
        raw = np.nan_to_num(prob) >= 0.5
        res = {"bbox": bbox, "image": str(img_date), "ems_flood_km2": round(float((ref & dom).sum()) * 1e-4, 2),
               "analysed_km2": round(float(dom.sum()) * 1e-4, 1)}
        for key, pred in (("rule", rule), ("model", raw), ("combined", comb)):
            res[key] = {k: round(float(v), 3) for k, v in score(pred, ref, dom).items()}
        results[name] = res
        print(f"{name} [{time.time() - t0:.0f}s] EMS flood {res['ems_flood_km2']} km² of {res['analysed_km2']} km² | "
              + " | ".join(f"{k} F1 {res[k]['f1']:.3f} (P {res[k]['precision']:.2f} R {res[k]['recall']:.2f})" for k in ("rule", "model", "combined")), flush=True)
        np.savez_compressed(ROOT / "experiments" / "out" / f"live_{name.split()[0]}.npz", ref=ref, dom=dom, prob=prob, rule=rule, comb=comb)
        out_json.write_text(json.dumps(results, indent=1), encoding="utf8")     # after every area: a crash loses nothing
        del tracks, prob, rule, comb, raw, ter
    if results:
        keys = ("rule", "model", "combined")
        print("MEAN F1:", {k: round(float(np.mean([r[k]["f1"] for r in results.values()])), 3) for k in keys})


if __name__ == "__main__":
    main()
