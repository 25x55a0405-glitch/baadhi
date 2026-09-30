"""Step 2 feature study: which signals separate flood/debris pixels from the rest?

Scores each candidate feature by ROC AUC against the EMSR927 reference (checking data), inside the
reference AOIs only. AUC 0.5 = useless, 1.0 = perfect; values below 0.5 mean "lower = affected".
usage: python experiments/feature_study.py upper
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

name = sys.argv[1] if len(sys.argv) > 1 else "upper"
z = np.load(Path(__file__).resolve().parents[1] / "data" / "cache" / f"dev_{name}.npz")
lab, dom = z["label"], z["domain"]
tracks = list(z["tracks"])
print(f"{name}: tracks {tracks}; checked area {dom.sum()*1e-4:.2f} km², affected {lab.sum()*1e-4:.2f} km² ({lab[dom].mean():.1%})")

rng = np.random.default_rng(0)


def auc(feat, mask=None, n=200_000):
    m = dom & np.isfinite(feat) & (mask if mask is not None else True)
    y, x = lab[m], feat[m]
    if y.sum() < 50 or (~y).sum() < 50:
        return np.nan, m.sum()
    if len(y) > n:
        i = rng.choice(len(y), n, replace=False)
        y, x = y[i], x[i]
    return roc_auc_score(y, x), m.sum()


rows = []
feats = {}
for t in tracks:
    g = z[f"t{t}_good"]
    for k in ("d_vv", "d_vh", "z_vv", "z_vh", "post_vv", "post_vh", "pre_vv", "pre_vh"):
        a = z[f"t{t}_{k}"]
        feats[f"t{t} {k}"] = (a, g)
        if k.startswith(("d_", "z_")):
            feats[f"t{t} |{k}|"] = (np.abs(a), g)
# multi-track fusion: strongest absolute change among tracks that see the ground well
zs = np.stack([np.where(z[f"t{t}_good"], np.abs(z[f"t{t}_z_vh"]), np.nan) for t in tracks])
feats["max|z_vh| over good tracks"] = (np.nanmax(zs, axis=0), None)
zv = np.stack([np.where(z[f"t{t}_good"], np.abs(z[f"t{t}_z_vv"]), np.nan) for t in tracks])
feats["max|z_vv| over good tracks"] = (np.nanmax(zv, axis=0), None)
dvh = np.stack([np.where(z[f"t{t}_good"], z[f"t{t}_d_vh"], np.nan) for t in tracks])
feats["mean d_vh over good tracks"] = (np.nanmean(dvh, axis=0), None)
for k in ("hand", "hand_major", "slope", "catch"):
    feats[f"terrain {k}"] = (z[k], None)
if "s2post_ndvi" in z and "s2pre_ndvi" in z:
    for k in ("ndvi", "bsi", "bright", "mndwi"):
        feats[f"S2 d{k}"] = (z[f"s2post_{k}"] - z[f"s2pre_{k}"], None)
        feats[f"S2 post {k}"] = (z[f"s2post_{k}"], None)

for k, (a, g) in feats.items():
    s, n = auc(a, g)
    rows.append((k, s, n))
rows.sort(key=lambda r: -abs((r[1] if np.isfinite(r[1]) else 0.5) - 0.5))
print(f"\n{'feature':34s} {'AUC':>6s}  {'pixels':>8s}   (AUC<0.5: low values mean affected)")
for k, s, n in rows:
    print(f"{k:34s} {s:6.3f}  {n:8d}")

# How much of the affected area can each track see well?
for t in tracks:
    g = z[f"t{t}_good"]
    print(f"track {t}: good geometry over {g[dom].mean():.0%} of checked area, {g[lab].mean():.0%} of affected area")
if "s2post_ndvi" in z:
    ok = np.isfinite(z["s2post_ndvi"]) & np.isfinite(z["s2pre_ndvi"])
    print(f"Sentinel-2 before+after both clear over {ok[dom].mean():.0%} of checked area, {ok[lab].mean():.0%} of affected area")
print(f"HAND-major in affected area: p50 {np.nanpercentile(z['hand_major'][lab], 50):.1f} m, p90 {np.nanpercentile(z['hand_major'][lab], 90):.1f} m, p99 {np.nanpercentile(z['hand_major'][lab], 99):.1f} m")
