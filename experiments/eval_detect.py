"""Check the detector against EMSR927 (checking only — never an input).

usage: python experiments/eval_detect.py upper [bidur phosretar]
Reports, inside the reference AOIs: precision, recall, F1 and IoU of "affected" (classes 1–3), plus
how much each evidence source contributes. Saves a picture per area in experiments/out/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baadhi.detect import detect, Params  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
OUT = Path(__file__).parent / "out"


def load(name):
    z = np.load(CACHE / f"dev_{name}.npz")
    terrain = {"hand_major": z["hand_major"], "slope": z["slope"]}
    tracks = [{"z_vv": z[f"t{t}_z_vv"], "z_vh": z[f"t{t}_z_vh"], "good": z[f"t{t}_good"]} for t in z["tracks"]]
    s2 = lambda tag: {k: z[f"s2{tag}_{k}"] for k in ("ndvi", "bsi", "bright", "mndwi")} if f"s2{tag}_ndvi" in z else None  # noqa: E731
    return z, terrain, tracks, s2("pre"), s2("post")


def scores(pred, lab, dom):
    tp = (pred & lab & dom).sum(); fp = (pred & ~lab & dom).sum(); fn = (~pred & lab & dom).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9), iou=tp / max(tp + fp + fn, 1))


def run(name, p=Params(), show=True):
    z, terrain, tracks, pre, post = load(name)
    r = detect(terrain, tracks, pre, post, p=p)
    lab, dom = z["label"], z["domain"]
    s = scores(r["cls"] > 0, lab, dom)
    s_nochan = scores((r["cls"] == 1) | (r["cls"] == 2), lab, dom)
    if show:
        print(f"{name:10s} affected (1–3): P {s['precision']:.2f} R {s['recall']:.2f} F1 {s['f1']:.2f} IoU {s['iou']:.2f} | "
              f"without channel: F1 {s_nochan['f1']:.2f} | area mapped {(r['cls'] > 0).sum() * 1e-4:.2f} km² (reference {lab.sum() * 1e-4:.2f} km² in AOI)")
        ev = r["evidence"]
        for k in ("optical_change", "optical_water", "radar_change"):
            e = ev[k]
            print(f"      evidence {k:15s}: recall {e[lab & dom].mean():.2f}, flags {e[~lab & dom].mean():.2%} of unaffected AOI pixels")
        OUT.mkdir(exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(13, 9), dpi=80)
        rgb = np.nan_to_num(np.clip(z["s2post_rgb"] * 3.2, 0, 1)) if "s2post_rgb" in z else np.zeros(lab.shape + (3,))
        ax[0].imshow(rgb); ax[0].contour(lab, [0.5], colors="cyan", linewidths=0.6); ax[0].set_title(f"{name}: after (S2) + EMSR927 outline (cyan)")
        show_cls = np.ma.masked_equal(r["cls"], 0)
        ax[1].imshow(rgb * 0.5); ax[1].imshow(show_cls, cmap=matplotlib.colors.ListedColormap(["#3a8dde", "#d9822b", "#9fb8c9"]), vmin=1, vmax=3, alpha=0.85)
        ax[1].contour(lab, [0.5], colors="cyan", linewidths=0.6); ax[1].contour(dom, [0.5], colors="white", linewidths=0.4)
        ax[1].set_title("Baadhi: water (blue), debris (orange), channel (grey)")
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.savefig(OUT / f"detect_{name}.png"); plt.close()
    return s


if __name__ == "__main__":
    for n in sys.argv[1:] or ["upper"]:
        if (CACHE / f"dev_{n}.npz").exists():
            run(n)
        else:
            print(f"{n}: not prepared yet")
