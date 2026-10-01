"""Figures for the report (EMS maps used for CHECKING only).

usage: python experiments/figures.py agreement upper|bidur|phosretar
       python experiments/figures.py model
Writes docs/img/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baadhi.detect import detect  # noqa: E402
import eval_detect as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
TITLES = {"upper": "EMSR927 AOI 1–2 · Syabru Bensi–Timure (development)", "bidur": "EMSR927 AOI 3 · Bidur (development)",
          "phosretar": "EMSR927 AOI 5 · Phosretar (held out)"}
C_TP, C_FP, C_FN = "#1b9e77", "#d95f02", "#7570b3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def agreement(name: str):
    z, terrain, tracks, pre, post = E.load(name)
    r = detect(terrain, tracks, pre, post)
    lab, dom = z["label"].astype(bool), z["domain"].astype(bool)
    ours = r["cls"] > 0
    s = E.scores(ours, lab, dom)
    rows, cols = np.nonzero(dom)
    pad = 40
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, dom.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, dom.shape[1])
    sl = (slice(r0, r1), slice(c0, c1))
    rgb = np.nan_to_num(np.clip(z["s2post_rgb"][sl] * 3.2, 0, 1)) if "s2post_rgb" in z else np.full((r1 - r0, c1 - c0, 3), 0.5)
    rgba = np.zeros((r1 - r0, c1 - c0, 4))
    d, o, L = dom[sl], ours[sl], lab[sl]
    for mask, col in ((o & L & d, C_TP), (o & ~L & d, C_FP), (~o & L & d, C_FN)):
        rgba[mask] = matplotlib.colors.to_rgba(col, 0.9)
    h, w = rgba.shape[:2]
    fig, ax = plt.subplots(figsize=(6.4, 6.4 * h / w + 0.5), dpi=220)
    ax.imshow(0.35 + 0.45 * rgb)
    ax.imshow(rgba, interpolation="nearest")
    ax.contour(d, [0.5], colors="white", linewidths=0.6)
    ax.set_title(f"{TITLES[name]}\nprecision {s['precision']:.2f} · recall {s['recall']:.2f} · F1 {s['f1']:.2f} · IoU {s['iou']:.2f}", fontsize=8.5)
    ax.legend(handles=[Patch(color=C_TP, label="both: Baadhi and EMS"), Patch(color=C_FP, label="Baadhi only"),
                       Patch(color=C_FN, label="EMS only (missed)")], loc="lower right", fontsize=7, framealpha=0.9)
    ax.axis("off")
    scale_px = 1000 / 10                                     # 1 km at 10 m
    ax.plot([10, 10 + scale_px], [h - 12, h - 12], color="white", lw=2)
    ax.text(10 + scale_px / 2, h - 18, "1 km", color="white", ha="center", fontsize=7)
    IMG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(IMG / f"agreement_{name}.png")
    plt.close(fig)
    print(name, {k: round(float(v), 3) for k, v in s.items()})


def model_chart():
    """Rule vs model F1 on data the model never saw (Kuro Siwo test events + live EMSR838)."""
    live = json.loads((ROOT / "experiments" / "out" / "live_water.json").read_text(encoding="utf8"))
    test_p = ROOT / "experiments" / "out" / "kurosiwo_test.json"
    rows = []
    if test_p.exists():
        t = json.loads(test_p.read_text(encoding="utf8"))
        for k, v in t.items():
            if k == "all":
                continue
            rows.append((f"Kuro Siwo test · {v['event']}", v["rule_flood_f1"], v["model_flood_f1"]))
    for k, v in live.items():
        rows.append((f"Live S-1 · EMSR838 {k}", v["rule"]["f1"], v["model"]["f1"]))
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 0.8), dpi=220)
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.19, [r[1] for r in rows], height=0.36, color="#b9b3a8", label="radar threshold rule")
    ax.barh(y - 0.19, [r[2] for r in rows], height=0.36, color="#256fd9", label="flood model (Baadhi)")
    for yi, r in zip(y, rows):
        ax.text(r[2] + 0.01, yi - 0.19, f"{r[2]:.2f}", va="center", fontsize=7, color="#256fd9")
        ax.text(r[1] + 0.01, yi + 0.19, f"{r[1]:.2f}", va="center", fontsize=7, color="#6b665d")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("flood F1 (higher is better)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout(pad=0.3)
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / "model_vs_rule.png")
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "agreement"
    if what == "agreement":
        for n in sys.argv[2:] or ["upper", "phosretar"]:
            agreement(n)
    elif what == "model":
        model_chart()
