"""Score a trained flood model on Kuro Siwo events it never saw (official test events, incl. Nepal).

For comparison, the same tiles are also scored with the plain radar rule the detector uses without
the model (flood = very dark after AND ≥ 3 dB darker than before) — this shows what the model adds.

usage: python -m baadhi.ml.evaluate models/runs/main/best.pt [--split test|val] [--tta]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .infer import FloodModel
from .train import IGNORE, TEST_ACTS, VAL_ACTS, Store, confusion, scores

EVENT_NAMES = {1111007: "Nepal (Koshi plains)", 1111013: "USA"}   # others: Kuro Siwo gives only event ids


def rule_flood(x: np.ndarray) -> np.ndarray:
    """The detector's radar-water rule on the model's own inputs (undo the channel scaling)."""
    pv, ph = x[:, 0] * 6 - 15, x[:, 1] * 6 - 22
    dv, dh = x[:, 4] * 3, x[:, 5] * 3
    dark = (pv <= -15) & (ph <= -22)
    drop = (dv <= -3) | (dh <= -3)
    return np.where(dark & drop, 2, 0)


def f1_of(cm: np.ndarray, c: int = 2) -> tuple[float, float]:
    tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
    return 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "experiments" / "out" / "kurosiwo_test.json"),
                    help="write the results as JSON here (after every event)")
    a = ap.parse_args()
    fm = FloodModel.load(a.weights, tta=a.tta, threads=a.threads)       # .pt (PyTorch) or .onnx (onnxruntime, faster)
    store = Store()
    acts = TEST_ACTS if a.split == "test" else VAL_ACTS
    rows = store.rows
    results, total_m, total_r = {}, np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64)
    for act in sorted(acts):
        idx = np.where(rows[:, 2].astype(np.int64) == act)[0]
        if len(idx) == 0:
            continue
        cm_m, cm_r = np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64)
        for s in range(0, len(idx), 16):
            xs, ys = zip(*(store.get(k) for k in idx[s:s + 16]))
            x, y = np.stack(xs), np.stack(ys)
            p = fm._softmax(fm._logits(x))
            if a.tta:
                p = (p + fm._softmax(fm._logits(np.ascontiguousarray(x[..., ::-1])))[..., ::-1]) / 2
            cm_m += confusion(p.argmax(1).ravel(), y.ravel())
            # the rule only knows "flood" vs "not"; count permanent water as "not flood" for its score
            yr = np.where(y == 1, 0, y)
            cm_r += confusion(rule_flood(x).ravel(), yr.ravel())
        sm = scores(cm_m)
        fr, ir = f1_of(cm_r)
        flood_px = int(cm_m[2].sum())
        results[act] = {"event": EVENT_NAMES.get(act, str(act)), "tiles": len(idx), "flood_pixels": flood_px,
                        "model_flood_f1": round(sm["flood_f1"], 3), "model_flood_iou": round(sm["flood_iou"], 3),
                        "model_water_iou": round(sm["water_iou"], 3), "rule_flood_f1": round(fr, 3), "rule_flood_iou": round(ir, 3)}
        total_m += cm_m
        total_r += cm_r
        r = results[act]
        print(f"{act:>8} {r['event']:<22} tiles {r['tiles']:5d} flood px {flood_px:9d} | model F1 {r['model_flood_f1']:.3f} "
              f"IoU {r['model_flood_iou']:.3f} water IoU {r['model_water_iou']:.3f} | rule F1 {r['rule_flood_f1']:.3f}", flush=True)
        if a.out:
            Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf8")
    sm = scores(total_m)
    fr, ir = f1_of(total_r)
    results["all"] = {"model_flood_f1": round(sm["flood_f1"], 3), "model_flood_iou": round(sm["flood_iou"], 3),
                      "model_water_iou": round(sm["water_iou"], 3), "rule_flood_f1": round(fr, 3), "rule_flood_iou": round(ir, 3)}
    print(f"{'ALL':>8} {'':<22} | model F1 {sm['flood_f1']:.3f} IoU {sm['flood_iou']:.3f} water IoU {sm['water_iou']:.3f} | rule F1 {fr:.3f} IoU {ir:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
