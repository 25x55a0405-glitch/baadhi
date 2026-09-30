"""Train the flood segmentation model on Kuro Siwo (CPU-friendly).

Splits follow Kuro Siwo's official event lists (configs/train/data_config.json in their repo), so
validation and test events — including the Nepal event (1111007) — are never seen in training.

usage: python -m baadhi.ml.train --model floodunet --epochs 30 --name run1
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import CLASSES, build_model

ROOT = Path(__file__).resolve().parents[2]
ARRAYS = ROOT / "data" / "train" / "kurosiwo" / "arrays"
RUNS = ROOT / "models" / "runs"

# Official Kuro Siwo split (their configs/train/data_config.json). Most official validation events are in
# shards we did not download (only 197 tiles here), so four training events from different climates
# — Peru, Madagascar, Belgium, Germany — are moved to validation. Official test events stay untouched.
OFFICIAL_TRAIN = {130, 470, 555, 118, 174, 324, 421, 554, 427, 518, 502, 498, 497, 496, 492, 147, 267, 273, 275, 417, 567,
                  1111011, 1111004, 1111009, 1111010, 1111006, 1111005}
OFFICIAL_VAL = {514, 559, 279, 520, 437, 1111003, 1111008}
EXTRA_VAL = {1111010, 1111006, 518, 497}
TRAIN_ACTS = OFFICIAL_TRAIN - EXTRA_VAL
VAL_ACTS = OFFICIAL_VAL | EXTRA_VAL
TEST_ACTS = {321, 561, 445, 562, 411, 1111002, 277, 1111007, 205, 1111013}
IGNORE = 255


class Store:
    """All converted chunks, memory-mapped; index = (chunk, row, actid, flood fraction)."""

    def __init__(self, root: Path = ARRAYS):
        self.X, self.Y, rows = [], [], []
        for xf in sorted(root.glob("X_*.npy")):
            tag = xf.stem[2:]
            self.X.append(np.load(xf, mmap_mode="r"))
            self.Y.append(np.load(root / f"Y_{tag}.npy", mmap_mode="r"))
            meta = [json.loads(l) for l in (root / f"meta_{tag}.jsonl").read_text(encoding="utf8").splitlines() if l.strip()]
            c = len(self.X) - 1
            rows += [(c, i, int(m["actid"]), float(m.get("pflood") or 0)) for i, m in enumerate(meta)]
        self.rows = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4))

    def select(self, acts: set[int]) -> np.ndarray:
        return np.where(np.isin(self.rows[:, 2].astype(np.int64), list(acts)))[0]

    def get(self, k: int):
        c, i = int(self.rows[k, 0]), int(self.rows[k, 1])
        return np.asarray(self.X[c][i], dtype=np.float32), np.asarray(self.Y[c][i], dtype=np.int64)


def augment(x: np.ndarray, y: np.ndarray, crop: int, rng: np.random.Generator):
    h, w = y.shape
    i, j = rng.integers(0, h - crop + 1), rng.integers(0, w - crop + 1)
    x, y = x[:, i:i + crop, j:j + crop].copy(), y[i:i + crop, j:j + crop].copy()
    k = rng.integers(0, 4)
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    # independent calibration offsets on the after / before images (dB), kept consistent in the Δ channels
    a, b = rng.normal(0, 0.7), rng.normal(0, 0.7)
    x[0:2] += a / 6
    x[2:4] += b / 6
    x[4:6] += (a - b) / 3
    x[:6] += rng.normal(0, 0.03, size=x[:6].shape).astype(np.float32)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def batches(store: Store, idx: np.ndarray, n: int, bs: int, crop: int, rng: np.random.Generator):
    """n crops per epoch; tiles with ≥ 1 % flood are drawn 4× more often."""
    pflood = store.rows[idx, 3]
    w = np.where(pflood >= 1.0, 4.0, 1.0)
    pick = rng.choice(idx, size=n, p=w / w.sum())
    for s in range(0, n, bs):
        xs, ys = zip(*(augment(*store.get(k), crop, rng) for k in pick[s:s + bs]))
        yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def dice_loss(logits, y, cls: int = 2):
    valid = (y != IGNORE).float()
    p = torch.softmax(logits, 1)[:, cls] * valid
    t = (y == cls).float() * valid
    inter = (p * t).sum()
    return 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)


def confusion(pred: np.ndarray, y: np.ndarray, n: int = 3) -> np.ndarray:
    m = y != IGNORE
    return np.bincount(n * y[m] + pred[m], minlength=n * n).reshape(n, n)


def scores(cm: np.ndarray) -> dict:
    out = {}
    for c, name in enumerate(CLASSES):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        out[f"{name}_iou"] = tp / max(tp + fp + fn, 1)
        out[f"{name}_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    # "any water after the event" (permanent + flood) — what a rescuer sees as water
    wt = cm[1:, 1:].sum(); wfp = cm[0, 1:].sum(); wfn = cm[1:, 0].sum()
    out["water_iou"] = wt / max(wt + wfp + wfn, 1)
    return out


@torch.no_grad()
def evaluate(model, store: Store, idx: np.ndarray, limit: int = 1200) -> dict:
    model.eval()
    cm = np.zeros((3, 3), np.int64)
    sel = idx if len(idx) <= limit else np.random.default_rng(1).choice(idx, limit, replace=False)
    for s in range(0, len(sel), 16):
        xs, ys = zip(*(store.get(k) for k in sel[s:s + 16]))
        pred = model(torch.from_numpy(np.stack(xs))).argmax(1).numpy()
        cm += confusion(pred.ravel(), np.stack(ys).ravel())
    model.train()
    return scores(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="floodunet")
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--crops", type=int, default=6000, help="crops per epoch")
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--name", default="run")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    store = Store()
    tr, va = store.select(TRAIN_ACTS), store.select(VAL_ACTS)
    print(f"samples: {len(store.rows)} total | train {len(tr)} | val {len(va)} | test {len(store.select(TEST_ACTS))}", flush=True)
    kw = dict(base=a.base) if a.model == "floodunet" else dict(decoder_channels=(128, 64, 32, 16, 16))
    model = build_model(a.model, **kw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (a.crops // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    weights = torch.tensor([1.0, 1.5, 3.0])
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a) | {"train_acts": sorted(TRAIN_ACTS), "val_acts": sorted(VAL_ACTS)}, indent=1))
    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["epoch", "seconds", "loss", "val_flood_f1", "val_flood_iou", "val_water_iou", "val_perm_iou"])
    best = -1.0
    for ep in range(1, a.epochs + 1):
        t0, tot, nb = time.time(), 0.0, 0
        model.train()
        for x, y in batches(store, tr, a.crops, a.bs, a.crop, rng):
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights, ignore_index=IGNORE) + 0.5 * dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        v = evaluate(model, store, va)
        wr.writerow([ep, round(time.time() - t0), round(tot / nb, 4), round(v["flood_f1"], 4), round(v["flood_iou"], 4), round(v["water_iou"], 4), round(v["permanent_water_iou"], 4)])
        log.flush()
        print(f"epoch {ep:2d} | {time.time() - t0:5.0f}s | loss {tot / nb:.3f} | val flood F1 {v['flood_f1']:.3f} IoU {v['flood_iou']:.3f} | water IoU {v['water_iou']:.3f}", flush=True)
        torch.save(model.state_dict(), out / "last.pt")
        if v["flood_f1"] > best:
            best = v["flood_f1"]
            torch.save(model.state_dict(), out / "best.pt")
    print(f"done — best val flood F1 {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
