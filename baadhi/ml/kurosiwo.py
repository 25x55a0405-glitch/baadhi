"""Convert Kuro Siwo GRD webdataset shards into compact training arrays.

Kuro Siwo (Bountos et al., NeurIPS 2024, CC BY 4.0): 43 flood events, one post-flood and two pre-flood
Sentinel-1 images per tile, expert labels 0 = no water, 1 = permanent water, 2 = flood.

Output per shard: X_<shard>.npy float16 (N, 8, 224, 224), Y_<shard>.npy uint8 (N, 224, 224) with
255 = ignore, and meta_<shard>.jsonl (activation id, flood/water fractions).
usage: python -m baadhi.ml.kurosiwo data/train/kurosiwo/train_GRD data/train/kurosiwo/arrays
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

from .features import build

NEEDED = ("flood_vv", "flood_vh", "sec1_vv", "sec1_vh", "sec2_vv", "sec2_vh", "dem", "mask", "valid_mask", "info")


def samples(tar_path: Path):
    """Yield one dict per sample, reading the tar sequentially (members of a sample are adjacent)."""
    cur_key, cur = None, {}
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, _, rest = m.name.partition(".")
            field = rest.rsplit(".", 1)[0]
            if key != cur_key:
                if cur_key is not None and all(k in cur for k in NEEDED):
                    yield cur
                cur_key, cur = key, {}
            data = tf.extractfile(m).read()
            cur[field] = json.loads(data) if field == "info" else np.load(io.BytesIO(data))
        if cur_key is not None and all(k in cur for k in NEEDED):
            yield cur


def convert(tar_path: Path, out_dir: Path, min_valid: float = 0.5, chunk: int = 1000) -> int:
    """Writes X_<shard>_<k>.npy / Y_<shard>_<k>.npy / meta_<shard>_<k>.jsonl in chunks of `chunk`
    samples, so memory stays under ~1 GB whatever the shard size."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tar_path.stem
    done = out_dir / f"done_{stem}"
    if done.exists():
        return -1
    X, Y, meta, k, total = [], [], [], 0, 0

    def flush():
        nonlocal X, Y, meta, k
        if X:
            np.save(out_dir / f"X_{stem}_{k:02d}.npy", np.stack(X))
            np.save(out_dir / f"Y_{stem}_{k:02d}.npy", np.stack(Y))
            (out_dir / f"meta_{stem}_{k:02d}.jsonl").write_text("\n".join(json.dumps(m) for m in meta), encoding="utf8")
            X, Y, meta, k = [], [], [], k + 1

    for s in samples(tar_path):
        valid = s["valid_mask"][0] > 0.5
        if valid.mean() < min_valid:
            continue
        pre_vv = np.nanmean([s["sec1_vv"][0], s["sec2_vv"][0]], axis=0)
        pre_vh = np.nanmean([s["sec1_vh"][0], s["sec2_vh"][0]], axis=0)
        x = build(s["flood_vv"][0], s["flood_vh"][0], pre_vv, pre_vh, s["dem"][0])
        y = s["mask"][0].astype("uint8")
        y[(y == 3) | ~valid] = 255  # 3 = "could not be labelled" in Kuro Siwo (their ignore_index)
        X.append(x.astype("float16"))
        Y.append(y)
        info = s["info"]
        meta.append({"actid": info.get("actid"), "pflood": info.get("pflood"), "pwater": info.get("pwater"), "grid_id": info.get("grid_id")})
        total += 1
        if len(X) >= chunk:
            flush()
    flush()
    done.write_text(str(total))
    return total


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    for tar in sorted(src.glob("*.tar")):
        n = convert(tar, dst)
        print(f"{tar.name}: {'already converted' if n < 0 else f'{n} samples'}", flush=True)
