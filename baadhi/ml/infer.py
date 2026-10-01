"""Run the trained flood model on a live scene (any area, any date).

Live radar from Planetary Computer is terrain-flattened gamma0 without speckle filtering; the model
learned from Kuro Siwo's sigma0 with a Lee filter. Each image is therefore converted first:
sigma0 ≈ gamma0 × cos(incidence), then Lee-filtered, then turned into exactly the same 8 feature
channels as in training (baadhi.ml.features.build).

    model = FloodModel.load()                  # models/flood_model.pt
    prob = model(tracks, terrain, grid)        # flood-water probability, NaN where no radar

Several radar tracks: a pixel's probability is the highest among the tracks that see it properly
(no layover/shadow) — flood extent is what any post-event pass saw flooded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import features as FE
from .features import CLASSES

# PyTorch is imported only for .pt weights (training, export): the dashboard runs the ONNX export
# with onnxruntime alone.
MODELS = Path(__file__).resolve().parents[2] / "models"
DEFAULT = MODELS / "flood_model.onnx"


class FloodModel:
    """The trained network behind one interface: PyTorch weights (.pt) or an ONNX export (.onnx, faster on CPU)."""

    def __init__(self, net=None, session=None, tile: int = 512, overlap: int = 64, tta: bool = True,
                 threads: int | None = None, name: str = "flood model"):
        self.net = net.eval() if net is not None else None
        self.session = session
        self.name = name
        self.tile, self.overlap, self.tta = tile, overlap, tta
        if threads and net is not None:
            import torch
            torch.set_num_threads(threads)

    @staticmethod
    def _config(path: Path) -> dict:
        cfg_path = path.with_suffix(".json") if path.with_suffix(".json").exists() else path.parent / "config.json"
        return json.loads(cfg_path.read_text(encoding="utf8"))

    @classmethod
    def load(cls, path: str | Path = DEFAULT, threads: int | None = None, **kw) -> "FloodModel":
        """Weights (.pt, with the trainer's config.json or <name>.json beside it) or an ONNX export (.onnx + <name>.json)."""
        path = Path(path)
        cfg = cls._config(path)
        kw.setdefault("name", f"{cfg['model']} U-Net trained on Kuro Siwo")
        if path.suffix == ".onnx":
            import os

            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2) - 2)
            return cls(session=ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"]), **kw)
        import torch

        from .model import build_model
        mk = dict(base=cfg["base"]) if cfg["model"] == "floodunet" else dict(decoder_channels=(128, 64, 32, 16, 16))
        net = build_model(cfg["model"], **mk)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return cls(net, threads=threads, **kw)

    def _logits(self, x: np.ndarray) -> np.ndarray:
        """(N, 8, t, t) float32 → (N, 3, t, t) logits."""
        if self.session is not None:
            return self.session.run(None, {self.session.get_inputs()[0].name: x})[0]
        import torch
        with torch.no_grad():
            return self.net(torch.from_numpy(x)).numpy()

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    # ------------------------------------------------------------------ raw prediction on a feature stack
    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (8, H, W) features → (3, H, W) class probabilities, sliding windows blended smoothly."""
        _, h, w = x.shape
        t, ov = self.tile, self.overlap
        step = t - ov
        ph, pw = max(h, t), max(w, t)
        xp = np.zeros((x.shape[0], ph, pw), np.float32)
        xp[:, :h, :w] = x
        out = np.zeros((len(CLASSES), ph, pw), np.float32)
        wsum = np.zeros((ph, pw), np.float32)
        ramp = np.minimum(np.arange(t) + 1, np.arange(t)[::-1] + 1).astype(np.float32)
        win = np.minimum(np.minimum.outer(ramp, ramp) / max(ov // 2, 1), 1.0)   # 1 inside, fades at the edges
        ys = list(range(0, ph - t + 1, step)) + ([ph - t] if (ph - t) % step else [])
        xs = list(range(0, pw - t + 1, step)) + ([pw - t] if (pw - t) % step else [])
        for i in ys:
            for j in xs:
                patch = np.ascontiguousarray(xp[None, :, i:i + t, j:j + t])
                if self.tta:                          # the patch and its mirror image in one batch, averaged
                    pr = self._softmax(self._logits(np.concatenate([patch, np.ascontiguousarray(patch[..., ::-1])])))
                    p = (pr[0] + pr[1][..., ::-1]) / 2
                else:
                    p = self._softmax(self._logits(patch))[0]
                out[:, i:i + t, j:j + t] += p * win
                wsum[i:i + t, j:j + t] += win
        return out[:, :h, :w] / np.maximum(wsum[:h, :w], 1e-6)

    # ------------------------------------------------------------------ one radar track → features
    @staticmethod
    def track_features(sd, dem: np.ndarray, incidence: np.ndarray, res: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        """StackData (gamma0 RTC, linear) → (8, H, W) features and a validity mask."""
        cos = np.cos(np.radians(incidence)).astype("float32")

        def sigma0(g0):
            return FE.lee_sigma_like(np.where(np.asarray(g0) > 0, g0, np.nan) * cos)

        post_vv, post_vh = sigma0(sd.post_vv), sigma0(sd.post_vh)
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pre_vv = np.nanmean([sigma0(a) for a in sd.pre_vv], axis=0)
                pre_vh = np.nanmean([sigma0(a) for a in sd.pre_vh], axis=0)
        valid = np.isfinite(post_vv) & np.isfinite(post_vh) & np.isfinite(pre_vv) & np.isfinite(pre_vh) & np.isfinite(dem)
        return FE.build(post_vv, post_vh, pre_vv, pre_vh, dem, res), valid

    # ------------------------------------------------------------------ pipeline hook
    def __call__(self, tracks: list[dict], terrain, grid, return_all: bool = False):
        """Flood-water probability on the grid (max over tracks that see each pixel properly)."""
        from .. import terrain as T

        best = np.full(grid.shape, np.nan, np.float32)
        perm = np.full(grid.shape, np.nan, np.float32)
        per_track = []
        for tr in tracks:
            sd = tr["stack"]
            inc = T.incidence_from_footprint(grid, sd.stack.post.items[0].geometry, sd.stack.direction)
            x, valid = self.track_features(sd, terrain.dem, inc, grid.res)
            p = self.predict(x)
            ok = valid & tr["good"]
            fl = np.where(ok, p[2], np.nan)
            pw = np.where(ok, p[1], np.nan)
            per_track.append(fl)
            best = np.fmax(best, fl)
            perm = np.fmax(perm, pw)
        if return_all:
            return best, perm, per_track
        return best


def export_onnx(pt_path: str | Path, onnx_path: str | Path, tile: int = 512) -> Path:
    """Export .pt weights to ONNX (any tile size divisible by 32) and copy the config beside it."""
    import shutil

    import torch
    pt_path, onnx_path = Path(pt_path), Path(onnx_path)
    fm = FloodModel.load(pt_path, tta=False)
    dummy = torch.zeros(1, FE.N_CHANNELS, tile, tile)
    torch.onnx.export(fm.net, dummy, str(onnx_path), input_names=["x"], output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"x": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}})
    cfg = pt_path.with_suffix(".json") if pt_path.with_suffix(".json").exists() else pt_path.parent / "config.json"
    if cfg.resolve() != onnx_path.with_suffix(".json").resolve():
        shutil.copy(cfg, onnx_path.with_suffix(".json"))
    return onnx_path


if __name__ == "__main__":      # python -m baadhi.ml.infer export models/flood_model.pt models/flood_model.onnx
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "export":
        print(export_onnx(sys.argv[2], sys.argv[3]))
