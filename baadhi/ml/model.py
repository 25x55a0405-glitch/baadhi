"""Flood segmentation network.

A compact U-Net built for CPU training (this project trains on a laptop, no GPU): residual conv
blocks, 4 down-sampling stages, ~2–8 M parameters depending on `base`. Input: the 8 feature channels
of baadhi.ml.features; output: 3 classes (no water, permanent water, flood water).
Trained from scratch on Kuro Siwo only — no ImageNet or other pre-trained weights.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .features import CLASSES, N_CHANNELS  # noqa: F401 — CLASSES re-exported for the trainer


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + self.skip(x), inplace=True)


class FloodUNet(nn.Module):
    def __init__(self, in_ch: int = N_CHANNELS, n_classes: int = len(CLASSES), base: int = 24):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 12]
        self.stem = ResBlock(in_ch, ch[0])
        self.down = nn.ModuleList(ResBlock(ch[i], ch[i + 1]) for i in range(4))
        self.up = nn.ModuleList(ResBlock(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4)))
        self.drop = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = [self.stem(x)]
        for d in self.down:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        y = self.drop(skips.pop())
        for u in self.up:
            s = skips.pop()
            y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
            y = u(torch.cat([y, s], 1))
        return self.head(y)


def build_model(kind: str = "floodunet", **kw) -> nn.Module:
    if kind == "floodunet":
        return FloodUNet(**kw)
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=kind, encoder_weights=None, in_channels=N_CHANNELS, classes=len(CLASSES), **kw)
