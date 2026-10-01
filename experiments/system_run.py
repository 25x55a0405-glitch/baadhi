"""Helpers so the checks against EMSR927 can score the DEPLOYED system (detector + flood model) instead of the detector alone.

    python experiments/eval_damage.py bidur runs/trishuli-bidur      # footprint = the run's classes.tif
    python experiments/eval_access.py bidur runs/trishuli-bidur

The pre-event OpenStreetMap comes from the same function and the same padded box as the pipeline uses, so a check
sees exactly the map data its run saw (both read the same disk cache). EMS data is still used for scoring only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baadhi.sources import osm  # noqa: E402


def run_arg(index: int = 2) -> Path | None:
    """The optional run folder on the command line (None = use the detector alone, as the first versions did)."""
    if len(sys.argv) > index:
        p = Path(sys.argv[index])
        if not p.is_absolute():
            p = ROOT / p
        if not (p / "classes.tif").exists():
            raise SystemExit(f"no classes.tif in {p}")
        return p
    return None


def footprint(run_dir: Path) -> np.ndarray:
    """The run's classes (0 none, 1 water, 2 debris, 3 channel); nodata counts as not affected."""
    with rasterio.open(run_dir / "classes.tif") as src:
        cls = src.read(1)
    return np.where(cls == 255, 0, cls).astype(np.uint8)


def pre_event(bbox, event, pad: float = 0.03):
    """(area layers, routing context, source name) — exactly what pipeline.analyse asks for."""
    obox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    got = osm.fetch_pre_event(obox, event, deadline=900.0, say=lambda m: print("  ", m, flush=True))
    return got["layers"], got["context"], f"{got['source']} {got['snapshot']}"
