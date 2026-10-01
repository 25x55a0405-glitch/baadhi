"""Score a saved run (the whole system, flood model included) against EMSR927 — CHECKING ONLY.

usage: python experiments/eval_run.py runs/<run-id> upper|bidur|phosretar
Reads the run's classes.tif and compares "affected" (classes 1–3) with the EMS observed-event polygons
inside the EMS areas of interest (minus "not analysed"), exactly like eval_detect.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baadhi.grid import Grid  # noqa: E402
import reference as R  # noqa: E402
from eval_detect import scores  # noqa: E402

AOIS = {"upper": (1, 2), "bidur": (3,), "phosretar": (5,)}


def main():
    run_dir, area = Path(sys.argv[1]), sys.argv[2]
    with rasterio.open(run_dir / "classes.tif") as src:
        cls = src.read(1)
        grid = Grid(src.crs, src.transform, src.width, src.height, tuple(json.loads((run_dir / "stats.json").read_text())["bbox"]))
    lab, dom = R.rasters(grid, AOIS[area])
    s = scores(cls > 0, lab, dom)
    stats = json.loads((run_dir / "stats.json").read_text(encoding="utf8"))
    print(f"{run_dir.name} vs EMSR927 {area}: P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} IoU {s['iou']:.3f} "
          f"| model: {stats.get('model') or 'none'} | evidence km²: {stats.get('evidence_km2')}")


if __name__ == "__main__":
    main()
