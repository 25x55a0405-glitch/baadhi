"""Run a case-study area through the full system into runs/<preset id>/ (the dashboard lists it).

usage: python experiments/run_pipeline.py trishuli-upper [model.pt | none]
"""
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baadhi.runner import MODEL_PATH, execute, load_model  # noqa: E402
from baadhi.server import PRESETS  # noqa: E402

pid = sys.argv[1] if len(sys.argv) > 1 else "trishuli-upper"
model_arg = sys.argv[2] if len(sys.argv) > 2 else str(MODEL_PATH)
preset = next(p for p in PRESETS if p["id"] == pid)
out = Path(__file__).resolve().parents[1] / "runs" / pid
t = time.time()
meta = execute(preset["bbox"], dt.date.fromisoformat(preset["date"]), out, preset["name"],
               progress=lambda m, f: print(f"[{time.time() - t:6.1f}s {f:4.0%}] {m}", flush=True),
               model=None if model_arg == "none" else load_model(Path(model_arg)), run_id=pid)
print(meta)
