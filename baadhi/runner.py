"""One complete analysis → a run folder the dashboard can open (used by the server and the command line).

    python -m baadhi.runner 85.30 28.13 85.40 28.29 2026-08-26 --name "Rasuwa" --out runs/rasuwa

Writes: classes/confidence GeoTIFFs, GeoJSON layers, map overlays (PNG), stats.json, the one-page
situation report (HTML + PDF when Chrome/Edge is available), meta.json and steps.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the ONNX export runs ~1.6× faster on a CPU than the PyTorch weights (identical results); either works
MODEL_PATH = next((p for p in (ROOT / "models" / "flood_model.onnx", ROOT / "models" / "flood_model.pt") if p.exists()),
                  ROOT / "models" / "flood_model.onnx")
CHROME = [Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe"),
          Path(r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
          Path(r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
          Path(r"C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
          Path("/usr/bin/google-chrome"), Path("/usr/bin/chromium"), Path("/usr/bin/chromium-browser"),
          Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")]


def load_model(path: Path = MODEL_PATH):
    """The trained segmentation model, or None (then the physical rules run alone)."""
    if not Path(path).exists():
        return None
    from .ml.infer import FloodModel
    return FloodModel.load(path)


def area_name(run) -> str:
    """Name a run after the most important named place near what was affected."""
    import numpy as np
    from shapely.geometry import shape

    rank = {"city": 0, "town": 1, "village": 2, "suburb": 3, "hamlet": 4, "neighbourhood": 5, "locality": 6, "isolated_dwelling": 7}
    feats = [f for f in (run.osm_layers.get("places") or {}).get("features", []) if f["properties"].get("name:en") or f["properties"].get("name")]
    if not feats:
        return f"Area around {(run.bbox[1] + run.bbox[3]) / 2:.3f}°N, {(run.bbox[0] + run.bbox[2]) / 2:.3f}°E"
    if run.cls is not None and (run.cls > 0).any():
        rr, cc = np.nonzero(run.cls > 0)
        x = run.grid.transform.c + (cc.mean() + 0.5) * run.grid.res
        y = run.grid.transform.f - (rr.mean() + 0.5) * run.grid.res
        cx, cy = run.grid.to_lonlat().transform(x, y)
    else:
        cx, cy = (run.bbox[0] + run.bbox[2]) / 2, (run.bbox[1] + run.bbox[3]) / 2

    def score(f):
        p = shape(f["geometry"]).representative_point()
        km = math.hypot((p.x - cx) * 111.32 * math.cos(math.radians(cy)), (p.y - cy) * 110.57)
        return km + 4 * rank.get(f["properties"].get("place"), 8)   # a town 8 km away beats a hamlet next door

    best = min(feats, key=score)
    return f"{best['properties'].get('name:en') or best['properties'].get('name')} area"


def to_pdf(html: Path, pdf: Path, timeout: float = 90.0) -> bool:
    """Print the HTML report to PDF with a headless Chrome/Edge, if one is installed. Chrome can take
    most of a minute to shut down after printing, so it is stopped as soon as the PDF is complete."""
    html, pdf = Path(html).resolve(), Path(pdf).resolve()
    exe = next((c for c in CHROME if c.exists()), None)
    if exe is None:
        return False
    prof = ROOT / ".work" / "chrome-pdf"      # kept between runs: a warm profile starts faster
    prof.mkdir(parents=True, exist_ok=True)
    pdf.unlink(missing_ok=True)
    args = [str(exe), "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-pdf-header-footer", "--no-first-run",
            "--no-default-browser-check", "--disable-extensions", "--disable-component-update", "--disable-background-networking",
            "--disable-sync", f"--print-to-pdf={pdf}", html.as_uri()]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return False
    t0, last = time.time(), -1
    try:
        while time.time() - t0 < timeout and proc.poll() is None:
            size = pdf.stat().st_size if pdf.exists() else -1
            if size > 0 and size == last:          # written and no longer growing
                break
            last = size
            time.sleep(0.5)
    finally:
        if proc.poll() is None:
            if sys.platform == "win32":             # the whole Chrome process tree
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    return pdf.exists() and pdf.stat().st_size > 0


def execute(bbox, date: dt.date, out: Path, name: str | None = None, progress=None, model=None, run_id: str | None = None,
            pdf: bool = True) -> dict:
    from . import report
    from .pipeline import analyse

    out = Path(out)
    t0 = time.time()
    steps = []

    def say(msg, frac):
        steps.append({"t": round(time.time() - t0, 1), "msg": msg})
        if progress:
            progress(msg, frac)

    run = analyse(tuple(bbox), date, out_dir=out, progress=say, ml_model=model)
    name = name or area_name(run)
    say("Writing the situation report", 0.97)
    report.map_png(run, out / "map.png")
    (out / "sitrep.html").write_text(report.sitrep_html(run, name, map_png="map.png"), encoding="utf8")
    if pdf:
        to_pdf(out / "sitrep.html", out / "sitrep.pdf")
    meta = {"id": run_id or out.name, "name": name, "bbox": [float(v) for v in bbox], "date": date.isoformat(),
            "created": dt.datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1), "model": run.model}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf8")
    (out / "steps.json").write_text(json.dumps(steps), encoding="utf8")
    return meta


def main():
    ap = argparse.ArgumentParser(description="Map flood/debris damage for an area and date.")
    ap.add_argument("west", type=float)
    ap.add_argument("south", type=float)
    ap.add_argument("east", type=float)
    ap.add_argument("north", type=float)
    ap.add_argument("date", type=dt.date.fromisoformat, help="flood date, YYYY-MM-DD")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None, help="run folder (default runs/<date>-<lon>-<lat>)")
    ap.add_argument("--model", default=str(MODEL_PATH), help="flood model weights ('' = rules only)")
    a = ap.parse_args()
    bbox = (a.west, a.south, a.east, a.north)
    out = Path(a.out) if a.out else ROOT / "runs" / f"{a.date}-{a.west:.3f}-{a.south:.3f}"
    t = time.time()
    meta = execute(bbox, a.date, out, a.name, progress=lambda m, f: print(f"[{time.time() - t:6.1f}s {f:4.0%}] {m}", flush=True),
                   model=load_model(Path(a.model)) if a.model else None)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
