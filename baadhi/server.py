"""Baadhi dashboard: choose an area and a flood date on the map, get the flood/debris map, the damage,
the settlements cut off from hospitals and a one-page situation report.

    python -m baadhi.server                 # then open http://127.0.0.1:8000

One analysis runs at a time (it needs most of the laptop's memory and CPU); others wait in a queue.
Every run is saved under runs/<id>/ and stays available after a restart.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import re
import threading
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
WEB = ROOT / "web"
from .pipeline import NoDataError  # noqa: E402
from .runner import MODEL_PATH  # noqa: E402 — ONNX export if present, else the PyTorch weights
MAX_KM2, MIN_KM2 = 600.0, 1.0
FIRST_DATE = dt.date(2015, 1, 1)          # Sentinel-1/-2 era
FILES = {"classes.tif", "confidence.tif", "stats.json", "layers.json", "map.png", "sitrep.html", "sitrep.pdf", "meta.json",
         "classes.png", "confidence.png", "s2_before.png", "s2_after.png", "radar_change.png", "model_prob.png"}
GEO = {"buildings_affected", "cut_roads", "roads_blocked", "bridges", "health", "settlements", "buildings"}

PRESETS = [
    {"id": "trishuli-upper", "name": "Rasuwa — Timure to Syabru Bensi (Bhote Koshi)", "short": "Timure–Syabru Bensi", "bbox": [85.30, 28.13, 85.40, 28.29],
     "date": "2026-08-26", "note": "Trishuli GLOF case study · EMSR927 AOI 1–2", "group": "Case study · Trishuli GLOF, 26 Aug 2026"},
    {"id": "trishuli-bidur", "name": "Nuwakot — Bidur / Trishuli Bazar", "short": "Bidur", "bbox": [85.09, 27.85, 85.20, 28.02],
     "date": "2026-08-26", "note": "Trishuli GLOF case study · EMSR927 AOI 3", "group": "Case study · Trishuli GLOF, 26 Aug 2026"},
    {"id": "trishuli-phosretar", "name": "Nuwakot/Dhading — Trishuli at Phosretar", "short": "Phosretar", "bbox": [84.93, 27.78, 85.12, 27.875],
     "date": "2026-08-26", "note": "Trishuli GLOF case study · EMSR927 AOI 5", "group": "Case study · Trishuli GLOF, 26 Aug 2026"},
    {"id": "punjab-chenab", "name": "Punjab — Chenab floodplain near Rasoo Nagar", "short": "Punjab 2025 (Chenab)", "bbox": [73.5128, 32.1999, 73.6528, 32.3399],
     "date": "2025-08-27", "note": "2025 Punjab floods · standing water on a plain (EMSR838 AOI01 for checking)", "group": "Other floods"},
    {"id": "chamoli", "name": "Chamoli — Rishiganga / Dhauliganga debris flow", "short": "Chamoli 2021", "bbox": [79.55, 30.44, 79.78, 30.57],
     "date": "2021-02-07", "note": "Uttarakhand rock-ice avalanche and debris flow", "group": "Other floods"},
    {"id": "teesta", "name": "North Sikkim — Teesta at Chungthang", "short": "Teesta GLOF 2023", "bbox": [88.52, 27.38, 88.68, 27.62],
     "date": "2023-10-04", "note": "South Lhonak glacial-lake outburst", "group": "Other floods"},
]

app = FastAPI(title="Baadhi", docs_url="/api/docs")


@app.middleware("http")
async def _fresh_assets(request, call_next):
    """Browsers must revalidate the page's own code, so an update is never hidden behind a stale cache."""
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".js", ".css", ".html")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
QUEUE = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
_model = {"obj": None, "tried": False}


_model_lock = threading.Lock()


def flood_model():
    """The trained segmentation model, loaded once (the server preloads it at start-up, so the first
    analysis does not wait ~1 min for PyTorch). Missing weights → the physical rules run alone."""
    with _model_lock:
        if not _model["tried"]:
            from .runner import load_model
            try:
                _model["obj"] = load_model(MODEL_PATH)
            finally:
                _model["tried"] = True
    return _model["obj"]


@app.on_event("startup")
def _preload():
    threading.Thread(target=flood_model, name="model-preload", daemon=True).start()


class RunRequest(BaseModel):
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="west, south, east, north (degrees)")
    date: dt.date
    name: str | None = Field(None, max_length=120)


class FlowRequest(BaseModel):
    lon: float
    lat: float
    date: dt.date


def area_km2(b) -> float:
    w, s, e, n = b
    return abs(e - w) * 111.32 * math.cos(math.radians((s + n) / 2)) * abs(n - s) * 110.57


def check(req: RunRequest):
    w, s, e, n = req.bbox
    if not (-180 <= w < e <= 180 and -85 <= s < n <= 85):
        raise HTTPException(400, "The area must be west < east and south < north, in degrees.")
    a = area_km2(req.bbox)
    if a < MIN_KM2:
        raise HTTPException(400, f"The area is {a:.1f} km² — draw at least {MIN_KM2:.0f} km².")
    if a > MAX_KM2:
        raise HTTPException(400, f"The area is {a:.0f} km² — the limit is {MAX_KM2:.0f} km² (about 25 × 25 km). Draw a smaller box.")
    if not (FIRST_DATE <= req.date <= dt.date.today()):
        raise HTTPException(400, f"The flood date must be between {FIRST_DATE} and today.")


def _work(job_id: str, req: RunRequest):
    from .runner import execute

    job = JOBS[job_id]
    out = RUNS / job_id
    job.update(status="running", started=time.time())

    def progress(msg, frac):
        with LOCK:
            job["steps"].append({"t": round(time.time() - job["started"], 1), "msg": msg})
            job["frac"] = frac

    try:
        progress("Loading the flood model", 0.01)
        meta = execute(req.bbox, req.date, out, req.name, progress=progress, model=flood_model(), run_id=job_id, pdf=False)
        job.update(status="done", frac=1.0, name=meta["name"])
        threading.Thread(target=_ensure_pdf, args=(job_id,), daemon=True).start()   # ready by the time anyone clicks
    except Exception as e:  # noqa: BLE001 — report any failure to the user instead of hanging
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf8")
        job.update(status="error", error=str(e) if isinstance(e, NoDataError) else f"{type(e).__name__}: {e}")


_pdf_locks: dict[str, threading.Lock] = {}


def _ensure_pdf(job_id: str) -> bool:
    """Print the run's report to PDF once (in the background after a run, or on the first request)."""
    from .runner import to_pdf
    d = RUNS / job_id
    with LOCK:
        lock = _pdf_locks.setdefault(job_id, threading.Lock())
    with lock:
        if not (d / "sitrep.pdf").exists() and (d / "sitrep.html").exists():
            to_pdf(d / "sitrep.html", d / "sitrep.pdf")
    return (d / "sitrep.pdf").exists()


# ---------------------------------------------------------------------------------------------- API
@app.get("/api/presets")
def presets():
    return PRESETS


@app.post("/api/runs")
def start(req: RunRequest):
    check(req)
    job_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    with LOCK:
        ahead = sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))
        JOBS[job_id] = {"id": job_id, "status": "queued", "steps": [], "frac": 0.0, "bbox": req.bbox, "date": req.date.isoformat(),
                        "name": req.name, "created": dt.datetime.now().isoformat(timespec="seconds"), "ahead": ahead}
    QUEUE.submit(_work, job_id, req)
    return {"id": job_id, "ahead": ahead}


def _saved(job_id: str) -> dict | None:
    d = RUNS / job_id
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf8"))
    steps = json.loads((d / "steps.json").read_text(encoding="utf8")) if (d / "steps.json").exists() else []
    return {**meta, "status": "done", "frac": 1.0, "steps": steps}


def _valid_id(job_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", job_id):
        raise HTTPException(404, "No such run")


@app.get("/api/runs")
def runs():
    items = []
    for d in sorted(RUNS.glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items.append(json.loads(d.read_text(encoding="utf8")))
        except Exception:  # noqa: BLE001
            continue
    return items


@app.get("/api/runs/{job_id}")
def status(job_id: str):
    _valid_id(job_id)
    job = JOBS.get(job_id)
    if job is None or job["status"] == "done":
        saved = _saved(job_id)
        if saved is None:
            if job is None:
                raise HTTPException(404, "No such run")
        else:
            job = {**(job or {}), **saved}
    out = dict(job)
    d = RUNS / job_id
    if out["status"] == "done":
        out["stats"] = json.loads((d / "stats.json").read_text(encoding="utf8"))
        out["layers"] = json.loads((d / "layers.json").read_text(encoding="utf8")) if (d / "layers.json").exists() else {}
        out["has_pdf"] = (d / "sitrep.pdf").exists()
    return out


@app.get("/api/runs/{job_id}/geo/{name}.geojson")
def geo(job_id: str, name: str):
    _valid_id(job_id)
    if name not in GEO:
        raise HTTPException(404, "No such layer")
    p = RUNS / job_id / f"{name}.geojson"
    if not p.exists():
        raise HTTPException(404, "Layer not produced for this run")
    return FileResponse(p, media_type="application/geo+json")


@app.get("/api/runs/{job_id}/bundle.zip")
def bundle(job_id: str):
    _valid_id(job_id)
    d = RUNS / job_id
    if not (d / "stats.json").exists():
        raise HTTPException(404, "No such run")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix in (".tif", ".geojson", ".json", ".html", ".pdf", ".png"):
                z.write(p, p.name)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="baadhi-{job_id}.zip"'})


@app.get("/api/runs/{job_id}/{file}")
def run_file(job_id: str, file: str):
    _valid_id(job_id)
    if file not in FILES:
        raise HTTPException(404, "No such file")
    p = RUNS / job_id / file
    if file == "sitrep.pdf" and not p.exists() and not _ensure_pdf(job_id):
        raise HTTPException(404, "No PDF printer (Chrome/Edge) found — open the web-page version and print it")
    if not p.exists():
        raise HTTPException(404, "Not produced for this run")
    if file == "sitrep.html":
        return HTMLResponse(p.read_text(encoding="utf8"))
    return FileResponse(p)


@app.post("/api/flowpath")
def flowpath(req: FlowRequest):
    from .flowpath import trace
    if not (-180 <= req.lon <= 180 and -85 <= req.lat <= 85):
        raise HTTPException(400, "Point outside the map")
    t = time.time()
    fp = trace(req.lon, req.lat, req.date)
    return {"line": fp.line, "length_km": fp.length_km, "settlements": fp.settlements, "seconds": round(time.time() - t, 1)}


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL_PATH.exists(), "queued": sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB), name="web")


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    print(f"Baadhi dashboard on http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
