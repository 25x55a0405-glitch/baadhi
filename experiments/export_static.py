"""Freeze saved analyses into plain files — a public demo for any static host (Cloudflare Pages, GitHub Pages, ...).

    python experiments/export_static.py [--out site] [--runs id,id,...] [--repo owner/name]

The Baadhi dashboard normally talks to a Python server (baadhi/server.py). A static host has no Python, and a live
analysis needs gigabytes of satellite data, so the demo shows SAVED analyses only. This script asks a running server
(started if none is) for everything a saved analysis needs and writes each response to the path it has under /api, so
the unchanged web/ code runs on plain files; window.BAADHI_DEMO switches the live controls off.

    site/index.html, app.js, style.css, vendor/      the dashboard
    site/api/presets, api/runs, api/health            JSON (as <path>/index.html: the host serves it for /api/...)
    site/api/runs/<id>/                               the run's JSON, rasters, GeoJSON layers, situation report, bundle (if small)

Cloudflare Pages allows 25 MiB per file; bigger files are left out (the bundle is simply not offered).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "video"))
from make_video import ensure_server  # noqa: E402  (finds a running Baadhi server or starts one)

DEFAULT_RUNS = ["trishuli-upper", "trishuli-bidur", "trishuli-phosretar", "practice-chamoli", "practice-teesta",
                "practice-melamchi", "practice-silchar", "practice-wayanad"]
LIMIT = 24 * 1024 * 1024


def write(path: Path, data: bytes | str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    ap.add_argument("--repo", default="25x55a0405-glitch/baadhi", help="owner/name; linked from the page only if it is public")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    base = ensure_server()
    from baadhi import server as S

    get = lambda path, **kw: requests.get(base + path, timeout=300, **kw)  # noqa: E731
    wanted = [r for r in a.runs.split(",") if r]
    listing = {it["id"]: it for it in get("/api/runs").json()}
    runs = [r for r in wanted if r in listing]
    missing = [r for r in wanted if r not in listing]
    if missing:
        print("not found (skipped):", ", ".join(missing))

    total = 0
    for rid in runs:
        job = get(f"/api/runs/{rid}").json()
        d = out / "api" / "runs" / rid
        n_bytes = 0
        for f in sorted(S.FILES):
            r = get(f"/api/runs/{rid}/{f}")
            if r.status_code != 200:
                continue
            if len(r.content) > LIMIT:
                print(f"  {rid}/{f}: {len(r.content) / 2**20:.1f} MiB — over the host's limit, left out")
                continue
            write(d / f, r.content)
            n_bytes += len(r.content)
        for g in sorted(S.GEO):
            r = get(f"/api/runs/{rid}/geo/{g}.geojson")
            if r.status_code == 200 and len(r.content) <= LIMIT:
                write(d / "geo" / f"{g}.geojson", r.content)
                n_bytes += len(r.content)
        z = get(f"/api/runs/{rid}/bundle.zip")
        job["bundle"] = z.status_code == 200 and len(z.content) <= LIMIT
        if job["bundle"]:
            write(d / "bundle.zip", z.content)
            n_bytes += len(z.content)
        write(d / "index.html", json.dumps(job))                        # GET /api/runs/<id>
        total += n_bytes
        print(f"{rid}: {n_bytes / 2**20:6.1f} MiB  bundle={'yes' if job['bundle'] else 'no (too big)'}")

    # the history list and the presets: only analyses that are included
    write(out / "api" / "runs" / "index.html", json.dumps([listing[r] for r in runs]))
    presets = []
    for p in get("/api/presets").json():
        run = next((c for c in (p["id"], "practice-" + p["id"]) if c in runs), None)
        if run:
            presets.append({**p, "run": run})
    write(out / "api" / "presets" / "index.html", json.dumps(presets))
    write(out / "api" / "health" / "index.html", json.dumps({"ok": True, "app": "baadhi", "static": True}))

    # the dashboard itself, with the demo switch (and the repository link, once the repository is public)
    public = requests.get(f"https://api.github.com/repos/{a.repo}", timeout=30).status_code == 200
    flag = "<script>window.BAADHI_DEMO = true;" + (f' window.BAADHI_REPO = "https://github.com/{a.repo}";' if public else "") + "</script>\n"
    html = (ROOT / "web" / "index.html").read_text(encoding="utf8")
    anchor = '<script src="/vendor/maplibre-gl.js"></script>'
    assert anchor in html
    write(out / "index.html", html.replace(anchor, flag + anchor, 1))
    for name in ("app.js", "style.css"):
        shutil.copy2(ROOT / "web" / name, out / name)
    shutil.copytree(ROOT / "web" / "vendor", out / "vendor", dirs_exist_ok=True)
    write(out / "_headers", "/api/*\n  Cache-Control: public, max-age=300\n/vendor/*\n  Cache-Control: public, max-age=86400\n")

    files = [p for p in out.rglob("*") if p.is_file()]
    print(f"\n{len(runs)} analyses, {len(files)} files, {sum(p.stat().st_size for p in files) / 2**20:.0f} MiB in {out}")
    print("repository link on the page:", "yes (public)" if public else "no (the repository is private)")


if __name__ == "__main__":
    main()
