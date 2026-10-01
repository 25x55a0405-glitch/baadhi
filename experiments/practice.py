"""Step 7 practice: other well-known floods through the whole system, cold (no cached imagery), timed.

Judges will pick their own areas and dates, so the system must cope with events it was never tuned on:
glacial-lake outbursts, debris flows, landslides and plains river floods, in any season.

usage: python experiments/practice.py [event ...]          (default: all)
Writes runs/practice-<event>/ (the dashboard lists them) and experiments/out/practice.json.
"""
import datetime as dt
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baadhi.runner import execute, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVENTS = {
    # name: (bbox W S E N, flood date, description)
    "chamoli": ((79.55, 30.44, 79.78, 30.57), dt.date(2021, 2, 7), "Chamoli, Uttarakhand — Rishiganga / Dhauliganga rock-ice avalanche and debris flow"),
    "teesta": ((88.52, 27.38, 88.68, 27.62), dt.date(2023, 10, 4), "North Sikkim — Teesta at Chungthang, South Lhonak glacial-lake outburst"),
    "melamchi": ((85.52, 27.78, 85.62, 27.95), dt.date(2021, 6, 15), "Sindhupalchok — Melamchi debris flood"),
    "wayanad": ((76.08, 11.42, 76.20, 11.53), dt.date(2024, 7, 30), "Wayanad, Kerala — Mundakkai / Chooralmala landslide debris flow"),
    "silchar": ((92.72, 24.75, 92.88, 24.88), dt.date(2022, 6, 20), "Silchar, Assam — Barak river flood"),
}


def main():
    names = sys.argv[1:] or list(EVENTS)
    model = load_model()
    out_json = ROOT / "experiments" / "out" / "practice.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_json.read_text(encoding="utf8")) if out_json.exists() else {}
    for name in names:
        bbox, date, desc = EVENTS[name]
        out = ROOT / "runs" / f"practice-{name}"
        t0 = time.time()
        print(f"=== {name}: {desc} ({date})", flush=True)
        try:
            meta = execute(bbox, date, out, desc, model=model, run_id=f"practice-{name}",
                           progress=lambda m, f: print(f"  [{time.time() - t0:6.1f}s {f:4.0%}] {m}", flush=True))
            stats = json.loads((out / "stats.json").read_text(encoding="utf8"))
            results[name] = {"ok": True, "seconds": meta["seconds"], "date": str(date), "bbox": bbox,
                             **{k: stats.get(k) for k in ("affected_km2", "water_km2", "debris_km2", "buildings_hit", "roads_blocked_km",
                                                          "bridges_at_risk", "access_cut_off", "access_settlements", "tracks", "optical_dates")}}
        except Exception as e:  # noqa: BLE001 — record and carry on with the next event
            traceback.print_exc()
            results[name] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"}
        out_json.write_text(json.dumps(results, indent=1, default=str), encoding="utf8")
        r = results[name]
        print(f"=== {name}: {'OK' if r['ok'] else 'FAILED'} in {r['seconds']:.0f}s — "
              + (f"{r['affected_km2']} km² affected, {r['buildings_hit']} buildings hit, {r['access_cut_off']} cut off" if r["ok"] else r["error"]),
              flush=True)


if __name__ == "__main__":
    main()
