"""One-page situation report, generated from a pipeline run. Every number comes from the run's maps;
nothing is typed in by hand or invented.

    html = sitrep_html(run, area_name="Rasuwa — Bhote Koshi / Trishuli corridor")
"""
from __future__ import annotations

import datetime as dt
import html

ATTRIBUTION = [
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 "
    "provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "© OpenStreetMap contributors (pre-event snapshot, ODbL).",
    "Flood model trained on Kuro Siwo (Bountos et al., NeurIPS 2024, CC BY 4.0).",
]

LIMITS = [
    "Sentinel satellites revisit an area only every few days: what happened between passes, and water that drained "
    "before the next pass, is not seen. No warning ahead of an event is possible.",
    "Radar cannot see steep slopes facing away from or towards the satellite (shadow, layover); optical images need clear sky.",
    "At 10 m resolution a building \"hit\" means it lies in the flood/debris footprint, not that it collapsed.",
    "Buildings, roads and settlements come from OpenStreetMap before the event; anything not mapped is not counted.",
    "Travel times assume typical speeds per road type; roads outside the analysed area are assumed passable.",
]


def _n(x, unit=""):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:,.2f}{unit}"
    return f"{x:,}{unit}"


def _name(props):
    nm = props.get("label") or props.get("name:en") or props.get("name")
    return html.escape(nm) if nm else f"<i>unnamed {html.escape(props.get('place') or 'place')}</i>"


def sitrep_html(run, area_name: str, map_png: str | None = None, generated: dt.datetime | None = None) -> str:
    s = run.stats
    gen = (generated or dt.datetime.now()).strftime("%d %b %Y, %H:%M")
    cut = [f for f in run.access.settlements if f["properties"]["status"] == "cut_off"]
    maybe = [f for f in run.access.settlements if f["properties"]["status"] == "possibly_cut_off"]
    detour = [f for f in run.access.settlements if f["properties"]["status"] == "long_detour"]
    cut.sort(key=lambda f: (f["properties"].get("hospital_min_before") or 999))
    rows = "".join(
        f"<tr><td>{_name(f['properties'])}</td><td>{_n(f['properties'].get('hospital_min_before'))} min</td>"
        f"<td>{'none by road' if f['properties'].get('hospital_min_after') is None else _n(f['properties'].get('hospital_min_after')) + ' min'}</td>"
        f"<td>{'no mapped path' if f['properties'].get('hospital_on_foot_min') is None else _n(f['properties'].get('hospital_on_foot_min')) + ' min'}</td></tr>"
        for f in cut[:10])
    health_hit = [f for f in run.damage.health if f["properties"]["status"] in ("hit", "possibly hit")]
    health_line = ", ".join(_name(f["properties"]) for f in health_hit) or "none of the mapped facilities"
    tracks = "; ".join(html.escape(t) for t in run.tracks)
    od = run.optical_dates
    # blocked roads by name: where to send the machinery first
    by_road: dict[str, list] = {}
    for f in run.damage.stretches if run.damage else []:
        p = f["properties"]
        if p.get("blocks"):
            key = p.get("name") or p.get("ref") or f"unnamed {p.get('highway') or 'road'}"
            by_road.setdefault(key, []).append(p["length_m"])
    road_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{len(v)}</td><td>{sum(v) / 1000:.1f} km</td></tr>"
                        for k, v in sorted(by_road.items(), key=lambda kv: -sum(kv[1]))[:5])
    ev = s.get("evidence_km2") or {}
    found = " · ".join(f"{label} {ev[k]:.2f}" for k, label in (("model_water", "AI model"), ("radar_change", "radar change"),
                       ("radar_water", "radar water"), ("optical_change", "optical change"), ("optical_water", "optical water"),
                       ("terrain_completed", "terrain fill")) if ev.get(k))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Situation report — {html.escape(area_name)}</title>
<style>
@page {{ size: A4; margin: 12mm; }}
body {{ font: 10.5px/1.45 'Segoe UI', system-ui, sans-serif; color: #15191c; margin: 0; }}
h1 {{ font-size: 18px; margin: 0 0 2px; }} .sub {{ color: #5a646b; margin-bottom: 10px; }}
.grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin: 8px 0 10px; }}
.k {{ border: 1px solid #d5dade; border-radius: 6px; padding: 6px 8px; }} .k b {{ display: block; font-size: 17px; }}
.k span {{ color: #5a646b; }} .warn {{ border-color: #c0392b; }} .warn b {{ color: #b3261e; }}
table {{ border-collapse: collapse; width: 100%; margin: 4px 0 8px; }} td, th {{ border-bottom: 1px solid #e3e7ea; padding: 3px 4px; text-align: left; }}
th {{ color: #5a646b; font-weight: 600; }} h2 {{ font-size: 12px; margin: 10px 0 3px; }}
.cols {{ display: grid; grid-template-columns: 1.1fr 1fr; gap: 12px; }} img {{ width: 100%; border: 1px solid #d5dade; border-radius: 4px; }}
.small {{ font-size: 9px; color: #5a646b; }} ul {{ margin: 2px 0 6px 16px; padding: 0; }}
.badge {{ display: inline-block; background: #fff3cd; border: 1px solid #e0c46c; border-radius: 4px; padding: 1px 6px; font-size: 9.5px; }}
</style></head><body>
<h1>Flood &amp; debris situation report — {html.escape(area_name)}</h1>
<div class="sub">Event date {run.event:%d %B %Y} · generated {gen} by Baadhi from satellite data · <span class="badge">Educational prototype — verify on the ground</span></div>
{f'<div class="badge" style="display:block;margin:0 0 8px;background:#fdecea;border-color:#e6a39b">{html.escape(run.osm_note)}</div>' if getattr(run, "osm_note", "") else ''}
<div class="grid">
  <div class="k warn"><b>{_n(s['affected_km2'])} km²</b><span>flooded or debris-covered ({_n(s['water_km2'])} km² water, {_n(s['debris_km2'])} km² debris/mud)</span></div>
  <div class="k warn"><b>{_n(s['buildings_hit'])}</b><span>buildings inside the footprint (+{_n(s['buildings_possibly_hit'])} at its edge) of {_n(s['buildings_total'])} mapped</span></div>
  <div class="k warn"><b>{_n(s['access_cut_off'])}</b><span>settlements cut off by road from the nearest hospital or town ({_n(s['access_foot_only'])} still reachable on foot)</span></div>
  <div class="k"><b>{_n(s.get('roads_blocked_km', 0))} km · {_n(s['bridges_at_risk'])}</b><span>of road under water/debris where it blocks the way · bridges over an affected reach (verify)</span></div>
</div>
<div class="cols"><div>
<h2>Settlements cut off by road</h2>
<table><tr><th>Settlement</th><th>Hospital/clinic before</th><th>After</th><th>On foot now</th></tr>{rows or '<tr><td colspan=4>None found in the analysed area.</td></tr>'}</table>
<div class="small">{len(cut)} cut off in total{f', {len(detour)} with long detours' if detour else ''}. Travel times from OpenStreetMap roads (typical speeds), nearest facility within ~25 km.</div>
{f'<div class="small"><b>Possibly cut off — verify ({len(maybe)}):</b> ' + ', '.join(_name(f["properties"]) for f in maybe[:15]) + ('…' if len(maybe) > 15 else '') + ' — still linked only by roads that touch the footprint or run within 20 m of it.</div>' if maybe else ''}
<h2>Roads blocked (longest first)</h2>
<table><tr><th>Road</th><th>Blocked stretches</th><th>Under water/debris</th></tr>{road_rows or '<tr><td colspan=3>No road blocked in the analysed area.</td></tr>'}</table>
<h2>Health facilities</h2><div>In or next to the footprint: {health_line}.</div>
<h2>Data used</h2>
<ul><li>Sentinel-1 radar: {tracks or '—'}</li>
<li>Sentinel-2 optical (clear pixels): before {', '.join(od.get('before', [])) or '—'}; after {', '.join(od.get('after', [])) or '—'}</li>
<li>AI flood model: {html.escape(run.model) if run.model else 'not used'}</li>
<li>Copernicus DEM GLO-30 · OpenStreetMap as of {run.osm_snapshot} (before the event{', ' + html.escape(run.osm_source) if getattr(run, 'osm_source', '') else ''})</li>
{f'<li>What found the affected area (km², kinds overlap): {found}</li>' if found else ''}</ul>
</div><div>{f'<img src="{map_png}" alt="Map of the affected area">' if map_png else ''}
<h2>What this report cannot tell you</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in LIMITS)}</ul></div></div>
<div class="small">{' '.join(html.escape(a) for a in ATTRIBUTION)}</div>
</body></html>"""


def _hillshade(dem, res: float, az: float = 315.0, alt: float = 45.0):
    import numpy as np
    d = np.where(np.isfinite(dem), dem, np.nanmean(dem) if np.isfinite(dem).any() else 0)
    gy, gx = np.gradient(d, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, z = np.radians(360 - az + 90), np.radians(90 - alt)
    hs = np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def map_png(run, path, osm_layers: dict | None = None):
    """Overview map for the report: hillshade, flood/debris footprint, blocked roads, settlements by
    road access (cut-off ones named), health facilities, legend, scale bar and north arrow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform

    g = run.grid
    fwd = g.from_lonlat().transform
    l, b, r, t = g.bounds
    w_in = 4.4
    fig, ax = plt.subplots(figsize=(w_in, w_in * g.height / g.width), dpi=170)
    if run.terrain is not None:
        ax.imshow(_hillshade(run.terrain.dem, g.res), cmap="Greys_r", vmin=-0.3, vmax=1.2, extent=(l, r, b, t), alpha=0.55)
    cls = np.ma.masked_equal(run.cls, 0)
    ax.imshow(cls, cmap=matplotlib.colors.ListedColormap(["#256fd9", "#d67820", "#78aac8"]), vmin=1, vmax=3,
              extent=(l, r, b, t), interpolation="nearest")
    for f in run.damage.stretches if run.damage else []:
        if not f["properties"].get("blocks"):
            continue
        x, y = shp_transform(fwd, shape(f["geometry"])).xy
        ax.plot(x, y, color="#c0392b", lw=1.8, solid_capstyle="round", zorder=3)
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    for f in run.damage.health if run.damage else []:
        p = shp_transform(fwd, shape(f["geometry"])).representative_point()
        if l <= p.x <= r and b <= p.y <= t:
            ax.plot(p.x, p.y, marker="P", ms=6, color="#b3261e", mec="white", mew=0.7, zorder=5)
    colors = {"cut_off": "#c0392b", "possibly_cut_off": "#ef8a6a", "long_detour": "#d99a00", "connected": "#2e7d32"}
    named = []
    for f in run.access.settlements if run.access else []:
        st = f["properties"]["status"]
        if st not in colors:
            continue
        p = shp_transform(fwd, shape(f["geometry"]))
        ax.plot(p.x, p.y, "o", ms=3.8, color=colors[st], mec="white", mew=0.6, zorder=4)
        nm = f["properties"].get("label")
        if st == "cut_off" and nm:
            named.append((f["properties"].get("hospital_min_before") or 999, nm, p.x, p.y))
    for _, nm, x, y in sorted(named)[:14]:      # label the cut-off settlements nearest to care first
        ax.text(x + (r - l) * 0.012, y, nm, fontsize=5.6, color="#7d1d14", va="center", zorder=6, path_effects=halo)
    # scale bar (bottom left) and north arrow
    span = (r - l) * 0.25
    km = max(1, int(round(span / 1000)))
    x0, y0 = l + (r - l) * 0.05, b + (t - b) * 0.035
    ax.plot([x0, x0 + km * 1000], [y0, y0], color="#15191c", lw=2.2, solid_capstyle="butt", zorder=7)
    ax.text(x0 + km * 500, y0 + (t - b) * 0.012, f"{km} km", ha="center", fontsize=6, zorder=7, path_effects=halo)
    ax.annotate("N", xy=(r - (r - l) * 0.06, t - (t - b) * 0.04), xytext=(r - (r - l) * 0.06, t - (t - b) * 0.10),
                ha="center", fontsize=7, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="#15191c", lw=1), zorder=7)
    handles = [Patch(color="#256fd9", label="Flood water"), Patch(color="#d67820", label="Debris / mud"),
               Patch(color="#78aac8", label="Affected river channel"), Line2D([], [], color="#c0392b", lw=1.8, label="Road blocked"),
               Line2D([], [], marker="o", ls="", color="#c0392b", mec="white", ms=4.5, label="Settlement cut off"),
               Line2D([], [], marker="o", ls="", color="#ef8a6a", mec="white", ms=4.5, label="Possibly cut off (verify)"),
               Line2D([], [], marker="o", ls="", color="#2e7d32", mec="white", ms=4.5, label="Still connected"),
               Line2D([], [], marker="P", ls="", color="#b3261e", mec="white", ms=5.5, label="Hospital / clinic")]
    ax.legend(handles=handles, loc="lower right", fontsize=5.4, frameon=True, framealpha=0.92, edgecolor="#d5dade", borderpad=0.6, handlelength=1.4)
    ax.set_xlim(l, r); ax.set_ylim(b, t); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#9aa3a9"); sp.set_linewidth(0.6)
    fig.tight_layout(pad=0.15)
    fig.savefig(path)
    plt.close(fig)
