# Baadhi <sub>बाढी</sub> — flood & debris damage from space

**Pick an area and a flood date. Baadhi maps where flood water and debris hit, counts the buildings,
roads and bridges in its path, and finds the settlements cut off from hospitals — from raw satellite
data, in a few minutes, and writes a one-page situation report.**

Built for the IIT Mandi *Multimodal AI Hackathon 2026*, Track B "Mapping Flood Damage from Space".
Case study: the 26 August 2026 glacial-lake outburst on the Bhote Koshi–Trishuli (Nepal), checked
against Copernicus EMS activation **EMSR927**.

![Dashboard: debris footprint, blocked roads, bridges at risk and cut-off settlements along the Bhote Koshi](docs/img/dashboard.png)

> Educational prototype — not an operational tool. Every result should be verified on the ground.

---

## What it answers

| Question | How | Output |
|---|---|---|
| **Where did the flood and debris hit?** | Sentinel-1 radar before/after (every usable orbit track), Sentinel-2 optical before/after when the sky is clear, a flood segmentation model, terrain rules from the Copernicus DEM | map of *flood water*, *debris / mud / scoured ground* and the *affected river channel*, with a confidence layer |
| **What was damaged?** | overlay on OpenStreetMap **as it was before the event** | buildings hit / at the edge, road stretches under water or debris, bridges over an affected reach, health facilities |
| **Who is cut off?** | road routing before vs after, to the nearest hospital/clinic and town (network extends 25 km beyond the area) | each settlement: *cut off* / *long detour* / *connected*, minutes before and after, and whether it is still reachable on foot |
| **Bonus: where would a flood go?** | steepest-descent flow path on the Copernicus DEM from any upstream point | the path and the settlements along it, with distance downstream and height above the river |

All of it lands in a **map dashboard** and a **one-page situation report** (HTML + PDF) where every
number comes from the maps — nothing is typed in by hand.

## Data — and the rules we keep

| Input | Used for | Source (no account needed) |
|---|---|---|
| Sentinel-1 GRD, radiometrically terrain-corrected (γ⁰) | change and water detection, the flood model | Microsoft Planetary Computer `sentinel-1-rtc` |
| Sentinel-2 L2A | optical change (vegetation stripped, fresh sediment, new water) | Planetary Computer `sentinel-2-l2a` |
| Copernicus DEM GLO-30 / GLO-90 | height above river, slopes, radar blind spots, flood path | Planetary Computer `cop-dem-glo-30`, `cop-dem-glo-90` |
| OpenStreetMap **before the event** | buildings, roads, bridges, places, health facilities | Overpass API "attic" query at *event date − 2 days, 00:00 UTC*; if that server is slow, the newest dated Geofabrik regional snapshot *before* the event |
| Kuro Siwo (training only) | training the flood model | Kuro Siwo dataset (CC BY 4.0) |

**Never used as inputs:** Copernicus EMS, UNOSAT or any other published flood/damage map, and OSM
edits made after the event. EMS maps (EMSR927, EMSR838) are used **only to score results**; that code
lives in `experiments/` and nothing in the `baadhi` package imports it. No detector threshold was fitted
to EMS data (see *Honesty notes*).

## How it works

```
area + date ─┬─ Sentinel-1 stacks (per track: up to 3 before, first after) ──┐
             ├─ Sentinel-2 clear-pixel composites before / after ─────────────┤
             ├─ Copernicus DEM → height above river, slope, radar geometry ──┤──► detector ──► flood water / debris / channel
             └─ OSM snapshot before the event ───┐                           │        ▲
                                                 │   flood model (U-Net) ────┘────────┘
                                                 ├──► damage: buildings, roads, bridges, health facilities
                                                 └──► access: routing before/after → cut-off settlements ──► dashboard + one-page report
```

1. **Radar change** — for every orbit track: γ⁰ log-ratio after/before, divided by each pixel's own
   pass-to-pass variability; layover and shadow masked from the DEM and the satellite's viewing geometry.
2. **Optical change** — clear pixels only (scene classification mask): vegetation index drop, bare-soil
   and brightness rise, new water. Where Sentinel-2 saw the ground clearly and unchanged, radar noise
   (e.g. a maize harvest) cannot overrule it.
3. **Flood model** — flood-water probability from the segmentation network (below).
4. **Detector** — physical rules with fixed thresholds: terrain gates (height above the main river,
   slope), clean-up, connection to the river corridor, and filling valley floor below the observed flood
   level where clouds or radar shadow hid the ground.
5. **Damage and access** — footprints on the pre-event OSM; routing on the road graph with the cut
   stretches and at-risk bridges removed; walking paths as a separate "on foot" layer. A settlement that is
   still linked only through roads touching the footprint or running within 20 m of it is flagged
   **possibly cut off — verify** (a 10 m map cannot place the footprint's edge exactly, and bank roads get undercut).

### The AI component: a flood segmentation model

- **Task:** per-pixel *no water / permanent water / flood water* from Sentinel-1 (VV, VH) after and
  before the event plus slope and relative height from the DEM (8 channels).
- **Network:** U-Net with a ResNet-18 encoder, **trained from scratch** on Kuro Siwo only (no ImageNet
  weights, so no data outside the allowed lists), on a laptop CPU.
- **Training data:** Kuro Siwo GRD, official event splits; four training events from different climates
  moved to validation; **official test events never seen**, including the Nepal event (1111007).
- **Live data made to look like the training data:** Planetary Computer gives terrain-flattened γ⁰
  without speckle filtering, Kuro Siwo is σ⁰ with a Lee filter — so live images are converted
  (σ⁰ ≈ γ⁰·cos θ) and Lee-filtered before the same feature code runs (`baadhi/ml/features.py`).
- **Deployment:** exported to ONNX (`models/flood_model.onnx`), run with onnxruntime — identical
  output to PyTorch, 1.6× faster; the dashboard does not need PyTorch.

**Results on data the model never saw** *(numbers for the final model are filled in by `baadhi/ml/evaluate.py` and `experiments/eval_live_water.py`)*

| Test | Plain radar rule (F1) | Model (F1) |
|---|---|---|
| Kuro Siwo test event 1111007, Nepal (Koshi plains) | 0.632 | 0.764 ⁽¹⁾ |
| **Live Sentinel-1**, EMSR838 Punjab floods, AOI01 Rasoo Nagar (143 km² flooded) | 0.631 | **0.945** |
| **Live Sentinel-1**, EMSR838, AOI02 Sharaqpur (65 km² flooded) | 0.313 | **0.906** |

⁽¹⁾ early checkpoint; updated with the final model. The live tests use exactly the Sentinel-1 pass
EMS mapped from and score only inside the EMS area of interest.

## Case study: Trishuli GLOF, 26 Aug 2026 vs Copernicus EMSR927

EMSR927 areas 1–3 were used while developing the detector; **area 5 (Phosretar) was held out** and
evaluated once.

| Area | Flood/debris map: precision · recall · F1 | Buildings EMS graded damaged/destroyed that we flag | Damaged road length inside our footprint | Damaged bridges we flag (of those in OSM) |
|---|---|---|---|---|
| AOI 1–2 Syabru Bensi–Timure (dev) | 0.99 · 0.81 · 0.89 | 95 % | 85 % | 5 / 5 |
| AOI 3 Bidur (dev) | 0.97 · 0.88 · 0.92 | 94 % | 84 % | 20 / 20 |
| **AOI 5 Phosretar (held out)** | **0.97 · 0.77 · 0.86** | **63 %** | **55 %** | **24 / 30** |

In the upper valley the Pasang Lhamu highway is blocked in many places: **22 settlements lose their
road link to the nearest hospital or town** (Syabru Bensi had a clinic 24 minutes away before the event).

**Is the cut-off list right?** The same routing, run with roads cut where *EMS* graded them damaged instead of
by our footprint, gives the same status for **34/34** settlements in the upper valley and **119/119** at Bidur
(development areas), and **167/182 (92 %)** in the held-out Phosretar area — where our list is conservative
(EMS-based routing cuts off 17 settlements, ours 4, 3 in common), because 45 % of the damaged road length there
lies outside our footprint.

## Setup

**Prerequisites:** Python 3.12, about 5 GB free disk for caches, an internet connection (satellite
data and OpenStreetMap are read on demand). Optional: Google Chrome or Microsoft Edge, used to print the
report to PDF. No GPU and no accounts are needed.

```bash
git clone https://github.com/<user>/baadhi.git
cd baadhi
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install numpy scipy "rasterio>=1.4" "shapely>=2.0" pyproj networkx pysheds requests pystac-client planetary-computer scikit-image pillow matplotlib fastapi "uvicorn[standard]" onnxruntime osmium
```

(`pip install -r requirements.txt` also installs PyTorch and friends — only needed to retrain the model.)

## Run

**Dashboard**

```bash
python serve.py
```

Open http://127.0.0.1:8000, then either click a case-study area or *Draw on map*, choose the flood
date and press **Analyse this area**. A 150–200 km² valley takes about 3–5 minutes on a laptop
(most of it downloading imagery; repeat runs are faster). Then use *Before / After / Radar change* to
see the evidence, click settlements for travel times, download the report or the map data, and use
*Where would a flood go?* to trace a flood path from any point.

**Command line**

```bash
python -m baadhi.runner 85.30 28.13 85.40 28.29 2026-08-26 --name "Rasuwa — Bhote Koshi"
```

writes a run folder under `runs/` (GeoTIFFs, GeoJSON layers, `stats.json`, `sitrep.html/pdf`) that the
dashboard also lists.

**Checks and training** (need the full `requirements.txt`)

```bash
python -m baadhi.preflight                         # data sources reachable?
python experiments/eval_detect.py upper            # detector vs EMSR927 (checking only)
python experiments/eval_damage.py phosretar        # buildings / roads / bridges vs EMSR927
python experiments/eval_live_water.py              # flood model on live Sentinel-1 vs EMSR838
python -m baadhi.ml.train --model resnet18 --epochs 30 --crops 3000 --lr 1e-3 --name main
python -m baadhi.ml.evaluate models/runs/main/best.pt      # Kuro Siwo test events
python -m baadhi.ml.infer export models/runs/main/best.pt models/flood_model.onnx
```

## Repository layout

```
baadhi/            the system
  pipeline.py      area + date → maps, damage, access (parallel downloads)
  sar.py optical.py terrain.py detect.py damage.py access.py flowpath.py
  report.py        one-page situation report · render.py map overlays
  server.py        dashboard API · runner.py one complete analysis
  ml/              features, network, Kuro Siwo conversion, training, evaluation, inference (ONNX)
  sources/         Planetary Computer; OpenStreetMap pre-event snapshot (Overpass history, Geofabrik fallback)
web/               dashboard (MapLibre GL, no build step)
experiments/       evaluation against EMS maps (checking only), screenshots
models/            flood_model.onnx + its config
serve.py           start the dashboard
```

## Honesty notes and limitations

- **Revisit gaps.** Sentinel-1 passes every 6–12 days per track; water that drained before the first
  pass after the event is not seen. Nothing here predicts a flood before it happens.
- **Radar blind spots.** Steep slopes facing towards or away from the satellite (layover, shadow) are
  masked; optical images need clear sky, which the monsoon rarely gives.
- **"Hit" is not "destroyed".** At 10 m a building inside the footprint may still stand. EMS grades
  damage from very-high-resolution images; we flag exposure.
- **OpenStreetMap completeness.** Unmapped buildings, roads, bridges and villages are not counted; 9 of
  the 39 bridges EMS graded at Phosretar are not in OSM at all.
- **Travel times** use typical speeds per road class; roads beyond the analysed area are assumed passable.
- **OpenStreetMap history comes from the public Overpass server**, which can be slow at busy hours. All
  map layers are fetched in one query per area. If it has not answered after a minute, the newest dated
  Geofabrik regional snapshot from *before* the event (1st of recent months, 1 January of earlier years) is cut
  out locally in parallel and the first complete source is used — the report says which one and its date. If
  neither is ready after 10 minutes, the flood/debris map is delivered anyway and damage/access are marked as
  not computed; the downloads continue and are cached, so running the same area again completes them.
- **Where it was tuned.** Detector thresholds are physical and fixed; they were checked on EMSR927 areas
  1–3, which therefore count as development areas. Area 5 was evaluated once. On EMSR838 the model alone
  is an independent test; two rules for plains (model water not forced to touch the river corridor;
  unknown height-above-river accepted on flat ground) were added after looking at AOI01, so the
  *combined* detector's EMSR838 numbers are a development check.
- **Kuro Siwo's Nepal event is in the Koshi plains**, not the mountains; mountain performance is shown
  on the EMSR927 case study.

## Attribution

- Contains modified Copernicus Sentinel data 2026 (and 2025 for the EMSR838 tests), processed by
  Microsoft Planetary Computer.
- Copernicus DEM: produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and
  Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- © OpenStreetMap contributors, ODbL — pre-event snapshots via the Overpass API and Geofabrik regional extracts.
- Kuro Siwo: N. I. Bountos, M. Sdraka, A. Zavras, et al., *Kuro Siwo: 33 billion m² under the water. A
  global multi-temporal satellite dataset for rapid flood mapping*, NeurIPS 2024 Datasets and Benchmarks
  (CC BY 4.0).
- Checking only: Copernicus Emergency Management Service © European Union — EMSR927, EMSR838.
- Basemap: © OpenFreeMap, © OpenMapTiles, data © OpenStreetMap contributors.
- Map library: MapLibre GL JS (BSD-3-Clause).
