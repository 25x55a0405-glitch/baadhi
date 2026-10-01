# Devpost submission — Baadhi (draft)

**Tagline:** Pick a place and a flood date — see where the water and debris went, what they hit, and who is cut off from a hospital. From raw satellite data, in minutes.

## Inspiration

On 26 August 2026 a glacial lake burst above the Bhote Koshi–Trishuli valley in Nepal. In valleys like this one road
washes away and whole villages lose their only route to a clinic. Responders need three answers fast: where did the
flood and debris go, what did it hit, and who is now cut off. Free satellite data (Sentinel-1 radar sees through
cloud) can answer all three — if someone turns it into a map and a list of names quickly.

## What it does

- **Where:** maps flood water and debris/mud from Sentinel-1 radar before/after (every usable orbit track), Sentinel-2
  optical when the sky is clear, a flood segmentation model we trained, and terrain rules from the Copernicus DEM.
- **What:** overlays the map on OpenStreetMap *as it was before the event* — buildings hit, road stretches blocked,
  bridges over an affected reach, health facilities.
- **Who:** routes every settlement to its nearest hospital/clinic and town before and after the flood — cut off, long
  detour, or still connected — and whether it can still be reached on foot.
- **Bonus:** click any point upstream (a glacial lake, a landslide dam) to trace where a flood would go and which
  villages lie in its path.
- A map dashboard (before/after imagery, radar change, evidence layers) and a **one-page situation report** where every
  number comes from the maps.

## How we built it

- Python pipeline reading Sentinel-1 RTC, Sentinel-2 L2A and Copernicus DEM from Microsoft Planetary Computer (no
  account needed) and pre-event OpenStreetMap through the Overpass API's history ("attic") queries.
- Radar change normalised by each pixel's own pass-to-pass variability; layover/shadow masks from the DEM and the
  satellite geometry; terrain rules (height above the river, slope) from our own hydrology on the DEM.
- **AI:** a U-Net (ResNet-18 encoder, 12.5 M parameters) trained **from scratch on a laptop CPU** on the Kuro Siwo
  dataset, with the official event splits. Live Sentinel-1 is converted to look like the training data (γ⁰→σ⁰, Lee
  filter) before the same feature code runs. Exported to ONNX for fast CPU inference.
- Road access with networkx (multi-source Dijkstra before/after cuts), flood path with D8 flow directions.
- FastAPI backend + MapLibre GL dashboard (no build step), situation report as HTML → PDF.

## Results

- Trishuli case vs Copernicus EMSR927 (used only for checking), deployed system: flood/debris map **F1 0.90 and 0.92** on the
  development areas and **0.89 on the held-out area** (precision 0.86, recall 0.93); 94–95 % of the buildings EMS graded as damaged
  lie in or at the edge of our footprint on the development areas (80 % on the held-out area).
- Cut-off settlements: routing with our footprint and routing with EMS's own road-damage grading give the **same status for all
  119 settlements checked** on the development areas (30 + 89) and **92 %** on the held-out area, where we flag 6 more as
  "possibly cut off — verify".
- The flood model on live Sentinel-1 scenes of the 2025 Punjab floods (EMSR838, never used in training): mean flood
  **F1 0.77 vs 0.52** for a radar threshold rule, up to 0.93. On the five Kuro Siwo test events it never saw, pooled F1
  **0.78 vs 0.71** — it wins four and loses one (0.39 vs 0.66, where it calls new flood water "permanent water"); we show that too.
- A new 150–250 km² area runs end to end in 4–5 minutes on a laptop when the public map server answers quickly (measured
  cold on two areas); 1–2½ minutes when cached; dense areas take about 5–8 minutes via the snapshot fallback.

## Challenges we ran into

- **Data rules:** EMS and other published maps may only be used to check results — so everything is built from raw
  Sentinel data, and the OpenStreetMap snapshot is taken two days before the event (the flood began on 25 August UTC).
- **Radar in the Himalaya:** steep slopes hide ground from the satellite; we mask layover and shadow per track and fill
  hidden valley floor from the terrain only where no image saw it unchanged.
- **Farmland looks like damage** to radar after a harvest — optical evidence, where the sky is clear, now outranks it.
- **Training on a CPU:** no GPU, so a compact network, 128-pixel crops and weight averaging; checkpoints that survive
  interruptions.
- **OpenStreetMap history** comes from a free public server that is sometimes overloaded. We fetch every layer in one
  query, rotate over the three official servers, and — if none answers within 2½ minutes — cut the area out of the
  newest dated Geofabrik snapshot from *before* the event; the report says which source and date it used.

## Accomplishments that we're proud of

- The model beats the rule on every live test area and on 4 of 5 unseen test events, runs in seconds on a CPU — and we publish where it fails.
- Honest evaluation: a held-out area, development areas labelled as such, and every limitation written down.

## What we learned

Free satellite data is enough to answer "who is cut off?" within minutes — the hard part is being honest about what a
10 m pixel can and cannot see.

## What's next

- Nepali-language situation reports; alerts when a new Sentinel-1 pass arrives over a watched valley.
- Train on more mountain events; add damage grading from very-high-resolution imagery where it is free.

## Built with

python · fastapi · maplibre-gl · pytorch · onnxruntime · segmentation-models-pytorch · rasterio · shapely · pysheds ·
networkx · planetary-computer · pystac-client · openstreetmap · overpass-api · sentinel-1 · sentinel-2 · copernicus-dem · kuro-siwo
