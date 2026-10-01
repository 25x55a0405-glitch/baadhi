# Example output

`trishuli-upper/` is a real run of the deployed system (flood model included) on the upper Bhote Koshi valley,
Nepal, for the 26 August 2026 glacial-lake outburst — the same analysis the dashboard shows for the preset
*Timure–Syabru Bensi*. Pre-event OpenStreetMap as of 24 Aug 2026.

| File | What it is |
|---|---|
| `sitrep.pdf` / `sitrep.html` | the one-page situation report (every number comes from the maps) |
| `stats.json` | the headline numbers and the evidence behind them |
| `classes.tif` | per-pixel result: 0 none · 1 flood water · 2 debris/mud · 3 affected channel (10 m grid) |
| `settlements.geojson` | each settlement: minutes to the nearest hospital/clinic and town before and after, status |
| `roads_blocked.geojson`, `bridges.geojson`, `buildings_affected.geojson` | the damage layers |

Open the GeoJSON files in any GIS (for example QGIS or geojson.io). These files are outputs, never inputs:
Copernicus EMS maps were used only to score results (see `experiments/`).
