# Live demo runbook (for the judging session)

Everything below runs on this laptop. It needs the internet (satellite images and maps are downloaded on demand).

## 15 minutes before

1. Plug in the charger and close heavy apps (a browser with many tabs, games, video calls you don't need).
2. Open **PowerShell** and start the dashboard:
   ```
   cd D:\claude-code\baadhi
   .venv\Scripts\python.exe serve.py
   ```
   Wait for `Baadhi dashboard on http://127.0.0.1:8765` (if it says another port, use that one). Leave this window open.
3. Open **Google Chrome** at http://127.0.0.1:8765 (or the address printed above)
4. Warm-up: click **Saved analyses → Rasuwa — Timure to Syabru Bensi**. The map should show orange debris along the
   river, red blocked roads, and red village dots. If it does, everything works.

## The 3-minute story (what to click, what to say)

| Click | Say |
|---|---|
| The saved Rasuwa run | "This is the 26 August Trishuli glacial-lake flood. Baadhi mapped it from raw Sentinel-1 and Sentinel-2 images and the pre-flood OpenStreetMap — no Copernicus maps as input." |
| The four numbers at the top | "4.6 km² under debris, 563 buildings in the footprint, 30 km of road blocked, and 22 villages cut off from the nearest hospital." |
| **Before / After / Radar change** buttons at the top of the map | "Every result can be checked against the evidence. Radar sees through clouds; red means the ground got darker, cyan brighter." |
| Zoom to **Syabru Bensi**, click its red dot | "Before the flood the clinic was 24 minutes away; now there is no road." |
| Layers → **AI model: flood-water probability** | "Our AI component: a flood segmentation network we trained from scratch on the Kuro Siwo dataset, on this laptop." |
| **Where would a flood go? → Pick a start point**, click the river near the top of the valley | "From any upstream point, it traces where a flood would go and which villages lie in its path." |
| **Situation report (PDF)** | "Every analysis ends with a one-page report; every number comes from the maps." |

## When the judges give you an area and a date

1. Type the date in **Flood date**.
2. Click **Draw on map**, then drag a box over the valley or floodplain. Keep it under ~25 × 25 km (the box turns red
   if it is too big). **Smaller is faster** — a new 10 × 15 km valley takes about 3–4 minutes.
3. Press **Analyse this area**. The steps appear on the left while it works — talk the judges through them:
   *downloading radar tracks → terrain → optical images → AI model → flood map → damage → cut-off villages → report*.
4. When the results appear: numbers, then **Before/After**, then the cut-off table, then the PDF.

## If something goes wrong

| What you see | What to do |
|---|---|
| The map background stays blank | Nothing is wrong with the analysis — the basemap server is slow. After 8 seconds the map switches to plain OpenStreetMap by itself. |
| "The area is … km² — the limit is 600 km²" | Draw a smaller box. |
| "OpenStreetMap history server is slow" appears in the steps | The free map server is busy; the system is already preparing a map snapshot from before the event in parallel (it names the source it used). Keep talking through a saved run meanwhile. |
| "Stopped: …" in red | Read the message aloud (it is honest about what failed), then run a slightly different box or a saved case. |
| "No usable satellite image of this area was found after …" | No Sentinel-1/-2 image yet after that date (new images appear 2–8 days after the pass) — pick an earlier date, or explain the revisit gap (it is in the limitations). |
| The page looks stale after an update | Press Ctrl+Shift+R in Chrome. |

## Answers to have ready

See `docs/qa-prep.md`.
