// Baadhi dashboard — vanilla JS + MapLibre GL. Talks to baadhi/server.py.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 0) => (x == null || Number.isNaN(x) ? "—" : Number(x).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
const MAX_KM2 = 600, MIN_KM2 = 1;

const COLORS = {
  water: "#256fd9", debris: "#d67820", channel: "#78aac8",
  cut_off: "#c0392b", possibly_cut_off: "#ef8a6a", long_detour: "#d99a00", connected: "#2e7d32", no_mapped_road: "#8a8f94",
  hit: "#1f1f1f", possible: "#8b8b8b", blocked: "#c0392b", bridge: "#8e44ad", health: "#b3261e", flow: "#0b5cad",
};
const STATUS_LABEL = { cut_off: "Cut off", possibly_cut_off: "Possibly cut off — verify", long_detour: "Long detour", connected: "Connected", no_mapped_road: "No mapped road" };

// ------------------------------------------------------------------ map
const STYLE_URL = "https://tiles.openfreemap.org/styles/positron";   // free, no API key
const GLYPHS = "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf";
const FALLBACK_STYLE = {
  version: 8, glyphs: GLYPHS,
  sources: { osm: { type: "raster", tileSize: 256, maxzoom: 19, tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' } },
  layers: [{ id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.85, "raster-brightness-min": 0.22, "raster-contrast": -0.15 } }],
};
const map = new maplibregl.Map({
  container: "map", style: STYLE_URL,
  center: [85.25, 28.0], zoom: 9, maxPitch: 0, attributionControl: { compact: true },
});
window.baadhiMap = map;           // handy for debugging from the browser console
window.baadhiState = () => state;
let fellBack = false;
const firstSymbolId = () => (map.getStyle().layers || []).find((l) => l.type === "symbol")?.id;
const FONT_BOLD = ["Noto Sans Bold"], FONT = ["Noto Sans Regular"];
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
map.dragRotate.disable(); map.touchZoomRotate.disableRotation();

// a small white cross on red for health facilities (no font needed)
function crossIcon() {
  const s = 36, c = document.createElement("canvas"); c.width = c.height = s;
  const g = c.getContext("2d");
  g.fillStyle = "#fff"; g.beginPath(); g.arc(s / 2, s / 2, s / 2 - 1, 0, Math.PI * 2); g.fill();
  g.fillStyle = COLORS.health; g.beginPath(); g.arc(s / 2, s / 2, s / 2 - 4, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#fff"; g.fillRect(s / 2 - 3, 9, 6, s - 18); g.fillRect(9, s / 2 - 3, s - 18, 6);
  return g.getImageData(0, 0, s, s);
}
function styleExtras() {             // things every basemap style needs from us
  if (!map.hasImage("cross")) map.addImage("cross", crossIcon(), { pixelRatio: 2 });
  if (!map.getSource("draft")) {
    map.addSource("draft", { type: "geojson", data: state.bbox && !state.run ? boxFeature(state.bbox) : empty() });
    map.addLayer({ id: "draft-fill", type: "fill", source: "draft", paint: { "fill-color": "#15191c", "fill-opacity": 0.06 } });
    map.addLayer({ id: "draft-line", type: "line", source: "draft", paint: { "line-color": "#15191c", "line-width": 1.6, "line-dasharray": [3, 2] } });
  }
}
// The basemap is decided once: the vector style if it is parsed within 8 s, otherwise plain OpenStreetMap
// tiles. Results are drawn only after that (they must not wait for every basemap tile to arrive).
const mapReady = new Promise((resolve) => {
  let done = false;
  const finish = () => { if (done) return; done = true; styleExtras(); resolve(); };
  map.once("style.load", finish);
  setTimeout(() => {
    if (done) return;
    fellBack = true;
    map.once("style.load", finish);
    map.setStyle(FALLBACK_STYLE, { diff: false });
  }, 8000);
});
function empty() { return { type: "FeatureCollection", features: [] }; }
function boxFeature(b) {
  const [w, s, e, n] = b;
  return { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } }] };
}

// ------------------------------------------------------------------ area input
const state = { bbox: null, run: null, poll: null, drawing: false, picking: false, overlays: [], markers: [], flowMarkers: [], view: "map" };

function km2(b) {
  const [w, s, e, n] = b, lat = ((s + n) / 2) * Math.PI / 180;
  return { wkm: Math.abs(e - w) * 111.32 * Math.cos(lat), hkm: Math.abs(n - s) * 110.57 };
}
function parseBbox(txt) {
  const v = txt.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (v.length !== 4 || v.some((x) => !Number.isFinite(x))) return null;
  const [w, s, e, n] = v;
  if (!(w < e && s < n && w >= -180 && e <= 180 && s >= -85 && n <= 85)) return null;
  return v;
}
function setBbox(b, { fit = false, fromInput = false } = {}) {
  state.bbox = b;
  const info = $("#area-size");
  if (!b) { info.textContent = "—"; $("#bbox").classList.toggle("invalid", !!$("#bbox").value.trim()); updateRunBtn(); return; }
  if (!fromInput) $("#bbox").value = b.map((x) => x.toFixed(4)).join(", ");
  const { wkm, hkm } = km2(b), a = wkm * hkm;
  info.textContent = `${fmt(wkm, 1)} × ${fmt(hkm, 1)} km · ${fmt(a)} km²`;
  const bad = a > MAX_KM2 || a < MIN_KM2;
  info.style.color = bad ? "var(--cut)" : "";
  $("#bbox").classList.toggle("invalid", bad);
  // rough time from the practice runs (downloads dominate; a repeat run of the same area is faster)
  const lo = Math.max(2, Math.round((60 + 0.9 * a) / 60)), hi = Math.round((150 + 1.4 * a) / 60);
  $("#form-msg").textContent = a > MAX_KM2 ? `Too large — the limit is ${MAX_KM2} km² (about 25 × 25 km).` : a < MIN_KM2 ? "Too small — draw at least 1 km²."
    : `About ${lo}–${Math.max(hi, lo + 1)} minutes for this area.`;
  mapReady.then(() => map.getSource("draft").setData(boxFeature(b)));
  if (fit) map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 60, duration: 600 });
  updateRunBtn();
}
function updateRunBtn() {
  const b = state.bbox, ok = b && (() => { const { wkm, hkm } = km2(b); const a = wkm * hkm; return a <= MAX_KM2 && a >= MIN_KM2; })();
  $("#run-btn").disabled = !(ok && $("#date").value);
}
$("#bbox").addEventListener("change", (e) => setBbox(parseBbox(e.target.value), { fit: true, fromInput: true }));
$("#date").addEventListener("input", updateRunBtn);
$("#date").max = new Date().toISOString().slice(0, 10);
$("#date").min = "2015-01-01";

// drawing a box: press, drag, release
let drawStart = null;
function setDrawing(on) {
  state.drawing = on;
  $("#draw-btn").setAttribute("aria-pressed", String(on));
  $(".mapwrap").classList.toggle("drawing", on);
  $("#draw-tip").classList.toggle("hidden", !on);
  on ? map.dragPan.disable() : map.dragPan.enable();
  if (on) setPicking(false);
}
$("#draw-btn").addEventListener("click", () => setDrawing(!state.drawing));
map.on("mousedown", (e) => { if (!state.drawing) return; e.preventDefault(); drawStart = e.lngLat; });
map.on("mousemove", (e) => {
  if (!state.drawing || !drawStart) return;
  const b = [Math.min(drawStart.lng, e.lngLat.lng), Math.min(drawStart.lat, e.lngLat.lat), Math.max(drawStart.lng, e.lngLat.lng), Math.max(drawStart.lat, e.lngLat.lat)];
  setBbox(b);
});
map.on("mouseup", (e) => {
  if (!state.drawing || !drawStart) return;
  const b = [Math.min(drawStart.lng, e.lngLat.lng), Math.min(drawStart.lat, e.lngLat.lat), Math.max(drawStart.lng, e.lngLat.lng), Math.max(drawStart.lat, e.lngLat.lat)];
  drawStart = null;
  setDrawing(false);
  setBbox(b[2] - b[0] > 1e-4 && b[3] - b[1] > 1e-4 ? b : null);
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { drawStart = null; setDrawing(false); setPicking(false); } });

// presets
fetch("/api/presets").then((r) => r.json()).then((ps) => {
  const box = $("#presets");
  const groups = [...new Set(ps.map((p) => p.group || ""))];
  for (const g of groups) {
    const label = document.createElement("span");
    label.className = "fine"; label.textContent = g;
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const p of ps.filter((x) => (x.group || "") === g)) {
      const b = document.createElement("button");
      b.className = "chip"; b.type = "button"; b.textContent = p.short || p.name; b.title = `${p.name} · ${p.note}`;
      b.addEventListener("click", () => { $("#date").value = p.date; $("#name").value = p.name; setBbox(p.bbox, { fit: true }); });
      chips.appendChild(b);
    }
    box.append(label, chips);
  }
});

// ------------------------------------------------------------------ running an analysis
$("#run-btn").addEventListener("click", async () => {
  const body = { bbox: state.bbox, date: $("#date").value, name: $("#name").value.trim() || null };
  $("#run-btn").disabled = true;
  $("#form-msg").textContent = "";
  try {
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Could not start");
    watch(j.id);
  } catch (err) {
    $("#form-msg").textContent = err.message;
    updateRunBtn();
  }
});

function watch(id) {
  clearInterval(state.poll);
  $("#progress-card").classList.remove("hidden");
  $("#results").classList.add("hidden");
  $("#steps").innerHTML = "";
  $("#bar").style.width = "2%";
  const t0 = Date.now();
  location.hash = `run=${id}`;
  const tick = async () => {
    $("#elapsed").textContent = `${Math.round((Date.now() - t0) / 1000)} s`;
    let j;
    try { j = await (await fetch(`/api/runs/${id}`)).json(); } catch { return; }
    $("#bar").style.width = `${Math.max(2, (j.frac || 0) * 100)}%`;
    const steps = j.steps || [];
    if (j.status === "queued") $("#steps").innerHTML = `<li><span class="t">—</span><span>Waiting for ${j.ahead || 1} earlier analysis to finish</span></li>`;
    else $("#steps").innerHTML = steps.map((s) => `<li><span class="t">${fmt(s.t, 0)} s</span><span>${esc(s.msg)}</span></li>`).join("");
    if (j.status === "done") { clearInterval(state.poll); $("#progress-card").classList.add("hidden"); updateRunBtn(); showRun(j); loadHistory(); }
    if (j.status === "error") {
      clearInterval(state.poll); updateRunBtn();
      $("#steps").insertAdjacentHTML("beforeend", `<li class="err"><span class="t">!</span><span>Stopped: ${esc(j.error)}</span></li>`);
    }
  };
  tick();
  state.poll = setInterval(tick, 1500);
}

// ------------------------------------------------------------------ results
async function showRun(j) {
  try { await drawRun(j); } catch (err) { console.error("showRun failed", err); window.baadhiLastError = String(err?.stack || err); }
}
async function drawRun(j) {
  state.run = j;
  const s = j.stats, id = j.id;
  $("#results").classList.remove("hidden");
  $("#res-title").textContent = j.name || "Analysis";
  $("#res-sub").textContent = `Flood date ${fmtDate(j.date)} · ${fmt(j.seconds ?? s.seconds)} s · ${bboxText(j.bbox)}`;
  $("#osm-note").textContent = s.osm_note || "";
  $("#osm-note").classList.toggle("hidden", !s.osm_note);

  $("#kpis").innerHTML = [
    kpi(`${fmt(s.affected_km2, 2)}<small>km²</small>`, `flooded or under debris — <span class="dot" style="background:${COLORS.water}"></span>${fmt(s.water_km2, 2)} water, <span class="dot" style="background:${COLORS.debris}"></span>${fmt(s.debris_km2, 2)} debris/mud`),
    kpi(fmt(s.buildings_hit), `buildings inside the footprint, +${fmt(s.buildings_possibly_hit)} at its edge (of ${fmt(s.buildings_total)} mapped)`),
    kpi(fmt(s.access_cut_off), `settlements cut off by road from hospital or town${s.access_possibly_cut_off ? ` · ${fmt(s.access_possibly_cut_off)} more possibly (verify)` : ""}${s.access_long_detour ? ` · ${fmt(s.access_long_detour)} long detours` : ""}`, s.access_cut_off > 0),
    kpi(`${fmt(s.roads_blocked_km ?? 0, 1)}<small>km</small>`, `of road under water/debris where it blocks the way · ${fmt(s.bridges_at_risk)} bridges over an affected reach`),
  ].join("");

  const d = `/api/runs/${id}`;
  $("#dl-html").href = `${d}/sitrep.html`;
  $("#dl-zip").href = `${d}/bundle.zip`;
  $("#dl-pdf").href = `${d}/sitrep.pdf`;   // printed in the background after the run (or on first click)

  const ev = [];
  for (const t of s.tracks || []) ev.push(`Sentinel-1 ${esc(t)}`);
  const od = s.optical_dates || {};
  ev.push(`Sentinel-2 clear pixels — before: ${esc((od.before || []).join(", ") || "none")}; after: ${esc((od.after || []).join(", ") || "none")}`);
  ev.push(`AI flood model: ${esc(s.model || j.model || "not used for this run")}`);
  ev.push(`Copernicus DEM GLO-30 · OpenStreetMap as of ${esc(s.osm_snapshot)} (before the event${s.osm_source ? `, ${esc(s.osm_source)}` : ""})`);
  $("#evidence").innerHTML = ev.map((x) => `<li>${x}</li>`).join("");

  const E = s.evidence_km2;
  const found = E && E.affected ? [["AI flood model (water)", E.model_water, "#2828c8"], ["Radar change", E.radar_change, COLORS.cut_off],
    ["Radar water", E.radar_water, COLORS.water], ["Optical change", E.optical_change, COLORS.debris], ["Optical water", E.optical_water, "#1aa3a3"],
    ["Filled from terrain (hidden ground)", E.terrain_completed, COLORS.no_mapped_road]].filter((i) => i[1] > 0) : [];
  $("#found-block").classList.toggle("hidden", !found.length);
  $("#found").innerHTML = found.map(([label, v, c]) =>
    `<div class="evbar"><span>${label}</span><i><em style="width:${Math.min(100, (100 * v) / E.affected).toFixed(1)}%;background:${c}"></em></i><b>${fmt(v, 2)}</b></div>`).join("");

  await mapReady;
  clearOverlays();
  const [settle, health, bld, blocked, bridges] = await Promise.all(
    ["settlements", "health", "buildings_affected", "roads_blocked", "bridges"].map((n) => fetch(`${d}/geo/${n}.geojson`).then((r) => (r.ok ? r.json() : empty())).catch(empty)));
  addRasterOverlays(j);
  addVectorOverlays({ settle, health, bld, blocked, bridges });
  fillTables(settle, health);
  buildLayerPanel();
  const b = j.bbox;
  const at = location.hash.match(/at=(-?[\d.]+),(-?[\d.]+),([\d.]+)/);   // shared view: #run=…&at=lon,lat,zoom
  if (at) map.jumpTo({ center: [+at[1], +at[2]], zoom: +at[3] });
  else map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: { top: 70, bottom: 40, left: 40, right: 260 }, duration: 700 });
  map.getSource("draft").setData(empty());
}
function kpi(big, text, alert = false) { return `<div class="kpi${alert ? " alert" : ""}"><b>${big}</b><span>${text}</span></div>`; }
function fmtDate(iso) { return new Date(`${iso}T00:00:00`).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" }); }
function bboxText(b) { return b ? `${b[1].toFixed(2)}–${b[3].toFixed(2)}°N, ${b[0].toFixed(2)}–${b[2].toFixed(2)}°E` : ""; }

function placeName(p) { return p.label || p["name:en"] || p.name || null; }
function fillTables(settle, health) {
  const order = { cut_off: 0, possibly_cut_off: 1, long_detour: 2 };
  const rows = settle.features.filter((f) => f.properties.status in order)
    .sort((a, b) => (order[a.properties.status] - order[b.properties.status]) || ((a.properties.hospital_min_before ?? 999) - (b.properties.hospital_min_before ?? 999)));
  const nCut = rows.filter((f) => f.properties.status === "cut_off").length;
  const nMaybe = rows.filter((f) => f.properties.status === "possibly_cut_off").length;
  const nDet = rows.length - nCut - nMaybe;
  $("#cut-count").textContent = rows.length ? [`${nCut} cut off`, nMaybe && `${nMaybe} possibly`, nDet && `${nDet} long detour`].filter(Boolean).join(" · ") : "";
  const tb = $("#cut-table tbody");
  tb.innerHTML = rows.length ? rows.map((f, i) => {
    const p = f.properties, nm = placeName(p);
    const now = p.status === "cut_off" ? `<span class="none">no road</span>`
      : p.status === "possibly_cut_off" ? `<span class="maybe">verify</span> ${fmt(p.hospital_min_after)} min` : `${fmt(p.hospital_min_after)} min`;
    return `<tr data-i="${i}"><td>${nm ? esc(nm) : `<span class="unnamed">unnamed ${esc(p.place || "place")}</span>`}</td>
      <td class="num">${p.hospital_min_before == null ? "—" : `${fmt(p.hospital_min_before)} min`}</td><td class="num">${now}</td>
      <td class="num">${p.hospital_on_foot_min == null ? "no path" : `${fmt(p.hospital_on_foot_min)} min`}</td></tr>`;
  }).join("") : `<tr><td colspan="4">No settlement lost its road link in this area.</td></tr>`;
  tb.querySelectorAll("tr[data-i]").forEach((tr) => tr.addEventListener("click", () => {
    const f = rows[+tr.dataset.i];
    map.flyTo({ center: f.geometry.coordinates, zoom: Math.max(map.getZoom(), 13.5), duration: 700 });
    settlementPopup(f, f.geometry.coordinates);
  }));
  const bad = health.features.filter((f) => f.properties.status !== "outside");
  $("#health-line").textContent = bad.length
    ? `In or next to the footprint: ${bad.map((f) => placeName(f.properties) || "unnamed facility").join(", ")}.`
    : `None of the ${health.features.length} mapped hospitals or clinics in the area is in or next to the footprint.`;
}

// ------------------------------------------------------------------ overlays
const RASTERS = [
  { key: "classes", name: "Flood water · debris · channel", group: "Result", on: true, opacity: 0.85 },
  { key: "confidence", name: "Confidence", group: "Result", on: false, opacity: 0.8 },
  { key: "model_prob", name: "AI model: flood-water probability", group: "Result", on: false, opacity: 0.85 },
  { key: "s2_before", name: "Sentinel-2 before", group: "Imagery", on: false, opacity: 1, view: true },
  { key: "s2_after", name: "Sentinel-2 after", group: "Imagery", on: false, opacity: 1, view: true },
  { key: "radar_change", name: "Radar change", group: "Imagery", on: false, opacity: 1, view: true },
];
function clearOverlays() {
  for (const id of state.overlays) { if (map.getLayer(id)) map.removeLayer(id); }
  for (const id of state.overlays) { if (map.getSource(id)) map.removeSource(id); }
  state.overlays = [];
  for (const m of state.markers) m.remove();
  state.markers = [];
  setView("map", true);
}
function addRasterOverlays(j) {
  const L = j.layers || {};
  // imagery first (lowest), then results — all below the basemap labels
  for (const r of [...RASTERS.filter((x) => x.view), ...RASTERS.filter((x) => !x.view)]) {
    const l = L[r.key];
    if (!l) continue;
    const [w, s, e, n] = l.bounds, id = `r-${r.key}`;
    map.addSource(id, { type: "image", url: `/api/runs/${j.id}/${l.file}`, coordinates: [[w, n], [e, n], [e, s], [w, s]] });
    map.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": r.opacity, "raster-fade-duration": 0, "raster-resampling": r.view ? "linear" : "nearest" },
      layout: { visibility: r.on ? "visible" : "none" } }, firstSymbolId());
    state.overlays.push(id);
  }
  document.querySelectorAll("#view-switch button").forEach((b) => { if (b.dataset.view !== "map") b.disabled = !L[b.dataset.view]; });
}
function addVectorOverlays({ settle, health, bld, blocked, bridges }) {
  const add = (id, src, layer) => {
    if (!map.getSource(id)) map.addSource(id, { type: "geojson", data: src });
    map.addLayer({ id: layer.id || id, source: id, ...layer });
    state.overlays.push(layer.id || id);
    if (!state.overlays.includes(id)) state.overlays.push(id);
  };
  add("v-buildings", bld, { id: "v-buildings", type: "fill", minzoom: 11,
    paint: { "fill-color": ["match", ["get", "status"], "hit", COLORS.hit, COLORS.possible], "fill-opacity": 0.9, "fill-outline-color": "#ffffff" } });
  add("v-buildings", bld, { id: "v-buildings-pt", type: "circle", maxzoom: 11,
    paint: { "circle-radius": 1.6, "circle-color": ["match", ["get", "status"], "hit", COLORS.hit, COLORS.possible] } });
  add("v-blocked", blocked, { id: "v-blocked", type: "line", filter: ["==", ["get", "blocks"], true],
    paint: { "line-color": COLORS.blocked, "line-width": ["interpolate", ["linear"], ["zoom"], 10, 2.5, 15, 5] }, layout: { "line-cap": "round" } });
  add("v-blocked", blocked, { id: "v-touched", type: "line", filter: ["==", ["get", "blocks"], false],
    paint: { "line-color": COLORS.blocked, "line-width": 1.6, "line-dasharray": [1.5, 1.5] } });
  const risky = { type: "FeatureCollection", features: bridges.features.filter((f) => f.properties.status === "at risk") };
  add("v-bridges", risky, { id: "v-bridges", type: "line", filter: ["==", ["geometry-type"], "LineString"],
    paint: { "line-color": COLORS.bridge, "line-width": ["interpolate", ["linear"], ["zoom"], 10, 4, 15, 8] }, layout: { "line-cap": "round" } });
  add("v-bridges", risky, { id: "v-bridges-pt", type: "circle", filter: ["==", ["geometry-type"], "Point"],
    paint: { "circle-radius": 5, "circle-color": COLORS.bridge, "circle-stroke-color": "#fff", "circle-stroke-width": 1.5 } });
  add("v-health", health, { id: "v-health", type: "symbol", layout: { "icon-image": "cross", "icon-size": 1, "icon-allow-overlap": true } });
  const shown = { type: "FeatureCollection", features: settle.features.filter((f) => f.properties.status !== "no_mapped_road") };
  add("v-settle", shown, { id: "v-settle", type: "circle",
    paint: { "circle-radius": ["match", ["get", "place"], ["city", "town"], 7, "village", 5.5, 4.5],
      "circle-color": ["match", ["get", "status"], "cut_off", COLORS.cut_off, "possibly_cut_off", COLORS.possibly_cut_off, "long_detour", COLORS.long_detour, "connected", COLORS.connected, COLORS.no_mapped_road],
      "circle-stroke-color": "#ffffff", "circle-stroke-width": 1.6 } });
  // names of cut-off settlements: MapLibre places what fits and drops overlapping ones (nearest to care first)
  add("v-settle", shown, { id: "v-settle-label", type: "symbol", filter: ["all", ["==", ["get", "status"], "cut_off"], ["has", "label"]],
    layout: { "text-field": ["get", "label"], "text-font": FONT_BOLD, "text-size": 11.5, "text-anchor": "left", "text-offset": [0.7, 0],
      "text-optional": true, "symbol-sort-key": ["coalesce", ["get", "hospital_min_before"], 999] },
    paint: { "text-color": "#7d1d14", "text-halo-color": "#ffffff", "text-halo-width": 1.6 } });
}

// popups
function settlementPopup(f, lngLat) {
  const p = f.properties, c = COLORS[p.status] || COLORS.no_mapped_road;
  const rows = [];
  if (p.hospital_min_before != null) rows.push(`hospital/clinic: ${fmt(p.hospital_min_before)} → ${p.hospital_min_after == null ? "no road" : `${fmt(p.hospital_min_after)} min`}`);
  if (p.town_min_before != null) rows.push(`town: ${fmt(p.town_min_before)} → ${p.town_min_after == null ? "no road" : `${fmt(p.town_min_after)} min`}`);
  if (p.status === "cut_off") rows.push(p.hospital_on_foot_min == null ? "on foot: no mapped path" : `on foot now: ${fmt(p.hospital_on_foot_min)} min`);
  new maplibregl.Popup({ offset: 10, maxWidth: "280px" }).setLngLat(lngLat)
    .setHTML(`<div class="pop-title">${esc(placeName(p) || `Unnamed ${p.place || "place"}`)}</div><span class="pop-status" style="background:${c}">${STATUS_LABEL[p.status] || p.status}</span>${rows.map((r) => `<div class="pop-row">${esc(r)}</div>`).join("")}`)
    .addTo(map);
}
map.on("click", "v-settle", (e) => { if (!state.picking) settlementPopup(e.features[0], e.lngLat); });
map.on("click", "v-health", (e) => {
  if (state.picking) return;
  const p = e.features[0].properties;
  new maplibregl.Popup({ offset: 12 }).setLngLat(e.lngLat).setHTML(`<div class="pop-title">${esc(placeName(p) || "Health facility")}</div><div class="pop-row">${esc(p.amenity || p.healthcare || "")} · ${p.status === "outside" ? "outside the footprint" : esc(p.status)}</div>`).addTo(map);
});
for (const id of ["v-blocked", "v-touched"]) map.on("click", id, (e) => {
  if (state.picking) return;
  const p = e.features[0].properties;
  new maplibregl.Popup({ offset: 8 }).setLngLat(e.lngLat).setHTML(`<div class="pop-title">${esc(p.name || p.highway || "Road")}</div><div class="pop-row">${fmt(p.length_m)} m under water/debris${p.blocks ? " — blocks the road" : ""}</div>`).addTo(map);
});
for (const id of ["v-settle", "v-health", "v-blocked", "v-touched"]) {
  map.on("mouseenter", id, () => { if (!state.drawing && !state.picking) map.getCanvas().style.cursor = "pointer"; });
  map.on("mouseleave", id, () => { if (!state.drawing && !state.picking) map.getCanvas().style.cursor = ""; });
}

// ------------------------------------------------------------------ layer panel + legend + view switch
const VECTORS = [
  { ids: ["v-buildings", "v-buildings-pt"], name: "Buildings hit / at the edge", group: "Damage", sw: COLORS.hit },
  { ids: ["v-blocked", "v-touched"], name: "Road stretches under water/debris", group: "Damage", sw: COLORS.blocked },
  { ids: ["v-bridges", "v-bridges-pt"], name: "Bridges over an affected reach", group: "Damage", sw: COLORS.bridge },
  { ids: ["v-health"], name: "Hospitals and clinics", group: "Access", sw: COLORS.health },
  { ids: ["v-settle"], name: "Settlements by road access", group: "Access", sw: COLORS.cut_off },
];
function buildLayerPanel() {
  const body = $("#layers-body");
  if (!state.run) { body.innerHTML = `<p class="empty">Run or open an analysis to see its layers.</p>`; return; }
  const L = state.run?.layers || {};
  let html = "", group = "";
  const items = [...RASTERS.filter((r) => !r.view && L[r.key]).map((r) => ({ ids: [`r-${r.key}`], name: r.name, group: r.group, sw: r.key === "classes" ? `linear-gradient(90deg, ${COLORS.water} 50%, ${COLORS.debris} 50%)` : r.key === "confidence" ? "linear-gradient(90deg,#fae178,#be3220)" : "linear-gradient(90deg,#96aaff,#2828c8)" })), ...VECTORS];
  items.forEach((it, i) => {
    if (it.group !== group) { html += `<h4>${it.group}</h4>`; group = it.group; }
    const vis = it.ids.some((id) => map.getLayer(id) && map.getLayoutProperty(id, "visibility") !== "none");
    html += `<label><input type="checkbox" data-i="${i}" ${vis ? "checked" : ""}><span class="sw" style="background:${it.sw}"></span>${esc(it.name)}</label>`;
  });
  body.innerHTML = html || `<p class="empty">Run or open an analysis to see its layers.</p>`;
  body.querySelectorAll("input").forEach((cb) => cb.addEventListener("change", () => {
    for (const id of items[+cb.dataset.i].ids) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", cb.checked ? "visible" : "none");
    drawLegend();
  }));
  drawLegend();
}
function visible(id) { return map.getLayer(id) && map.getLayoutProperty(id, "visibility") !== "none"; }
function drawLegend() {
  const rows = [];
  const chip = (c, t) => `<span class="li"><span class="sw" style="background:${c}"></span>${t}</span>`;
  const dot = (c, t) => `<span class="li"><span class="pt" style="background:${c}"></span>${t}</span>`;
  const line = (c, t) => `<span class="li"><span class="ln" style="border-color:${c}"></span>${t}</span>`;
  if (visible("r-classes")) rows.push(chip(COLORS.water, "Flood water") + chip(COLORS.debris, "Debris / mud") + chip(COLORS.channel, "Affected channel"));
  if (visible("r-confidence")) rows.push(`<span class="li wide">Confidence<span class="ramp" style="background:linear-gradient(90deg,#fae178,#be3220)"></span></span>`);
  if (visible("r-model_prob")) rows.push(`<span class="li wide">AI flood probability<span class="ramp" style="background:linear-gradient(90deg,#96aaff,#2828c8)"></span></span>`);
  if (visible("v-buildings") || visible("v-buildings-pt")) rows.push(chip(COLORS.hit, "Building hit") + chip(COLORS.possible, "At the edge"));
  const lines = [];
  if (visible("v-blocked")) lines.push(line(COLORS.blocked, "Road blocked"));
  if (visible("v-bridges")) lines.push(line(COLORS.bridge, "Bridge at risk"));
  if (lines.length) rows.push(lines.join(""));
  if (visible("v-settle")) rows.push(dot(COLORS.cut_off, "Cut off") + dot(COLORS.possibly_cut_off, "Possibly (verify)") + dot(COLORS.long_detour, "Detour") + dot(COLORS.connected, "Connected"));
  if (visible("v-health")) rows.push(`<span class="li"><span class="pt cross"></span>Hospital / clinic</span>`);
  if (map.getLayer("flow-line")) rows.push(line(COLORS.flow, "Traced flood path"));
  if (state.view === "radar_change") rows.push(`<span class="li wide"><b style="color:#c0392b">■</b>&nbsp;darker after (water, mud) · <b style="color:#1aa3a3">■</b>&nbsp;brighter (rough debris)</span>`);
  $("#legend").innerHTML = rows.map((r) => `<div class="row">${r}</div>`).join("");
}

function setView(v, silent = false) {
  state.view = v;
  document.querySelectorAll("#view-switch button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === v)));
  for (const r of RASTERS.filter((x) => x.view)) if (map.getLayer(`r-${r.key}`)) map.setLayoutProperty(`r-${r.key}`, "visibility", r.key === v ? "visible" : "none");
  if (!silent) drawLegend();
}
document.querySelectorAll("#view-switch button").forEach((b) => b.addEventListener("click", () => { if (!b.disabled) setView(b.dataset.view); }));
$("#layers-toggle").addEventListener("click", () => {
  const open = $("#layers-toggle").getAttribute("aria-expanded") !== "true";
  $("#layers-toggle").setAttribute("aria-expanded", String(open));
  $("#layers-body").classList.toggle("hidden", !open);
});

// ------------------------------------------------------------------ flood path (bonus)
function setPicking(on) {
  state.picking = on;
  $("#flow-btn").setAttribute("aria-pressed", String(on));
  map.getCanvas().style.cursor = on ? "crosshair" : "";
  if (on && state.drawing) setDrawing(false);
}
$("#flow-btn").addEventListener("click", () => setPicking(!state.picking));
map.on("click", async (e) => {
  if (!state.picking) return;
  setPicking(false);
  const date = $("#date").value || state.run?.date || new Date().toISOString().slice(0, 10);
  $("#flow-out").classList.remove("hidden");
  $("#flow-sum").textContent = `Tracing from ${e.lngLat.lat.toFixed(4)}°N, ${e.lngLat.lng.toFixed(4)}°E … (about a minute)`;
  $("#flow-table tbody").innerHTML = "";
  $("#flow-btn").disabled = true;
  try {
    const r = await fetch("/api/flowpath", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ lon: e.lngLat.lng, lat: e.lngLat.lat, date }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Tracing failed");
    showFlow(j, e.lngLat);
  } catch (err) {
    $("#flow-sum").textContent = err.message;
  } finally { $("#flow-btn").disabled = false; }
});
function showFlow(j, start) {
  for (const m of state.flowMarkers) m.remove();
  state.flowMarkers = [];
  const line = { type: "Feature", properties: {}, geometry: j.line };
  if (map.getSource("flow")) map.getSource("flow").setData(line);
  else {
    map.addSource("flow", { type: "geojson", data: line });
    map.addLayer({ id: "flow-casing", type: "line", source: "flow", paint: { "line-color": "#ffffff", "line-width": 7, "line-opacity": 0.9 }, layout: { "line-cap": "round", "line-join": "round" } });
    map.addLayer({ id: "flow-line", type: "line", source: "flow", paint: { "line-color": COLORS.flow, "line-width": 3.5, "line-dasharray": [2, 1.2] }, layout: { "line-cap": "round", "line-join": "round" } });
  }
  const startEl = document.createElement("div");
  startEl.style.cssText = `width:14px;height:14px;border-radius:50%;background:${COLORS.flow};border:3px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.3)`;
  state.flowMarkers.push(new maplibregl.Marker({ element: startEl }).setLngLat(start).addTo(map));
  const pts = { type: "FeatureCollection", features: j.settlements.map((x) => ({ type: "Feature", geometry: { type: "Point", coordinates: [x.lon, x.lat] },
    properties: { label: `${x.name} · ${fmt(x.km_downstream, 1)} km`, direct: x.exposure === "directly in the path", km: x.km_downstream } })) };
  if (map.getSource("flow-pts")) map.getSource("flow-pts").setData(pts);
  else {
    map.addSource("flow-pts", { type: "geojson", data: pts });
    map.addLayer({ id: "flow-pts", type: "circle", source: "flow-pts", paint: { "circle-radius": 5,
      "circle-color": ["case", ["get", "direct"], COLORS.cut_off, COLORS.long_detour], "circle-stroke-color": "#fff", "circle-stroke-width": 1.5 } });
    map.addLayer({ id: "flow-labels", type: "symbol", source: "flow-pts", layout: { "text-field": ["get", "label"], "text-font": FONT_BOLD,
      "text-size": 11, "text-anchor": "left", "text-offset": [0.8, 0], "text-optional": true, "symbol-sort-key": ["get", "km"] },
      paint: { "text-color": "#0b3b6e", "text-halo-color": "#fff", "text-halo-width": 1.6 } });
  }
  $("#flow-sum").textContent = `${fmt(j.length_km, 1)} km traced · ${j.settlements.length} settlement${j.settlements.length === 1 ? "" : "s"} along the path · ${fmt(j.seconds)} s`;
  $("#flow-table tbody").innerHTML = j.settlements.length ? j.settlements.map((s) =>
    `<tr><td>${esc(s.name)}<span class="tag-exp">${esc(s.exposure)}</span></td><td class="num">${fmt(s.km_downstream, 1)}</td><td class="num">${fmt(s.height_above_river_m)} m</td></tr>`).join("")
    : `<tr><td colspan="3">No mapped settlement near the path.</td></tr>`;
  const cs = j.line.coordinates;
  if (cs.length > 1) {
    const xs = cs.map((c) => c[0]), ys = cs.map((c) => c[1]);
    map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: 80, duration: 700 });
  }
  drawLegend();
}

// ------------------------------------------------------------------ saved runs
async function loadHistory() {
  let items = [];
  try { items = await (await fetch("/api/runs")).json(); } catch { return; }
  const ul = $("#history");
  ul.innerHTML = items.length ? items.slice(0, 12).map((it) =>
    `<li><button data-id="${esc(it.id)}"><span class="h-name">${esc(it.name)}</span><span class="h-meta">${esc(it.date)} · ${esc(it.created?.replace("T", " ").slice(0, 16))} · ${fmt(it.seconds)} s</span></button></li>`).join("")
    : `<li class="fine">None yet.</li>`;
  ul.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => openRun(b.dataset.id)));
}
async function openRun(id) {
  const r = await fetch(`/api/runs/${id}`);
  if (!r.ok) return;
  const j = await r.json();
  if (!location.hash.includes(`run=${id}`)) location.hash = `run=${id}`;   // keep extras such as &at=…
  if (j.status === "done") showRun(j); else watch(id);
}

loadHistory();
buildLayerPanel();                 // shows its hint until a run is opened
function fromHash() {
  const m = location.hash.match(/run=([\w-]+)/);
  if (m && m[1] !== state.run?.id) openRun(m[1]);
}
window.addEventListener("hashchange", fromHash);
fromHash();
