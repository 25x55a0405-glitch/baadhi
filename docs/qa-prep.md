# Q&A preparation — short, honest answers

## Data and rules

**Did you use Copernicus EMS or any published flood map as input?**
No. Inputs are only Sentinel-1, Sentinel-2, Copernicus DEM, OpenStreetMap from before the event, and Kuro Siwo for
training. EMS maps are used only to score our results; that code is in `experiments/` and the system never imports it.

**How do you make sure OpenStreetMap is "before the event"?**
We ask the Overpass API for the map *as it stood* at a moment in time ("attic" query). We use two days before the
flood date at 00:00 UTC, because the Trishuli flood began on 25 August in UTC (22:00), which is 26 August in Nepal.
If the public history servers are slow, we fall back to the newest dated Geofabrik regional snapshot from *before* the
event (1st of recent months, 1 January of earlier years), cut out locally — the report names the source and its date.

**Why Planetary Computer and not the Copernicus Data Space?**
Same Sentinel data, no account needed, and it offers radar that is already terrain-corrected (RTC), which the
Himalaya needs. The Copernicus Data Space would work the same way.

## Flood mapping

**How do you find debris, not just water?**
Radar: we compare the image after the flood with up to three images before, from the same orbit, and divide the change
by how much each pixel normally varies between passes — a real change stands out. Optical (when clear): vegetation
stripped, fresh bare sediment. Terrain rules keep only changes near the river and connected to the river corridor.

**Radar in steep valleys?**
Slopes facing towards or away from the satellite are distorted (layover) or hidden (shadow). We compute, per orbit
track, which pixels it can see properly from the DEM and the satellite's viewing angle, and use only those. Using
several tracks (ascending + descending) covers more of the valley.

**What if it's cloudy?**
Radar sees through cloud. Optical evidence is used only where Sentinel-2 had clear pixels; otherwise radar and the
model carry the map.

**How accurate is it?**
Against the Copernicus EMS delineation of the Trishuli flood, with the deployed system (rules + flood model): F1 0.90 and 0.92 on
the development areas and 0.89 on the area we held out (precision 0.86, recall 0.93). Before we added the flood model the held-out
score was 0.86 (precision 0.97, recall 0.77): the model adds recall in the wide, braided reach at some cost in precision.

## The AI model

**What is the AI component?**
A flood segmentation network (U-Net, ResNet-18 encoder, 12.5 M parameters) we trained from scratch on the Kuro Siwo
dataset, on this laptop's CPU — no GPU, no ImageNet weights. It labels each pixel no water / permanent water / flood water
from the radar images before and after plus slope and height.

**How do you know it works on new places?**
Kuro Siwo's official test events are never seen in training (five events in the shards we downloaded, including a Nepal one): pooled
F1 0.78 against 0.71 for the classic radar threshold; it wins four events and loses one. On live Sentinel-1 images of the 2025 Punjab
floods, scored against the EMS maps, the mean F1 is 0.77 against 0.52 — better in all four areas.

**Your training data is sigma-nought, your live data is gamma-nought — how do you handle that?**
We convert the live images to look like the training data (σ⁰ ≈ γ⁰ × cos of the incidence angle) and apply the same kind
of speckle filter (Lee), then run exactly the same feature code as in training.

**Does the model matter in the Trishuli case?**
It does add there: with the model on, the held-out area's recall rises from 0.77 to 0.93 (precision falls from 0.97 to 0.86), and the
model supports 2.9 of the 5.1 km² mapped in the upper valley (the evidence layers overlap). Debris and scoured ground still come
mainly from radar and optical change and the terrain rules; on plains floods the model is the strongest evidence we have.

## Damage and access

**Is a "hit" building destroyed?**
Not necessarily — it lies inside the flood/debris footprint at 10 m. EMS grades damage from very-high-resolution
images; we flag exposure. In the upper valley, 94 % of the buildings EMS graded damaged are in or at the edge of our
footprint (80 % in the held-out area).

**How do you decide who is cut off?**
We build the road network from pre-flood OpenStreetMap (including 25 km around the area so the nearest hospital can be
outside it), remove road stretches with ≥ 20 m under water/debris and bridges over an affected reach, and compare the
travel time to the nearest hospital/clinic and town before and after. No route after = cut off; time doubled and +30 min =
long detour. Footpaths are walked separately to tell whether a village can still be reached on foot.

**How do you know the cut-off list is right?**
We ran the same routing with the roads EMS graded damaged: same status for 30/30 settlements in the upper valley and 89/89 at Bidur,
and 92 % (93/101) in the held-out Phosretar area. There EMS-based routing cuts off 17 settlements; we call 11 cut off and flag the
other 6 as "possibly cut off — verify", and EMS-based routing still finds a road for 2 we call cut off.

## Honesty questions

**Where does the AI fail?**
On Kuro Siwo test event 562 it labels 70 % of the flood pixels as permanent water (the before-images already showed water): F1 0.39
against 0.66 for the plain rule. Where the before-images already show water, the mapped flood extent is a lower bound. It also
struggles in braided rivers (Punjab AOI05 Trimmu: F1 0.39).

**Which checkpoint did you deploy, and was it chosen on the test data?**
The final epoch (30 of 30, averaged weights) — a rule fixed before any evaluation. Validation flood F1 peaked at epoch 2 (0.662) and
stayed flat, while water IoU kept improving; the README shows both checkpoints on every test, and we kept the final
one after seeing both sets of results.

**You looked at Phosretar more than once — is it still held out?**
It was held out of the detector's tuning. We scored the rule-only detector there first and the deployed system later; both numbers are
reported and nothing was chosen from them.

## Limitations (say these before they ask)

- Satellites pass every 6–12 days per track; water that drained before the first pass is not seen. Nothing is predicted.
- Steep slopes can be hidden from radar; optical needs clear sky.
- Unmapped buildings, roads, bridges and villages (OpenStreetMap gaps) are not counted.
- Travel times use typical speeds; roads beyond the analysed area are assumed open.
- Detector thresholds were checked on development areas; only one EMS area was held out.
- It is an educational prototype — every result should be verified on the ground.

## Speed

**How long does it take?** 4–5 minutes for a new 150–250 km² area on this laptop (no GPU; measured cold on Silchar and Melamchi: 250 s and 286 s),
1–2½ minutes when the downloads are cached. Radar and optical downloads take about 2 minutes; the flood model takes
30–80 s per area with ONNX Runtime on the CPU. The free OpenStreetMap history server is the one external dependency:
if it is slow we fall back to a Geofabrik snapshot from before the event, and if both fail we still deliver the flood map
and say that damage and access were not computed.
