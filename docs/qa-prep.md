# Q&A preparation — short, honest answers

## Data and rules

**Did you use Copernicus EMS or any published flood map as input?**
No. Inputs are only Sentinel-1, Sentinel-2, Copernicus DEM, OpenStreetMap from before the event, and Kuro Siwo for
training. EMS maps are used only to score our results; that code is in `experiments/` and the system never imports it.

**How do you make sure OpenStreetMap is "before the event"?**
We ask the Overpass API for the map *as it stood* at a moment in time ("attic" query). We use two days before the
flood date at 00:00 UTC, because the Trishuli flood began on 25 August in UTC (22:00), which is 26 August in Nepal.

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
Against the Copernicus EMS delineation of the Trishuli flood: F1 0.89–0.92 on development areas and 0.86 on the area
we held out (precision 0.97, recall 0.77). The misses are the upper flood trace on the banks, which leaves little
visible change at 10 m.

## The AI model

**What is the AI component?**
A flood segmentation network (U-Net, ResNet-18 encoder, 12.5 M parameters) we trained from scratch on the Kuro Siwo
dataset, on this laptop's CPU — no GPU, no ImageNet weights. It labels each pixel no water / permanent water / flood water
from the radar images before and after plus slope and height.

**How do you know it works on new places?**
Kuro Siwo's official test events are never seen in training (including a Nepal event). And on live Sentinel-1 images of
the 2025 Punjab floods, scored against the EMS maps, it reaches a mean F1 of 0.80 against 0.52 for the classic radar
threshold — better in all four test areas.

**Your training data is sigma-nought, your live data is gamma-nought — how do you handle that?**
We convert the live images to look like the training data (σ⁰ ≈ γ⁰ × cos of the incidence angle) and apply the same kind
of speckle filter (Lee), then run exactly the same feature code as in training.

**Why doesn't the model add much in the Trishuli case?**
That flood left mostly debris and scoured ground, not standing water — the model maps water. With the model on, the
Trishuli scores do not change (no false water in the mountains); on plains floods it is the strongest evidence we have.

## Damage and access

**Is a "hit" building destroyed?**
Not necessarily — it lies inside the flood/debris footprint at 10 m. EMS grades damage from very-high-resolution
images; we flag exposure. In the upper valley, 95 % of the buildings EMS graded damaged are in or at the edge of our
footprint.

**How do you decide who is cut off?**
We build the road network from pre-flood OpenStreetMap (including 25 km around the area so the nearest hospital can be
outside it), remove road stretches with ≥ 20 m under water/debris and bridges over an affected reach, and compare the
travel time to the nearest hospital/clinic and town before and after. No route after = cut off; time doubled and +30 min =
long detour. Footpaths are walked separately to tell whether a village can still be reached on foot.

**How do you know the cut-off list is right?**
We ran the same routing with the roads EMS graded damaged: same status for 34/34 settlements in the upper valley, 119/119
at Bidur, and 92 % in the held-out Phosretar area, where our list is conservative (misses some villages behind damaged
bank-side roads).

## Limitations (say these before they ask)

- Satellites pass every 6–12 days per track; water that drained before the first pass is not seen. Nothing is predicted.
- Steep slopes can be hidden from radar; optical needs clear sky.
- Unmapped buildings, roads, bridges and villages (OpenStreetMap gaps) are not counted.
- Travel times use typical speeds; roads beyond the analysed area are assumed open.
- Detector thresholds were checked on development areas; only one EMS area was held out.
- It is an educational prototype — every result should be verified on the ground.

## Speed

**How long does it take?** About 3–5 minutes for a 150–200 km² valley on this laptop (no GPU), most of it downloading.
The model itself runs in seconds per image with ONNX Runtime.
