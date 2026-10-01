"""Where did the flood hit? Flood water and debris/mud from before/after evidence.

Every rule is physical, with fixed thresholds — nothing is fitted to reference damage maps (the
hackathon allows Copernicus EMS maps only for checking results).

Evidence, per pixel:
  Optical (Sentinel-2, only where the ground is seen clearly before AND after)
    stripped    vegetation index NDVI drops by ≥ 0.20
    sediment    bare-soil index rises ≥ 0.05 (or brightness ≥ 0.03) together with an NDVI drop ≥ 0.10
    new water   water index MNDWI turns positive where it was negative
  Radar (Sentinel-1, per track, only where that track sees the ground properly — no layover/shadow)
    changed     |change| ≥ 3 × the pixel's own pass-to-pass variability, in VV or VH
    agreed      two tracks each show ≥ 2 ×
    water       very dark after (VV ≤ −15 dB, VH ≤ −22 dB) and ≥ 3 dB darker than before
  Model         flood-water probability from the Kuro Siwo-trained network (optional)
Terrain gate: water ≤ 30 m and debris ≤ 80 m above the main river, slope ≤ 35° — or up to 60°
on banks within 60 m of the river (debris flows scour and undercut gorge banks).
Clean-up: 3×3 majority vote, drop blobs < 0.3 ha, fill holes < 0.3 ha, keep only zones that touch
the main river corridor (flash floods and debris flows travel down rivers).
Terrain completion: per river reach, fill connected valley floor below the observed flood level
(where cloud or radar shadow hid the ground) — see complete_with_terrain().

Classes: 0 not affected · 1 flood water · 2 debris / mud / scoured ground · 3 river channel in the
affected reach (already water or bare gravel before the event).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

NONE, WATER, DEBRIS, CHANNEL = 0, 1, 2, 3
CLASS_NAMES = {NONE: "not affected", WATER: "flood water", DEBRIS: "debris / mud", CHANNEL: "river channel (affected reach)"}


@dataclass
class Params:
    ndvi_drop: float = 0.20
    ndvi_drop_weak: float = 0.10
    bsi_rise: float = 0.05
    bright_rise: float = 0.03
    z_strong: float = 3.0
    z_agree: float = 2.0
    ml_prob: float = 0.5
    water_vv_db: float = -15.0   # open water is darker than this in VV…
    water_vh_db: float = -22.0   # …and in VH
    water_drop_db: float = 3.0   # and at least this much darker than before the event
    hand_water: float = 30.0
    hand_debris: float = 80.0
    max_slope: float = 35.0
    flat_slope: float = 5.0      # "flat ground" where height above the river is unknown
    bank_hand: float = 60.0      # steep banks this close above the river can be scoured/undercut
    bank_slope: float = 60.0
    min_blob_ha: float = 0.3
    corridor_m: float = 150.0
    channel_hand: float = 6.0
    complete: bool = True
    channel_zone: float = 10.0   # within this height of the river any vegetation loss counts
    bare_ndvi: float = 0.20      # further up, the ground must end (nearly) bare…
    bare_bright: float = 0.08    # …and bright, like fresh sand, gravel or scoured rock


def _clean(mask: np.ndarray, pixel_ha: float, p: Params) -> np.ndarray:
    m = ndi.uniform_filter(mask.astype("float32"), 3) > 0.5               # majority vote: speckle out
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1)) * pixel_ha
        keep = np.concatenate([[False], sizes >= p.min_blob_ha])
        m = keep[lab]
    holes = ndi.binary_fill_holes(m) & ~m
    lab, n = ndi.label(holes)
    if n:
        sizes = ndi.sum(holes, lab, index=np.arange(1, n + 1)) * pixel_ha
        small = np.concatenate([[False], sizes < p.min_blob_ha])
        m |= small[lab]
    return m


def complete_with_terrain(affected: np.ndarray, hand: np.ndarray, slope: np.ndarray, res: float,
                          reach_m: float = 500.0, radius_m: float = 300.0, q: float = 0.75,
                          max_level: float = 40.0, min_px: int = 30) -> np.ndarray:
    """A flood's surface is roughly level across the valley. For each ~500 m reach of the main river,
    estimate how high the detected flood/debris climbs the banks (75th percentile of its height above
    the river, capped at 40 m) and add connected valley-floor ground below that level that the
    satellites could not see (cloud, radar shadow). Returns the added pixels only."""
    river = hand <= 1.0
    if not river.any() or not affected.any():
        return np.zeros_like(affected)
    dist, (ri, rj) = ndi.distance_transform_edt(~river, return_indices=True)
    dist *= res
    block = int(round(reach_m / res))
    reach = (ri // block) * 100000 + (rj // block)            # reach id = block of the nearest river pixel
    near = dist <= radius_m
    sel = affected & near
    if sel.sum() < min_px:
        return np.zeros_like(affected)
    ids, hvals = reach[sel], hand[sel]
    order = np.argsort(ids, kind="stable")
    ids, hvals = ids[order], hvals[order]
    uniq, start, counts = np.unique(ids, return_index=True, return_counts=True)
    level = {}
    for u, s, c in zip(uniq, start, counts):
        if c >= min_px:
            level[u] = min(float(np.quantile(hvals[s:s + c], q)), max_level)
    if not level:
        return np.zeros_like(affected)
    lut_keys = np.fromiter(level.keys(), dtype=np.int64)
    lut_vals = np.fromiter(level.values(), dtype=np.float64)
    idx = np.searchsorted(lut_keys, reach.ravel())
    idx = np.clip(idx, 0, len(lut_keys) - 1)
    lvl = np.where(lut_keys[idx] == reach.ravel(), lut_vals[idx], -1.0).reshape(hand.shape)
    candidate = near & (hand <= lvl) & (slope <= 30) & ~affected
    lab, n = ndi.label(candidate | affected | river, structure=np.ones((3, 3)))
    seeded = np.zeros(n + 1, bool)
    seeded[np.unique(lab[affected])] = True
    seeded[0] = False
    return candidate & seeded[lab]


def detect(terrain: dict, tracks: list[dict], s2pre: dict | None, s2post: dict | None,
           ml_flood_prob: np.ndarray | None = None, res: float = 10.0, p: Params = Params()) -> dict:
    """terrain: hand_major, slope (arrays). tracks: [{z_vv, z_vh, good}]. s2pre/s2post: {ndvi, bsi,
    bright, mndwi} with NaN where not clear. Returns {"cls", "conf", "evidence"}."""
    shape = terrain["hand_major"].shape
    hand = np.nan_to_num(terrain["hand_major"], nan=1e4)
    slope = np.nan_to_num(terrain["slope"], nan=90)
    # Debris flows scour and undercut the banks of a gorge, so steep ground is allowed close to the river
    gate_debris = (hand <= p.hand_debris) & ((slope <= p.max_slope) | ((hand <= p.bank_hand) & (slope <= p.bank_slope)))
    # Height above the river can be unknown (on wide plains the flow never reaches a major river inside
    # the analysis window); water pools on flat ground, so there flatness alone lets water through.
    hand_unknown = ~np.isfinite(terrain["hand_major"])
    gate_water = ((hand <= p.hand_water) | (hand_unknown & (slope <= p.flat_slope))) & (slope <= p.max_slope)

    ev = {}
    # ---- optical
    if s2pre is not None and s2post is not None:
        seen = np.isfinite(s2pre["ndvi"]) & np.isfinite(s2post["ndvi"])
        dndvi = np.nan_to_num(s2post["ndvi"] - s2pre["ndvi"])
        dbsi = np.nan_to_num(s2post["bsi"] - s2pre["bsi"])
        dbright = np.nan_to_num(s2post["bright"] - s2pre["bright"])
        stripped = seen & (dndvi <= -p.ndvi_drop)
        sediment = seen & (dndvi <= -p.ndvi_drop_weak) & ((dbsi >= p.bsi_rise) | (dbright >= p.bright_rise))
        new_water = seen & (np.nan_to_num(s2post["mndwi"], nan=-1) > 0) & (np.nan_to_num(s2pre["mndwi"], nan=1) < 0)
        # Away from the channel, a harvested field also "loses vegetation" (maize harvest is Aug–Sep in
        # the mid-hills). Fresh flood sediment leaves ground fully bare and bright, so require that there.
        fresh = (np.nan_to_num(s2post["ndvi"], nan=1) <= p.bare_ndvi) & (np.nan_to_num(s2post["bright"]) >= p.bare_bright)
        near_channel = hand <= p.channel_zone
        ev["optical_seen"] = seen
        ev["optical_change"] = (stripped | sediment) & (near_channel | fresh)
        ev["optical_water"] = new_water
    else:
        ev["optical_seen"] = np.zeros(shape, bool)
        ev["optical_change"] = ev["optical_water"] = np.zeros(shape, bool)

    # ---- radar (per track, good geometry only)
    strong = np.zeros(shape, bool)
    agree = np.zeros(shape, "int16")
    sar_seen = np.zeros(shape, bool)
    radar_water = np.zeros(shape, bool)
    for t in tracks:
        good = t["good"] & np.isfinite(t["z_vv"])
        zmax = np.maximum(np.abs(np.nan_to_num(t["z_vv"])), np.abs(np.nan_to_num(t["z_vh"])))
        strong |= good & (zmax >= p.z_strong)
        agree += (good & (zmax >= p.z_agree)).astype("int16")
        sar_seen |= good
        if "post_vv" in t:
            # open water mirrors the radar pulse away: very dark after, and much darker than before
            dark = (np.nan_to_num(t["post_vv"], nan=0) <= p.water_vv_db) & (np.nan_to_num(t["post_vh"], nan=0) <= p.water_vh_db)
            drop = (np.nan_to_num(t["d_vv"]) <= -p.water_drop_db) | (np.nan_to_num(t["d_vh"]) <= -p.water_drop_db)
            radar_water |= good & dark & drop
    ev["radar_water"] = radar_water
    # Optical outranks radar where it can see: farmland makes radar change noisy, so where Sentinel-2
    # saw the ground clearly before and after, radar change alone does not count.
    ev["radar_change"] = (strong | (agree >= 2)) & ~(ev["optical_seen"] & ~ev["optical_change"])
    ev["radar_seen"] = sar_seen

    # ---- model
    ev["model_water"] = (np.nan_to_num(ml_flood_prob) >= p.ml_prob) if ml_flood_prob is not None else np.zeros(shape, bool)

    pixel_ha = res * res / 1e4
    water = _clean((ev["model_water"] | ev["optical_water"] | ev["radar_water"]) & gate_water, pixel_ha, p)
    changed = _clean((ev["optical_change"] | ev["radar_change"]) & gate_debris & ~water, pixel_ha, p)

    # keep only zones connected to the main river corridor (flash floods and debris flows run down rivers).
    # Water the trained model found stands on its own: on a plain, flood water breaks into many patches
    # between dry fields, and the model was validated on unseen events — the corridor rule is there to
    # suppress noisy rule-based evidence, not a learned water detector.
    near_river = ndi.distance_transform_edt(hand > 1.0) * res <= p.corridor_m
    lab, n = ndi.label(water | changed)
    if n:
        touches = np.zeros(n + 1, bool)
        touches[np.unique(lab[near_river & (lab > 0)])] = True
        touches[0] = False
        keep = touches[lab]
        wl, wn = ndi.label(water)
        by_model = np.zeros(wn + 1, bool)
        by_model[np.unique(wl[ev["model_water"] & (wl > 0)])] = True
        by_model[0] = False
        water &= keep | by_model[wl]
        changed &= keep

    # terrain completion: valley floor below the observed flood level, hidden from the satellites
    if p.complete:
        # only fill what the satellites could not see — ground seen clearly and unchanged stays out
        added = complete_with_terrain(water | changed, hand, slope, res) & ~(ev["optical_seen"] & ~ev["optical_change"])
        changed |= added
        ev["terrain_completed"] = added

    # river channel within the affected reach: lowest valley floor next to detected change
    affected = water | changed
    reach = ndi.binary_dilation(affected, iterations=int(round(100 / res)))
    channel = reach & (hand <= p.channel_hand) & ~affected

    cls = np.zeros(shape, "uint8")
    cls[changed] = DEBRIS
    cls[water] = WATER
    cls[channel] = CHANNEL

    n_ev = (ev["optical_change"] | ev["optical_water"]).astype("float32") + ev["radar_change"] + ev["model_water"]
    conf = np.clip(0.35 + 0.25 * n_ev, 0, 1).astype("float32")
    conf[cls == NONE] = 0
    conf[cls == CHANNEL] = 0.5
    return {"cls": cls, "conf": conf, "evidence": ev}
