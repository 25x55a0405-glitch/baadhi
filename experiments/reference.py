"""EMSR927 reference maps — FOR CHECKING ONLY.

Hackathon rule: Copernicus EMS maps may be used to check results, never as an input. This module
lives in experiments/, outside the `baadhi` package, and nothing in `baadhi` imports it.
Credit: European Union, Copernicus Emergency Management Service data.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from shapely import wkt
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

ROOT = Path(__file__).resolve().parents[1] / "data" / "reference" / "EMSR927"
AOI_NAMES = {1: "Syapru Besi", 2: "Timure", 3: "Bidur", 5: "Phosretar"}


def _latest_package(aoi: int) -> Path:
    """Monitoring products supersede the first delineation; take the newest package for the AOI."""
    pk = sorted(ROOT.glob(f"EMSR927_AOI{aoi:02d}_GRA_*"), key=lambda p: ("MONIT" in p.name, p.name))
    pk = [p for p in pk if p.is_dir()]
    return pk[-1]


def aoi_polygon(aoi: int):
    a = json.loads((ROOT.parent / "EMSR927.json").read_text(encoding="utf8"))["results"][0]
    for x in a["aois"]:
        if x["number"] == aoi:
            return wkt.loads(x["extent"])
    raise KeyError(aoi)


def layer(aoi: int, name: str) -> list:
    files = glob.glob(str(_latest_package(aoi) / f"*_{name}_v*.json"))
    feats = []
    for f in files:
        feats += json.loads(Path(f).read_text(encoding="utf8"))["features"]
    return feats


def rasters(grid, aois=(1, 2)):
    """(label, domain) on the grid. label = inside an observed-event polygon; domain = inside the AOIs
    and not marked "not analysed" (only there can we say a pixel is not affected)."""
    fwd = grid.from_lonlat().transform
    g = lambda geom: shp_transform(fwd, geom)  # noqa: E731
    dom_shapes, lab_shapes, na_shapes = [], [], []
    for a in aois:
        dom_shapes.append((g(aoi_polygon(a)), 1))
        lab_shapes += [(g(shape(f["geometry"])), 1) for f in layer(a, "observedEventA")]
        na_shapes += [(g(shape(f["geometry"])), 1) for f in layer(a, "notAnalysedA")]
    kw = dict(out_shape=grid.shape, transform=grid.transform)
    domain = rasterize(dom_shapes, **kw).astype(bool)
    if na_shapes:
        domain &= ~rasterize(na_shapes, **kw).astype(bool)
    label = rasterize(lab_shapes, **kw).astype(bool) if lab_shapes else np.zeros(grid.shape, bool)
    return label & domain, domain


def buildings(aoi: int) -> list[dict]:
    """Damage-graded building points (latest package)."""
    return [{"lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1], "grade": f["properties"].get("damage_gra")}
            for f in layer(aoi, "builtUpP")]
