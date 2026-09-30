"""Who is cut off? Road access from every settlement to the nearest hospital/clinic and town,
before the flood and after it, on the pre-event OpenStreetMap road network.

Method
  1. Build a graph from OSM roads (pre-event). Travel time per edge from road class; footpaths are a
     separate "on foot" layer — in the hills a village cut off by road may still be reached on foot.
  2. Remove edges whose stretch runs through the flood/debris footprint (≥ 20 m inside), and bridges
     at risk (over an affected reach).
  3. From all destinations at once (multi-source Dijkstra), compute each settlement's travel time to
     the nearest hospital/clinic and to the nearest town — before and after.
  4. Status: cut off (no road route after), long detour (≥ 2× and ≥ +30 min), connected.
Main roads, hospitals and towns up to ~25 km beyond the area are included (osm.fetch_context), so the
nearest hospital can lie outside it; roads outside the analysed area are assumed intact (no flood map
there) — stated as a limitation in the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform as shp_transform

from .grid import Grid

SPEED_KMH = {"motorway": 60, "trunk": 50, "primary": 40, "secondary": 35, "tertiary": 30, "unclassified": 25, "road": 25,
             "residential": 20, "living_street": 15, "service": 15, "track": 12, "motorway_link": 40, "trunk_link": 35,
             "primary_link": 30, "secondary_link": 25, "tertiary_link": 20}
FOOT = {"path", "footway", "steps", "bridleway"}
FOOT_KMH = 3.0
SNAP_M = 300.0   # a settlement/destination further than this from any road is "no mapped road"


@dataclass
class AccessResult:
    settlements: list = field(default_factory=list)   # GeoJSON features with before/after minutes and status
    cut_edges: list = field(default_factory=list)     # GeoJSON of removed road stretches
    stats: dict = field(default_factory=dict)


def _key(x, y):
    return (round(x, 1), round(y, 1))


def build_graph(roads: list, grid: Grid, foot: bool = False) -> nx.Graph:
    """Undirected graph in grid (metre) coordinates. Edges carry minutes and the OSM feature index."""
    fwd = grid.from_lonlat().transform
    G = nx.Graph()
    for idx, f in enumerate(roads):
        hw = f["properties"].get("highway")
        if hw is None or ((hw in FOOT) != foot and not (foot and hw in SPEED_KMH)):
            continue
        g = shp_transform(fwd, shape(f["geometry"]))
        lines = [g] if isinstance(g, LineString) else list(getattr(g, "geoms", []))
        kmh = FOOT_KMH if foot else SPEED_KMH.get(hw, 20)
        for line in lines:
            cs = list(line.coords)
            for a, b in zip(cs[:-1], cs[1:]):
                d = math.dist(a, b)
                if d == 0:
                    continue
                G.add_edge(_key(*a), _key(*b), minutes=d / 1000 / kmh * 60, road=idx, length=d)
    return G


def _snap(G: nx.Graph, pts: list[tuple[float, float]]):
    """Nearest graph node for each point (brute force over a KD-tree)."""
    from scipy.spatial import cKDTree
    nodes = list(G.nodes)
    if not nodes or not pts:
        return [None] * len(pts)
    tree = cKDTree(np.array(nodes))
    d, i = tree.query(np.array(pts))
    return [nodes[j] if dist <= SNAP_M else None for dist, j in zip(d, i)]


def _times(G: nx.Graph, sources: list) -> dict:
    src = [s for s in sources if s is not None and s in G]
    if not src:
        return {}
    return nx.multi_source_dijkstra_path_length(G, src, weight="minutes")


def edges_cut(G: nx.Graph, road_status: list, cut_roads: set[int]) -> list:
    return [(u, v) for u, v, d in G.edges(data=True) if d["road"] in cut_roads]


def display_name(props: dict) -> str | None:
    return props.get("name:en") or props.get("name")


def analyse(osm_layers: dict, road_status: list, bridge_status: list, affected: np.ndarray, grid: Grid,
            context: dict | None = None) -> AccessResult:
    """`osm_layers`: pre-event layers for the analysis area. `context`: main roads, health facilities and
    towns ~25 km around it (osm.fetch_context), so the nearest hospital can lie outside the area.
    Roads outside the analysis area are assumed intact — we have no flood map there."""
    roads = list(osm_layers.get("roads", {}).get("features", []))
    n_area = len(roads)
    fwd = grid.from_lonlat().transform
    # which road features are cut: the damage step's "cut" roads, plus roads that ARE at-risk bridges
    cut_roads = {i for i, f in enumerate(road_status) if f["properties"]["status"] == "cut"}
    risky_bridge_ids = {f["properties"].get("osm_id") for f in bridge_status if f["properties"]["status"] == "at risk"}
    for i, f in enumerate(roads):
        if f["properties"].get("bridge") not in (None, "no") and f["properties"].get("osm_id") in risky_bridge_ids:
            cut_roads.add(i)
    health = list(osm_layers.get("health", {}).get("features", []))
    towns = [f for f in osm_layers.get("places", {}).get("features", []) if f["properties"].get("place") in ("city", "town")]
    if context:
        seen = {f["properties"].get("osm_id") for f in roads}
        roads += [f for f in context.get("context_roads_major", {}).get("features", []) if f["properties"].get("osm_id") not in seen]
        ids = {f["properties"].get("osm_id") for f in health}
        health += [f for f in context.get("context_health", {}).get("features", []) if f["properties"].get("osm_id") not in ids]
        ids = {f["properties"].get("osm_id") for f in towns}
        towns += [f for f in context.get("context_towns", {}).get("features", []) if f["properties"].get("osm_id") not in ids]

    G = build_graph(roads, grid)
    Gf = build_graph(roads, grid, foot=True)            # roads + paths, walked
    Ga, Gfa = G.copy(), Gf.copy()
    for H in (Ga, Gfa):
        H.remove_edges_from([(u, v) for u, v, d in H.edges(data=True) if d["road"] in cut_roads])

    places = osm_layers.get("places", {}).get("features", [])
    pt = lambda f: shp_transform(fwd, shape(f["geometry"])).representative_point().coords[0]  # noqa: E731
    settle_pts = [pt(f) for f in places]
    targets = {"hospital": [pt(f) for f in health], "town": [pt(f) for f in towns]}

    res = AccessResult()
    before, after, after_foot, nearest = {}, {}, {}, {}
    for name, pts in targets.items():
        before[name] = _times(G, _snap(G, pts))
        after[name] = _times(Ga, _snap(Ga, pts))
        after_foot[name] = _times(Gfa, _snap(Gfa, pts))

    snapped = _snap(G, settle_pts)
    snapped_f = _snap(Gfa, settle_pts)
    counts = {"cut_off": 0, "long_detour": 0, "connected": 0, "no_mapped_road": 0, "foot_only": 0}
    for f, node, node_f in zip(places, snapped, snapped_f):
        props = {k: v for k, v in f["properties"].items()}
        props["label"] = display_name(props)
        foot_h = after_foot["hospital"].get(node_f) if node_f is not None else None
        if node is None:
            status = "no_mapped_road"
        else:
            t0h, t1h = before["hospital"].get(node), after["hospital"].get(node)
            t0t, t1t = before["town"].get(node), after["town"].get(node)
            if t0h is None and t0t is None:
                status = "no_mapped_road"
            elif (t0h is not None and t1h is None) or (t0t is not None and t1t is None):
                status = "cut_off"
            else:
                pairs = [(t0, t1) for t0, t1 in ((t0h, t1h), (t0t, t1t)) if t0 is not None]
                worst = max(t1 / max(t0, 1) for t0, t1 in pairs)
                extra = max(t1 - t0 for t0, t1 in pairs)
                status = "long_detour" if (worst >= 2 and extra >= 30) else "connected"
            props.update(hospital_min_before=None if t0h is None else round(t0h), hospital_min_after=None if t1h is None else round(t1h),
                         town_min_before=None if t0t is None else round(t0t), town_min_after=None if t1t is None else round(t1t))
        props.update(status=status, hospital_on_foot_min=None if foot_h is None else round(foot_h))
        if status == "cut_off" and foot_h is not None:
            counts["foot_only"] += 1
        counts[status] += 1
        res.settlements.append({"type": "Feature", "geometry": f["geometry"], "properties": props})

    inv = grid.to_lonlat().transform
    for u, v, d in G.edges(data=True):
        if d["road"] in cut_roads:
            rp = roads[d["road"]]["properties"]
            res.cut_edges.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [list(inv(*u)), list(inv(*v))]},
                                  "properties": {"highway": rp.get("highway"), "name": display_name(rp)}})
    res.stats = {**counts, "settlements": len(places), "health_facilities": len(health), "towns": len(towns),
                 "road_edges": G.number_of_edges(), "road_edges_cut": sum(1 for _, _, d in G.edges(data=True) if d["road"] in cut_roads),
                 "context_km": 25 if context else 0}
    return res
