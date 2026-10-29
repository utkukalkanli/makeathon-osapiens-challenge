"""Pure functions turning label rasters and prediction GeoJSON into web assets."""
import copy

import numpy as np
import shapely
from pyproj import Geod
from rasterio.transform import from_bounds
from shapely.geometry import mapping, shape

RAMP = {1: "#fde68a", 2: "#f97316", 3: "#dc2626"}
_GEOD = Geod(ellps="WGS84")


def _rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def tile_grid(bounds, size: int = 1000):
    w, s, e, n = bounds
    width = round(size * (e - w) / (n - s))
    return (size, width), from_bounds(w, s, e, n, width, size)


def colorize_mask(mask: np.ndarray, hex_color: str, alpha: int = 220) -> np.ndarray:
    rgba = np.zeros(mask.shape + (4,), np.uint8)
    rgba[mask.astype(bool)] = (*_rgb(hex_color), alpha)
    return rgba


def _votes(m1, m2, m3) -> np.ndarray:
    return m1.astype(np.uint8) + m2.astype(np.uint8) + m3.astype(np.uint8)


def colorize_agreement(m1, m2, m3) -> np.ndarray:
    votes = _votes(m1, m2, m3)
    rgba = np.zeros(votes.shape + (4,), np.uint8)
    for n, color in RAMP.items():
        rgba[votes == n] = (*_rgb(color), 220)
    return rgba


def label_stats(m1, m2, m3) -> dict:
    votes = _votes(m1, m2, m3)
    alerted = int((votes >= 1).sum())
    if alerted == 0:
        return {"alertedPx": 0, "singleSourcePct": 0.0, "allAgreePct": 0.0,
                "glads2Pct": 0.0, "gladlPct": 0.0, "raddPct": 0.0}

    def pct(count, total):
        return round(100.0 * float(count) / total, 1)

    return {
        "alertedPx": alerted,
        "singleSourcePct": pct((votes == 1).sum(), alerted),
        "allAgreePct": pct((votes == 3).sum(), alerted),
        "glads2Pct": pct(m1.sum(), votes.size),
        "gladlPct": pct(m2.sum(), votes.size),
        "raddPct": pct(m3.sum(), votes.size),
    }


def assign_tile_ids(fc: dict, tiles: dict) -> dict:
    out = copy.deepcopy(fc)
    for feat in out["features"]:
        pt = shape(feat["geometry"]).representative_point()
        feat.setdefault("properties", {})
        feat["properties"]["tile_id"] = next((tid for tid, poly in tiles.items() if poly.contains(pt)), None)
    return out


def hectares_by_tile(fc: dict) -> dict:
    totals: dict = {}
    for feat in fc["features"]:
        tid = (feat.get("properties") or {}).get("tile_id")
        if tid is None:
            continue
        area, _ = _GEOD.geometry_area_perimeter(shape(feat["geometry"]))
        totals[tid] = totals.get(tid, 0.0) + abs(area) / 10_000
    return {k: round(v, 1) for k, v in totals.items()}


def simplify_fc(fc: dict, tolerance: float = 0.00003, precision: int = 6) -> dict:
    out = copy.deepcopy(fc)
    for feat in out["features"]:
        geom = shape(feat["geometry"]).simplify(tolerance, preserve_topology=True)
        geom = shapely.transform(geom, lambda a: np.round(a, precision))
        feat["geometry"] = mapping(geom)
    return out
