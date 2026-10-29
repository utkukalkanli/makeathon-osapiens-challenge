import numpy as np
from shapely.geometry import box, mapping

from tools.web_data.countries import TILE_COUNTRY
from tools.web_data.pipeline import (
    assign_tile_ids,
    colorize_agreement,
    hectares_by_tile,
    label_stats,
    tile_grid,
)


def _square(x, y):
    return {"type": "Feature", "properties": {}, "geometry": mapping(box(x, y, x + 0.1, y + 0.1))}


def test_colorize_agreement_maps_votes_to_ramp():
    m1 = np.array([[1, 1, 1, 0]], bool); m2 = np.array([[0, 1, 1, 0]], bool); m3 = np.array([[0, 0, 1, 0]], bool)
    rgba = colorize_agreement(m1, m2, m3)
    assert tuple(rgba[0, 0, :3]) == (0xfd, 0xe6, 0x8a)
    assert tuple(rgba[0, 1, :3]) == (0xf9, 0x73, 0x16)
    assert tuple(rgba[0, 2, :3]) == (0xdc, 0x26, 0x26)
    assert rgba[0, 3, 3] == 0 and rgba[0, 0, 3] > 0


def test_label_stats_relative_to_alerted_pixels():
    s = label_stats(np.array([[1, 1, 0, 0]], bool), np.array([[1, 0, 0, 0]], bool), np.array([[1, 0, 0, 0]], bool))
    assert s["alertedPx"] == 2 and s["allAgreePct"] == 50.0 and s["singleSourcePct"] == 50.0 and s["glads2Pct"] == 50.0


def test_label_stats_empty_is_all_zero():
    z = np.zeros((3, 3), bool)
    assert label_stats(z, z, z) == {"alertedPx": 0, "singleSourcePct": 0.0, "allAgreePct": 0.0, "glads2Pct": 0.0, "gladlPct": 0.0, "raddPct": 0.0}


def test_assign_tile_ids_by_containment():
    tiles = {"A": box(0, 0, 1, 1), "B": box(2, 0, 3, 1)}
    fc = {"type": "FeatureCollection", "features": [_square(0.2, 0.2), _square(2.2, 0.2), _square(5, 5)]}
    ids = [f["properties"]["tile_id"] for f in assign_tile_ids(fc, tiles)["features"]]
    assert ids == ["A", "B", None]


def test_hectares_by_tile_one_hectare_square_near_equator():
    # 100 m x 100 m ~ 0.000898 deg at the equator
    d = 100 / 111_320
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"tile_id": "A"}, "geometry": mapping(box(0, 0, d, d))}]}
    assert abs(hectares_by_tile(fc)["A"] - 1.0) < 0.05


def test_tile_grid_matches_bounds():
    shape, tr = tile_grid((10.0, 0.0, 10.09, 0.09), size=1000)
    assert shape == (1000, 1000) and abs(tr.c - 10.0) < 1e-9 and abs(tr.f - 0.09) < 1e-9


def test_country_lookup_covers_all_21_tiles():
    assert len(TILE_COUNTRY) == 21 and TILE_COUNTRY["33NTE_5_1"] == "Cameroon"
