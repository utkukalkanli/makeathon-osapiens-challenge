"""Tile id -> country lookup for the 21 challenge tiles."""

_GROUPS = {
    "Colombia": "18NXH_6_8 18NWG_6_6 18NWJ_8_9 18NXJ_7_6 18NWH_1_4 18NYH_9_9 19NBD_4_4 18NWM_9_4 18NVJ_1_6 18NYH_2_1",
    "Thailand": "47QMB_0_8 47QMA_6_2",
    "Thailand–Laos border": "47QQV_2_4",
    "Laos": "48QWD_2_2 48PXC_7_7 48QVE_3_0",
    "Vietnam": "48PYB_3_6",
    "Cambodia": "48PWV_7_8 48PUT_0_8 48PWA_0_6",
    "Cameroon": "33NTE_5_1",
}

TILE_COUNTRY = {tile: country for country, tiles in _GROUPS.items() for tile in tiles.split()}
