"""Decode and fuse the three weak-label sources into a binary post-2020
deforestation mask, reprojected onto a reference grid (typically the AEF tile).

Sources (see challenge.ipynb §5):
  - GLAD-S2: single alert.tif (conf 0-4) + alertDate.tif (days since 2019-01-01)
  - GLAD-L : per-year alert{YY}.tif (0=no, 2=probable, 3=confirmed) + alertDate{YY}.tif
  - RADD   : single labels.tif; value = conf_digit*10000 + days_since_2014-12-31
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .inventory import LABELS_DIR

# Day offsets for "after 2020-12-31", i.e. events occurring in 2021 or later.
GLADS2_EPOCH_DAYS_2021 = 731  # (2021-01-01) - (2019-01-01) in days
RADD_EPOCH_DAYS_2021 = 2192   # (2021-01-01) - (2014-12-31) in days

# Thresholds: only "medium" confidence and above count as deforestation.
GLADS2_MIN_CONF = 3   # 3=medium, 4=high
GLADL_MIN_CONF = 2    # 2=probable, 3=confirmed
RADD_MIN_CONF = 2     # 2=low, 3=high (both accepted)


def _reproject_to(
    src_path: Path,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    band: int = 1,
    dtype=np.int32,
) -> np.ndarray:
    """Read one band from src_path and reproject it onto the destination grid."""
    out = np.zeros(dst_shape, dtype=dtype)
    with rasterio.open(src_path) as src:
        data = src.read(band).astype(dtype)
        reproject(
            source=data,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return out


def glads2_mask(tile_id: str, dst_shape, dst_transform, dst_crs) -> np.ndarray:
    """Binary post-2020 deforestation mask from GLAD-S2."""
    alert_p = LABELS_DIR / "glads2" / f"glads2_{tile_id}_alert.tif"
    date_p = LABELS_DIR / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    if not alert_p.exists() or not date_p.exists():
        return np.zeros(dst_shape, dtype=bool)
    alert = _reproject_to(alert_p, dst_shape, dst_transform, dst_crs)
    days = _reproject_to(date_p, dst_shape, dst_transform, dst_crs)
    return (alert >= GLADS2_MIN_CONF) & (days > GLADS2_EPOCH_DAYS_2021)


def gladl_mask(
    tile_id: str,
    dst_shape,
    dst_transform,
    dst_crs,
    years=(21, 22, 23, 24, 25),
) -> np.ndarray:
    """Binary post-2020 deforestation mask from GLAD-L (union over years ≥2021)."""
    acc = np.zeros(dst_shape, dtype=bool)
    for yy in years:
        p = LABELS_DIR / "gladl" / f"gladl_{tile_id}_alert{yy}.tif"
        if not p.exists():
            continue
        alert = _reproject_to(p, dst_shape, dst_transform, dst_crs)
        acc |= alert >= GLADL_MIN_CONF
    return acc


def radd_mask(tile_id: str, dst_shape, dst_transform, dst_crs) -> np.ndarray:
    """Binary post-2020 deforestation mask from RADD."""
    p = LABELS_DIR / "radd" / f"radd_{tile_id}_labels.tif"
    if not p.exists():
        return np.zeros(dst_shape, dtype=bool)
    raw = _reproject_to(p, dst_shape, dst_transform, dst_crs)
    conf = raw // 10000
    days = raw % 10000
    return (conf >= RADD_MIN_CONF) & (days > RADD_EPOCH_DAYS_2021)


def fused_labels(
    tile_id: str,
    dst_shape,
    dst_transform,
    dst_crs,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (hard_label, soft_label) as float32 arrays of dst_shape.

    hard_label: majority vote across the 3 sources (>=2 of 3 agree). Robust to
      any single noisy source.
    soft_label: agreement rate in {0, 1/3, 2/3, 1} — useful for soft BCE.
    """
    m1 = glads2_mask(tile_id, dst_shape, dst_transform, dst_crs)
    m2 = gladl_mask(tile_id, dst_shape, dst_transform, dst_crs)
    m3 = radd_mask(tile_id, dst_shape, dst_transform, dst_crs)
    votes = m1.astype(np.uint8) + m2.astype(np.uint8) + m3.astype(np.uint8)
    hard = (votes >= 2).astype(np.float32)
    soft = votes.astype(np.float32) / 3.0
    return hard, soft


def load_reference_grid(aef_path: Path):
    """Return (shape, transform, crs) of the AEF tile to use as the reference grid."""
    with rasterio.open(aef_path) as src:
        return src.shape, src.transform, src.crs


if __name__ == "__main__":
    # Smoke test: fuse labels for one tile and print statistics.
    from .inventory import train_tiles

    t = train_tiles()[0]
    shape, transform, crs = load_reference_grid(t.aef_path(2020))
    print(f"Tile {t.tile_id}: grid {shape} CRS {crs}")
    hard, soft = fused_labels(t.tile_id, shape, transform, crs)
    print(f"  hard positives: {hard.sum():.0f} / {hard.size} ({100*hard.mean():.2f}%)")
    print(f"  soft mean (non-zero): {soft[soft > 0].mean():.3f}")
    print(f"  agreement breakdown: "
          f"0={np.isclose(soft, 0).sum()} "
          f"1/3={np.isclose(soft, 1/3).sum()} "
          f"2/3={np.isclose(soft, 2/3).sum()} "
          f"3/3={np.isclose(soft, 1.0).sum()}")
