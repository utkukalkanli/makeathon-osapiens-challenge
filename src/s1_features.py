"""Annual S1 RTC (VV, gamma-0) composites on the AEF reference grid.

S1 adds cloud-penetrating radar evidence that S2 alone cannot provide —
especially valuable in persistently cloudy tropical test tiles (Congo,
Indonesia) where the S2 annual median is noisy.

Per (tile, year) we emit 2 channels, mirroring [src/s2_features.py](src/s2_features.py):

  * VV_ASC  = per-pixel median of VV over ascending passes that year (dB)
  * VV_DESC = per-pixel median of VV over descending passes that year (dB)

Rationale:
  - Only VV is shipped (no VH) — dual-pol features are unavailable.
  - Keeping ASC/DESC separate preserves the view-geometry signal
    (incidence-angle differences change forest backscatter).
  - dB (10*log10) stabilises the long-tailed linear distribution and keeps
    values in a compact range around [-25, +5] dB for forest/non-forest.
  - Per-pixel annual median is robust to speckle without discarding signal.

Missing-month / missing-direction tiles get zero-filled channels; the
network learns to ignore them by magnitude.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .inventory import S1_DIR, Tile
from .labels import load_reference_grid

_S1_NAME_RE = re.compile(
    r"^(?P<tile>.+)_s1_rtc_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<direction>ascending|descending)\.tif$"
)

# Clamp applied before dB conversion so log10 is safe and rare bright
# corner-reflector-like returns don't dominate the median.
_LINEAR_EPS = 1e-4
_LINEAR_CAP = 10.0


def _s1_dir_for(tile: Tile) -> Path:
    return S1_DIR / tile.split / f"{tile.tile_id}__s1_rtc"


def _month_paths(tile: Tile, year: int, direction: str) -> list[Path]:
    d = _s1_dir_for(tile)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob(f"*_s1_rtc_{year}_*_{direction}.tif")):
        m = _S1_NAME_RE.match(p.name)
        if m and int(m["year"]) == year and m["direction"] == direction:
            out.append(p)
    return out


def _read_vv_db(path: Path) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS] | None:
    """Return (vv_dB, transform, crs) with NaN for invalid pixels."""
    with rasterio.open(path) as src:
        vv = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    valid = np.isfinite(vv) & (vv > 0)
    if not valid.any():
        return None
    vv = np.clip(vv, _LINEAR_EPS, _LINEAR_CAP)
    db = 10.0 * np.log10(vv, where=valid, out=np.full_like(vv, np.nan))
    db[~valid] = np.nan
    return db, transform, crs


def _annual_median_native(
    tile: Tile, year: int, direction: str
) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS] | None:
    """Per-pixel annual median VV (dB) in the S1 native UTM grid for one direction."""
    paths = _month_paths(tile, year, direction)
    stack = []
    ref_transform = None
    ref_crs = None
    ref_shape = None
    for p in paths:
        got = _read_vv_db(p)
        if got is None:
            continue
        db, transform, crs = got
        if ref_shape is None:
            ref_transform, ref_crs, ref_shape = transform, crs, db.shape
        elif db.shape != ref_shape:
            # Different month, same tile, same direction → expect same grid.
            # Skip any outlier rather than mis-aligning.
            continue
        stack.append(db)
    if not stack:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.stack(stack, 0), axis=0).astype(np.float32)
    med = np.nan_to_num(med, nan=0.0)  # pixels with no valid pass in the year
    return med, ref_transform, ref_crs


def s1_annual_features(tile: Tile, year: int) -> np.ndarray:
    """Return (2, H_aef, W_aef) float32 array of [VV_asc_dB, VV_desc_dB]
    reprojected to the AEF reference grid (EPSG:4326). Empty directions are
    zero-filled."""
    ref_shape, ref_transform, ref_crs = load_reference_grid(tile.aef_path(year))
    out = np.zeros((2, *ref_shape), dtype=np.float32)
    for idx, direction in enumerate(("ascending", "descending")):
        got = _annual_median_native(tile, year, direction)
        if got is None:
            continue
        src_arr, src_transform, src_crs = got
        reproject(
            source=src_arr,
            destination=out[idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
    return out


if __name__ == "__main__":
    from .inventory import train_tiles

    t = train_tiles()[0]
    print(f"testing on {t.tile_id}")
    for y in t.years:
        f = s1_annual_features(t, y)
        asc, dsc = f[0], f[1]
        print(
            f"  {y}: shape={f.shape}  "
            f"VV_asc mean/std={asc.mean():.2f}/{asc.std():.2f} dB  "
            f"VV_desc mean/std={dsc.mean():.2f}/{dsc.std():.2f} dB  "
            f"asc_nonzero={100*(asc!=0).mean():.0f}%  desc_nonzero={100*(dsc!=0).mean():.0f}%"
        )
