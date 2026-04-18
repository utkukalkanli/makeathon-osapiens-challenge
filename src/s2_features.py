"""Annual cloud-masked S2 composites on the AEF reference grid.

For each (tile, year) we compute two features:
  * NDVI = (B08 - B04) / (B08 + B04)  — greenness
  * NBR  = (B08 - B12) / (B08 + B12)  — burn / biomass loss

Pipeline per tile/year:
  1. For each monthly L2A file, mask clouds with a simple B02-brightness rule
     (no SCL band is shipped with the challenge data) and nodata (0).
  2. Compute per-pixel median NDVI and NBR across the year's valid months.
  3. Reproject the 2-channel composite from UTM to the AEF tile's EPSG:4326
     grid so it lines up pixel-for-pixel with the AEF embeddings.

Rationale for the simple cloud rule: L2A surface reflectance is stored in
unscaled uint16 (x10000). B02 > 2000 (i.e. blue reflectance > 0.20) catches
bright clouds and cloud shadows miss less often than a single-band rule for
the darker Amazon/Congo canopies we care about. Paired with the annual
median, residual cloudiness washes out — we only need a *representative*
view of the year, not per-date reconstruction.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .inventory import S2_DIR, Tile
from .labels import load_reference_grid

# Band order in the shipped L2A rasters (1-indexed for rasterio):
#   1: B01  2: B02  3: B03  4: B04  5: B05  6: B06  7: B07
#   8: B08  9: B8A 10: B09 11: B11 12: B12
_B02 = 2
_B04 = 4
_B08 = 8
_B12 = 12

CLOUD_B02_THRESHOLD = 2000  # ~0.20 reflectance. Above => likely cloud.

_S2_NAME_RE = re.compile(r"^(?P<tile>.+)_s2_l2a_(?P<year>\d{4})_(?P<month>\d{1,2})\.tif$")


def _s2_dir_for(tile: Tile) -> Path:
    return S2_DIR / tile.split / f"{tile.tile_id}__s2_l2a"


def _monthly_paths(tile: Tile, year: int) -> list[Path]:
    d = _s2_dir_for(tile)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob(f"*_s2_l2a_{year}_*.tif")):
        m = _S2_NAME_RE.match(p.name)
        if m and int(m["year"]) == year:
            out.append(p)
    return out


def _ndvi_nbr_from_monthly(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read B02/B04/B08/B12 from one monthly image, mask clouds+nodata,
    return (ndvi, nbr) with NaN for invalid pixels. Returns None if file
    is unusable (all invalid)."""
    with rasterio.open(path) as src:
        b02 = src.read(_B02).astype(np.float32)
        b04 = src.read(_B04).astype(np.float32)
        b08 = src.read(_B08).astype(np.float32)
        b12 = src.read(_B12).astype(np.float32)
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
    valid = (b02 > 0) & (b04 > 0) & (b08 > 0) & (b12 > 0)
    clear = valid & (b02 < CLOUD_B02_THRESHOLD)
    if not clear.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (b08 - b04) / (b08 + b04)
        nbr = (b08 - b12) / (b08 + b12)
    ndvi[~clear] = np.nan
    nbr[~clear] = np.nan
    return ndvi, nbr, transform, crs, shape  # type: ignore[return-value]


def _annual_median_native(tile: Tile, year: int):
    """Annual per-pixel median NDVI/NBR in the S2 native UTM grid.

    Returns (ndvi_med, nbr_med, transform, crs) or None if no usable months.
    """
    months = _monthly_paths(tile, year)
    ndvi_stack, nbr_stack = [], []
    ref_transform = None
    ref_crs = None
    ref_shape = None
    for p in months:
        got = _ndvi_nbr_from_monthly(p)
        if got is None:
            continue
        ndvi, nbr, transform, crs, shape = got
        if ref_shape is None:
            ref_transform, ref_crs, ref_shape = transform, crs, shape
        elif shape != ref_shape:
            # Monthly files for the same tile should all share a grid —
            # skip any outlier rather than mis-aligning.
            continue
        ndvi_stack.append(ndvi)
        nbr_stack.append(nbr)
    if not ndvi_stack:
        return None
    import warnings
    with warnings.catch_warnings():
        # Pixels cloudy every month produce all-NaN slices — handled below.
        warnings.simplefilter("ignore", RuntimeWarning)
        ndvi_med = np.nanmedian(np.stack(ndvi_stack, 0), axis=0).astype(np.float32)
        nbr_med = np.nanmedian(np.stack(nbr_stack, 0), axis=0).astype(np.float32)
    # Fill remaining NaNs (pixels cloudy all year) with neutral 0.
    ndvi_med = np.nan_to_num(ndvi_med, nan=0.0)
    nbr_med = np.nan_to_num(nbr_med, nan=0.0)
    return ndvi_med, nbr_med, ref_transform, ref_crs


def s2_annual_features(tile: Tile, year: int) -> np.ndarray:
    """Return (2, H_aef, W_aef) float32 array of [NDVI, NBR] reprojected to
    the AEF reference grid (EPSG:4326). If no S2 data is available, returns
    zeros."""
    ref_shape, ref_transform, ref_crs = load_reference_grid(tile.aef_path(year))
    out = np.zeros((2, *ref_shape), dtype=np.float32)
    got = _annual_median_native(tile, year)
    if got is None:
        return out
    ndvi_nat, nbr_nat, src_transform, src_crs = got

    for idx, src_arr in enumerate((ndvi_nat, nbr_nat)):
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
        f = s2_annual_features(t, y)
        finite = f[f != 0]
        print(
            f"  {y}: shape={f.shape}  "
            f"ndvi mean/std={f[0].mean():.3f}/{f[0].std():.3f}  "
            f"nbr mean/std={f[1].mean():.3f}/{f[1].std():.3f}  "
            f"nonzero={100*(f!=0).mean():.1f}%"
        )
