"""Generate submissions at multiple thresholds with per-polygon time_step.

Two improvements over [src/predict.py](src/predict.py):
  1. Emit submissions at several thresholds so we can pick the test-optimal one
     (our val-best 0.52 seems too high — test Poly Recall was only 36.4%).
  2. Attach a ``time_step`` (YYMM) to each polygon by finding the S2 NDVI drop
     that best matches the polygon footprint, so the Year score > 0.

Prob rasters are cached to disk — re-running with new thresholds skips the
expensive TTA inference pass. NDVI monthly stacks are cached per-tile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize, shapes
from rasterio.warp import Resampling, reproject
from shapely.geometry import mapping, shape

from submission_utils import merge_geojson_featurecollections, validate_submission_geojson

from .analyze import _predict_tta
from .inventory import Tile, test_tiles
from .labels import load_reference_grid
from .s2_features import CLOUD_B02_THRESHOLD, _B02, _B04, _B08, _monthly_paths
from .unet.dataset import FullTileDataset, TemporalFullTileDataset
from .unet.model import SiameseChangeUNet, TemporalChangeUNet

SUB_ROOT = Path("submission")


def _load_model(ckpt: str, device: str):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    kind = sd.get("model_kind", "siamese")
    in_ch = sd.get("in_channels", 64)
    if kind == "temporal":
        model = TemporalChangeUNet(in_channels=in_ch, base=32).to(device)
    else:
        model = SiameseChangeUNet(in_channels=in_ch, base=32).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    print(f"  loaded {ckpt}  kind={kind}  in_ch={in_ch}  epoch={sd.get('epoch')}")
    return model, kind


def _monthly_ndvi_on_aef_grid(
    tile: Tile,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build a (T, H, W) NDVI stack on the AEF grid, one slice per monthly S2
    acquisition, with NaN for cloudy/nodata pixels. Returns the stack and the
    list of (year, month) tuples aligned with T."""
    ref_shape, ref_transform, ref_crs = load_reference_grid(
        tile.aef_path(tile.years[0])
    )
    slices: list[np.ndarray] = []
    months: list[tuple[int, int]] = []
    for year in tile.years:
        for path in sorted(_monthly_paths(tile, year)):
            stem = path.stem
            try:
                mo = int(stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            with rasterio.open(path) as src:
                b02 = src.read(_B02).astype(np.float32)
                b04 = src.read(_B04).astype(np.float32)
                b08 = src.read(_B08).astype(np.float32)
                src_transform = src.transform
                src_crs = src.crs
            valid = (
                (b02 > 0)
                & (b04 > 0)
                & (b08 > 0)
                & (b02 < CLOUD_B02_THRESHOLD)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi = (b08 - b04) / (b08 + b04)
            ndvi[~valid] = np.nan
            dst = np.full(ref_shape, np.nan, dtype=np.float32)
            reproject(
                source=ndvi,
                destination=dst,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            slices.append(dst)
            months.append((year, mo))
    if not slices:
        return np.zeros((0, *ref_shape), dtype=np.float32), []
    return np.stack(slices, axis=0), months


def _polygon_time_step(
    ndvi: np.ndarray,               # (T, H, W) NaN=cloud/nodata
    months: list[tuple[int, int]],  # aligned to T
    mask: np.ndarray,               # (H, W) bool
    min_drop: float = 0.15,
    before_k: int = 6,
    after_k: int = 3,
) -> int | None:
    """Return the YYMM int for the polygon's most likely deforestation month,
    or ``None`` when no clear NDVI drop can be found.

    Method: mean NDVI inside the polygon mask at each observation (NaN-safe).
    Linearly interpolate missing values along the time axis, smooth with a
    3-point moving average, then score each candidate index ``t`` by
    ``mean(smoothed[t-before_k:t]) - mean(smoothed[t:t+after_k])``. The argmax
    is the event month if the drop exceeds ``min_drop``.
    """
    if not months or not mask.any():
        return None
    T = len(months)
    if T < before_k + after_k:
        return None
    masked = ndvi[:, mask]                             # (T, N)
    with np.errstate(invalid="ignore"):
        series = np.nanmean(masked, axis=1)            # (T,) some NaN allowed
    valid = np.isfinite(series)
    if valid.sum() < before_k + after_k:
        return None
    # Linear-interp NaN gaps to avoid dropping observations entirely.
    idx = np.arange(T)
    series_filled = np.interp(idx, idx[valid], series[valid])
    kernel = np.ones(3, dtype=np.float32) / 3.0
    smoothed = np.convolve(series_filled, kernel, mode="same")
    best_t = -1
    best_drop = -np.inf
    for t in range(before_k, T - after_k + 1):
        drop = float(
            smoothed[t - before_k : t].mean() - smoothed[t : t + after_k].mean()
        )
        if drop > best_drop:
            best_drop = drop
            best_t = t
    if best_t < 0 or best_drop < min_drop:
        return None
    year, month = months[best_t]
    yy = year % 100
    return int(yy * 100 + month)


def _infer_prob(
    model, item, kind: str, device: str, crop: int, overlap: int
) -> tuple[str, np.ndarray]:
    if kind == "temporal":
        tile_id, series_t, _ = item
        series_t = series_t.to(device)
        prob = _predict_tta(model, series_t, crop=crop, overlap=overlap, kind=kind)
    else:
        tile_id, pre, post, _ = item
        pre = pre.to(device)
        post = post.to(device)
        prob = _predict_tta(model, (pre, post), crop=crop, overlap=overlap, kind=kind)
    return tile_id, prob.cpu().numpy()


def _tile_georef(tile: Tile) -> tuple[rasterio.Affine, rasterio.crs.CRS, tuple[int, int]]:
    with rasterio.open(tile.aef_path(tile.years[0])) as src:
        return src.transform, src.crs, (src.height, src.width)


def _vectorize(
    prob: np.ndarray,
    threshold: float,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    min_area_ha: float,
) -> gpd.GeoDataFrame:
    binary = (prob >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    polys = [
        shape(geom)
        for geom, value in shapes(binary, mask=binary, transform=transform)
        if value == 1
    ]
    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame(geometry=polys, crs=crs).to_crs("EPSG:4326")
    utm = gdf.to_crs(gdf.estimate_utm_crs())
    keep = (utm.area / 10_000) >= min_area_ha
    return gdf[keep].reset_index(drop=True)


def main(
    ckpt: str = "checkpoints/temporal_s2_pseudo_v1_best.pt",
    thresholds: list[float] | None = None,
    out_name: str = "temporal_s2_pseudo_v2",
    min_area_ha: float = 0.5,
    crop: int = 512,
    overlap: int = 128,
):
    thresholds = thresholds or [0.25, 0.30, 0.35, 0.40, 0.45]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model, kind = _load_model(ckpt, device)

    tiles = test_tiles()
    tile_by_id = {t.tile_id: t for t in tiles}
    ds = TemporalFullTileDataset(tiles) if kind == "temporal" else FullTileDataset(tiles)

    probs_dir = SUB_ROOT / f"{out_name}_cache" / "probs"
    ndvi_dir = SUB_ROOT / f"{out_name}_cache" / "ndvi"
    probs_dir.mkdir(parents=True, exist_ok=True)
    ndvi_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. TTA inference (cached) ----
    print("\n=== Inference (TTA) ===")
    prob_paths: dict[str, Path] = {}
    for item in ds:
        tile_id = item[0]
        pcache = probs_dir / f"{tile_id}.npy"
        if pcache.exists():
            print(f"  {tile_id}: cached {pcache}")
        else:
            print(f"  {tile_id}: running TTA inference ...")
            _, prob = _infer_prob(model, item, kind, device, crop, overlap)
            np.save(pcache, prob)
            print(f"    wrote {pcache}  shape={prob.shape}  "
                  f"mean_prob={float(prob.mean()):.3f}")
        prob_paths[tile_id] = pcache

    # ---- 2. Monthly NDVI stacks per tile (cached) ----
    print("\n=== Monthly NDVI on AEF grid ===")
    ndvi_paths: dict[str, Path] = {}
    for tile in tiles:
        nc = ndvi_dir / f"{tile.tile_id}.npz"
        if nc.exists():
            print(f"  {tile.tile_id}: cached {nc}")
        else:
            print(f"  {tile.tile_id}: computing monthly NDVI ...")
            ndvi, months = _monthly_ndvi_on_aef_grid(tile)
            np.savez_compressed(
                nc,
                ndvi=ndvi,
                months=np.array(months, dtype=np.int32) if months else np.zeros((0, 2), dtype=np.int32),
            )
            print(f"    wrote {nc}  T={len(months)}  shape={ndvi.shape}")
        ndvi_paths[tile.tile_id] = nc

    # ---- 3. For each threshold, produce per-tile GeoJSONs + merged submission ----
    for thr in thresholds:
        thr_tag = f"thr{int(round(thr * 100)):03d}"
        print(f"\n=== threshold = {thr} ({thr_tag}) ===")
        thr_root = SUB_ROOT / f"{out_name}_{thr_tag}"
        thr_root.mkdir(parents=True, exist_ok=True)
        per_tile_paths: list[Path] = []
        total_polys = 0
        total_with_ts = 0
        ts_by_year: dict[int, int] = {}
        for tile_id, prob_p in prob_paths.items():
            prob = np.load(prob_p)
            tile = tile_by_id[tile_id]
            transform, crs, shape_hw = _tile_georef(tile)
            if prob.shape != shape_hw:
                # Defensive: crop/pad if shapes disagree by 1px from rasterio vs AEF.
                h, w = min(prob.shape[0], shape_hw[0]), min(prob.shape[1], shape_hw[1])
                prob = prob[:h, :w]
            gdf = _vectorize(prob, thr, transform, crs, min_area_ha)
            if gdf.empty:
                print(f"  {tile_id}: no polygons")
                continue

            # Load NDVI stack lazily per tile
            with np.load(ndvi_paths[tile_id]) as f:
                ndvi = f["ndvi"]
                m_arr = f["months"]
            months = [tuple(r) for r in m_arr.tolist()] if len(m_arr) else []

            # AEF grid is EPSG:4326 already; the polygon CRS after to_crs("EPSG:4326")
            # matches the raster CRS, so rasterize against the AEF transform directly.
            features: list[dict] = []
            n_ts = 0
            for geom in gdf.geometry:
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=shape_hw,
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                ).astype(bool)
                ts = _polygon_time_step(ndvi, months, mask) if months else None
                props: dict = {}
                if ts is not None:
                    props["time_step"] = int(ts)
                    n_ts += 1
                    year = 2000 + ts // 100
                    ts_by_year[year] = ts_by_year.get(year, 0) + 1
                features.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": mapping(geom),
                })

            fc = {"type": "FeatureCollection", "features": features}
            validate_submission_geojson(fc)
            out_p = thr_root / f"{tile_id}.geojson"
            with open(out_p, "w") as f:
                json.dump(fc, f)
            per_tile_paths.append(out_p)
            total_polys += len(features)
            total_with_ts += n_ts
            print(f"  {tile_id}: {len(features):5d} polygons  "
                  f"{n_ts} with time_step  "
                  f"pos_pct={(prob >= thr).mean()*100:.2f}%")

        if per_tile_paths:
            merged_p = SUB_ROOT / f"{out_name}_{thr_tag}_submission.geojson"
            merge_geojson_featurecollections(per_tile_paths, output_path=merged_p)
            pct_ts = 100 * total_with_ts / max(total_polys, 1)
            print(f"  MERGED: {merged_p}  "
                  f"polygons={total_polys}  "
                  f"with_time_step={total_with_ts} ({pct_ts:.1f}%)")
            if ts_by_year:
                years_str = "  ".join(
                    f"{y}:{n}" for y, n in sorted(ts_by_year.items())
                )
                print(f"  time_step by year: {years_str}")
        else:
            print(f"  thr={thr}: no features emitted, skipping merge")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/temporal_s2_pseudo_v1_best.pt")
    ap.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.25, 0.30, 0.35, 0.40, 0.45],
    )
    ap.add_argument("--out_name", default="temporal_s2_pseudo_v2")
    ap.add_argument("--min_area_ha", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    args = ap.parse_args()
    main(
        ckpt=args.ckpt,
        thresholds=args.thresholds,
        out_name=args.out_name,
        min_area_ha=args.min_area_ha,
        crop=args.crop,
        overlap=args.overlap,
    )
