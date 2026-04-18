"""Raw S2 (12-band) + S1 (VV) loader on the AEF reference grid (EPSG:4326).

For each (tile, year) we build an annual median composite:
  * S2: 12 reflectance bands (B01..B12 in the order shipped), cloud-masked
    via a simple B02-brightness rule (the shipped L2A has no SCL band).
  * S1: VV backscatter median across all ascending+descending acquisitions.

Everything is reprojected onto the AEF tile grid so labels and pseudo-labels
from the existing pipeline apply unchanged. Output per tile:
    pre  : (13, H, W)  — S2 bands 0..11, S1 VV in channel 12
    post : (13, H, W)
    lat, lon : scalar centroid (for Clay metadata conditioning)

Two-timestep layout (pre=2020, post=2025) mirrors the siamese path. The
temporal extension reads a full year stack when needed (see `series`).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, reproject
from torch.utils.data import Dataset

from ..inventory import S1_DIR, S2_DIR, Tile, train_tiles
from ..labels import fused_labels, load_reference_grid

CACHE_DIR = Path("cache_clay")
CACHE_VERSION = 1

S2_NUM_BANDS = 12
TOTAL_BANDS = S2_NUM_BANDS + 1  # +1 for S1 VV

CLOUD_B02_THRESHOLD = 2000  # matches src/s2_features.py
_B02_IDX = 2  # 1-indexed in rasterio.read

_S2_NAME_RE = re.compile(r"^(?P<tile>.+)_s2_l2a_(?P<year>\d{4})_(?P<month>\d{1,2})\.tif$")


def _s2_dir(tile: Tile) -> Path:
    return S2_DIR / tile.split / f"{tile.tile_id}__s2_l2a"


def _s1_dir(tile: Tile) -> Path:
    return S1_DIR / tile.split / f"{tile.tile_id}__s1_rtc"


def _s2_monthly_paths(tile: Tile, year: int) -> list[Path]:
    d = _s2_dir(tile)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob(f"*_s2_l2a_{year}_*.tif")):
        m = _S2_NAME_RE.match(p.name)
        if m and int(m["year"]) == year:
            out.append(p)
    return out


def _s1_yearly_paths(tile: Tile, year: int) -> list[Path]:
    d = _s1_dir(tile)
    if not d.exists():
        return []
    return sorted(d.glob(f"*_s1_rtc_{year}_*.tif"))


def _s2_annual_median_native(tile: Tile, year: int):
    """Return (stack, transform, crs) with stack=(12, H, W) float32 median
    composite in the S2 native UTM grid, or None if no usable months."""
    months = _s2_monthly_paths(tile, year)
    if not months:
        return None
    per_band_stacks: list[list[np.ndarray]] = [[] for _ in range(S2_NUM_BANDS)]
    ref_transform = ref_crs = ref_shape = None
    for p in months:
        with rasterio.open(p) as src:
            if src.count < S2_NUM_BANDS:
                continue
            if ref_shape is None:
                ref_transform, ref_crs, ref_shape = src.transform, src.crs, src.shape
            elif src.shape != ref_shape:
                continue
            b02 = src.read(_B02_IDX).astype(np.float32)
            valid = b02 > 0
            clear = valid & (b02 < CLOUD_B02_THRESHOLD)
            if not clear.any():
                continue
            for i in range(S2_NUM_BANDS):
                band = src.read(i + 1).astype(np.float32)
                band[~clear] = np.nan
                per_band_stacks[i].append(band)
    if not per_band_stacks[0]:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        meds = [
            np.nanmedian(np.stack(per_band_stacks[i], 0), axis=0).astype(np.float32)
            for i in range(S2_NUM_BANDS)
        ]
    stack = np.stack(meds, axis=0)  # (12, H, W)
    stack = np.nan_to_num(stack, nan=0.0)
    return stack, ref_transform, ref_crs


def _s1_annual_median_native(tile: Tile, year: int):
    """Return (vv, transform, crs) with vv=(H, W) float32 median VV across all
    ascending+descending acquisitions in the year. None if nothing available."""
    paths = _s1_yearly_paths(tile, year)
    if not paths:
        return None
    ref_transform = ref_crs = ref_shape = None
    vv_stack: list[np.ndarray] = []
    for p in paths:
        with rasterio.open(p) as src:
            if src.count < 1:
                continue
            if ref_shape is None:
                ref_transform, ref_crs, ref_shape = src.transform, src.crs, src.shape
            elif src.shape != ref_shape:
                continue
            vv = src.read(1).astype(np.float32)
            vv[vv <= 0] = np.nan
            vv_stack.append(vv)
    if not vv_stack:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vv_med = np.nanmedian(np.stack(vv_stack, 0), axis=0).astype(np.float32)
    vv_med = np.nan_to_num(vv_med, nan=0.0)
    return vv_med, ref_transform, ref_crs


def _reproject_to_ref(
    src_arr: np.ndarray,
    src_transform,
    src_crs,
    dst_shape,
    dst_transform,
    dst_crs,
    resampling=Resampling.bilinear,
) -> np.ndarray:
    """Reproject (C, H, W) or (H, W) array onto dst grid."""
    if src_arr.ndim == 2:
        out = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=src_arr, destination=out,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=resampling,
        )
        return out
    C = src_arr.shape[0]
    out = np.zeros((C, *dst_shape), dtype=np.float32)
    for i in range(C):
        reproject(
            source=src_arr[i], destination=out[i],
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=resampling,
        )
    return out


def _year_stack_on_ref(
    tile: Tile, year: int, ref_shape, ref_transform, ref_crs
) -> np.ndarray:
    """Build (13, H, W) stack on the AEF grid: 12 S2 bands + 1 S1 VV."""
    out = np.zeros((TOTAL_BANDS, *ref_shape), dtype=np.float32)
    got_s2 = _s2_annual_median_native(tile, year)
    if got_s2 is not None:
        s2_stack, s2_t, s2_crs = got_s2
        out[:S2_NUM_BANDS] = _reproject_to_ref(
            s2_stack, s2_t, s2_crs, ref_shape, ref_transform, ref_crs,
            resampling=Resampling.bilinear,
        )
    got_s1 = _s1_annual_median_native(tile, year)
    if got_s1 is not None:
        vv, s1_t, s1_crs = got_s1
        out[S2_NUM_BANDS] = _reproject_to_ref(
            vv, s1_t, s1_crs, ref_shape, ref_transform, ref_crs,
            resampling=Resampling.bilinear,
        )
    return out


def build_tile_tensors_clay(
    tile: Tile,
    pre_year: int = 2020,
    post_year: int = 2025,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Load and cache (pre, post, lat, lon, hard, soft) per tile.

    Keys:
      - pre, post : (13, H, W) float32
      - lat, lon  : () float32 centroid in WGS84
      - hard, soft: (H, W) float32  (train only)
      - version   : cache schema marker
    """
    cache_p = CACHE_DIR / tile.split / f"{tile.tile_id}.npz"
    if cache and cache_p.exists():
        with np.load(cache_p) as f:
            keys = set(f.files)
            version = int(f["version"]) if "version" in keys else 0
            if version >= CACHE_VERSION and {"pre", "post", "lat", "lon"} <= keys:
                return {k: f[k] for k in f.files if k != "version"}

    ref_shape, ref_transform, ref_crs = load_reference_grid(tile.aef_path(pre_year))
    pre = _year_stack_on_ref(tile, pre_year, ref_shape, ref_transform, ref_crs)
    post = _year_stack_on_ref(tile, post_year, ref_shape, ref_transform, ref_crs)

    with rasterio.open(tile.aef_path(pre_year)) as src:
        b = src.bounds
    lat = np.float32((b.bottom + b.top) / 2)
    lon = np.float32((b.left + b.right) / 2)

    out: dict[str, np.ndarray] = {"pre": pre, "post": post, "lat": lat, "lon": lon}
    if tile.split == "train":
        hard, soft = fused_labels(tile.tile_id, ref_shape, ref_transform, ref_crs)
        out["hard"] = hard
        out["soft"] = soft

    if cache:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_p, version=np.int32(CACHE_VERSION), **out)
    return out


class ClayCropDataset(Dataset):
    """Random crops of (pre, post) on the AEF grid + fused label.

    Yields (pre, post, target, weight, meta) where meta is a dict with
    lat/lon scalars — used by the Clay encoder's metadata conditioning.
    """

    def __init__(
        self,
        tiles: list[Tile],
        crop_size: int = 256,
        crops_per_tile: int = 16,
        positive_crop_prob: float = 0.75,
        augment: bool = True,
        pre_year: int = 2020,
        post_year: int = 2025,
        use_soft_target: bool = False,
    ):
        self.tiles = tiles
        self.crop = crop_size
        self.crops_per_tile = crops_per_tile
        self.positive_crop_prob = positive_crop_prob
        self.augment = augment
        self.use_soft_target = use_soft_target
        self._data = [
            build_tile_tensors_clay(t, pre_year=pre_year, post_year=post_year)
            for t in tiles
        ]

    def __len__(self) -> int:
        return len(self.tiles) * self.crops_per_tile

    def _sample_crop(self, d, rng):
        _, h, w = d["pre"].shape
        cs = self.crop
        if (
            "hard" in d
            and rng.random() < self.positive_crop_prob
            and d["hard"].sum() > 0
        ):
            ys, xs = np.where(d["hard"] > 0)
            idx = rng.integers(0, len(ys))
            cy, cx = int(ys[idx]), int(xs[idx])
            y = int(np.clip(cy - cs // 2, 0, h - cs))
            x = int(np.clip(cx - cs // 2, 0, w - cs))
        else:
            y = int(rng.integers(0, h - cs + 1))
            x = int(rng.integers(0, w - cs + 1))
        return y, x

    def __getitem__(self, idx):
        tile_idx = idx % len(self.tiles)
        d = self._data[tile_idx]
        rng = np.random.default_rng()
        y, x = self._sample_crop(d, rng)
        cs = self.crop

        pre = d["pre"][:, y : y + cs, x : x + cs]
        post = d["post"][:, y : y + cs, x : x + cs]
        if "hard" in d:
            hard = d["hard"][y : y + cs, x : x + cs]
            soft = d["soft"][y : y + cs, x : x + cs]
        else:
            hard = np.zeros((cs, cs), dtype=np.float32)
            soft = hard

        if self.augment:
            k = int(rng.integers(0, 4))
            pre = np.rot90(pre, k=k, axes=(1, 2)).copy()
            post = np.rot90(post, k=k, axes=(1, 2)).copy()
            hard = np.rot90(hard, k=k, axes=(0, 1)).copy()
            soft = np.rot90(soft, k=k, axes=(0, 1)).copy()
            if rng.random() < 0.5:
                pre = pre[:, :, ::-1].copy()
                post = post[:, :, ::-1].copy()
                hard = hard[:, ::-1].copy()
                soft = soft[:, ::-1].copy()

        target = soft if self.use_soft_target else hard
        weight = np.ones_like(target, dtype=np.float32)
        return (
            torch.from_numpy(pre).float(),
            torch.from_numpy(post).float(),
            torch.from_numpy(target).float().unsqueeze(0),
            torch.from_numpy(weight).float().unsqueeze(0),
            torch.tensor([float(d["lat"]), float(d["lon"])], dtype=torch.float32),
        )


class ClayFullTileDataset(Dataset):
    """Deterministic full-tile loader for validation / inference."""

    def __init__(
        self,
        tiles: list[Tile],
        pre_year: int = 2020,
        post_year: int = 2025,
    ):
        self.tiles = tiles
        self._data = [
            build_tile_tensors_clay(t, pre_year=pre_year, post_year=post_year)
            for t in tiles
        ]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        t = self.tiles[idx]
        d = self._data[idx]
        pre = torch.from_numpy(d["pre"]).float()
        post = torch.from_numpy(d["post"]).float()
        hard = torch.from_numpy(
            d.get("hard", np.zeros(d["pre"].shape[-2:], dtype=np.float32))
        ).float()
        meta = torch.tensor([float(d["lat"]), float(d["lon"])], dtype=torch.float32)
        return t.tile_id, pre, post, hard, meta


def s2_coverage(tile: Tile, pre_year: int = 2020, post_year: int = 2025) -> float:
    """Fraction of pixels in the pre-year S2 stack that are non-zero.

    Used to skip tiles where every month got cloud-filtered — Clay can't
    learn anything useful from an all-zero S2 input. S1 coverage is
    irrelevant here because the Clay branch is S2-only.
    """
    d = build_tile_tensors_clay(tile, pre_year=pre_year, post_year=post_year)
    return float((d["pre"][:S2_NUM_BANDS] != 0).mean())


def filter_by_s2_coverage(
    tiles: list[Tile], min_coverage: float = 0.10, verbose: bool = True
) -> tuple[list[Tile], list[str]]:
    """Return (kept_tiles, skipped_tile_ids) based on S2 coverage threshold."""
    kept, skipped = [], []
    for t in tiles:
        cov = s2_coverage(t)
        if cov < min_coverage:
            skipped.append(t.tile_id)
            if verbose:
                print(f"  skip {t.tile_id}: S2 coverage {cov*100:.1f}% < {min_coverage*100:.0f}%")
        else:
            kept.append(t)
    return kept, skipped


def _load_pseudo(pseudo_dir: Path, tile_id: str) -> dict[str, np.ndarray]:
    p = Path(pseudo_dir) / f"{tile_id}.npz"
    with np.load(p) as f:
        return {"hard": f["hard"].astype(np.float32),
                "weight": f["weight"].astype(np.float32)}


class ClayPseudoCropDataset(Dataset):
    """Siamese crops from test tiles using ensemble pseudo-labels.

    Yields (pre, post, target, weight, meta) — weight=0 on ignored pixels.
    """

    def __init__(
        self,
        tiles: list[Tile],
        pseudo_dir: str | Path,
        crop_size: int = 256,
        crops_per_tile: int = 16,
        positive_crop_prob: float = 0.75,
        augment: bool = True,
        pre_year: int = 2020,
        post_year: int = 2025,
    ):
        self.tiles = tiles
        self.crop = crop_size
        self.crops_per_tile = crops_per_tile
        self.positive_crop_prob = positive_crop_prob
        self.augment = augment
        self._data = []
        for t in tiles:
            d = dict(build_tile_tensors_clay(t, pre_year=pre_year, post_year=post_year))
            d.update(_load_pseudo(pseudo_dir, t.tile_id))
            self._data.append(d)

    def __len__(self) -> int:
        return len(self.tiles) * self.crops_per_tile

    def _sample_crop(self, d, rng):
        _, h, w = d["pre"].shape
        cs = self.crop
        pos_mask = (d["hard"] > 0) & (d["weight"] > 0)
        if rng.random() < self.positive_crop_prob and pos_mask.any():
            ys, xs = np.where(pos_mask)
            idx = rng.integers(0, len(ys))
            cy, cx = int(ys[idx]), int(xs[idx])
            y = int(np.clip(cy - cs // 2, 0, h - cs))
            x = int(np.clip(cx - cs // 2, 0, w - cs))
        else:
            y = int(rng.integers(0, h - cs + 1))
            x = int(rng.integers(0, w - cs + 1))
        return y, x

    def __getitem__(self, idx):
        tile_idx = idx % len(self.tiles)
        d = self._data[tile_idx]
        rng = np.random.default_rng()
        y, x = self._sample_crop(d, rng)
        cs = self.crop

        pre = d["pre"][:, y : y + cs, x : x + cs]
        post = d["post"][:, y : y + cs, x : x + cs]
        hard = d["hard"][y : y + cs, x : x + cs]
        weight = d["weight"][y : y + cs, x : x + cs]

        if self.augment:
            k = int(rng.integers(0, 4))
            pre = np.rot90(pre, k=k, axes=(1, 2)).copy()
            post = np.rot90(post, k=k, axes=(1, 2)).copy()
            hard = np.rot90(hard, k=k, axes=(0, 1)).copy()
            weight = np.rot90(weight, k=k, axes=(0, 1)).copy()
            if rng.random() < 0.5:
                pre = pre[:, :, ::-1].copy()
                post = post[:, :, ::-1].copy()
                hard = hard[:, ::-1].copy()
                weight = weight[:, ::-1].copy()

        return (
            torch.from_numpy(pre).float(),
            torch.from_numpy(post).float(),
            torch.from_numpy(hard).float().unsqueeze(0),
            torch.from_numpy(weight).float().unsqueeze(0),
            torch.tensor([float(d["lat"]), float(d["lon"])], dtype=torch.float32),
        )


def precompute_cache(split: str = "train", pre_year: int = 2020, post_year: int = 2025):
    from ..inventory import list_tiles
    for t in list_tiles(split):
        print(f"[{split}] caching clay {t.tile_id} ...", flush=True)
        d = build_tile_tensors_clay(t, pre_year=pre_year, post_year=post_year)
        s2_nonzero = float((d["pre"][:S2_NUM_BANDS] != 0).mean())
        s1_nonzero = float((d["pre"][S2_NUM_BANDS:] != 0).mean())
        print(
            f"  shape={d['pre'].shape}  lat/lon=({float(d['lat']):.3f},{float(d['lon']):.3f})  "
            f"S2_nonzero={s2_nonzero*100:.1f}%  S1_nonzero={s1_nonzero*100:.1f}%"
            + (f"  pos={d['hard'].mean()*100:.2f}%" if "hard" in d else "")
        )


if __name__ == "__main__":
    ts = train_tiles()[:1]
    ds = ClayCropDataset(ts, crop_size=256, crops_per_tile=2)
    pre, post, tgt, w, meta = ds[0]
    print("clay crop:", pre.shape, post.shape, tgt.shape, w.shape, "meta:", meta.tolist())
    print(f"  pre S2 range: [{pre[:12].min():.1f}, {pre[:12].max():.1f}]")
    print(f"  pre S1 range: [{pre[12].min():.4f}, {pre[12].max():.4f}]")
    print(f"  target mean: {tgt.mean():.3f}")
