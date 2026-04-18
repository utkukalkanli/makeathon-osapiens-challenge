"""PyTorch dataset: AEF embeddings (pre=2020, post=2025) + fused labels.

The AEF tile is the reference grid (EPSG:4326). Labels from RADD, GLAD-L,
GLAD-S2 are reprojected onto this grid at load time and cached per-tile.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .inventory import Tile, train_tiles
from .labels import fused_labels, load_reference_grid

CACHE_DIR = Path("cache")


def _read_aef(path: Path) -> np.ndarray:
    """Read a 64-band AEF tiff as float32 array (64, H, W). Replaces NaN with 0."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _pad_or_crop_like(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Pad with zeros or crop an (..., H, W) array to exactly target_shape."""
    h_t, w_t = target_shape
    h, w = arr.shape[-2:]
    # Crop
    arr = arr[..., : min(h, h_t), : min(w, w_t)]
    # Pad
    ph, pw = h_t - arr.shape[-2], w_t - arr.shape[-1]
    if ph > 0 or pw > 0:
        pad_spec = [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)]
        arr = np.pad(arr, pad_spec, constant_values=0.0)
    return arr


def build_tile_tensors(
    tile: Tile,
    pre_year: int = 2020,
    post_year: int = 2025,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Load and align everything for one tile. Cached as .npz on disk.

    Keys written:
      - pre   : (64, H, W) AEF for pre_year
      - post  : (64, H, W) AEF for post_year
      - series: (T, 64, H, W) AEF stack across all tile.years (temporal input)
      - hard  : (H, W) majority-vote binary label  (train only)
      - soft  : (H, W) agreement-rate soft label   (train only)
    """
    cache_p = CACHE_DIR / tile.split / f"{tile.tile_id}.npz"
    if cache and cache_p.exists():
        with np.load(cache_p) as f:
            keys = set(f.files)
            # Rebuild if cache was written by an older version without "series".
            if "series" in keys:
                return {k: f[k] for k in f.files}

    # Reference grid = AEF pre-year tile.
    ref_path = tile.aef_path(pre_year)
    ref_shape, ref_transform, ref_crs = load_reference_grid(ref_path)

    pre = _read_aef(ref_path)  # (64, H, W)
    post = _read_aef(tile.aef_path(post_year))
    post = _pad_or_crop_like(post, ref_shape)  # tiny size differences happen

    # Full temporal series across all years available for this tile.
    series_frames = []
    for y in tile.years:
        arr = _read_aef(tile.aef_path(y))
        arr = _pad_or_crop_like(arr, ref_shape)
        series_frames.append(arr)
    series = np.stack(series_frames, axis=0)  # (T, 64, H, W)

    out: dict[str, np.ndarray] = {"pre": pre, "post": post, "series": series}

    if tile.split == "train":
        hard, soft = fused_labels(tile.tile_id, ref_shape, ref_transform, ref_crs)
        out["hard"] = hard
        out["soft"] = soft

    if cache:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_p, **out)
    return out


def precompute_cache(split: str = "train", pre_year: int = 2020, post_year: int = 2025):
    from .inventory import list_tiles

    for t in list_tiles(split):
        print(f"[{split}] caching {t.tile_id} ...", flush=True)
        d = build_tile_tensors(t, pre_year, post_year)
        shape = d["pre"].shape
        if "hard" in d:
            print(f"  pre={shape}  pos={d['hard'].mean()*100:.2f}%")
        else:
            print(f"  pre={shape}")


class AEFCropDataset(Dataset):
    """Random crops of pre/post AEF embeddings + fused label.

    One epoch = ``crops_per_tile`` samples per tile. Crops are weighted-sampled:
    with probability ``positive_crop_prob`` we pick a crop that contains at
    least one positive label pixel (when available) to combat class imbalance.
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
            build_tile_tensors(t, pre_year=pre_year, post_year=post_year)
            for t in tiles
        ]

    def __len__(self) -> int:
        return len(self.tiles) * self.crops_per_tile

    def _sample_crop(self, d: dict[str, np.ndarray], rng: np.random.Generator):
        _, h, w = d["pre"].shape
        cs = self.crop
        if h < cs or w < cs:
            # pad d["pre"]/d["post"] to at least cs
            pass  # AEF tiles are ~1000x1000 so unreachable in practice
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
            # Random flips and 90deg rotations are label-preserving.
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
        return (
            torch.from_numpy(pre).float(),
            torch.from_numpy(post).float(),
            torch.from_numpy(target).float().unsqueeze(0),
        )


class FullTileDataset(Dataset):
    """Deterministic full-tile loader for validation / inference (no crops)."""

    def __init__(
        self,
        tiles: list[Tile],
        pre_year: int = 2020,
        post_year: int = 2025,
    ):
        self.tiles = tiles
        self._data = [
            build_tile_tensors(t, pre_year=pre_year, post_year=post_year)
            for t in tiles
        ]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        t = self.tiles[idx]
        d = self._data[idx]
        pre = torch.from_numpy(d["pre"]).float()
        post = torch.from_numpy(d["post"]).float()
        hard = torch.from_numpy(d.get("hard", np.zeros(d["pre"].shape[-2:], dtype=np.float32))).float()
        return t.tile_id, pre, post, hard


class TemporalAEFCropDataset(Dataset):
    """Random crops of a multi-year AEF stack + fused label.

    Yields ``(series, target)`` where ``series`` is ``(T, 64, crop, crop)``
    with T = number of AEF years available per tile (typically 6).
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
            build_tile_tensors(t, pre_year=pre_year, post_year=post_year)
            for t in tiles
        ]

    def __len__(self) -> int:
        return len(self.tiles) * self.crops_per_tile

    def _sample_crop(self, d, rng):
        _, _, h, w = d["series"].shape
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

        series = d["series"][:, :, y : y + cs, x : x + cs]  # (T, 64, cs, cs)
        if "hard" in d:
            hard = d["hard"][y : y + cs, x : x + cs]
            soft = d["soft"][y : y + cs, x : x + cs]
        else:
            hard = np.zeros((cs, cs), dtype=np.float32)
            soft = hard

        if self.augment:
            k = int(rng.integers(0, 4))
            series = np.rot90(series, k=k, axes=(2, 3)).copy()
            hard = np.rot90(hard, k=k, axes=(0, 1)).copy()
            soft = np.rot90(soft, k=k, axes=(0, 1)).copy()
            if rng.random() < 0.5:
                series = series[:, :, :, ::-1].copy()
                hard = hard[:, ::-1].copy()
                soft = soft[:, ::-1].copy()

        target = soft if self.use_soft_target else hard
        return (
            torch.from_numpy(series).float(),
            torch.from_numpy(target).float().unsqueeze(0),
        )


class TemporalFullTileDataset(Dataset):
    """Deterministic full-tile temporal loader for validation / inference."""

    def __init__(
        self,
        tiles: list[Tile],
        pre_year: int = 2020,
        post_year: int = 2025,
    ):
        self.tiles = tiles
        self._data = [
            build_tile_tensors(t, pre_year=pre_year, post_year=post_year)
            for t in tiles
        ]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        t = self.tiles[idx]
        d = self._data[idx]
        series = torch.from_numpy(d["series"]).float()  # (T, 64, H, W)
        hard = torch.from_numpy(
            d.get("hard", np.zeros(d["series"].shape[-2:], dtype=np.float32))
        ).float()
        return t.tile_id, series, hard


if __name__ == "__main__":
    ts = train_tiles()
    ds = AEFCropDataset(ts[:2], crop_size=256, crops_per_tile=4)
    pre, post, tgt = ds[0]
    print("siamese crop:", pre.shape, post.shape, tgt.shape, "target mean:", tgt.mean().item())

    tds = TemporalAEFCropDataset(ts[:2], crop_size=256, crops_per_tile=4)
    series, tgt = tds[0]
    print("temporal crop:", series.shape, tgt.shape, "target mean:", tgt.mean().item())
