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

from ..inventory import Tile, train_tiles
from ..labels import fused_labels, load_reference_grid
from ..s2_features import s2_annual_features

CACHE_DIR = Path("cache")
# Bump when the cache schema changes so stale .npz files are rebuilt.
CACHE_VERSION = 2  # v2 = added per-year S2 NDVI/NBR features


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
      - pre   : (66, H, W) AEF+S2 for pre_year  (64 AEF + 2 S2 NDVI/NBR)
      - post  : (66, H, W) AEF+S2 for post_year
      - series: (T, 66, H, W) AEF+S2 stack across all tile.years
      - hard  : (H, W) majority-vote binary label  (train only)
      - soft  : (H, W) agreement-rate soft label   (train only)
      - version: scalar, cache schema version
    """
    cache_p = CACHE_DIR / tile.split / f"{tile.tile_id}.npz"
    if cache and cache_p.exists():
        with np.load(cache_p) as f:
            keys = set(f.files)
            version = int(f["version"]) if "version" in keys else 0
            if version >= CACHE_VERSION and {"pre", "post", "series"} <= keys:
                return {k: f[k] for k in f.files if k != "version"}

    # Reference grid = AEF pre-year tile.
    ref_path = tile.aef_path(pre_year)
    ref_shape, ref_transform, ref_crs = load_reference_grid(ref_path)

    def _aef_plus_s2(year: int) -> np.ndarray:
        aef = _read_aef(tile.aef_path(year))         # (64, H, W)
        aef = _pad_or_crop_like(aef, ref_shape)
        s2 = s2_annual_features(tile, year)          # (2, H, W)
        s2 = _pad_or_crop_like(s2, ref_shape)
        return np.concatenate([aef, s2], axis=0).astype(np.float32)  # (66, H, W)

    pre = _aef_plus_s2(pre_year)
    post = _aef_plus_s2(post_year)

    # Full temporal series across all years available for this tile.
    series = np.stack([_aef_plus_s2(y) for y in tile.years], axis=0)  # (T, 66, H, W)

    out: dict[str, np.ndarray] = {"pre": pre, "post": post, "series": series}

    if tile.split == "train":
        hard, soft = fused_labels(tile.tile_id, ref_shape, ref_transform, ref_crs)
        out["hard"] = hard
        out["soft"] = soft

    if cache:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_p, version=np.int32(CACHE_VERSION), **out)
    return out


def precompute_cache(split: str = "train", pre_year: int = 2020, post_year: int = 2025):
    from ..inventory import list_tiles

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
        weight = np.ones_like(target, dtype=np.float32)
        return (
            torch.from_numpy(pre).float(),
            torch.from_numpy(post).float(),
            torch.from_numpy(target).float().unsqueeze(0),
            torch.from_numpy(weight).float().unsqueeze(0),
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
        weight = np.ones_like(target, dtype=np.float32)
        return (
            torch.from_numpy(series).float(),
            torch.from_numpy(target).float().unsqueeze(0),
            torch.from_numpy(weight).float().unsqueeze(0),
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


def _load_pseudo(pseudo_dir: Path, tile_id: str) -> dict[str, np.ndarray]:
    p = Path(pseudo_dir) / f"{tile_id}.npz"
    with np.load(p) as f:
        return {"hard": f["hard"].astype(np.float32),
                "weight": f["weight"].astype(np.float32)}


class PseudoAEFCropDataset(Dataset):
    """Siamese crops from test tiles using ensemble pseudo-labels.

    Yields (pre, post, target, weight) — weight is 0 on ignored pixels so
    the masked BCE/Tversky skip them entirely.
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
            d = dict(build_tile_tensors(t, pre_year=pre_year, post_year=post_year))
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
        )


class PseudoTemporalAEFCropDataset(Dataset):
    """Temporal crops from test tiles with ensemble pseudo-labels + ignore mask."""

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
            d = dict(build_tile_tensors(t, pre_year=pre_year, post_year=post_year))
            d.update(_load_pseudo(pseudo_dir, t.tile_id))
            self._data.append(d)

    def __len__(self) -> int:
        return len(self.tiles) * self.crops_per_tile

    def _sample_crop(self, d, rng):
        _, _, h, w = d["series"].shape
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

        series = d["series"][:, :, y : y + cs, x : x + cs]
        hard = d["hard"][y : y + cs, x : x + cs]
        weight = d["weight"][y : y + cs, x : x + cs]

        if self.augment:
            k = int(rng.integers(0, 4))
            series = np.rot90(series, k=k, axes=(2, 3)).copy()
            hard = np.rot90(hard, k=k, axes=(0, 1)).copy()
            weight = np.rot90(weight, k=k, axes=(0, 1)).copy()
            if rng.random() < 0.5:
                series = series[:, :, :, ::-1].copy()
                hard = hard[:, ::-1].copy()
                weight = weight[:, ::-1].copy()

        return (
            torch.from_numpy(series).float(),
            torch.from_numpy(hard).float().unsqueeze(0),
            torch.from_numpy(weight).float().unsqueeze(0),
        )


if __name__ == "__main__":
    ts = train_tiles()
    ds = AEFCropDataset(ts[:2], crop_size=256, crops_per_tile=4)
    pre, post, tgt, w = ds[0]
    print("siamese crop:", pre.shape, post.shape, tgt.shape, w.shape, "target mean:", tgt.mean().item())

    tds = TemporalAEFCropDataset(ts[:2], crop_size=256, crops_per_tile=4)
    series, tgt, w = tds[0]
    print("temporal crop:", series.shape, tgt.shape, w.shape, "target mean:", tgt.mean().item())
