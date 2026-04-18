"""Compute per-channel mean/std across train tiles and save to JSON.

Reads the v3 cache (68-channel raw feature tensors: 64 AEF + 2 S2 NDVI/NBR +
2 S1 VV ASC/DESC dB) and computes stats used by the dataset for z-score
normalization. S2/S1 stats are computed over *valid* pixels only (mask=1)
so zero-filled cells (missing data) don't pull means toward 0.

Output:  checkpoints/channel_stats.json  with keys:
    mean:  list[float]  length 68  — per-channel mean
    std:   list[float]  length 68  — per-channel std
    count: list[int]    length 68  — valid pixel count per channel

Run:  .venv/bin/python3 -m src.compute_channel_stats
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .inventory import train_tiles

FEATURE_CH = 68
AEF_END = 64
S2_CH = (64, 65)
S1_ASC_CH = 66
S1_DESC_CH = 67
_EPS = 1e-6


def _channel_accum(sums, sq_sums, counts, ch: int, values: np.ndarray):
    vf = values.astype(np.float64, copy=False)
    sums[ch] += vf.sum()
    sq_sums[ch] += (vf * vf).sum()
    counts[ch] += vf.size


def compute(cache_root: str = "cache", out: str = "checkpoints/channel_stats.json"):
    sums = np.zeros(FEATURE_CH, dtype=np.float64)
    sq_sums = np.zeros(FEATURE_CH, dtype=np.float64)
    counts = np.zeros(FEATURE_CH, dtype=np.int64)

    tiles = train_tiles()
    for t in tiles:
        p = Path(cache_root) / t.split / f"{t.tile_id}.npz"
        if not p.exists():
            print(f"skip missing: {p}")
            continue
        with np.load(p) as f:
            series = f["series"]  # (T, 68, H, W)
        T = series.shape[0]
        print(f"{t.tile_id}  T={T}  shape={series.shape}")
        # AEF: all pixels valid
        for c in range(AEF_END):
            _channel_accum(sums, sq_sums, counts, c, series[:, c])
        # S2: valid where any S2 channel nonzero
        s2_mask = np.abs(series[:, S2_CH[0] : S2_CH[1] + 1]).sum(axis=1) > _EPS
        for c in S2_CH:
            v = series[:, c][s2_mask]
            _channel_accum(sums, sq_sums, counts, c, v)
        # S1 ASC / DESC independent masks
        asc_mask = np.abs(series[:, S1_ASC_CH]) > _EPS
        v = series[:, S1_ASC_CH][asc_mask]
        _channel_accum(sums, sq_sums, counts, S1_ASC_CH, v)
        desc_mask = np.abs(series[:, S1_DESC_CH]) > _EPS
        v = series[:, S1_DESC_CH][desc_mask]
        _channel_accum(sums, sq_sums, counts, S1_DESC_CH, v)

    n = np.clip(counts, 1, None)
    means = sums / n
    variances = np.maximum(sq_sums / n - means * means, 0.0)
    stds = np.sqrt(variances)
    # Guard against zero std (degenerate constant channel).
    stds = np.where(stds < 1e-6, 1.0, stds)

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(
            {
                "mean": means.tolist(),
                "std": stds.tolist(),
                "count": counts.tolist(),
            },
            f,
            indent=2,
        )
    print(f"\nsaved {out_p}")
    print(f"  AEF  ch0: mean={means[0]:+.3f} std={stds[0]:.3f}  (n={counts[0]:,})")
    print(f"  AEF  ch31: mean={means[31]:+.3f} std={stds[31]:.3f}")
    print(f"  S2   NDVI: mean={means[64]:+.3f} std={stds[64]:.3f}  (n={counts[64]:,})")
    print(f"  S2   NBR : mean={means[65]:+.3f} std={stds[65]:.3f}")
    print(f"  S1   ASC : mean={means[66]:+.3f} std={stds[66]:.3f}  (n={counts[66]:,})")
    print(f"  S1   DESC: mean={means[67]:+.3f} std={stds[67]:.3f}  (n={counts[67]:,})")


if __name__ == "__main__":
    compute()
