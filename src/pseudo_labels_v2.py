"""2-teacher ensemble pseudo-labels for the test tiles.

Fixes a flaw in [src/pseudo_labels.py](src/pseudo_labels.py): the v1 pseudo-labels
were generated from two S2-only teachers, so any deforestation visible only
via S1 was forced to "negative" — punishing the new S2+S1 student at exactly
the pixels where S1 should add value.

v2 uses the cached TTA probs from:
  * ``temporal_s2_pseudo_v1_best.pt``     — S2-only teacher (sees nothing in cloudy years)
  * ``temporal_s2s1_norm_v1_best.pt``     — new S2+S1 teacher with per-channel norm,
                                            modality masks and dropout

Rules (all applied pixel-wise):
  * ``pos``:    both teachers >= ``pos_thr``  AND  |p_s2 - p_s2s1| <= ``agree_thr``
  * ``neg``:    both teachers <= ``neg_thr``  AND  |p_s2 - p_s2s1| <= ``agree_thr``
  * ``ignore``: everything else (single-modality disagreement, mid-confidence, …)

Pixels in "ignore" get weight=0 and the loss skips them — the student is not
punished for predicting positive where S1 alone detects a change.

Output per test tile: ``<out_dir>/<tile_id>.npz`` with
    hard   (H, W) uint8    : 0 / 1 pseudo-label (0 when ignore)
    weight (H, W) uint8    : 1 = use this pixel, 0 = ignore
    prob   (H, W) float32  : mean of the two probs (for inspection)

Run:
  .venv/bin/python3 -m src.pseudo_labels_v2 --out_dir pseudo_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .inventory import test_tiles


def _load_probs(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def main(
    s2_cache: str = "submission/temporal_s2_pseudo_v2_cache/probs",
    s2s1_cache: str = "submission/temporal_s2s1_norm_v1_cache/probs",
    out_dir: str = "pseudo_v2",
    pos_thr: float = 0.60,
    neg_thr: float = 0.20,
    agree_thr: float = 0.30,
):
    s2_dir = Path(s2_cache)
    s2s1_dir = Path(s2s1_cache)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    per_tile_summary = {}
    totals = {"pos": 0, "neg": 0, "ignore": 0, "total": 0}

    for tile in test_tiles():
        tid = tile.tile_id
        p_s2 = s2_dir / f"{tid}.npy"
        p_s2s1 = s2s1_dir / f"{tid}.npy"
        if not (p_s2.exists() and p_s2s1.exists()):
            print(f"skip {tid}: missing cached probs")
            continue
        prob_s2 = _load_probs(p_s2)
        prob_s2s1 = _load_probs(p_s2s1)
        # Defensive: match sizes by cropping to smaller (1-2 px drift possible).
        H = min(prob_s2.shape[0], prob_s2s1.shape[0])
        W = min(prob_s2.shape[1], prob_s2s1.shape[1])
        prob_s2 = prob_s2[:H, :W]
        prob_s2s1 = prob_s2s1[:H, :W]

        agree = np.abs(prob_s2 - prob_s2s1) <= agree_thr
        pos = (prob_s2 >= pos_thr) & (prob_s2s1 >= pos_thr) & agree
        neg = (prob_s2 <= neg_thr) & (prob_s2s1 <= neg_thr) & agree
        weight = (pos | neg).astype(np.uint8)
        hard = pos.astype(np.uint8)
        ignore = ~(pos | neg)

        prob_mean = 0.5 * (prob_s2 + prob_s2s1)

        out_p = out_root / f"{tid}.npz"
        np.savez_compressed(out_p, hard=hard, weight=weight, prob=prob_mean.astype(np.float32))

        n = hard.size
        n_pos, n_neg, n_ign = int(pos.sum()), int(neg.sum()), int(ignore.sum())
        pct = lambda x: 100 * x / max(n, 1)
        print(f"  {tid}: pos={pct(n_pos):.3f}%  neg={pct(n_neg):.2f}%  ignore={pct(n_ign):.2f}%  "
              f"(wrote {out_p})")

        per_tile_summary[tid] = {
            "pos_pct": n_pos / n,
            "neg_pct": n_neg / n,
            "ignore_pct": n_ign / n,
        }
        totals["pos"] += n_pos
        totals["neg"] += n_neg
        totals["ignore"] += n_ign
        totals["total"] += n

    tot_n = max(totals["total"], 1)
    summary = {
        "per_tile": per_tile_summary,
        "totals": {
            "pos_pct": totals["pos"] / tot_n,
            "neg_pct": totals["neg"] / tot_n,
            "ignore_pct": totals["ignore"] / tot_n,
        },
        "thresholds": {"pos_thr": pos_thr, "neg_thr": neg_thr, "agree_thr": agree_thr},
        "teachers": {"s2_cache": s2_cache, "s2s1_cache": s2s1_cache},
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {out_root / 'summary.json'}")
    print(f"  totals:  pos={100*summary['totals']['pos_pct']:.3f}%  "
          f"neg={100*summary['totals']['neg_pct']:.2f}%  "
          f"ignore={100*summary['totals']['ignore_pct']:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_cache", default="submission/temporal_s2_pseudo_v2_cache/probs")
    ap.add_argument("--s2s1_cache", default="submission/temporal_s2s1_norm_v1_cache/probs")
    ap.add_argument("--out_dir", default="pseudo_v2")
    ap.add_argument("--pos_thr", type=float, default=0.60)
    ap.add_argument("--neg_thr", type=float, default=0.20)
    ap.add_argument("--agree_thr", type=float, default=0.30)
    args = ap.parse_args()
    main(**vars(args))
