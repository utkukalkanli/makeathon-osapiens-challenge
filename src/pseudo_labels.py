"""Generate pseudo-labels for the test tiles using the S2 ensemble.

A pixel earns a pseudo-label only when *both* models agree and are confident:
  - ``prob_ensemble >= pos_thr``   (positive) OR
  - ``prob_ensemble <= neg_thr``   (negative)
  AND ``|prob_siamese - prob_temporal| <= agree_thr``   (agreement)
Everything else is flagged ``ignore`` and masked out of the loss.

Output per test tile: ``<out_dir>/<tile_id>.npz`` with keys
  - hard   (H, W) uint8  : 0/1 pseudo-label
  - weight (H, W) uint8  : 1 = use this pixel, 0 = ignore
  - prob   (H, W) float32: ensemble probability (kept for inspection)

Run:
  .venv/bin/python3 -m src.pseudo_labels \
      --siamese_ckpt checkpoints/siamese_s2_v1_best.pt \
      --temporal_ckpt checkpoints/temporal_s2_v1_best.pt \
      --out_dir pseudo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import torch

from .inventory import test_tiles
from .unet.dataset import FullTileDataset, TemporalFullTileDataset
from .unet.model import SiameseChangeUNet, TemporalChangeUNet
from .unet.train import predict_full, predict_full_temporal


def _load(ckpt: str, device: str):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    kind = sd.get("model_kind", "siamese")
    in_ch = sd.get("in_channels", 64)
    if kind == "temporal":
        model = TemporalChangeUNet(in_channels=in_ch, base=32).to(device)
    else:
        model = SiameseChangeUNet(in_channels=in_ch, base=32).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    return model, kind


def main(
    siamese_ckpt: str = "checkpoints/siamese_s2_v1_best.pt",
    temporal_ckpt: str = "checkpoints/temporal_s2_v1_best.pt",
    out_dir: str = "pseudo",
    pos_thr: float = 0.80,
    neg_thr: float = 0.20,
    agree_thr: float = 0.30,
    crop: int = 512,
    overlap: int = 128,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    siamese, kind_s = _load(siamese_ckpt, device)
    temporal, kind_t = _load(temporal_ckpt, device)
    assert kind_s == "siamese" and kind_t == "temporal"

    mlflow.set_experiment("deforestation")
    mlflow.start_run(run_name=f"pseudo_labels:{Path(siamese_ckpt).stem}+{Path(temporal_ckpt).stem}")
    mlflow.set_tag("stage", "pseudo_labels")
    mlflow.log_params({
        "siamese_ckpt": siamese_ckpt,
        "temporal_ckpt": temporal_ckpt,
        "pos_thr": pos_thr,
        "neg_thr": neg_thr,
        "agree_thr": agree_thr,
    })

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tiles = test_tiles()
    ds_s = FullTileDataset(tiles)
    ds_t = TemporalFullTileDataset(tiles)

    summary = {}
    totals = {"pos": 0, "neg": 0, "ignore": 0, "pixels": 0}
    for (tid_s, pre, post, _), (tid_t, series, _) in zip(ds_s, ds_t):
        assert tid_s == tid_t
        pre = pre.to(device)
        post = post.to(device)
        series = series.to(device)
        prob_s = predict_full(siamese, pre, post, crop=crop, overlap=overlap).cpu().numpy()
        prob_t = predict_full_temporal(temporal, series, crop=crop, overlap=overlap).cpu().numpy()
        prob_e = 0.5 * (prob_s + prob_t)

        agree = np.abs(prob_s - prob_t) <= agree_thr
        pos = (prob_e >= pos_thr) & agree
        neg = (prob_e <= neg_thr) & agree
        weight = (pos | neg).astype(np.uint8)
        hard = pos.astype(np.uint8)
        ignore = (~(pos | neg))

        n = prob_e.size
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        n_ign = int(ignore.sum())
        totals["pos"] += n_pos
        totals["neg"] += n_neg
        totals["ignore"] += n_ign
        totals["pixels"] += n
        print(
            f"  {tid_s}: pos={n_pos/n*100:5.2f}%  neg={n_neg/n*100:5.2f}%  "
            f"ignore={n_ign/n*100:5.2f}%"
        )
        summary[tid_s] = {
            "pos_pct": n_pos / n, "neg_pct": n_neg / n, "ignore_pct": n_ign / n,
        }
        mlflow.log_metric(f"pos_pct_{tid_s}", n_pos / n)
        mlflow.log_metric(f"ignore_pct_{tid_s}", n_ign / n)

        np.savez_compressed(
            out_root / f"{tid_s}.npz",
            hard=hard, weight=weight, prob=prob_e.astype(np.float32),
        )

    overall = {
        "pos_pct": totals["pos"] / totals["pixels"],
        "neg_pct": totals["neg"] / totals["pixels"],
        "ignore_pct": totals["ignore"] / totals["pixels"],
    }
    print(
        f"\ntotals: pos={overall['pos_pct']*100:.2f}%  neg={overall['neg_pct']*100:.2f}%  "
        f"ignore={overall['ignore_pct']*100:.2f}%"
    )
    mlflow.log_metrics({
        "pos_pct_total": overall["pos_pct"],
        "neg_pct_total": overall["neg_pct"],
        "ignore_pct_total": overall["ignore_pct"],
    })

    summary_p = out_root / "summary.json"
    with open(summary_p, "w") as f:
        json.dump({"per_tile": summary, "totals": overall,
                   "thresholds": {"pos_thr": pos_thr, "neg_thr": neg_thr,
                                  "agree_thr": agree_thr}}, f, indent=2)
    mlflow.log_artifact(str(summary_p))
    mlflow.end_run()
    print(f"saved pseudo labels -> {out_root}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--siamese_ckpt", default="checkpoints/siamese_s2_v1_best.pt")
    ap.add_argument("--temporal_ckpt", default="checkpoints/temporal_s2_v1_best.pt")
    ap.add_argument("--out_dir", default="pseudo")
    ap.add_argument("--pos_thr", type=float, default=0.80)
    ap.add_argument("--neg_thr", type=float, default=0.20)
    ap.add_argument("--agree_thr", type=float, default=0.30)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    args = ap.parse_args()
    main(**vars(args))
