"""Post-training diagnostics on the validation set:

  1. Threshold sweep to find the best operating point.
  2. Per-tile F1/IoU breakdown at that threshold (finds the weakest region).
  3. Test-time augmentation (TTA: flip/rot averaging) gain vs no TTA.
  4. Probability histogram (positives vs negatives) for calibration sense.

Run:  .venv/bin/python3 -m src.analyze --ckpt checkpoints/siamese_unet_v1_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import FullTileDataset, TemporalFullTileDataset
from .inventory import train_tiles
from .model import SiameseChangeUNet, TemporalChangeUNet
from .train import DEFAULT_VAL_TILES, predict_full, predict_full_temporal


def _scores(prob: np.ndarray, target: np.ndarray, thr: float):
    p = prob >= thr
    t = target > 0.5
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "iou": iou}


@torch.no_grad()
def _predict_tta(model, inputs, crop, overlap, kind: str) -> torch.Tensor:
    """Average predictions over 8 D4 symmetries (4 rotations x {flip,no-flip}).

    ``inputs`` is either ``(pre, post)`` for siamese or ``series`` for temporal.
    """
    probs = []
    for k in range(4):
        for flip in (False, True):
            if kind == "temporal":
                series = inputs
                s = torch.rot90(series, k=k, dims=(2, 3))
                if flip:
                    s = torch.flip(s, dims=(3,))
                out = predict_full_temporal(model, s, crop=crop, overlap=overlap)
            else:
                pre, post = inputs
                p = torch.rot90(pre, k=k, dims=(1, 2))
                q = torch.rot90(post, k=k, dims=(1, 2))
                if flip:
                    p = torch.flip(p, dims=(2,))
                    q = torch.flip(q, dims=(2,))
                out = predict_full(model, p, q, crop=crop, overlap=overlap)
            if flip:
                out = torch.flip(out, dims=(1,))
            out = torch.rot90(out, k=-k, dims=(0, 1))
            probs.append(out)
    return torch.stack(probs).mean(dim=0)


def main(
    ckpt: str = "checkpoints/siamese_unet_v1_best.pt",
    crop: int = 512,
    overlap: int = 128,
    thresholds: tuple = tuple(round(x, 2) for x in np.arange(0.10, 0.91, 0.02)),
    out: str = "checkpoints/analysis_v1.json",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    kind = sd.get("model_kind", "siamese")
    if kind == "temporal":
        model = TemporalChangeUNet(in_channels=64, base=32).to(device)
    else:
        model = SiameseChangeUNet(in_channels=64, base=32).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    print(f"loaded {ckpt}  kind={kind}  (epoch={sd.get('epoch')})")

    val = [t for t in train_tiles() if t.tile_id in DEFAULT_VAL_TILES]
    ds = TemporalFullTileDataset(val) if kind == "temporal" else FullTileDataset(val)

    # Predict once per tile (plain + TTA), stash probs for sweep.
    tile_results: dict[str, dict] = {}
    for item in ds:
        if kind == "temporal":
            tile_id, series, hard = item
            series = series.to(device)
            prob_plain = predict_full_temporal(model, series, crop=crop, overlap=overlap).cpu().numpy()
            prob_tta = _predict_tta(model, series, crop=crop, overlap=overlap, kind=kind).cpu().numpy()
        else:
            tile_id, pre, post, hard = item
            pre = pre.to(device)
            post = post.to(device)
            prob_plain = predict_full(model, pre, post, crop=crop, overlap=overlap).cpu().numpy()
            prob_tta = _predict_tta(model, (pre, post), crop=crop, overlap=overlap, kind=kind).cpu().numpy()
        tile_results[tile_id] = {
            "prob_plain": prob_plain,
            "prob_tta": prob_tta,
            "target": hard.numpy(),
            "pos_rate": float(hard.float().mean()),
        }
        print(f"  predicted {tile_id}  pos_rate={tile_results[tile_id]['pos_rate']*100:.2f}%")

    # ---- 1. Threshold sweep (macro F1 over val tiles) ----
    def macro(key: str, thr: float):
        f1s = [_scores(r[key], r["target"], thr)["f1"] for r in tile_results.values()]
        ious = [_scores(r[key], r["target"], thr)["iou"] for r in tile_results.values()]
        return float(np.mean(f1s)), float(np.mean(ious))

    sweep = {"plain": [], "tta": []}
    for thr in thresholds:
        f1_p, iou_p = macro("prob_plain", thr)
        f1_t, iou_t = macro("prob_tta", thr)
        sweep["plain"].append({"thr": thr, "f1": f1_p, "iou": iou_p})
        sweep["tta"].append({"thr": thr, "f1": f1_t, "iou": iou_t})

    best_plain = max(sweep["plain"], key=lambda d: d["f1"])
    best_tta = max(sweep["tta"], key=lambda d: d["f1"])
    print("\n=== Threshold sweep (macro over val tiles) ===")
    print(f"  plain: best f1={best_plain['f1']:.3f} iou={best_plain['iou']:.3f} @ thr={best_plain['thr']}")
    print(f"  TTA  : best f1={best_tta['f1']:.3f} iou={best_tta['iou']:.3f} @ thr={best_tta['thr']}")

    # Stepwise table near the best threshold (easier to inspect calibration).
    print("\n thr  |  plain_f1  plain_iou  |   tta_f1   tta_iou")
    for p, t in zip(sweep["plain"], sweep["tta"]):
        marker_p = " *" if p is best_plain else "  "
        marker_t = " *" if t is best_tta else "  "
        print(f" {p['thr']:.2f} |   {p['f1']:.3f}     {p['iou']:.3f}  {marker_p} |   {t['f1']:.3f}    {t['iou']:.3f} {marker_t}")

    # ---- 2. Per-tile breakdown at TTA's best threshold ----
    thr = best_tta["thr"]
    print(f"\n=== Per-tile at thr={thr} (TTA) ===")
    per_tile = {}
    for tid, r in tile_results.items():
        s = _scores(r["prob_tta"], r["target"], thr)
        per_tile[tid] = s
        print(
            f"  {tid:15s}  pos={r['pos_rate']*100:5.2f}%  "
            f"F1={s['f1']:.3f}  IoU={s['iou']:.3f}  "
            f"P={s['precision']:.3f}  R={s['recall']:.3f}"
        )

    # ---- 3. Probability histogram: positives vs negatives (TTA) ----
    print(f"\n=== Probability distribution (TTA, pooled across val) ===")
    pos_probs = np.concatenate(
        [r["prob_tta"][r["target"] > 0.5].flatten() for r in tile_results.values()]
    )
    neg_probs = np.concatenate(
        [r["prob_tta"][r["target"] <= 0.5].flatten() for r in tile_results.values()]
    )
    bins = np.linspace(0.0, 1.0, 11)
    pos_h, _ = np.histogram(pos_probs, bins=bins)
    neg_h, _ = np.histogram(neg_probs, bins=bins)
    print("  bin        pos%       neg%")
    for i in range(len(bins) - 1):
        pp = 100 * pos_h[i] / max(pos_h.sum(), 1)
        nn = 100 * neg_h[i] / max(neg_h.sum(), 1)
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}   {pp:6.2f}%    {nn:6.2f}%")
    print(f"  positives: mean={pos_probs.mean():.3f}  median={np.median(pos_probs):.3f}")
    print(f"  negatives: mean={neg_probs.mean():.3f}  median={np.median(neg_probs):.3f}")

    # ---- save JSON summary ----
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump({
            "checkpoint": ckpt,
            "sweep": sweep,
            "best_plain": best_plain,
            "best_tta": best_tta,
            "per_tile_at_best_tta": per_tile,
        }, f, indent=2, default=float)
    print(f"\nsaved {out_p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/siamese_unet_v1_best.pt")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--out", default="checkpoints/analysis_v1.json")
    args = ap.parse_args()
    main(**vars(args))
