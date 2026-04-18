"""Ensemble two trained checkpoints by averaging their sigmoid probabilities.

Rationale: siamese (pre=2020, post=2025) and temporal (6-year stack) make
partially decorrelated errors — temporal helps on 18NXH but regresses on
48QVE, siamese is the inverse. A simple average often recovers the best of
both without any retraining.

Usage:
  .venv/bin/python3 -m src.ensemble \
      --siamese_ckpt checkpoints/siamese_unet_v1_best.pt \
      --temporal_ckpt checkpoints/temporal_v1_best.pt

Also supports prediction on the test split with ``--predict_test``: writes
GeoJSONs alongside a combined raster per tile under ``out_dir``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import FullTileDataset, TemporalFullTileDataset
from .inventory import test_tiles, train_tiles
from .model import SiameseChangeUNet, TemporalChangeUNet
from .train import (
    DEFAULT_VAL_TILES,
    predict_full,
    predict_full_temporal,
)


def _scores(prob: np.ndarray, target: np.ndarray, thr: float):
    p = prob >= thr
    t = target > 0.5
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou,
            "tp": tp, "fp": fp, "fn": fn}


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
    print(f"  loaded {ckpt}  kind={kind}  in_ch={in_ch}  epoch={sd.get('epoch')}")
    return model, kind


@torch.no_grad()
def _predict_both(
    siamese, temporal,
    tile_id, pre_tensor=None, post_tensor=None, series_tensor=None,
    crop=512, overlap=128,
):
    """Run both models on a tile and return (prob_siamese, prob_temporal)."""
    prob_s = predict_full(siamese, pre_tensor, post_tensor,
                          crop=crop, overlap=overlap).cpu().numpy()
    prob_t = predict_full_temporal(temporal, series_tensor,
                                   crop=crop, overlap=overlap).cpu().numpy()
    return prob_s, prob_t


def main(
    siamese_ckpt: str,
    temporal_ckpt: str,
    weight_siamese: float = 0.5,
    crop: int = 512,
    overlap: int = 128,
    predict_test: bool = False,
    out_dir: str = "submission/ensemble",
    threshold: float | None = None,  # None => sweep on val
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    siamese, kind_s = _load(siamese_ckpt, device)
    temporal, kind_t = _load(temporal_ckpt, device)
    assert kind_s == "siamese" and kind_t == "temporal", \
        f"Expected siamese + temporal checkpoints, got {kind_s} + {kind_t}"
    w_s = float(weight_siamese)
    w_t = 1.0 - w_s
    print(f"ensemble weights: siamese={w_s:.2f}  temporal={w_t:.2f}")

    # --- Val evaluation: load both dataset flavors over the same tiles. ---
    val_tiles_list = [t for t in train_tiles() if t.tile_id in DEFAULT_VAL_TILES]
    val_s = FullTileDataset(val_tiles_list)
    val_t = TemporalFullTileDataset(val_tiles_list)

    per_tile = {}
    for (tid_s, pre, post, hard), (tid_t, series, _) in zip(val_s, val_t):
        assert tid_s == tid_t
        pre = pre.to(device)
        post = post.to(device)
        series = series.to(device)
        prob_s, prob_t = _predict_both(
            siamese, temporal, tid_s,
            pre_tensor=pre, post_tensor=post, series_tensor=series,
            crop=crop, overlap=overlap,
        )
        prob_e = w_s * prob_s + w_t * prob_t
        per_tile[tid_s] = {
            "prob_siamese": prob_s,
            "prob_temporal": prob_t,
            "prob_ensemble": prob_e,
            "target": hard.numpy(),
            "pos_rate": float(hard.float().mean()),
        }
        print(f"  predicted {tid_s}  pos={per_tile[tid_s]['pos_rate']*100:.2f}%")

    # --- Threshold sweep: compare siamese / temporal / ensemble macro F1. ---
    thresholds = [round(x, 2) for x in np.arange(0.10, 0.91, 0.02)]
    def macro(key: str, thr: float):
        return float(np.mean([_scores(r[key], r["target"], thr)["f1"] for r in per_tile.values()]))
    sweep = {"siamese": [], "temporal": [], "ensemble": []}
    for thr in thresholds:
        sweep["siamese"].append({"thr": thr, "f1": macro("prob_siamese", thr)})
        sweep["temporal"].append({"thr": thr, "f1": macro("prob_temporal", thr)})
        sweep["ensemble"].append({"thr": thr, "f1": macro("prob_ensemble", thr)})

    best_s = max(sweep["siamese"], key=lambda d: d["f1"])
    best_t = max(sweep["temporal"], key=lambda d: d["f1"])
    best_e = max(sweep["ensemble"], key=lambda d: d["f1"])
    print("\n=== Macro F1 vs threshold (best starred) ===")
    print(" thr   siamese  temporal  ensemble")
    for s, t, e in zip(sweep["siamese"], sweep["temporal"], sweep["ensemble"]):
        ms = "*" if s is best_s else " "
        mt = "*" if t is best_t else " "
        me = "*" if e is best_e else " "
        print(f" {s['thr']:.2f}   {s['f1']:.3f}{ms}  {t['f1']:.3f}{mt}   {e['f1']:.3f}{me}")
    print(f"\nBest siamese  : F1 {best_s['f1']:.3f} @ thr={best_s['thr']}")
    print(f"Best temporal : F1 {best_t['f1']:.3f} @ thr={best_t['thr']}")
    print(f"Best ensemble : F1 {best_e['f1']:.3f} @ thr={best_e['thr']}  "
          f"(Δ vs siamese = {best_e['f1'] - best_s['f1']:+.3f})")

    thr_use = threshold if threshold is not None else best_e["thr"]
    print(f"\n=== Per-tile at thr={thr_use} ===")
    per_tile_out = {}
    for tid, r in per_tile.items():
        s_s = _scores(r["prob_siamese"], r["target"], thr_use)
        s_t = _scores(r["prob_temporal"], r["target"], thr_use)
        s_e = _scores(r["prob_ensemble"], r["target"], thr_use)
        per_tile_out[tid] = {"siamese": s_s, "temporal": s_t, "ensemble": s_e,
                             "pos_rate": r["pos_rate"]}
        print(f"  {tid:16s} pos={r['pos_rate']*100:5.2f}%  "
              f"siamese F1={s_s['f1']:.3f}  temporal F1={s_t['f1']:.3f}  "
              f"ensemble F1={s_e['f1']:.3f} "
              f"(P={s_e['precision']:.3f} R={s_e['recall']:.3f})")

    summary = {
        "siamese_ckpt": siamese_ckpt,
        "temporal_ckpt": temporal_ckpt,
        "weights": {"siamese": w_s, "temporal": w_t},
        "best_siamese": best_s, "best_temporal": best_t, "best_ensemble": best_e,
        "threshold_used": thr_use,
        "per_tile": per_tile_out,
    }
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/analysis_ensemble.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("saved checkpoints/analysis_ensemble.json")

    # --- Optional: ensemble predictions on the held-out test tiles. ---
    if predict_test:
        print(f"\n=== Predicting test tiles -> {out_dir} ===")
        from submission_utils import raster_to_geojson
        import rasterio

        def _write_binary_raster(path: Path, mask: np.ndarray, ref_aef_path: Path):
            with rasterio.open(ref_aef_path) as src:
                profile = src.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
            profile["width"] = mask.shape[1]
            profile["height"] = mask.shape[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(mask.astype("uint8"), 1)

        tiles = test_tiles()
        ds_s = FullTileDataset(tiles)
        ds_t = TemporalFullTileDataset(tiles)
        out_root = Path(out_dir)

        for (tid_s, pre, post, _), (tid_t, series, _) in zip(ds_s, ds_t):
            assert tid_s == tid_t
            pre = pre.to(device)
            post = post.to(device)
            series = series.to(device)
            prob_s, prob_t = _predict_both(
                siamese, temporal, tid_s,
                pre_tensor=pre, post_tensor=post, series_tensor=series,
                crop=crop, overlap=overlap,
            )
            prob = w_s * prob_s + w_t * prob_t
            binary = (prob >= thr_use).astype(np.uint8)
            pos_pct = binary.mean() * 100
            print(f"  {tid_s}: positives={binary.sum()} ({pos_pct:.2f}%)")
            if binary.sum() == 0:
                continue
            tile = next(t for t in tiles if t.tile_id == tid_s)
            tif_p = out_root / "rasters" / f"{tid_s}.tif"
            _write_binary_raster(tif_p, binary, tile.aef_path(2020))
            geo_p = out_root / f"{tid_s}.geojson"
            try:
                raster_to_geojson(tif_p, output_path=geo_p, min_area_ha=0.5)
                print(f"    wrote {geo_p}")
            except ValueError as e:
                print(f"    skip {tid_s}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--siamese_ckpt", default="checkpoints/siamese_unet_v1_best.pt")
    ap.add_argument("--temporal_ckpt", default="checkpoints/temporal_v1_best.pt")
    ap.add_argument("--weight_siamese", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--predict_test", action="store_true")
    ap.add_argument("--out_dir", default="submission/ensemble")
    ap.add_argument("--threshold", type=float, default=None,
                    help="If set, use this threshold (else use best from val sweep).")
    args = ap.parse_args()
    main(**vars(args))
