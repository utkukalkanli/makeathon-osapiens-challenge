"""Fine-tune the Clay v1.5 segmentation head on AEF-grid S2 crops.

Phase 1 default: encoder frozen, only the seg head is trained. Reuses the
same val tiles, loss (BCE+Tversky), and MLflow experiment as the U-Net path.

Pseudo-label support mirrors ``src.unet.train``: point ``--pseudo_dir`` at a
directory of ``<tile_id>.npz`` files and test-tile crops will be mixed in.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import ConcatDataset, DataLoader

from ..inventory import test_tiles, train_tiles
from ..losses import BCETverskyLoss
from .dataset import (
    ClayCropDataset,
    ClayFullTileDataset,
    ClayPseudoCropDataset,
    filter_by_s2_coverage,
)
from .model import ClaySiameseSegmenter, count_params

DEFAULT_VAL_TILES = ("18NXH_6_8", "48PXC_7_7", "48QVE_3_0")
CKPT_DIR = Path("checkpoints")


def _iou_f1(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    p = pred > thr
    t = target > 0.5
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    iou = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return iou, f1


def _tile_positions(H: int, W: int, crop: int, stride: int):
    ys = list(range(0, max(1, H - crop + 1), stride))
    xs = list(range(0, max(1, W - crop + 1), stride))
    if ys[-1] + crop < H:
        ys.append(H - crop)
    if xs[-1] + crop < W:
        xs.append(W - crop)
    return ys, xs


@torch.no_grad()
def predict_full_clay(
    model: ClaySiameseSegmenter,
    pre: torch.Tensor,
    post: torch.Tensor,
    latlon: torch.Tensor,
    crop: int = 256,
    overlap: int = 64,
) -> torch.Tensor:
    """Sliding-window inference. pre/post: (C, H, W); latlon: (4,).
    Returns (H, W) probability map."""
    _, H, W = pre.shape
    stride = crop - overlap
    prob = torch.zeros((H, W), device=pre.device)
    weight = torch.zeros((H, W), device=pre.device)
    ys, xs = _tile_positions(H, W, crop, stride)
    ll = latlon.unsqueeze(0)
    for y in ys:
        for x in xs:
            pre_c = pre[:, y : y + crop, x : x + crop].unsqueeze(0)
            post_c = post[:, y : y + crop, x : x + crop].unsqueeze(0)
            logits = model(pre_c, post_c, ll)[0, 0]
            prob[y : y + crop, x : x + crop] += torch.sigmoid(logits)
            weight[y : y + crop, x : x + crop] += 1
    return prob / weight.clamp_min(1)


@torch.no_grad()
def evaluate(model, val_ds, device: str, crop: int = 256, overlap: int = 64):
    model.eval()
    per_tile = {}
    for tile_id, pre, post, hard, meta in val_ds:
        pre = pre.to(device)
        post = post.to(device)
        meta = meta.to(device)
        prob = predict_full_clay(model, pre, post, meta, crop=crop, overlap=overlap)
        iou, f1 = _iou_f1(prob.cpu(), hard)
        per_tile[tile_id] = {"iou": iou, "f1": f1, "pos_rate": float(hard.mean())}
    macro_iou = float(np.mean([v["iou"] for v in per_tile.values()]))
    macro_f1 = float(np.mean([v["f1"] for v in per_tile.values()]))
    return {"per_tile": per_tile, "macro_iou": macro_iou, "macro_f1": macro_f1}


def main(
    val_tile_ids: tuple[str, ...] = DEFAULT_VAL_TILES,
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 1e-3,
    crop_size: int = 256,
    crops_per_tile: int = 32,
    num_workers: int = 4,
    seed: int = 0,
    out_name: str = "clay_frozen_v1",
    freeze_encoder: bool = True,
    encoder_lr: float | None = None,  # only used when freeze_encoder=False
    pretrained_ckpt: str | None = None,  # resolved to HF download if None
    pseudo_dir: str | None = None,
    pseudo_crops_per_tile: int = 16,
    eval_crop: int = 256,
    eval_overlap: int = 64,
    min_s2_coverage: float = 0.10,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_tiles = train_tiles()
    tr = [t for t in all_tiles if t.tile_id not in val_tile_ids]
    va = [t for t in all_tiles if t.tile_id in val_tile_ids]
    print(f"filtering train tiles by S2 coverage (min={min_s2_coverage*100:.0f}%)...")
    tr, skipped_train = filter_by_s2_coverage(tr, min_coverage=min_s2_coverage)
    _, skipped_val = filter_by_s2_coverage(va, min_coverage=min_s2_coverage)
    if skipped_val:
        print(f"  WARN: val tiles with no S2 coverage kept for reporting only: {skipped_val}")
    print(f"train tiles: {len(tr)}  val tiles: {[t.tile_id for t in va]}  "
          f"skipped_train={skipped_train}")

    real_ds = ClayCropDataset(
        tr, crop_size=crop_size, crops_per_tile=crops_per_tile,
        positive_crop_prob=0.75, augment=True,
    )
    val_ds = ClayFullTileDataset(va)

    if pseudo_dir is not None:
        pseudo_ds = ClayPseudoCropDataset(
            test_tiles(), pseudo_dir=pseudo_dir,
            crop_size=crop_size, crops_per_tile=pseudo_crops_per_tile,
            positive_crop_prob=0.75, augment=True,
        )
        train_ds = ConcatDataset([real_ds, pseudo_ds])
        print(f"pseudo: {len(pseudo_ds)} extra crops/epoch from test tiles")
    else:
        train_ds = real_ds
    print(f"total crops/epoch: {len(train_ds)}")

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
        persistent_workers=num_workers > 0,
    )

    model = ClaySiameseSegmenter(freeze_encoder=freeze_encoder).to(device)
    if pretrained_ckpt is None:
        pretrained_ckpt = hf_hub_download("made-with-clay/Clay", "v1.5/clay-v1.5.ckpt")
    print(f"loading Clay weights from {pretrained_ckpt}")
    model.load_pretrained(pretrained_ckpt)
    total, trainable = count_params(model)
    print(f"params: total={total/1e6:.1f}M  trainable={trainable/1e6:.1f}M  "
          f"freeze_encoder={freeze_encoder}")

    if freeze_encoder:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4,
        )
        enc_lr_eff = 0.0
    else:
        enc_lr_eff = encoder_lr if encoder_lr is not None else lr * 0.01
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": enc_lr_eff},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=1e-4,
        )
        print(f"  encoder_lr={enc_lr_eff:.2e}  head_lr={lr:.2e}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = BCETverskyLoss(bce_weight=1.0, tversky_weight=1.0,
                             alpha=0.3, beta=0.7, pos_weight=3.0)

    CKPT_DIR.mkdir(exist_ok=True)
    mlflow.set_experiment("deforestation")
    mlflow.start_run(run_name=out_name)
    mlflow.set_tags({"stage": "train", "model_kind": "clay"})
    mlflow.log_params({
        "model_kind": "clay",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "crop_size": crop_size,
        "crops_per_tile": crops_per_tile,
        "seed": seed,
        "val_tiles": ",".join(val_tile_ids),
        "freeze_encoder": freeze_encoder,
        "encoder_lr": enc_lr_eff,
        "pretrained_ckpt": Path(pretrained_ckpt).name,
        "params_total": total,
        "params_trainable": trainable,
        "pseudo_dir": pseudo_dir or "",
        "pseudo_crops_per_tile": pseudo_crops_per_tile if pseudo_dir else 0,
        "min_s2_coverage": min_s2_coverage,
        "skipped_train_tiles": ",".join(skipped_train),
    })
    best_f1 = -1.0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for batch in loader:
            pre, post, tgt, w, meta = batch
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            logits = model(pre, post, meta)
            loss = loss_fn(logits, tgt, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            losses.append(loss.item())
        sched.step()

        train_loss = float(np.mean(losses))
        metrics = evaluate(model, val_ds, device=device,
                           crop=eval_crop, overlap=eval_overlap)
        dt = time.time() - t0
        print(
            f"ep {ep:03d}  loss {train_loss:.4f}  "
            f"val_iou {metrics['macro_iou']:.3f}  val_f1 {metrics['macro_f1']:.3f}  "
            f"lr {opt.param_groups[0]['lr']:.2e}  ({dt:.1f}s)",
            flush=True,
        )
        history.append({"epoch": ep, "train_loss": train_loss, **metrics})

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_macro_iou": metrics["macro_iou"],
            "lr": opt.param_groups[0]["lr"],
        }, step=ep)
        for tid, m in metrics["per_tile"].items():
            mlflow.log_metric(f"f1_{tid}", m["f1"], step=ep)
            mlflow.log_metric(f"iou_{tid}", m["iou"], step=ep)

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            ckpt_path = CKPT_DIR / f"{out_name}_best.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": ep,
                "metrics": metrics,
                "val_tiles": list(val_tile_ids),
                "model_kind": "clay",
                "freeze_encoder": freeze_encoder,
            }, ckpt_path)
            print(f"  ↳ saved new best (f1={best_f1:.3f})")
            mlflow.log_metric("best_val_macro_f1", best_f1, step=ep)

    hist_path = CKPT_DIR / f"{out_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=float)
    print(f"done. best val_f1 = {best_f1:.3f}")
    mlflow.log_artifact(str(hist_path))
    mlflow.log_artifact(str(CKPT_DIR / f"{out_name}_best.pt"))
    mlflow.end_run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--crops_per_tile", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_name", default="clay_frozen_v1")
    ap.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--encoder_lr", type=float, default=None,
                    help="LR for encoder params when freeze_encoder=False. Default: lr*0.01.")
    ap.add_argument("--pretrained_ckpt", default=None)
    ap.add_argument("--pseudo_dir", default=None)
    ap.add_argument("--pseudo_crops_per_tile", type=int, default=16)
    ap.add_argument("--eval_crop", type=int, default=256)
    ap.add_argument("--eval_overlap", type=int, default=64)
    ap.add_argument("--min_s2_coverage", type=float, default=0.10,
                    help="Skip tiles where fewer than this fraction of S2 pixels are non-zero.")
    args = ap.parse_args()
    main(**vars(args))
