"""Train the siamese change-detection U-Net on AEF embeddings."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import AEFCropDataset, FullTileDataset
from .inventory import train_tiles
from .losses import BCETverskyLoss
from .model import SiameseChangeUNet, count_params

# Fixed val tiles chosen to span regions and positive-rate levels
# (see `python -m src.labels` / dataset inspection).
DEFAULT_VAL_TILES = ("18NXH_6_8", "48PXC_7_7", "48QVE_3_0")

CKPT_DIR = Path("checkpoints")


def _iou_f1(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    """Compute IoU and F1 for a single binary mask pair."""
    p = (pred > thr)
    t = target > 0.5
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    iou = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return iou, f1


@torch.no_grad()
def evaluate(model, val_ds: FullTileDataset, device: str, crop: int = 512, overlap: int = 64):
    """Tiled inference over full AEF tiles; returns dict of metrics."""
    model.eval()
    per_tile = {}
    for tile_id, pre, post, hard in val_ds:
        pre = pre.to(device)
        post = post.to(device)
        prob = predict_full(model, pre, post, crop=crop, overlap=overlap)
        iou, f1 = _iou_f1(prob.cpu(), hard)
        per_tile[tile_id] = {"iou": iou, "f1": f1, "pos_rate": float(hard.mean())}
    macro_iou = float(np.mean([v["iou"] for v in per_tile.values()]))
    macro_f1 = float(np.mean([v["f1"] for v in per_tile.values()]))
    return {"per_tile": per_tile, "macro_iou": macro_iou, "macro_f1": macro_f1}


@torch.no_grad()
def predict_full(
    model,
    pre: torch.Tensor,
    post: torch.Tensor,
    crop: int = 512,
    overlap: int = 64,
) -> torch.Tensor:
    """Sliding-window inference across a full tile. Returns H,W probability map."""
    _, H, W = pre.shape
    stride = crop - overlap
    prob = torch.zeros((H, W), device=pre.device)
    weight = torch.zeros((H, W), device=pre.device)
    ys = list(range(0, max(1, H - crop + 1), stride))
    xs = list(range(0, max(1, W - crop + 1), stride))
    if ys[-1] + crop < H:
        ys.append(H - crop)
    if xs[-1] + crop < W:
        xs.append(W - crop)
    for y in ys:
        for x in xs:
            pre_c = pre[:, y : y + crop, x : x + crop].unsqueeze(0)
            post_c = post[:, y : y + crop, x : x + crop].unsqueeze(0)
            logits = model(pre_c, post_c)[0, 0]
            prob[y : y + crop, x : x + crop] += torch.sigmoid(logits)
            weight[y : y + crop, x : x + crop] += 1
    return prob / weight.clamp_min(1)


def main(
    val_tile_ids: tuple[str, ...] = DEFAULT_VAL_TILES,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    crop_size: int = 256,
    crops_per_tile: int = 32,
    num_workers: int = 4,
    seed: int = 0,
    out_name: str = "siamese_unet",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_tiles = train_tiles()
    tr = [t for t in all_tiles if t.tile_id not in val_tile_ids]
    va = [t for t in all_tiles if t.tile_id in val_tile_ids]
    print(f"train tiles: {len(tr)}  val tiles: {[t.tile_id for t in va]}")

    train_ds = AEFCropDataset(
        tr, crop_size=crop_size, crops_per_tile=crops_per_tile,
        positive_crop_prob=0.75, augment=True,
    )
    val_ds = FullTileDataset(va)

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, persistent_workers=num_workers > 0,
    )

    model = SiameseChangeUNet(in_channels=64, base=32).to(device)
    print(f"model params: {count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = BCETverskyLoss(bce_weight=1.0, tversky_weight=1.0,
                             alpha=0.3, beta=0.7, pos_weight=3.0)

    CKPT_DIR.mkdir(exist_ok=True)
    best_f1 = -1.0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for pre, post, tgt in loader:
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            logits = model(pre, post)
            loss = loss_fn(logits, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        train_loss = float(np.mean(losses))
        metrics = evaluate(model, val_ds, device=device)
        dt = time.time() - t0
        print(
            f"ep {ep:03d}  loss {train_loss:.4f}  "
            f"val_iou {metrics['macro_iou']:.3f}  val_f1 {metrics['macro_f1']:.3f}  "
            f"lr {opt.param_groups[0]['lr']:.2e}  ({dt:.1f}s)",
            flush=True,
        )
        history.append({"epoch": ep, "train_loss": train_loss, **metrics})

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save({
                "model": model.state_dict(),
                "epoch": ep,
                "metrics": metrics,
                "val_tiles": list(val_tile_ids),
            }, CKPT_DIR / f"{out_name}_best.pt")
            print(f"  ↳ saved new best (f1={best_f1:.3f})")

    with open(CKPT_DIR / f"{out_name}_history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)
    print(f"done. best val_f1 = {best_f1:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--crops_per_tile", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_name", default="siamese_unet")
    args = ap.parse_args()
    main(**vars(args))
