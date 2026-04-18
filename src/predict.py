"""Predict deforestation masks for test tiles and emit submission GeoJSON."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

from .inventory import test_tiles, train_tiles
from .unet.dataset import FullTileDataset, TemporalFullTileDataset
from .unet.model import SiameseChangeUNet, TemporalChangeUNet
from .unet.train import predict_full, predict_full_temporal

SUBMISSION_DIR = Path("submission")


def _write_binary_raster(path: Path, mask: np.ndarray, ref_aef_path: Path):
    """Write a uint8 single-band GeoTIFF matching the AEF tile's georeference."""
    with rasterio.open(ref_aef_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    # AEF profile width/height may be for 64 bands; normalize.
    profile["width"] = mask.shape[1]
    profile["height"] = mask.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)


def main(
    ckpt: str = "checkpoints/siamese_unet_best.pt",
    threshold: float = 0.5,
    crop: int = 512,
    overlap: int = 128,
    split: str = "test",
    out_dir: str = "submission",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    kind = sd.get("model_kind", "siamese")
    in_ch = sd.get("in_channels", 64)
    if kind == "temporal":
        model = TemporalChangeUNet(in_channels=in_ch, base=32).to(device)
    else:
        model = SiameseChangeUNet(in_channels=in_ch, base=32).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    print(
        f"loaded {ckpt}  kind={kind}  epoch={sd.get('epoch')}  "
        f"metrics={sd.get('metrics', {}).get('macro_f1', 'n/a')}"
    )

    tiles = test_tiles() if split == "test" else train_tiles()
    ds = TemporalFullTileDataset(tiles) if kind == "temporal" else FullTileDataset(tiles)
    out_root = Path(out_dir)

    from submission_utils import raster_to_geojson

    for item in ds:
        if kind == "temporal":
            tile_id, series, _ = item
            series = series.to(device)
            prob = predict_full_temporal(model, series, crop=crop, overlap=overlap).cpu().numpy()
        else:
            tile_id, pre, post, _ = item
            pre = pre.to(device)
            post = post.to(device)
            prob = predict_full(model, pre, post, crop=crop, overlap=overlap).cpu().numpy()
        binary = (prob >= threshold).astype(np.uint8)
        print(f"{tile_id}: positives={binary.sum()} ({binary.mean()*100:.2f}%)")

        if binary.sum() == 0:
            print(f"  skip {tile_id}: no positives (threshold={threshold})")
            continue

        # Write binary raster aligned to AEF grid, then vectorise.
        tile = next(t for t in tiles if t.tile_id == tile_id)
        tif_p = out_root / "rasters" / f"{tile_id}.tif"
        _write_binary_raster(tif_p, binary, tile.aef_path(2020))

        geo_p = out_root / f"{tile_id}.geojson"
        try:
            raster_to_geojson(tif_p, output_path=geo_p, min_area_ha=0.5)
            print(f"  wrote {geo_p}")
        except ValueError as e:
            print(f"  skip {tile_id}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/siamese_unet_best.pt")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--out_dir", default="submission")
    args = ap.parse_args()
    main(**vars(args))
