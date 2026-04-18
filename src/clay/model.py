"""Clay v1.5 ViT-B encoder + siamese change-detection segmentation head.

Uses the pretrained Clay encoder (frozen by default) on Sentinel-2 bands.
The seg head is a lightweight conv decoder that upsamples from the 8x-subsampled
token grid back to the input resolution.

Siamese fusion: run the encoder on `pre` and `post` separately, drop CLS tokens,
reshape the remaining tokens into a spatial grid, fuse via
``concat([post_grid, |post_grid - pre_grid|])`` and decode to a 1-channel logit.

Band mapping: our loader provides 12 S2 bands (B01..B12). Clay uses 10 bands
(drops B01, B09). The wrapper reindexes accordingly. S1 is omitted from the Clay
branch for now — the siamese/temporal U-Nets cover the radar signal in the
final ensemble.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from box import Box
from einops import rearrange

from .upstream.model import Encoder

# Path to the metadata.yaml copied from clay-foundation/model.
METADATA_PATH = Path(__file__).parent / "metadata.yaml"
PLATFORM = "sentinel-2-l2a"

# Channel indices in our loader (src/clay/dataset.py) that map to Clay's 10 S2 bands.
# Loader band order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
# Clay band order : blue(B02), green(B03), red(B04), rededge1..3(B05,6,7),
#                   nir(B08), nir08(B8A), swir16(B11), swir22(B12)
CLAY_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]


def _load_clay_metadata():
    return Box(yaml.safe_load(open(METADATA_PATH)))


def _build_latlon_tensor(lat: float, lon: float, device, dtype=torch.float32):
    """Clay expects latlon=[sin(π*lat/180), cos(π*lat/180), sin(π*lon/180), cos(π*lon/180)]."""
    lat_r = math.pi * lat / 180.0
    lon_r = math.pi * lon / 180.0
    return torch.tensor(
        [math.sin(lat_r), math.cos(lat_r), math.sin(lon_r), math.cos(lon_r)],
        device=device, dtype=dtype,
    )


def _encode_latlon_batch(latlon_raw: torch.Tensor) -> torch.Tensor:
    """Batched equivalent: (B, 2) [lat, lon] -> (B, 4) sin/cos encoding."""
    lat_r = math.pi * latlon_raw[:, 0] / 180.0
    lon_r = math.pi * latlon_raw[:, 1] / 180.0
    return torch.stack(
        [torch.sin(lat_r), torch.cos(lat_r), torch.sin(lon_r), torch.cos(lon_r)],
        dim=-1,
    )


def _build_time_tensor(week: int, hour: int, device, dtype=torch.float32):
    """Clay time=[sin(2π*week/52), cos(2π*week/52), sin(2π*hour/24), cos(2π*hour/24)].

    For annual composites we pass the mid-year week (26) and mid-day hour (12).
    """
    w = 2 * math.pi * week / 52.0
    h = 2 * math.pi * hour / 24.0
    return torch.tensor(
        [math.sin(w), math.cos(w), math.sin(h), math.cos(h)],
        device=device, dtype=dtype,
    )


def _s2_normalize(bands: torch.Tensor, metadata: Box) -> torch.Tensor:
    """Normalize S2 reflectance with Clay's per-band mean/std.

    Input: (B, 10, H, W) in raw uint16 reflectance (already cast to float).
    """
    names = metadata[PLATFORM].band_order
    mean = torch.tensor([metadata[PLATFORM].bands.mean[n] for n in names],
                        device=bands.device, dtype=bands.dtype).view(1, -1, 1, 1)
    std = torch.tensor([metadata[PLATFORM].bands.std[n] for n in names],
                       device=bands.device, dtype=bands.dtype).view(1, -1, 1, 1)
    return (bands - mean) / std


class ClayEncoderWrapper(nn.Module):
    """Frozen-by-default Clay ViT-B encoder with mask_ratio=0 fine-tuning mode."""

    def __init__(
        self,
        patch_size: int = 8,
        dim: int = 1024,      # clay_mae_large (v1.5 checkpoint)
        depth: int = 24,
        heads: int = 16,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        freeze: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.encoder = Encoder(
            mask_ratio=0.0,
            patch_size=patch_size,
            shuffle=False,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
        )
        self.metadata = _load_clay_metadata()
        self._waves_cache = None  # lazily built from metadata
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    @property
    def waves(self) -> torch.Tensor:
        if self._waves_cache is None:
            names = self.metadata[PLATFORM].band_order
            wl = [self.metadata[PLATFORM].bands.wavelength[n] for n in names]
            self._waves_cache = torch.tensor(wl, dtype=torch.float32)
        return self._waves_cache

    @property
    def gsd(self) -> torch.Tensor:
        return torch.tensor(self.metadata[PLATFORM].gsd, dtype=torch.float32)

    def load_pretrained(self, ckpt_path: str, verbose: bool = True):
        """Load Clay v1.5 MAE encoder weights from a lightning .ckpt."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = sd["state_dict"] if "state_dict" in sd else sd
        enc_sd = {}
        prefix = "model.encoder."
        for k, v in state.items():
            if k.startswith(prefix):
                enc_sd[k[len(prefix):]] = v
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        if verbose:
            print(f"  loaded {len(enc_sd)} encoder tensors  "
                  f"missing={len(missing)}  unexpected={len(unexpected)}")
        return missing, unexpected

    def forward(self, pixels_raw: torch.Tensor, latlon: torch.Tensor,
                time: torch.Tensor | None = None) -> torch.Tensor:
        """pixels_raw: (B, 12, H, W) from the loader (B01..B12).
        latlon     : (B, 2) raw [lat, lon] degrees, or (B, 4) already encoded.
        time       : (B, 4) as built by _build_time_tensor. Defaults to mid-year.

        Returns token grid (B, D, H/ps, W/ps).
        """
        B, _, H, W = pixels_raw.shape
        bands = pixels_raw[:, CLAY_BAND_INDICES, :, :]  # (B, 10, H, W)
        bands = _s2_normalize(bands, self.metadata)
        if latlon.shape[-1] == 2:
            latlon = _encode_latlon_batch(latlon)
        if time is None:
            time = _build_time_tensor(26, 12, pixels_raw.device, pixels_raw.dtype).expand(B, 4)
        datacube = {
            "pixels": bands,
            "time": time,
            "latlon": latlon,
            "gsd": self.gsd.to(pixels_raw.device),
            "waves": self.waves.to(pixels_raw.device),
        }
        encoded, *_ = self.encoder(datacube)  # (B, 1 + L, D)
        tokens = encoded[:, 1:]               # drop CLS
        Hp, Wp = H // self.patch_size, W // self.patch_size
        grid = rearrange(tokens, "b (h w) d -> b d h w", h=Hp, w=Wp)
        return grid


class SegHead(nn.Module):
    """Simple conv decoder: fused tokens (B, 2D, Hp, Wp) -> (B, 1, H, W) logits."""

    def __init__(self, in_dim: int, patch_size: int = 8, mid: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, mid, 1),
            nn.GroupNorm(8, mid),
            nn.GELU(),
        )
        # Upsample by patch_size in log2 steps of 2 (8 -> 3 stages).
        n_stages = int(math.log2(patch_size))
        layers = []
        c_in = mid
        for _ in range(n_stages):
            c_out = max(mid // 2, 32)
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2),
                nn.GroupNorm(8, c_out),
                nn.GELU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.GELU(),
            ]
            c_in = c_out
        self.up = nn.Sequential(*layers)
        self.head = nn.Conv2d(c_in, 1, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.up(x)
        return self.head(x)


class ClaySiameseSegmenter(nn.Module):
    """Full pipeline: pre/post -> Clay encoder -> siamese fusion -> seg head."""

    def __init__(
        self,
        patch_size: int = 8,
        dim: int = 1024,
        freeze_encoder: bool = True,
        mid: int = 128,
    ):
        super().__init__()
        self.encoder = ClayEncoderWrapper(
            patch_size=patch_size, dim=dim, freeze=freeze_encoder,
        )
        self.head = SegHead(in_dim=2 * dim, patch_size=patch_size, mid=mid)

    def load_pretrained(self, ckpt_path: str, verbose: bool = True):
        return self.encoder.load_pretrained(ckpt_path, verbose=verbose)

    @property
    def _encoder_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.encoder.encoder.parameters())

    def forward(self, pre: torch.Tensor, post: torch.Tensor,
                latlon: torch.Tensor) -> torch.Tensor:
        """pre, post: (B, 12, H, W); latlon: (B, 4). Returns (B, 1, H, W) logits."""
        if self._encoder_frozen:
            with torch.no_grad():
                pre_grid = self.encoder(pre, latlon)
                post_grid = self.encoder(post, latlon)
        else:
            pre_grid = self.encoder(pre, latlon)
            post_grid = self.encoder(post, latlon)
        fused = torch.cat([post_grid, (post_grid - pre_grid).abs()], dim=1)
        return self.head(fused)


def count_params(m: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    m = ClaySiameseSegmenter(freeze_encoder=True).to(device)
    total, trainable = count_params(m)
    print(f"total params: {total/1e6:.1f}M  trainable: {trainable/1e6:.1f}M")
    pre = torch.randn(1, 12, 256, 256, device=device) * 1000 + 1500
    post = torch.randn(1, 12, 256, 256, device=device) * 1000 + 1500
    latlon = _build_latlon_tensor(1.22, -74.42, device).unsqueeze(0)
    with torch.no_grad():
        y = m(pre, post, latlon)
    print(f"output: {y.shape}")
