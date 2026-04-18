"""Siamese change-detection U-Net operating on 64-D AEF embeddings.

Design:
- A small shared encoder is applied independently to `pre` and `post`.
- At each encoder scale, features are combined via
  ``[post, |post - pre|]`` (concat) — this preserves the "post" semantics
  while making the change signal explicit.
- A U-Net decoder upsamples back to input resolution and produces a
  1-channel logit map.

Why this architecture (given only 16 training tiles):
- The AEF embeddings already encode strong semantic priors. The model
  only needs to learn "what kind of change looks like deforestation",
  not "what is forest". Keeping the encoder small (~0.5M params total)
  lowers the risk of overfitting.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.block = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd-sized skips via bilinear crop/pad.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SiameseChangeUNet(nn.Module):
    """Siamese U-Net for binary change detection on AEF embeddings."""

    def __init__(self, in_channels: int = 64, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        # Shared siamese encoder (applied to pre AND post separately).
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = Down(c1, c2)
        self.enc3 = Down(c2, c3)
        self.enc4 = Down(c3, c4)

        # Decoder fuses [post_i, |post_i - pre_i|] at each scale (=> 2*c_i).
        self.up3 = Up(in_ch=2 * c4, skip_ch=2 * c3, out_ch=c3)
        self.up2 = Up(in_ch=c3, skip_ch=2 * c2, out_ch=c2)
        self.up1 = Up(in_ch=c2, skip_ch=2 * c1, out_ch=c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e1, e2, e3, e4

    @staticmethod
    def _fuse(pre_f, post_f):
        return torch.cat([post_f, (post_f - pre_f).abs()], dim=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        pre1, pre2, pre3, pre4 = self._encode(pre)
        post1, post2, post3, post4 = self._encode(post)

        f4 = self._fuse(pre4, post4)
        f3 = self._fuse(pre3, post3)
        f2 = self._fuse(pre2, post2)
        f1 = self._fuse(pre1, post1)

        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        return self.head(d1)


class TemporalChangeUNet(nn.Module):
    """Temporal extension of SiameseChangeUNet.

    Input is ``(B, T, C, H, W)`` where T is the number of AEF years (e.g. 6
    for 2020..2025). A shared encoder runs independently on each year; at
    each encoder scale we fuse the time axis via

        [ f_{T-1},  max_{t>=1} |f_t - f_0|,  mean_{t>=1} |f_t - f_0| ]

    - ``f_{T-1}``: features of the latest observation (semantic "what's there now").
    - ``max |f_t - f_0|``: peak deviation from the pre-2020 reference — strong
      signal for "an event happened at some point".
    - ``mean |f_t - f_0|``: average deviation — helps distinguish transient
      disturbances (single spike) from sustained change (consistent deviation).

    The decoder is unchanged from SiameseChangeUNet.
    """

    def __init__(self, in_channels: int = 64, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = Down(c1, c2)
        self.enc3 = Down(c2, c3)
        self.enc4 = Down(c3, c4)

        # Fused channel count is 3 * c at each scale (last + max_diff + mean_diff).
        self.up3 = Up(in_ch=3 * c4, skip_ch=3 * c3, out_ch=c3)
        self.up2 = Up(in_ch=c3, skip_ch=3 * c2, out_ch=c2)
        self.up1 = Up(in_ch=c2, skip_ch=3 * c1, out_ch=c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e1, e2, e3, e4

    @staticmethod
    def _temporal_fuse(feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, C, H, W) -> (B, 3*C, H, W)."""
        # diffs with respect to the first frame (t=0, pre-year)
        f0 = feats[:, :1]                   # (B, 1, C, H, W)
        diffs = (feats[:, 1:] - f0).abs()  # (B, T-1, C, H, W)
        last = feats[:, -1]                # (B, C, H, W)
        max_diff, _ = diffs.max(dim=1)     # (B, C, H, W)
        mean_diff = diffs.mean(dim=1)      # (B, C, H, W)
        return torch.cat([last, max_diff, mean_diff], dim=1)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        # series: (B, T, C, H, W) -> reshape to run encoder once on B*T
        B, T, C, H, W = series.shape
        x = series.reshape(B * T, C, H, W)
        e1, e2, e3, e4 = self._encode(x)

        def un(e):
            return e.reshape(B, T, *e.shape[1:])

        f1 = self._temporal_fuse(un(e1))
        f2 = self._temporal_fuse(un(e2))
        f3 = self._temporal_fuse(un(e3))
        f4 = self._temporal_fuse(un(e4))

        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        return self.head(d1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = SiameseChangeUNet()
    print(f"siamese params: {count_params(m):,}")
    pre = torch.randn(2, 64, 256, 256)
    post = torch.randn(2, 64, 256, 256)
    y = m(pre, post)
    print("siamese output:", y.shape)

    t = TemporalChangeUNet()
    print(f"temporal params: {count_params(t):,}")
    series = torch.randn(2, 6, 64, 256, 256)
    y = t(series)
    print("temporal output:", y.shape)
