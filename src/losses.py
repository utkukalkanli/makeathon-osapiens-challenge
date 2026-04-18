"""Loss: BCE + Tversky. Tversky (β>0.5) punishes false negatives harder,
which matters when positives are rare and recall is the main failure mode."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCETverskyLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        tversky_weight: float = 1.0,
        alpha: float = 0.3,  # FP weight
        beta: float = 0.7,   # FN weight (higher => penalise missed deforestation more)
        pos_weight: float = 3.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.tv_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``weight`` broadcasts to ``target`` (0/1); pixels with weight=0
        are masked out of both BCE and Tversky (used for pseudo-label ignore)."""
        pw = torch.tensor(self.pos_weight, device=logits.device)
        if weight is None:
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
            probs = torch.sigmoid(logits)
            tp = (probs * target).sum(dim=(-2, -1))
            fp = (probs * (1 - target)).sum(dim=(-2, -1))
            fn = ((1 - probs) * target).sum(dim=(-2, -1))
        else:
            bce_map = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pw, reduction="none"
            )
            denom = weight.sum().clamp_min(1.0)
            bce = (bce_map * weight).sum() / denom
            probs = torch.sigmoid(logits)
            w = weight
            tp = (probs * target * w).sum(dim=(-2, -1))
            fp = (probs * (1 - target) * w).sum(dim=(-2, -1))
            fn = ((1 - probs) * target * w).sum(dim=(-2, -1))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tv_loss = 1.0 - tversky.mean()
        return self.bce_weight * bce + self.tv_weight * tv_loss
