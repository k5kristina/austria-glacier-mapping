"""Segmentation losses: weighted cross-entropy, soft Dice, and their combination."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationCriterion(nn.Module):
    """Segmentation objective supporting CE, Dice and Dice+CE."""

    def __init__(
        self,
        mode: str = "dice_ce",
        fg_weight: float = 2.2,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.fg_weight = fg_weight

    def _soft_dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs_fg = F.softmax(logits, dim=1)[:, 1, :, :]
        target_fg = (targets == 1).float()

        inter = (probs_fg * target_fg).sum(dim=(1, 2))
        denom = (probs_fg + target_fg).sum(dim=(1, 2))
        dice = (2.0 * inter + 1.0) / (denom + 1.0)
        return 1.0 - dice.mean()

    def _weighted_ce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor([1.0, self.fg_weight], device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, targets, weight=weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "ce":
            return self._weighted_ce_loss(logits, targets)
        if self.mode == "dice":
            return self._soft_dice_loss(logits, targets)
        if self.mode == "dice_ce":
            return 0.5 * self._weighted_ce_loss(logits, targets) + 0.5 * self._soft_dice_loss(
                logits, targets
            )
        raise ValueError(f"Unsupported loss mode: {self.mode}")
