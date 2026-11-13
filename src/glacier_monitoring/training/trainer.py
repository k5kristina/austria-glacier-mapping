"""Reproducible training engine with AdamW, cosine warmup schedule, and early stopping."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from glacier_monitoring.data.dataset import GlacierTileDataset
from glacier_monitoring.models.cnn import ResUNet
from glacier_monitoring.training.losses import SegmentationCriterion


@dataclass
class TrainingResult:
    """Summary of a completed model training run."""

    best_val_iou: float
    best_epoch: int
    train_losses: list[float]
    val_ious: list[float]


def set_global_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """CUDA if present, else Apple-silicon GPU (MPS), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_batch_iou(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).long()
    inter = ((preds == 1) & (targets == 1)).sum().item()
    union = ((preds == 1) | (targets == 1)).sum().item()
    return float(inter / max(union, 1))


def train_segmentation_model(
    model: ResUNet,
    train_features: np.ndarray,
    train_masks: np.ndarray,
    val_features: np.ndarray,
    val_masks: np.ndarray,
    epochs: int = 18,
    batch_size: int = 16,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    patience: int = 6,
    loss_mode: str = "dice_ce",
    seed: int = 42,
) -> TrainingResult:
    """Train a segmentation model with AdamW, warmup + cosine schedule and early stopping on val IoU."""
    set_global_seed(seed)
    device = select_device()
    model.to(device)

    train_ds = GlacierTileDataset(train_features, train_masks, augment=True, seed=seed)
    val_ds = GlacierTileDataset(val_features, val_masks, augment=False, seed=seed)

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = SegmentationCriterion(mode=loss_mode, fg_weight=2.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(warmup_epochs, 1))
        progress = float(epoch_idx - warmup_epochs) / float(max(epochs - warmup_epochs, 1))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_iou = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    train_losses: list[float] = []
    val_ious: list[float] = []
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(n_batches, 1)
        train_losses.append(epoch_loss)

        # Validation evaluation
        model.eval()
        all_probs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                v_probs = model.predict_proba(vx)
                all_probs.append(v_probs.cpu())
                all_targets.append(vy)

        probs_cat = torch.cat(all_probs, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        val_iou = _compute_batch_iou(probs_cat, targets_cat)
        val_ious.append(val_iou)

        if val_iou > best_val_iou + 1e-5:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        best_val_iou=best_val_iou,
        best_epoch=best_epoch,
        train_losses=train_losses,
        val_ious=val_ious,
    )
