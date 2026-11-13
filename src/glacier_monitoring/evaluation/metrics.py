"""Pixel, boundary and area metrics, plus a glacier-level bootstrap for pooled IoU."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def compute_pixel_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    """Compute IoU, F1 (Dice), Precision, and Recall on binary masks of arbitrary shape."""
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    tp = float(np.logical_and(pred, true).sum())
    fp = float(np.logical_and(pred, ~true).sum())
    fn = float(np.logical_and(~pred, true).sum())

    union = tp + fp + fn
    iou = tp / union if union > 0 else (1.0 if (fp == 0 and fn == 0) else 0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 1.0

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def _extract_boundary(mask_2d: np.ndarray) -> np.ndarray:
    """Extract 1-pixel morphological contour of a 2D binary mask."""
    b = mask_2d.astype(bool)
    if not np.any(b):
        return np.zeros_like(b, dtype=bool)
    eroded = binary_erosion(b, border_value=0)
    return b & (~eroded)


def compute_boundary_f1(
    pred_2d: np.ndarray,
    true_2d: np.ndarray,
    tolerance_px: int = 2,
) -> float:
    """Compute Boundary F1 (BF1) score within a pixel tolerance band."""
    b_pred = _extract_boundary(pred_2d)
    b_true = _extract_boundary(true_2d)

    if not np.any(b_pred) and not np.any(b_true):
        return 1.0
    if not np.any(b_pred) or not np.any(b_true):
        return 0.0

    struct = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), dtype=bool)
    dil_true = binary_dilation(b_true, structure=struct)
    dil_pred = binary_dilation(b_pred, structure=struct)

    prec = float(np.logical_and(b_pred, dil_true).sum()) / float(b_pred.sum())
    rec = float(np.logical_and(b_true, dil_pred).sum()) / float(b_true.sum())
    if prec + rec == 0.0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def evaluate_glacier(
    prob: np.ndarray,
    label: np.ndarray,
    target_region: np.ndarray,
    valid: np.ndarray,
    pixel_area_km2: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Metrics for one glacier image, ignoring cloudy / no-data pixels.

    IoU, F1 and boundary F1 are computed over the whole grid against all inventory ice. Area error
    is computed inside ``target_region`` only, i.e. for the glacier the grid was built around.
    """
    pred = (prob >= threshold) & valid
    true = label.astype(bool) & valid
    px = compute_pixel_metrics(pred[valid], true[valid])
    pred_area = float((pred & target_region).sum()) * pixel_area_km2
    true_area = float((true & target_region).sum()) * pixel_area_km2
    return {
        **px,
        "boundary_f1": compute_boundary_f1(pred, true, tolerance_px=2),
        "tp": float((pred & true).sum()),
        "fp": float((pred & ~true).sum()),
        "fn": float((~pred & true & valid).sum()),
        "pred_area_km2": pred_area,
        "true_area_km2": true_area,
        "area_rel_error_pct": 100.0 * (pred_area - true_area) / max(true_area, 1e-9),
    }


def pooled_iou(tp: Sequence[float], fp: Sequence[float], fn: Sequence[float]) -> float:
    """IoU over all pixels of several glaciers together."""
    tp_s, fp_s, fn_s = float(np.sum(tp)), float(np.sum(fp)), float(np.sum(fn))
    return tp_s / max(tp_s + fp_s + fn_s, 1.0)


def bootstrap_pooled_iou(
    tp: Sequence[float],
    fp: Sequence[float],
    fn: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% interval for pooled IoU, resampling whole glaciers (pixels within a glacier are not
    independent, so resampling pixels would give far too narrow intervals)."""
    tp_a, fp_a, fn_a = (np.asarray(v, dtype=np.float64) for v in (tp, fp, fn))
    rng = np.random.default_rng(seed)
    n = len(tp_a)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(pooled_iou(tp_a[idx], fp_a[idx], fn_a[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
