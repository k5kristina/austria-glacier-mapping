"""Model shapes, baselines and evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch

from glacier_monitoring.evaluation.metrics import (
    bootstrap_pooled_iou,
    compute_boundary_f1,
    compute_pixel_metrics,
    evaluate_glacier,
    pooled_iou,
)
from glacier_monitoring.models import NDSIThresholdModel, RandomForestPixelModel, ResUNet


def test_unet_output_shape_and_probabilities() -> None:
    net = ResUNet(in_channels=14, base_channels=8)
    x = torch.randn(2, 14, 64, 48)
    assert net(x).shape == (2, 2, 64, 48)
    p = net.predict_proba(x)
    assert p.shape == (2, 64, 48)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_ndsi_threshold_recovers_separating_value() -> None:
    feats = np.zeros((1, 14, 10, 10), dtype=np.float32)
    feats[0, 6, :, :5] = 0.7
    feats[0, 6, :, 5:] = 0.1
    mask = (feats[0, 6] > 0.4).astype(np.uint8)[None]
    model = NDSIThresholdModel()
    thr = model.fit(feats, mask)
    assert 0.1 < thr <= 0.7
    assert (model.predict(feats) == mask).all()


def test_random_forest_learns_simple_rule() -> None:
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(4, 14, 16, 16)).astype(np.float32)
    mask = (feats[:, 3] > 0).astype(np.uint8)
    rf = RandomForestPixelModel(n_estimators=20, max_depth=4, seed=0).fit(feats, mask)
    assert (rf.predict(feats) == mask).mean() > 0.95


def test_pixel_metrics() -> None:
    pred = np.array([1, 1, 0, 0], dtype=bool)
    true = np.array([1, 0, 1, 0], dtype=bool)
    m = compute_pixel_metrics(pred, true)
    assert np.isclose(m["iou"], 1 / 3)
    assert np.isclose(m["f1"], 0.5)


def test_boundary_f1_perfect_and_shifted() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    assert compute_boundary_f1(mask, mask) == 1.0
    assert compute_boundary_f1(np.roll(mask, 10, axis=1), mask, tolerance_px=2) < 0.8


def test_evaluate_glacier_ignores_invalid_pixels_and_measures_area() -> None:
    label = np.zeros((10, 10), dtype=np.uint8)
    label[:, :5] = 1
    prob = label.astype(float).copy()
    prob[:, 5] = 1.0  # one column too wide
    valid = np.ones((10, 10), dtype=bool)
    region = np.ones((10, 10), dtype=bool)
    m = evaluate_glacier(prob, label, region, valid, pixel_area_km2=0.0004)
    assert np.isclose(m["iou"], 50 / 60)
    assert np.isclose(m["area_rel_error_pct"], 20.0)
    valid[:, 5] = False  # the wrong column is cloudy -> ignored
    assert np.isclose(evaluate_glacier(prob, label, region, valid, 0.0004)["iou"], 1.0)


def test_pooled_iou_and_bootstrap_interval() -> None:
    tp, fp, fn = [90, 50, 70], [10, 30, 5], [0, 20, 25]
    iou = pooled_iou(tp, fp, fn)
    assert np.isclose(iou, 210 / 300)
    lo, hi = bootstrap_pooled_iou(tp, fp, fn, n_boot=500)
    assert lo <= iou <= hi
