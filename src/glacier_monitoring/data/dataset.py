"""Assemble glacier scenes, glacier-level splits, training tiles and normalisation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from glacier_monitoring.data.features import CHANNELS, build_feature_stack
from glacier_monitoring.data.ingest import SCL_INVALID, rasterize_on_grid

SPLIT_NAMES = {"train": "train", "val": "val", "test_a": "test"}


@dataclass
class GlacierData:
    """Everything known about one glacier: grid, inventory masks and per-year imagery."""

    meta: dict[str, Any]
    split: str
    dem: np.ndarray
    label: np.ndarray  # all GI 5 ice on the grid (uint8)
    target: np.ndarray  # this glacier's GI 5 outline (uint8)
    region: np.ndarray  # area used for the time series (uint8)
    austria: np.ndarray  # pixels inside Austria, where the inventory is complete (bool)
    bands: dict[int, np.ndarray] = field(default_factory=dict)  # year -> (6, H, W) uint16
    scl: dict[int, np.ndarray] = field(default_factory=dict)  # year -> (H, W) uint8
    scenes: dict[int, dict[str, Any]] = field(default_factory=dict)  # year -> scene metadata

    @property
    def id(self) -> str:
        return str(self.meta["id"])

    @property
    def label_year(self) -> int:
        return int(self.meta["gi5_year"])

    def features(self, year: int, pixel_size_m: float) -> np.ndarray:
        return build_feature_stack(self.bands[year], self.dem, pixel_size_m)

    def valid(self, year: int) -> np.ndarray:
        """Usable pixels: inside Austria and not cloud, cloud shadow or no data."""
        return self.austria & ~np.isin(self.scl[year], SCL_INVALID)


def load_glaciers(
    outlines: dict[str, dict[str, Any]], raw_dir: Path, years: Sequence[int], pixel_size_m: float
) -> dict[str, GlacierData]:
    """Load cached rasters for all glaciers (run ``ingest.download_all`` first)."""
    glaciers: dict[str, GlacierData] = {}
    for gid, gmeta in outlines.items():
        g = GlacierData(
            meta=gmeta,
            split=SPLIT_NAMES[gmeta["split_subset"]],
            dem=np.load(raw_dir / "dem" / f"{gid}.npy"),
            label=rasterize_on_grid(gmeta["gi5_all_ice_utm"], gmeta, pixel_size_m),
            target=rasterize_on_grid(gmeta["gi5_target_utm"], gmeta, pixel_size_m),
            region=rasterize_on_grid(gmeta["region_utm"], gmeta, pixel_size_m),
            austria=rasterize_on_grid(gmeta["austria_utm"], gmeta, pixel_size_m).astype(bool),
        )
        for yr in years:
            with np.load(raw_dir / "s2" / f"{gid}_{yr}.npz") as z:
                g.bands[yr] = z["bands"]
                g.scl[yr] = z["scl"]
                g.scenes[yr] = json.loads(str(z["meta"]))
        # The inventory only covers Austria, so the time-series region stops at the border
        g.region = (g.region.astype(bool) & g.austria).astype(np.uint8)
        glaciers[gid] = g
    return glaciers


def assert_no_glacier_leakage(glaciers: dict[str, GlacierData]) -> None:
    """Each glacier belongs to exactly one split and test regions never appear in training."""
    regions_by_split: dict[str, set[str]] = {}
    for g in glaciers.values():
        regions_by_split.setdefault(g.split, set()).add(str(g.meta["region"]))
    overlap = regions_by_split.get("test", set()) & (
        regions_by_split.get("train", set()) | regions_by_split.get("val", set())
    )
    if overlap:
        raise AssertionError(f"Test regions also used for training/validation: {overlap}")


def label_years(g: GlacierData, years: Sequence[int], window: int) -> list[int]:
    """Years whose imagery is paired with the GI 5 label (mapping year +/- ``window``)."""
    return [yr for yr in years if abs(yr - g.label_year) <= window]


def extract_tiles(
    glaciers: Sequence[GlacierData],
    years: Sequence[int],
    window: int,
    pixel_size_m: float,
    tile: int = 64,
    stride: int = 32,
    max_invalid: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut ``tile`` x ``tile`` training tiles from label-year imagery; drops cloudy tiles."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for g in glaciers:
        for yr in label_years(g, years, window):
            feats = g.features(yr, pixel_size_m)
            valid = g.valid(yr)
            n = g.label.shape[0]
            starts = list(range(0, max(n - tile, 0) + 1, stride))
            for r in starts:
                for c in starts:
                    if 1.0 - valid[r : r + tile, c : c + tile].mean() > max_invalid:
                        continue
                    xs.append(feats[:, r : r + tile, c : c + tile])
                    ys.append(g.label[r : r + tile, c : c + tile])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.uint8)


def compute_channel_stats(train_tiles: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-channel mean/std computed on training tiles only."""
    return {
        name: {
            "mean": float(train_tiles[:, i].mean()),
            "std": max(float(train_tiles[:, i].std()), 1e-6),
        }
        for i, name in enumerate(CHANNELS)
    }


def normalize(features: np.ndarray, stats: dict[str, dict[str, float]]) -> np.ndarray:
    """Standardise ``(N, C, H, W)`` or ``(C, H, W)`` features with training statistics."""
    mean = np.array([stats[c]["mean"] for c in CHANNELS], dtype=np.float32)
    std = np.array([stats[c]["std"] for c in CHANNELS], dtype=np.float32)
    shape = (1, -1, 1, 1) if features.ndim == 4 else (-1, 1, 1)
    return ((features - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32)


class GlacierTileDataset(Dataset):
    """PyTorch dataset with rotations, flips and a small brightness jitter."""

    def __init__(
        self, features: np.ndarray, masks: np.ndarray, augment: bool = False, seed: int = 42
    ):
        super().__init__()
        self.features = features.astype(np.float32)
        self.masks = masks.astype(np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx].copy()  # (C, H, W)
        y = self.masks[idx].copy()  # (H, W)

        if self.augment:
            k = int(self.rng.integers(0, 4))
            if k > 0:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                y = np.rot90(y, k=k, axes=(0, 1)).copy()
            if self.rng.random() < 0.5:
                x = np.flip(x, axis=2).copy()
                y = np.flip(y, axis=1).copy()
            gain = float(self.rng.uniform(0.95, 1.05))
            bias = float(self.rng.uniform(-0.03, 0.03))
            x = x * gain + bias

        return torch.from_numpy(x), torch.from_numpy(y)
