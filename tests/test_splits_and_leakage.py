"""Glacier-level splits, tile extraction and train-only normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from glacier_monitoring.data.dataset import (
    GlacierData,
    assert_no_glacier_leakage,
    compute_channel_stats,
    extract_tiles,
    label_years,
    normalize,
)
from glacier_monitoring.data.ingest import GLACIERS


def _glacier(gid: str, region: str, split: str, n: int = 128, gi5_year: int = 2022) -> GlacierData:
    rng = np.random.default_rng(abs(hash(gid)) % 1000)
    g = GlacierData(
        meta={"id": gid, "region": region, "gi5_year": gi5_year},
        split=split,
        dem=np.full((n, n), 3000.0) + rng.normal(0, 5, (n, n)),
        label=np.zeros((n, n), dtype=np.uint8),
        target=np.zeros((n, n), dtype=np.uint8),
        region=np.zeros((n, n), dtype=np.uint8),
        austria=np.ones((n, n), dtype=bool),
    )
    for yr in (2021, 2022, 2023, 2024):
        g.bands[yr] = rng.integers(500, 6000, (6, n, n)).astype(np.uint16)
        g.scl[yr] = np.full((n, n), 4, dtype=np.uint8)
    return g


def test_study_design_holds_out_whole_regions() -> None:
    regions = {
        s: {g["region"] for g in GLACIERS if g["split_subset"] == s}
        for s in ("train", "val", "test_a")
    }
    assert regions["test_a"] == {"Silvretta", "Dachstein"}
    assert not regions["test_a"] & (regions["train"] | regions["val"])
    assert len({g["id"] for g in GLACIERS}) == len(GLACIERS)


def test_leakage_check_rejects_shared_region() -> None:
    ok = {"a": _glacier("a", "Oetztal", "train"), "b": _glacier("b", "Silvretta", "test")}
    assert_no_glacier_leakage(ok)
    bad = {**ok, "c": _glacier("c", "Silvretta", "train")}
    with pytest.raises(AssertionError):
        assert_no_glacier_leakage(bad)


def test_label_year_window() -> None:
    g = _glacier("a", "Oetztal", "train", gi5_year=2022)
    assert label_years(g, [2017, 2021, 2022, 2023, 2025], 0) == [2022]
    assert label_years(g, [2017, 2021, 2022, 2023, 2025], 1) == [2021, 2022, 2023]


def test_tiles_skip_cloudy_windows() -> None:
    g = _glacier("a", "Oetztal", "train", n=128)
    x, y = extract_tiles([g], [2022], 0, 20.0, tile=64, stride=64)
    assert x.shape == (4, 14, 64, 64) and y.shape == (4, 64, 64)
    g.scl[2022][:64, :64] = 9  # high-probability cloud
    x, _ = extract_tiles([g], [2022], 0, 20.0, tile=64, stride=64)
    assert x.shape[0] == 3


def test_normalisation_uses_given_statistics_only() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 2.0, (20, 14, 8, 8)).astype(np.float32)
    test = rng.normal(50.0, 2.0, (5, 14, 8, 8)).astype(np.float32)
    stats = compute_channel_stats(train)
    assert abs(normalize(train, stats).mean()) < 1e-4
    assert normalize(test, stats).mean() > 10  # test data is not re-centred


def test_pixels_outside_austria_are_invalid() -> None:
    g = _glacier("a", "Silvretta", "test", n=64)
    g.austria[:, :32] = False  # Swiss side of the border
    assert g.valid(2022)[:, :32].sum() == 0
    assert g.valid(2022)[:, 32:].all()
