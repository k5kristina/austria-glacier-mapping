"""Spectral indices, terrain derivatives and the feature stack."""

from __future__ import annotations

import numpy as np

from glacier_monitoring.data.features import (
    CHANNELS,
    build_feature_stack,
    compute_spectral_indices,
    compute_topographic_features,
)


def test_ndsi_is_high_for_snow_and_low_for_rock() -> None:
    green = np.array([0.8, 0.10])
    swir = np.array([0.1, 0.25])
    idx = compute_spectral_indices(green, np.zeros(2), np.zeros(2), swir)
    assert idx["NDSI"][0] > 0.7
    assert idx["NDSI"][1] < 0.0


def test_slope_of_inclined_plane() -> None:
    # Elevation rises 20 m per 20 m pixel towards the east -> 45 degree slope, facing west
    dem = np.tile(np.arange(10, dtype=float) * 20.0, (10, 1))
    topo = compute_topographic_features(dem, pixel_size_m=20.0)
    assert np.allclose(topo["Slope_deg"][2:-2, 2:-2], 45.0, atol=0.01)
    assert np.allclose(topo["Aspect_sin"][2:-2, 2:-2], -1.0, atol=1e-6)


def test_feature_stack_shape_and_scaling() -> None:
    bands = np.full((6, 8, 8), 5000, dtype=np.uint16)
    dem = np.full((8, 8), 3000.0)
    feats = build_feature_stack(bands, dem)
    assert feats.shape == (len(CHANNELS), 8, 8)
    assert np.allclose(feats[0], 0.5)
    assert np.allclose(feats[CHANNELS.index("elevation_km")], 3.0)


def test_north_facing_slope_is_darker_than_south_facing() -> None:
    rows = np.arange(20, dtype=float)[:, None].repeat(20, axis=1)
    north_facing = rows * 15.0  # elevation rises towards the south -> faces north
    south_facing = -rows * 15.0
    hs_n = compute_topographic_features(north_facing)["Hillshade"].mean()
    hs_s = compute_topographic_features(south_facing)["Hillshade"].mean()
    assert np.isclose(compute_topographic_features(north_facing)["Aspect_cos"][5, 5], 1.0)
    assert hs_s > hs_n
