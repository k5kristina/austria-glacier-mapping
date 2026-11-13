"""Trend statistics and the snow / temperature regression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glacier_monitoring.timeseries.climate_link import (
    bright_snow_fraction,
    snow_temperature_regression,
)
from glacier_monitoring.timeseries.trends import mann_kendall_test, theil_sen_trend


def test_mann_kendall_detects_monotonic_decline() -> None:
    years = np.arange(2017, 2026)
    res = mann_kendall_test(years, 10 - 0.3 * (years - 2017))
    assert res["kendall_tau"] == -1.0
    assert res["p_value"] < 0.01


def test_theil_sen_is_robust_to_one_outlier_and_skips_nan() -> None:
    years = np.arange(2017, 2026)
    area = 5.0 - 0.1 * (years - 2017)
    area[3] = 9.0  # e.g. fresh snow
    area[5] = np.nan  # cloudy year
    res = theil_sen_trend(years, area)
    assert np.isclose(res["slope_per_year"], -0.1, atol=0.02)
    assert res["n_years"] == 8


def test_bright_snow_fraction() -> None:
    feats = np.zeros((14, 4, 4), dtype=np.float32)
    feats[0:3, :2] = 0.8  # bright
    feats[6] = 0.9  # high NDSI everywhere
    glacier = np.ones((4, 4), dtype=bool)
    assert bright_snow_fraction(feats, glacier, np.ones((4, 4), dtype=bool)) == 0.5


def test_regression_recovers_known_temperature_effect() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for g in range(10):
        base = rng.uniform(0.2, 0.6)
        for yr in range(2017, 2026):
            temp = rng.normal(1.5, 0.7)
            doy = rng.integers(200, 280)
            rows.append(
                {
                    "glacier_id": g,
                    "year": yr,
                    "jjas_anomaly_c": temp,
                    "day_of_year": doy,
                    "snow_fraction": base - 0.08 * temp - 0.001 * doy + rng.normal(0, 0.01),
                }
            )
    res = snow_temperature_regression(pd.DataFrame(rows))
    assert np.isclose(res["slope_per_degC"], -0.08, atol=0.01)
    assert res["p_value"] < 0.001
