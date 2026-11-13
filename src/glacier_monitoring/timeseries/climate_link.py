"""Relate late-summer snow cover on the glaciers to summer temperature."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bright_snow_fraction(features: np.ndarray, glacier: np.ndarray, valid: np.ndarray) -> float:
    """Share of the glacier covered by bright snow (high visible reflectance and high NDSI).

    Snow is much brighter than bare glacier ice in the visible bands, so the share of bright pixels
    on the glacier at the end of summer is a simple proxy for how much winter snow survived.
    """
    region = glacier.astype(bool) & valid
    if region.sum() == 0:
        return float("nan")
    visible = features[0:3].mean(axis=0)
    ndsi = features[6]
    snow = (visible > 0.5) & (ndsi > 0.4)
    return float(snow[region].mean())


def snow_temperature_regression(df: pd.DataFrame) -> dict[str, float]:
    """OLS of snow fraction on summer temperature anomaly and acquisition day-of-year.

    Snow fraction is demeaned per glacier (glacier fixed effects), so the model asks: in years that
    were warmer than usual *for that glacier's region*, was less snow left? Day-of-year is included
    because a September image naturally shows less snow than a July image of the same summer.
    Glacier-years from the same summer are not independent, so the p-value is optimistic.
    """
    d = df.dropna(subset=["snow_fraction", "jjas_anomaly_c", "day_of_year"]).copy()
    d["snow_dm"] = d["snow_fraction"] - d.groupby("glacier_id")["snow_fraction"].transform("mean")
    d["temp_dm"] = d["jjas_anomaly_c"] - d.groupby("glacier_id")["jjas_anomaly_c"].transform("mean")
    d["doy_dm"] = d["day_of_year"] - d.groupby("glacier_id")["day_of_year"].transform("mean")

    x = np.column_stack([d["temp_dm"], d["doy_dm"]])
    y = d["snow_dm"].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    n, k = len(y), x.shape[1] + d["glacier_id"].nunique()
    sigma2 = float(resid @ resid) / max(n - k, 1)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(x.T @ x)))
    t_temp = beta[0] / se[0]
    p_temp = float(2 * stats.t.sf(abs(t_temp), df=max(n - k, 1)))

    yearly = d.groupby("year")[["snow_dm", "temp_dm"]].mean()
    r_year, p_year = stats.pearsonr(yearly["temp_dm"], yearly["snow_dm"])
    return {
        "n_glacier_years": float(n),
        "slope_per_degC": float(beta[0]),
        "slope_se": float(se[0]),
        "p_value": p_temp,
        "slope_per_day": float(beta[1]),
        "n_years": float(len(yearly)),
        "yearly_mean_pearson_r": float(r_year),
        "yearly_mean_pearson_p": float(p_year),
    }
