"""Robust trend estimation: Theil-Sen slope and the Mann-Kendall test."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm, theilslopes


def mann_kendall_test(years: Sequence[float], values: Sequence[float]) -> dict[str, float]:
    """Perform the non-parametric Mann-Kendall monotonic trend test."""
    order = np.argsort(years)
    y = np.asarray(values, dtype=np.float64)[order]
    n = len(y)
    if n < 3:
        return {"s_stat": 0.0, "z_score": 0.0, "p_value": 1.0, "kendall_tau": 0.0}

    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = y[j] - y[k]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Tie-adjusted variance of S
    _, tie_counts = np.unique(y, return_counts=True)
    tie_term = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    var_s = max(float(var_s), 1e-9)

    if s > 0:
        z = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / np.sqrt(var_s)
    else:
        z = 0.0

    p_val = float(2.0 * (1.0 - norm.cdf(abs(z))))
    tau = float(s / (0.5 * n * (n - 1)))
    return {
        "s_stat": float(s),
        "z_score": float(z),
        "p_value": p_val,
        "kendall_tau": tau,
    }


def theil_sen_trend(years: Sequence[float], values: Sequence[float]) -> dict[str, float]:
    """Theil-Sen slope with its 95% interval, plus the Mann-Kendall test. NaN values are dropped."""
    t = np.asarray(years, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(y)
    t, y = t[keep], y[keep]
    if len(y) < 4:
        nan = float("nan")
        return {
            "slope_per_year": nan,
            "ci_low": nan,
            "ci_high": nan,
            "n_years": float(len(y)),
            "s_stat": nan,
            "z_score": nan,
            "p_value": nan,
            "kendall_tau": nan,
        }
    res = theilslopes(y, t, alpha=0.95)
    return {
        "slope_per_year": float(res.slope),
        "ci_low": float(res.low_slope),
        "ci_high": float(res.high_slope),
        "n_years": float(len(y)),
        **mann_kendall_test(t, y),
    }
