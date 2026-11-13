"""Figures for the README. Every panel is drawn from pipeline outputs; nothing is hand-set."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"ndsi": "#9aa5b1", "rf": "#e0a458", "unet": "#2f6f9f"}
LABELS = {"ndsi": "NDSI threshold", "rf": "Random Forest", "unet": "U-Net (ensemble)"}
INVENTORY_COLOR = "#ff3df2"  # magenta: visible on both snow and rock
MODEL_COLOR = "#2de1fc"


def _rgb(features: np.ndarray) -> np.ndarray:
    return np.clip(np.transpose(features[[2, 1, 0]], (1, 2, 0)) / 0.6, 0, 1)


def _outline(ax: Any, mask: np.ndarray, color: str, lw: float = 1.0) -> None:
    """Coloured contour with a dark halo so it stays visible on bright snow."""
    if mask.any() and not mask.all():
        cs = ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=lw)
        cs.set(path_effects=[pe.Stroke(linewidth=lw + 1.4, foreground="#141414"), pe.Normal()])


def plot_test_predictions(
    glaciers: dict[str, Any],
    predictors: dict[str, Callable[[Any, int], np.ndarray]],
    out: Path,
    pixel_size_m: float,
) -> None:
    """Test glaciers (unseen regions) at their inventory year: image, inventory and predictions."""
    test = [g for g in glaciers.values() if g.split == "test"]
    cols = ["Sentinel-2 RGB", "Inventory (GI 5)", *[LABELS[m] for m in ("ndsi", "rf", "unet")]]
    fig, axes = plt.subplots(len(test), len(cols), figsize=(2.6 * len(cols), 2.6 * len(test)))
    for r, g in enumerate(test):
        yr = g.label_year
        feats = g.features(yr, pixel_size_m)
        valid = g.valid(yr)
        panels = [None, g.label.astype(bool)] + [
            (predictors[m](g, yr) >= 0.5) & valid for m in ("ndsi", "rf", "unet")
        ]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(_rgb(feats))
            if panel is not None:
                overlay = np.zeros((*panel.shape, 4))
                overlay[panel] = (0.2, 0.75, 1.0, 0.45)
                ax.imshow(overlay)
                _outline(ax, g.label.astype(bool), INVENTORY_COLOR, 0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{g.meta['name']}\n{yr}", fontsize=9)
    fig.suptitle(
        "Test glaciers (Silvretta & Dachstein, never seen in training). Magenta line = inventory",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_model_comparison(summary_df: pd.DataFrame, eval_df: pd.DataFrame, out: Path) -> None:
    """Pooled test IoU with glacier-bootstrap 95% intervals, plus per-glacier dots."""
    test = summary_df[summary_df["split"] == "test"].set_index("model")
    models = ["ndsi", "rf", "unet"]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for i, m in enumerate(models):
        row = test.loc[m]
        ax.bar(i, row["pooled_iou"], color=COLORS[m], width=0.6)
        ax.errorbar(
            i,
            row["pooled_iou"],
            yerr=[
                [row["pooled_iou"] - row["pooled_iou_ci95_low"]],
                [row["pooled_iou_ci95_high"] - row["pooled_iou"]],
            ],
            color="black",
            capsize=5,
        )
        per = eval_df[(eval_df["model"] == m) & (eval_df["split"] == "test")]["iou"]
        ax.scatter(
            np.full(len(per), i) + np.linspace(-0.15, 0.15, len(per)),
            per,
            color="black",
            s=12,
            zorder=3,
            alpha=0.6,
        )
        ax.text(i, 0.02, f"{row['pooled_iou']:.2f}", ha="center", color="white", fontweight="bold")
    ax.set_xticks(range(len(models)), [LABELS[m] for m in models])
    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU on test glaciers")
    ax.set_title("Bars: pooled IoU with 95% bootstrap interval; dots: single glaciers", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_area_timeseries(ts_df: pd.DataFrame, trend_df: pd.DataFrame, out: Path) -> None:
    """Each glacier's mapped area relative to its own median, 2017-2025."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    rel_all = []
    for _, grp in ts_df.groupby("glacier_id"):
        rel = 100.0 * grp["area_km2"] / grp["area_km2"].median()
        rel_all.append(rel.to_numpy())
        ax.plot(grp["year"], rel, color="#9aa5b1", lw=0.8, alpha=0.7)
    years = sorted(ts_df["year"].unique())
    med = np.nanmedian(np.vstack(rel_all), axis=0)
    ax.plot(years, med, color=COLORS["unet"], lw=2.5, marker="o", label="Median of 19 glaciers")
    inv = trend_df["inventory_pct_per_yr_gi4_to_gi5"].median()
    mid = np.mean(years)
    ax.plot(
        years,
        100 + inv * (np.array(years) - mid),
        "--",
        color="#c0392b",
        label=f"Inventory rate GI 4→GI 5 (median {inv:.1f} %/yr)",
    )
    ax.set_ylabel("Mapped area (% of glacier's median)")
    ax.set_title("Glacier area from Sentinel-2, one late-summer scene per year", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_snow_vs_temperature(ts_df: pd.DataFrame, out: Path) -> None:
    """Late-summer bright-snow share on the glaciers vs. summer temperature anomaly."""
    d = ts_df.dropna(subset=["snow_fraction"]).copy()
    d["snow_dm"] = d["snow_fraction"] - d.groupby("glacier_id")["snow_fraction"].transform("mean")
    yearly = d.groupby("year").agg(snow=("snow_dm", "mean"), temp=("jjas_anomaly_c", "mean"))
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(
        d["jjas_anomaly_c"],
        100 * d["snow_dm"],
        color="#9aa5b1",
        s=10,
        alpha=0.6,
        label="glacier-year",
    )
    ax.scatter(
        yearly["temp"], 100 * yearly["snow"], color=COLORS["unet"], s=40, label="yearly mean"
    )
    for yr, row in yearly.iterrows():
        ax.annotate(
            str(yr),
            (row["temp"], 100 * row["snow"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Summer (Jun-Sep) temperature anomaly vs 1991-2020 [°C]")
    ax.set_ylabel("Snow-covered share of glacier\n(difference from glacier mean) [%-points]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_retreat_strip(
    glacier: Any,
    predict: Callable[[Any, int], np.ndarray],
    years: list[int],
    out: Path,
    pixel_size_m: float,
) -> None:
    """One glacier, every year: Sentinel-2 image with the model outline (cyan) and GI 5 (magenta)."""
    fig, axes = plt.subplots(1, len(years), figsize=(2.1 * len(years), 2.5))
    for ax, yr in zip(axes, years, strict=True):
        feats = glacier.features(yr, pixel_size_m)
        ax.imshow(_rgb(feats))
        _outline(ax, (predict(glacier, yr) >= 0.5) & glacier.valid(yr), MODEL_COLOR, 0.8)
        _outline(ax, glacier.label.astype(bool), INVENTORY_COLOR, 0.6)
        ax.set_title(f"{yr}\n{glacier.scenes[yr]['acquisition_date']}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{glacier.meta['name']}: model outline (cyan) vs inventory {glacier.label_year} (magenta)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_training_curves(curves_df: pd.DataFrame, out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for seed, grp in curves_df.groupby("seed"):
        a1.plot(grp["epoch"], grp["train_loss"], label=f"seed {seed}")
        a2.plot(grp["epoch"], grp["val_tile_iou"], label=f"seed {seed}")
    a1.set_title("Training loss")
    a2.set_title("Validation IoU (validation glaciers)")
    for a in (a1, a2):
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def generate_all_figures(
    glaciers: dict[str, Any],
    predictors: dict[str, Callable[[Any, int], np.ndarray]],
    best_model: str,
    eval_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    figures_dir: Path,
    years: list[int],
    pixel_size_m: float,
) -> None:
    plot_test_predictions(glaciers, predictors, figures_dir / "test_predictions.png", pixel_size_m)
    plot_model_comparison(summary_df, eval_df, figures_dir / "model_comparison.png")
    plot_area_timeseries(ts_df, trend_df, figures_dir / "area_timeseries.png")
    plot_snow_vs_temperature(ts_df, figures_dir / "snow_vs_temperature.png")
    plot_training_curves(curves_df, figures_dir / "training_curves.png")
    for gid in ("AT_HT_01", "AT_SIL_01"):  # Pasterze (training region), Jamtalferner (test region)
        plot_retreat_strip(
            glaciers[gid],
            predictors[best_model],
            years,
            figures_dir / f"strip_{gid}.png",
            pixel_size_m,
        )
