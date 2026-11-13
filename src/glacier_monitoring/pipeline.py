"""End-to-end pipeline: data -> training -> evaluation on unseen regions -> time series -> figures."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from glacier_monitoring.data.dataset import (
    GlacierData,
    assert_no_glacier_leakage,
    compute_channel_stats,
    extract_tiles,
    load_glaciers,
    normalize,
)
from glacier_monitoring.data.ingest import download_all, fetch_geosphere_climate
from glacier_monitoring.evaluation.metrics import bootstrap_pooled_iou, evaluate_glacier, pooled_iou
from glacier_monitoring.models.baselines import NDSIThresholdModel, RandomForestPixelModel
from glacier_monitoring.models.cnn import ResUNet
from glacier_monitoring.timeseries.climate_link import (
    bright_snow_fraction,
    snow_temperature_regression,
)
from glacier_monitoring.timeseries.trends import theil_sen_trend
from glacier_monitoring.training.trainer import set_global_seed, train_segmentation_model
from glacier_monitoring.visualization.gifs import build_gifs
from glacier_monitoring.visualization.plots import generate_all_figures

Predictor = Callable[[GlacierData, int], np.ndarray]

MODEL_NAMES = {
    "ndsi": "NDSI threshold",
    "rf": "Random Forest (per pixel)",
    "unet": "Residual U-Net (3-seed ensemble)",
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_predictors(
    ndsi: NDSIThresholdModel,
    rf: RandomForestPixelModel,
    unets: list[ResUNet],
    stats: dict[str, dict[str, float]],
    pixel_size_m: float,
) -> dict[str, Predictor]:
    def p_ndsi(g: GlacierData, yr: int) -> np.ndarray:
        return ndsi.predict_proba(g.features(yr, pixel_size_m)[None])[0]

    def p_rf(g: GlacierData, yr: int) -> np.ndarray:
        return rf.predict_proba(normalize(g.features(yr, pixel_size_m), stats)[None])[0]

    def p_unet(g: GlacierData, yr: int) -> np.ndarray:
        x = torch.from_numpy(normalize(g.features(yr, pixel_size_m), stats)[None])
        return np.mean([net.predict_proba(x)[0].numpy() for net in unets], axis=0)

    return {"ndsi": p_ndsi, "rf": p_rf, "unet": p_unet}


def run_full_pipeline(
    workspace_dir: Path, config_path: Path | None = None, quick: bool = False
) -> dict[str, Any]:
    """Run everything. ``quick`` trains tiny models for a fast end-to-end check."""
    cfg = load_yaml_config(config_path or workspace_dir / "configs" / "default.yaml")
    years: list[int] = list(cfg["data"]["years"])
    px = float(cfg["data"]["pixel_size_m"])
    pixel_area_km2 = px * px / 1e6
    tcfg = cfg["training"]
    seeds: list[int] = list(tcfg["seeds"])[:1] if quick else list(tcfg["seeds"])

    external_dir = workspace_dir / "data" / "external"
    raw_dir = workspace_dir / "data" / "raw"
    processed_dir = workspace_dir / "data" / "processed"
    ckpt_dir = workspace_dir / "models" / "checkpoints"
    tables_dir = workspace_dir / "reports" / "tables"
    figures_dir = workspace_dir / "reports" / "figures"
    for d in (processed_dir, ckpt_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------------------------------
    outlines = download_all(external_dir, raw_dir, years, px)
    glaciers = load_glaciers(outlines, raw_dir, years, px)
    assert_no_glacier_leakage(glaciers)
    climate = fetch_geosphere_climate(years, external_dir)
    climate.to_csv(tables_dir / "climate_jjas.csv", index=False)

    pd.DataFrame(
        [
            {
                "glacier_id": g.id,
                "glacier_name": g.meta["name"],
                "region": g.meta["region"],
                "split": g.split,
                "gi5_year": g.label_year,
                "area_gi5_km2": g.meta["area_gi5_km2"],
                "area_gi4_2015_km2": g.meta["area_gi4_km2"],
                "grid_size_px": g.meta["size_px"],
            }
            for g in glaciers.values()
        ]
    ).to_csv(tables_dir / "glaciers.csv", index=False)
    pd.DataFrame([s for g in glaciers.values() for s in g.scenes.values()]).to_csv(
        tables_dir / "scenes.csv", index=False
    )

    by_split = {s: [g for g in glaciers.values() if g.split == s] for s in ("train", "val", "test")}
    train_x, train_y = extract_tiles(by_split["train"], years, int(tcfg["label_year_window"]), px)
    val_x, val_y = extract_tiles(by_split["val"], years, 0, px, stride=64)
    stats = compute_channel_stats(train_x)
    (processed_dir / "channel_stats_train.json").write_text(json.dumps(stats, indent=2))
    train_xn, val_xn = normalize(train_x, stats), normalize(val_x, stats)
    print(f"[pipeline] {len(train_x)} training tiles, {len(val_x)} validation tiles")

    # ------------------------------------------------------------------------------------------
    # 2. Models
    # ------------------------------------------------------------------------------------------
    set_global_seed(int(cfg["project"]["seed"]))
    ndsi = NDSIThresholdModel()
    ndsi.fit(train_x, train_y)
    rf = RandomForestPixelModel(
        n_estimators=30 if quick else 150,
        max_depth=12,
        max_pixels_train=20000 if quick else 200000,
        seed=int(cfg["project"]["seed"]),
    )
    rf.fit(train_xn, train_y)

    unets: list[ResUNet] = []
    curves: list[dict[str, Any]] = []
    for seed in seeds:
        set_global_seed(seed)
        net = ResUNet(in_channels=train_x.shape[1], base_channels=int(tcfg["base_channels"]))
        result = train_segmentation_model(
            net,
            train_xn,
            train_y,
            val_xn,
            val_y,
            epochs=2 if quick else int(tcfg["epochs"]),
            batch_size=int(tcfg["batch_size"]),
            lr=float(tcfg["lr"]),
            patience=int(tcfg["patience"]),
            seed=seed,
        )
        torch.save(net.state_dict(), ckpt_dir / f"unet_seed{seed}.pt")
        unets.append(net)
        for ep, (loss, iou) in enumerate(zip(result.train_losses, result.val_ious, strict=True)):
            curves.append({"seed": seed, "epoch": ep + 1, "train_loss": loss, "val_tile_iou": iou})
        print(
            f"[pipeline] U-Net seed {seed}: best val IoU {result.best_val_iou:.3f} (epoch {result.best_epoch})"
        )
    pd.DataFrame(curves).to_csv(tables_dir / "unet_training_curves.csv", index=False)

    predictors = _build_predictors(ndsi, rf, unets, stats, px)
    for seed, net in zip(seeds, unets, strict=True):
        predictors[f"unet_seed{seed}"] = _build_predictors(ndsi, rf, [net], stats, px)["unet"]

    # ------------------------------------------------------------------------------------------
    # 3. Evaluation at the inventory year: validation (model selection) and test (unseen regions).
    #    Training glaciers are scored too, for the viewer only; those scores are not a fair test.
    # ------------------------------------------------------------------------------------------
    eval_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for g in by_split[split]:
            yr = g.label_year
            valid = g.valid(yr)
            for name, predict in predictors.items():
                m = evaluate_glacier(
                    predict(g, yr), g.label, g.region.astype(bool), valid, pixel_area_km2
                )
                eval_rows.append(
                    {
                        "model": name,
                        "split": split,
                        "glacier_id": g.id,
                        "glacier_name": g.meta["name"],
                        "year": yr,
                        **m,
                    }
                )
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(tables_dir / "per_glacier_metrics.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for (name, split), grp in eval_df.groupby(["model", "split"], sort=False):
        lo, hi = bootstrap_pooled_iou(grp["tp"], grp["fp"], grp["fn"])
        summary_rows.append(
            {
                "model": name,
                "split": split,
                "n_glaciers": len(grp),
                "pooled_iou": round(pooled_iou(grp["tp"], grp["fp"], grp["fn"]), 4),
                "pooled_iou_ci95_low": round(lo, 4),
                "pooled_iou_ci95_high": round(hi, 4),
                "mean_f1": round(float(grp["f1"].mean()), 4),
                "mean_boundary_f1": round(float(grp["boundary_f1"].mean()), 4),
                "median_abs_area_error_pct": round(
                    float(grp["area_rel_error_pct"].abs().median()), 2
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "model_comparison.csv", index=False)
    print(summary_df.to_string(index=False))

    val_scores = summary_df[(summary_df["split"] == "val") & summary_df["model"].isin(MODEL_NAMES)]
    best = str(val_scores.sort_values("pooled_iou", ascending=False)["model"].iloc[0])
    print(f"[pipeline] Best model on validation glaciers: {best}")

    # ------------------------------------------------------------------------------------------
    # 4. Time series 2017-2025 with the selected model, and the snow / temperature comparison
    # ------------------------------------------------------------------------------------------
    max_invalid = float(cfg["timeseries"]["max_invalid_fraction"])
    clim = climate.set_index(["region", "year"])
    ts_rows: list[dict[str, Any]] = []
    app_bundle: dict[str, np.ndarray] = {}
    for g in glaciers.values():
        region = g.region.astype(bool)
        for yr in years:
            feats = g.features(yr, px)
            valid = g.valid(yr)
            preds = {m: predictors[m](g, yr) >= 0.5 for m in MODEL_NAMES}
            pred = preds[best]
            invalid_in_region = 1.0 - float(valid[region].mean())
            usable = invalid_in_region <= max_invalid
            scene = g.scenes[yr]
            ts_rows.append(
                {
                    "glacier_id": g.id,
                    "glacier_name": g.meta["name"],
                    "region": g.meta["region"],
                    "split": g.split,
                    "year": yr,
                    "acquisition_date": scene["acquisition_date"],
                    "day_of_year": pd.Timestamp(scene["acquisition_date"]).dayofyear,
                    "invalid_fraction_in_region": round(invalid_in_region, 4),
                    "area_km2": (
                        round(float((pred & region).sum()) * pixel_area_km2, 4)
                        if usable
                        else np.nan
                    ),
                    **{
                        f"area_km2_{m}": (
                            round(float((p & region).sum()) * pixel_area_km2, 4)
                            if usable
                            else np.nan
                        )
                        for m, p in preds.items()
                    },
                    "snow_fraction": (
                        round(bright_snow_fraction(feats, g.target, valid), 4) if usable else np.nan
                    ),
                    "jjas_anomaly_c": float(clim.loc[(g.meta["region"], yr), "jjas_anomaly_c"]),
                }
            )
            rgb = np.clip(feats[[2, 1, 0]] / 0.6 * 255.0, 0, 255).astype(np.uint8)
            app_bundle[f"{g.id}_{yr}_rgb"] = rgb
            app_bundle[f"{g.id}_{yr}_invalid"] = np.packbits(~valid)
            for m, p in preds.items():
                app_bundle[f"{g.id}_{yr}_pred_{m}"] = np.packbits(p)
        app_bundle[f"{g.id}_label"] = np.packbits(g.label.astype(bool))
    ts_df = pd.DataFrame(ts_rows)
    ts_df.to_csv(tables_dir / "area_timeseries.csv", index=False)
    np.savez_compressed(processed_dir / "app_bundle.npz", **app_bundle)
    (processed_dir / "app_meta.json").write_text(
        json.dumps(
            {
                gid: {
                    k: g.meta[k]
                    for k in (
                        "name",
                        "region_display",
                        "size_px",
                        "gi5_year",
                        "gi4_matched",
                        "area_gi4_km2",
                        "area_gi5_km2",
                    )
                }
                | {"split": g.split}
                for gid, g in glaciers.items()
            }
            | {"_model": best, "_years": years},
            indent=2,
        )
    )

    trend_rows: list[dict[str, Any]] = []
    for gid, grp in ts_df.groupby("glacier_id", sort=False):
        g = glaciers[gid]
        ref = float(grp["area_km2"].median())
        tr = theil_sen_trend(grp["year"], grp["area_km2"])
        inv = (
            100.0
            * (g.meta["area_gi5_km2"] - g.meta["area_gi4_km2"])
            / g.meta["area_gi4_km2"]
            / (g.label_year - 2015)
            if g.meta["gi4_matched"]
            else np.nan
        )
        trend_rows.append(
            {
                "glacier_id": gid,
                "glacier_name": g.meta["name"],
                "split": g.split,
                "n_years": int(tr["n_years"]),
                "median_area_km2": round(ref, 3),
                "slope_km2_per_yr": round(tr["slope_per_year"], 4),
                "slope_pct_per_yr": round(100.0 * tr["slope_per_year"] / ref, 2),
                "slope_ci95_pct": f"[{100 * tr['ci_low'] / ref:.2f}, {100 * tr['ci_high'] / ref:.2f}]",
                "mann_kendall_tau": round(tr["kendall_tau"], 3),
                "mann_kendall_p": round(tr["p_value"], 4),
                "inventory_pct_per_yr_gi4_to_gi5": round(inv, 2),
            }
        )
    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(tables_dir / "area_trends.csv", index=False)

    climate_result = snow_temperature_regression(ts_df)
    pd.DataFrame([climate_result]).to_csv(tables_dir / "snow_vs_temperature.csv", index=False)

    # ------------------------------------------------------------------------------------------
    # 5. Figures and summary
    # ------------------------------------------------------------------------------------------
    generate_all_figures(
        glaciers=glaciers,
        predictors=predictors,
        best_model=best,
        eval_df=eval_df,
        summary_df=summary_df,
        ts_df=ts_df,
        trend_df=trend_df,
        curves_df=pd.DataFrame(curves),
        figures_dir=figures_dir,
        years=years,
        pixel_size_m=px,
    )

    build_gifs(processed_dir, figures_dir)

    test = summary_df[summary_df["split"] == "test"].set_index("model")
    seed_ious = [float(test.loc[f"unet_seed{s}", "pooled_iou"]) for s in seeds]
    summary = {
        "n_glaciers": len(glaciers),
        "n_training_tiles": int(len(train_x)),
        "selected_model": best,
        "ndsi_threshold": round(ndsi.best_threshold, 3),
        "test_pooled_iou": {m: float(test.loc[m, "pooled_iou"]) for m in MODEL_NAMES},
        "test_pooled_iou_ci95": {
            m: [
                float(test.loc[m, "pooled_iou_ci95_low"]),
                float(test.loc[m, "pooled_iou_ci95_high"]),
            ]
            for m in MODEL_NAMES
        },
        "unet_single_seed_test_iou_mean_sd": [
            round(float(np.mean(seed_ious)), 4),
            round(float(np.std(seed_ious)), 4),
        ],
        "median_glacier_trend_pct_per_yr": round(float(trend_df["slope_pct_per_yr"].median()), 2),
        "median_inventory_rate_pct_per_yr": round(
            float(trend_df["inventory_pct_per_yr_gi4_to_gi5"].median()), 2
        ),
        "snow_vs_temperature": {k: round(v, 4) for k, v in climate_result.items()},
    }
    (workspace_dir / "reports" / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
