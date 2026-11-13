"""Explore: glacier viewer with all models, model comparison, trends and climate."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from views.common import (
    ACCENT,
    ACCENT_DARK,
    COLORS,
    FIGURES_DIR,
    INK,
    MODEL_LABELS,
    MODELS,
    MUTED,
    PROCESSED_DIR,
    SPLIT_LABELS,
    eyebrow,
    glacier_ids,
    glacier_png,
    inject_css,
    legend_html,
    load_bundle,
    load_summary,
    load_table,
    outputs_available,
    show_gif,
)

inject_css()
if not outputs_available():
    st.error("No pipeline outputs found. Run `glacier-pipeline run` first.")
    st.stop()

meta, bundle = load_bundle()
summary = load_summary()
ts = load_table("area_timeseries.csv")
metrics = load_table("per_glacier_metrics.csv")
comparison = load_table("model_comparison.csv")
trends = load_table("area_trends.csv")
years: list[int] = meta["_years"]
model_scale = alt.Scale(domain=[MODEL_LABELS[m] for m in MODELS], range=[COLORS[m] for m in MODELS])

eyebrow("Interactive")
st.title("Explore the glaciers")
tab_view, tab_models, tab_trends, tab_method = st.tabs(
    ["🛰️ Glacier viewer", "📊 Model comparison", "📉 Trends & climate", "🔬 Method & data"]
)


# ------------------------------------------------------------------------------------------------
# Glacier viewer
# ------------------------------------------------------------------------------------------------
@st.fragment
def glacier_viewer() -> None:
    """Runs on its own, so moving the year slider does not rebuild the other tabs."""
    ids = sorted(
        glacier_ids(meta), key=lambda g: (["test", "val", "train"].index(meta[g]["split"]), g)
    )
    c1, c2, c3 = st.columns([1.6, 2.4, 1.2])
    gid = c1.selectbox(
        "Glacier",
        ids,
        format_func=lambda g: f"{meta[g]['name']} · {meta[g]['region_display']}",
    )
    g = meta[gid]
    year = c2.select_slider("Year", options=years, value=g["gi5_year"])
    view = c3.radio("View", ["Overlay", "Side by side"], horizontal=True)

    l1, l2, l3, l4, l5 = st.columns(5)
    show = {
        "inventory": l1.checkbox("Inventory", value=True),
        "ndsi": l2.checkbox("NDSI", value=view == "Side by side"),
        "rf": l3.checkbox("Random Forest", value=view == "Side by side"),
        "unet": l4.checkbox("U-Net", value=True),
    }
    show_clouds = l5.checkbox("Shade clouds", value=True)

    inv_layer = ("inventory",) if show["inventory"] else ()

    no_data = float((bundle[f"{gid}_{year}_rgb"].max(axis=0) == 0).mean())

    left, right = st.columns([3, 2], gap="large")
    with left:
        if no_data > 0.25:
            st.info(
                f"No usable satellite image of {g['name']} in {year}: "
                f"{no_data:.0%} of the area lies outside every clear Sentinel-2 scene of that "
                "summer. This year is left out of the trend. Pick another year."
            )
        elif view == "Overlay":
            layers = inv_layer + tuple(m for m in MODELS if show[m])
            st.image(glacier_png(gid, year, layers, show_clouds), width="stretch")
        else:
            chosen = [m for m in MODELS if show[m]] or ["unet"]
            cols = st.columns(len(chosen))
            for col, m in zip(cols, chosen, strict=True):
                png = glacier_png(gid, year, (*inv_layer, m), show_clouds, MODEL_LABELS[m], size=4)
                col.image(png, width="stretch")
        if no_data <= 0.25:
            keys = (["inventory"] if show["inventory"] else []) + [m for m in MODELS if show[m]]
            st.html(legend_html(keys))
        if year != g["gi5_year"] and no_data <= 0.25:
            st.caption(
                f"The inventory outline is from {g['gi5_year']}. In {year} the real glacier edge "
                "may differ, so treat it as a reference, not the truth for this year."
            )

    with right:
        row = ts[(ts["glacier_id"] == gid) & (ts["year"] == year)].iloc[0]
        cloud = row["invalid_fraction_in_region"]
        st.markdown(f"#### {g['name']}")
        if no_data > 0.25:
            status = "**no usable image this year**"
        else:
            status = (
                f"scene from **{row['acquisition_date']}** · clouds over the glacier: "
                f"**{'none' if cloud == 0 else f'{cloud:.1%}'}**"
            )
        st.markdown(f"**{SPLIT_LABELS[g['split']].capitalize()}** · {status}")
        with st.expander("Why are there almost never clouds?"):
            st.markdown(
                "For every glacier and year the pipeline checks up to 25 late-summer images and "
                "keeps the one with the fewest cloudy pixels over the glacier. In 128 of 171 "
                "glacier-years it found a completely cloud-free image. Only in 2017, when few "
                "processed images exist, are clouds common. Years with more than 3 % cloud over "
                "the glacier are left out of the trend."
            )

        m = metrics[(metrics["glacier_id"] == gid) & metrics["model"].isin(MODELS)].copy()
        m["Model"] = m["model"].map(MODEL_LABELS)
        st.markdown(f"**Accuracy against the inventory ({g['gi5_year']})**")
        st.dataframe(
            m[["Model", "iou", "boundary_f1", "area_rel_error_pct"]].rename(
                columns={
                    "iou": "IoU",
                    "boundary_f1": "Boundary F1",
                    "area_rel_error_pct": "Area error %",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "IoU": st.column_config.NumberColumn(format="%.2f"),
                "Boundary F1": st.column_config.NumberColumn(format="%.2f"),
                "Area error %": st.column_config.NumberColumn(format="%+.1f"),
            },
        )
        if g["split"] == "train":
            st.caption("⚠️ The models learned from this glacier, so these scores are optimistic.")

        g_ts = ts[ts["glacier_id"] == gid]
        long = g_ts.melt(
            id_vars=["year"], value_vars=[f"area_km2_{m}" for m in MODELS], value_name="area"
        )
        long["Model"] = long["variable"].str.replace("area_km2_", "").map(MODEL_LABELS)
        # Inventory ice inside the same region the model areas are measured in
        inv_area = float(m.loc[m["model"] == "unet", "true_area_km2"].iloc[0])
        inv_pts = pd.DataFrame(
            [{"year": g["gi5_year"], "area": inv_area, "Source": "Inventory GI 5"}]
        )
        lines = (
            alt.Chart(long)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("area:Q", title="Glacier area (km²)", scale=alt.Scale(zero=False)),
                color=alt.Color("Model:N", scale=model_scale, legend=alt.Legend(orient="bottom")),
                tooltip=["year", "Model", alt.Tooltip("area:Q", format=".2f")],
            )
        )
        pts = (
            alt.Chart(inv_pts)
            .mark_point(
                shape="diamond",
                size=160,
                filled=True,
                color=COLORS["inventory"],
                stroke="#141414",
                strokeWidth=1,
            )
            .encode(x="year:O", y="area:Q", tooltip=["Source", alt.Tooltip("area:Q", format=".2f")])
        )
        st.markdown("**Mapped area per year** (◆ = inventory, measured in the same region)")
        st.altair_chart((lines + pts).properties(height=260), width="stretch")

    gif = PROCESSED_DIR / "gifs" / f"{gid}.gif"
    if gif.exists():
        with st.expander(f"Animation: {g['name']} 2017–2025 (U-Net in cyan, inventory in magenta)"):
            col, _ = st.columns([1, 1])
            with col:
                show_gif(gif)


with tab_view:
    glacier_viewer()

# ------------------------------------------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------------------------------------------
with tab_models:
    st.markdown("""
All three models were evaluated on the **six test glaciers in Silvretta and Dachstein**: two
mountain ranges they never saw during training. **IoU** (intersection over union) measures the
overlap between the predicted glacier and the inventory outline: 1.0 is perfect, 0.5 means half
of the combined area disagrees.
""")
    test = comparison[(comparison["split"] == "test") & comparison["model"].isin(MODELS)].copy()
    test["Model"] = test["model"].map(MODEL_LABELS)
    per = metrics[(metrics["split"] == "test") & metrics["model"].isin(MODELS)].copy()
    per["Model"] = per["model"].map(MODEL_LABELS)

    base = alt.Chart(test).encode(
        x=alt.X("Model:N", sort=None, title=None, axis=alt.Axis(labelAngle=0))
    )
    bars = base.mark_bar(width=70, cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        y=alt.Y("pooled_iou:Q", title="IoU on test glaciers", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", scale=model_scale, legend=None),
        tooltip=[alt.Tooltip("pooled_iou:Q", format=".3f", title="pooled IoU")],
    )
    err = base.mark_rule(strokeWidth=2).encode(
        y="pooled_iou_ci95_low:Q", y2="pooled_iou_ci95_high:Q"
    )
    dots = (
        alt.Chart(per)
        .mark_circle(size=70, color=INK, opacity=0.7)
        .encode(
            x=alt.X("Model:N", sort=None, axis=alt.Axis(labelAngle=0)),
            y="iou:Q",
            xOffset=alt.XOffset("glacier_name:N"),
            tooltip=["glacier_name", alt.Tooltip("iou:Q", format=".2f")],
        )
    )
    a, b = st.columns([1.2, 1], gap="large")
    a.altair_chart((bars + err + dots).properties(height=360), width="stretch")
    a.caption(
        "Bars: IoU over all test pixels. Lines: 95 % interval (bootstrap over glaciers). Dots: single glaciers."
    )
    with b:
        st.markdown("#### What the numbers say")
        st.markdown(f"""
- The **U-Net** is best ({summary["test_pooled_iou"]["unet"]:.2f}), ahead of the Random Forest
  ({summary["test_pooled_iou"]["rf"]:.2f}) and the NDSI threshold ({summary["test_pooled_iou"]["ndsi"]:.2f}).
- It also gets glacier **area** closest (median error 5 % vs 10–11 %).
- Six test glaciers is a small sample, so the intervals overlap. The advantage is consistent,
  but its exact size is uncertain.
- The hardest cases are Jamtalferner and Vermuntgletscher, where much of the glacier is dark,
  dirty bare ice.
""")
    table = per.pivot_table(index="glacier_name", columns="Model", values="iou")[
        [MODEL_LABELS[m] for m in MODELS]
    ].rename_axis("Glacier")
    st.markdown("**IoU per test glacier**")
    st.dataframe(
        table,
        width="stretch",
        column_config={
            c: st.column_config.ProgressColumn(c, format="%.2f", min_value=0.0, max_value=1.0)
            for c in table.columns
        },
    )
    with st.expander("All test glaciers side by side (static figure)"):
        st.image(str(FIGURES_DIR / "test_predictions.png"), width="stretch")

# ------------------------------------------------------------------------------------------------
# Trends & climate
# ------------------------------------------------------------------------------------------------
with tab_trends:
    st.markdown("### How fast are the glaciers shrinking?")
    rel = ts.assign(
        rel=100 * ts["area_km2"] / ts.groupby("glacier_id")["area_km2"].transform("median")
    )
    med = rel.groupby("year", as_index=False)["rel"].median()
    sel = alt.selection_point(fields=["glacier_name"], on="pointerover", nearest=True, empty=False)
    gl_lines = (
        alt.Chart(rel)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "rel:Q", title="Area (% of the glacier's own median)", scale=alt.Scale(zero=False)
            ),
            detail="glacier_name:N",
            color=alt.condition(sel, alt.value(ACCENT_DARK), alt.value("#d3d8e0")),
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.6)),
            tooltip=["glacier_name", "year", alt.Tooltip("rel:Q", format=".1f")],
        )
        .add_params(sel)
    )
    med_line = (
        alt.Chart(med)
        .mark_line(point=True, strokeWidth=4, color=INK)
        .encode(x="year:O", y="rel:Q", tooltip=[alt.Tooltip("rel:Q", format=".1f", title="median")])
    )
    t1, t2 = st.columns([1.5, 1], gap="large")
    t1.altair_chart((gl_lines + med_line).properties(height=380), width="stretch")
    t1.caption(
        "Thin lines: single glaciers (hover to highlight). Dark line: median of all 19 glaciers."
    )
    with t2:
        st.markdown(f"""
- Median trend: **{summary["median_glacier_trend_pct_per_yr"]:.1f} % per year**. The decline is
  statistically clear (Mann–Kendall p < 0.05) for
  **{int((trends["mann_kendall_p"] < 0.05).sum())} of {len(trends)}** glaciers.
- The official inventories (2015 → 2021–23) give **{summary["median_inventory_rate_pct_per_yr"]:.1f} % per year**.
- The satellite trend is a bit steeper. Two possible reasons: 2022 and 2025 were extreme melt
  years that fall after the inventory, and leftover snow in early years can make glaciers look
  larger.
- Jumps of 10–20 % in single years usually mean fresh or leftover snow in that year's image.
""")
    show_cols = {
        "glacier_name": "Glacier",
        "split": "Split",
        "slope_pct_per_yr": "Satellite trend %/yr",
        "inventory_pct_per_yr_gi4_to_gi5": "Inventory %/yr",
        "mann_kendall_p": "Mann–Kendall p",
    }
    st.dataframe(
        trends[list(show_cols)]
        .assign(
            inventory_pct_per_yr_gi4_to_gi5=trends["inventory_pct_per_yr_gi4_to_gi5"].map(
                lambda v: "–" if pd.isna(v) else f"{v:.2f}"
            )
        )
        .rename(columns=show_cols),
        hide_index=True,
        width="stretch",
        column_config={
            "Inventory %/yr": st.column_config.TextColumn(help="–: no matching GI 4 outline")
        },
    )

    st.markdown("### Do warm summers leave less snow on the glaciers?")
    d = ts.dropna(subset=["snow_fraction"]).copy()
    d["snow_dm"] = 100 * (
        d["snow_fraction"] - d.groupby("glacier_id")["snow_fraction"].transform("mean")
    )
    yearly = d.groupby("year", as_index=False).agg(
        temp=("jjas_anomaly_c", "mean"), snow=("snow_dm", "mean")
    )
    scatter = (
        alt.Chart(d)
        .mark_circle(size=40, color=MUTED, opacity=0.5)
        .encode(
            x=alt.X("jjas_anomaly_c:Q", title="Summer temperature anomaly vs 1991–2020 (°C)"),
            y=alt.Y("snow_dm:Q", title="Snow on glacier vs its average (%-points)"),
            tooltip=["glacier_name", "year", alt.Tooltip("snow_dm:Q", format=".0f")],
        )
    )
    yearly_pts = (
        alt.Chart(yearly)
        .mark_circle(size=170, color=ACCENT, stroke=INK, strokeWidth=1.5)
        .encode(x="temp:Q", y="snow:Q", tooltip=["year", alt.Tooltip("temp:Q", format=".2f")])
    )
    yearly_txt = (
        alt.Chart(yearly)
        .mark_text(dx=18, dy=-8, fontSize=12)
        .encode(x="temp:Q", y="snow:Q", text="year:O")
    )
    reg = scatter.transform_regression("jjas_anomaly_c", "snow_dm").mark_line(
        color=INK, strokeWidth=2, strokeDash=[6, 4]
    )
    s1, s2 = st.columns([1.5, 1], gap="large")
    s1.altair_chart(
        (scatter + reg + yearly_pts + yearly_txt).properties(height=380), width="stretch"
    )
    clim = summary["snow_vs_temperature"]
    with s2:
        st.markdown(f"""
For every glacier-year I measured how much of the glacier is still covered by bright snow at
the end of summer, then compared it with that summer's temperature at the nearest GeoSphere
station.

- **{100 * clim["slope_per_degC"]:.0f} percentage points of snow per +1 °C** (± {100 * clim["slope_se"]:.0f}),
  after accounting for the image date (a September image naturally shows less snow than a July
  one).
- Small dots are single glacier-years, large mint dots are yearly averages, and the dashed\n  line is a simple fit.
- With only 9 summers, the yearly averages alone give r = {clim["yearly_mean_pearson_r"]:.2f}:
  the right direction, but not statistically significant on their own.
""")

# ------------------------------------------------------------------------------------------------
# Method & data
# ------------------------------------------------------------------------------------------------
with tab_method:
    st.markdown("""
### Data
| Source | Used for |
| --- | --- |
| **Sentinel-2 L2A** (Copernicus, via Element84 Earth Search) | 6 spectral bands, one late-summer image per glacier and year |
| **Copernicus DEM GLO-30** | elevation, slope, aspect, hillshade |
| **Austrian Glacier Inventory GI 5** (2021–23, PANGAEA) | training labels and test reference |
| **Austrian Glacier Inventory GI 4** (2015, PANGAEA) | trend region and inventory shrinkage rate |
| **GeoSphere Austria** station temperatures | summer temperature anomaly |
| **geoBoundaries** | the Austrian border (the inventory only covers Austria) |

### Models
- **NDSI threshold**: snow and ice reflect green light strongly and shortwave infrared weakly.
  One threshold on that ratio, tuned on the training glaciers.
- **Random Forest**: 150 decision trees looking at each pixel on its own (14 values: bands,
  indices, terrain).
- **U-Net**: a convolutional neural network that also sees the *shape and surroundings* of each
  pixel. About 0.9 M parameters, trained from scratch, 3 runs averaged.

### Fair testing
- Split by **region**: Ötztal, Hohe Tauern and Stubai for training and validation; Silvretta and
  Dachstein only for the final test.
- Normalisation statistics are computed on training data only.
- The best model is chosen on the validation glaciers, never on the test glaciers.

### Limitations
- One image per year means snow can make glaciers look larger.
- Labels from one inventory date (training uses ± 1 year around it).
- 20 m pixels: small glaciers are only a few hundred pixels.
- Only 6 test glaciers, so the intervals are wide.
""")
