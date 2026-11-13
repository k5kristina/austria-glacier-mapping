"""Landing page: hero with mountains and an animated data line, then a short introduction."""

from __future__ import annotations

import numpy as np
import streamlit as st
from views.common import (
    FIGURES_DIR,
    eyebrow,
    inject_css,
    load_summary,
    load_table,
    outputs_available,
)

inject_css()

if not outputs_available():
    st.error("No pipeline outputs found. Run `glacier-pipeline run` first.")
    st.stop()

summary = load_summary()
# Inline so it applies even when the shared style module is cached
GRID_CSS = """<style>.card-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; align-items: stretch; } @media (max-width: 800px) { .card-grid { grid-template-columns: 1fr; } }</style>"""
ts = load_table("area_timeseries.csv")

# Median glacier area (each glacier relative to its own median) -> the animated hero line
rel = ts.assign(rel=100 * ts["area_km2"] / ts.groupby("glacier_id")["area_km2"].transform("median"))
median = rel.groupby("year")["rel"].median()
years = median.index.to_list()
x0, x1, y_lo, y_hi = 70, 930, 86.0, 108.0
xs = np.linspace(x0, x1, len(years))
ys = 40 + (y_hi - median.to_numpy()) / (y_hi - y_lo) * 210
points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
dots = "".join(
    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="dot" style="animation-delay:{0.4 + 0.28 * i:.2f}s"/>'
    f'<text x="{x:.1f}" y="285" class="yr">{yr}</text>'
    for i, (x, y, yr) in enumerate(zip(xs, ys, years, strict=True))
)
# 2017 has few cloud-free glaciers, so measure the change from the first complete year
counts = ts.dropna(subset=["area_km2"]).groupby("year")["glacier_id"].nunique()
start = int(counts[counts == counts.max()].index.min())
change = median.iloc[-1] - median.loc[start]

hero_html = f"""
<style>
.hero {{ --ink: #0a0f1c; --paper: #ffffff; --accent: #9fe3c9; }}
.hero {{ position: relative; overflow: hidden; border-radius: 24px; height: 700px;
  border: 1px solid rgba(255,255,255,.08);
  margin-top: 8px; background: var(--ink); color: var(--paper);
  font-family: 'Inter', sans-serif; }}
/* faint dot grid */
.hero::before {{ content: ""; position: absolute; inset: 0; z-index: 0;
  background-image: radial-gradient(rgba(255,255,255,.07) 1px, transparent 1px);
  background-size: 22px 22px; }}
/* soft accent glow behind the chart */
.hero::after {{ content: ""; position: absolute; z-index: 0; width: 620px; height: 620px;
  right: -160px; top: 120px; border-radius: 50%;
  background: radial-gradient(circle, rgba(159,227,201,.13), transparent 65%); }}
.hero .text {{ position: relative; z-index: 3; padding: 52px 56px 0 56px; max-width: 760px;
  animation: rise 1s ease-out both; }}
.badge {{ display: inline-block; padding: 6px 14px; border-radius: 999px;
  border: 1px solid rgba(159,227,201,.45); color: var(--accent); font-size: .78rem;
  font-weight: 500; letter-spacing: .06em; }}
.hero .title {{ font-size: 3.5rem; font-weight: 800; line-height: 1.04; letter-spacing: -.035em;
  margin: 1.1rem 0 1rem 0; color: var(--paper); }}
.hero .title em {{ font-style: normal; color: var(--accent); }}
.hero p {{ font-size: 1.12rem; line-height: 1.55; color: rgba(255,255,255,.68); margin: 0;
  max-width: 560px; }}
.hero a.cta, .hero a.cta:visited {{ display: inline-block; margin-top: 1.6rem; padding: 12px 22px; border-radius: 12px;
  background: var(--accent); color: var(--ink) !important; font-weight: 800;
  text-decoration: none !important;
  font-size: .98rem; transition: transform .15s ease, box-shadow .15s ease; }}
.hero a.cta:hover {{ transform: translateY(-2px); box-shadow: 0 10px 30px rgba(159,227,201,.25); }}
.hero svg.chart {{ position: absolute; z-index: 2; left: 0; right: 0; bottom: 40px;
  width: 100%; height: 280px; }}
.hero svg.mountains {{ position: absolute; z-index: 1; left: 0; bottom: 0; width: 100%;
  height: 260px; }}
.line {{ fill: none; stroke: var(--accent); stroke-width: 4; stroke-linecap: round;
  stroke-linejoin: round; stroke-dasharray: 1400; stroke-dashoffset: 1400;
  filter: drop-shadow(0 0 10px rgba(159,227,201,.55)); animation: draw 3s ease-out .3s forwards; }}
.dot {{ fill: var(--ink); stroke: var(--accent); stroke-width: 2.5; opacity: 0;
  animation: pop .45s ease-out forwards; }}
.yr {{ fill: rgba(255,255,255,.45); font-size: 12px; text-anchor: middle;
  font-family: 'Inter', sans-serif; }}
.chart-label {{ fill: rgba(255,255,255,.6); font-size: 13px; font-family: 'Inter', sans-serif; }}
.chart-label tspan {{ fill: var(--accent); font-weight: 500; }}
@keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
@keyframes pop {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; }} }}
@media (max-width: 700px) {{ .hero .title {{ font-size: 2.3rem; }} .hero .text {{ padding: 32px 22px 0; }} }}
</style>
<div class="hero">
  <div class="text">
    <span class="badge">SENTINEL-2 · MACHINE LEARNING · AUSTRIAN ALPS</span>
    <div class="title">Austria's glaciers,<br/>seen from <em>space</em></div>
    <p>Teaching a computer to find glaciers in satellite images, then checking how much
    they shrank between 2017 and 2025.</p>
    <a class="cta" href="explore" target="_self">Explore the glaciers →</a>
  </div>
  <svg class="chart" viewBox="0 0 1000 300" preserveAspectRatio="none">
    <text x="{x0}" y="22" class="chart-label">Median glacier area, 19 glaciers ·
      <tspan>{change:+.0f} % since {start}</tspan></text>
    <polyline class="line" points="{points}"/>
    {dots}
  </svg>
  <svg class="mountains" viewBox="0 0 1000 300" preserveAspectRatio="none">
    <path fill="#ffffff" fill-opacity=".04" stroke="#ffffff" stroke-opacity=".12" stroke-width="1"
      d="M0 190 L80 120 L140 160 L230 70 L300 140 L370 95 L450 150 L540 60 L610 130 L690 85
      L780 150 L860 90 L940 140 L1000 110 L1000 300 L0 300 Z"/>
    <path fill="#ffffff" fill-opacity=".06" stroke="#ffffff" stroke-opacity=".16" stroke-width="1"
      d="M0 230 L60 190 L130 220 L210 140 L260 185 L330 150 L420 215 L500 130 L560 175
      L640 120 L720 200 L800 150 L880 205 L950 170 L1000 190 L1000 300 L0 300 Z"/>
    <path fill="#0a0f1c" d="M0 265 L90 240 L170 258 L260 225 L350 255 L440 232 L520 260
      L610 236 L700 262 L790 238 L880 258 L1000 240 L1000 300 L0 300 Z"/>
  </svg>
</div>
"""
# Markdown treats indented or blank lines as code / paragraph breaks, so send it as one line
st.markdown(" ".join(line.strip() for line in hero_html.splitlines()), unsafe_allow_html=True)


st.write("")
eyebrow("Why")
st.markdown("## Why this project?")
st.markdown("""
Glaciers are among the clearest signs of a warming climate. Austria has an official glacier
inventory, but it is mapped carefully by hand from aerial images, so a new one appears only every
several years (the last two cover 2015 and 2021–2023). The European **Sentinel-2** satellites
photograph the Alps every few days, and the images are free. If a model can reliably draw
glacier outlines from these images, glacier change can be followed **every year**.
""")

st.write("")
eyebrow("Goal")
st.markdown("## Three questions")
st.html(GRID_CSS + """<div class="card-grid">
<div class="step"><div class="n">1</div><h4>Which method finds glaciers best?</h4>
    <div class="muted">A simple snow index (NDSI), a Random Forest or a neural network
    (U-Net), and does it still work on mountains it has never seen?</div></div>
<div class="step"><div class="n">2</div><h4>How fast are they shrinking?</h4>
    <div class="muted">Measured with the best model for 19 glaciers from 2017 to 2025, and
    compared with the official inventory.</div></div>
<div class="step"><div class="n">3</div><h4>Do warm summers leave less snow?</h4>
    <div class="muted">End-of-summer snow on the glaciers compared with GeoSphere Austria
    temperature records.</div></div>
</div>""")

st.write("")
eyebrow("Method")
st.markdown("## How it works")
st.html(GRID_CSS + """<div class="card-grid">
<div class="step"><div class="n">🛰</div><h4>Satellite images</h4>
    <div class="muted">One clear late-summer Sentinel-2 image per glacier and year, 20 m pixels,
    plus elevation and slope from the Copernicus DEM.</div></div>
<div class="step"><div class="n">⚙</div><h4>Three models</h4>
    <div class="muted">NDSI threshold, Random Forest and a U-Net, all trained on outlines from
    the Austrian Glacier Inventory (GI 5).</div></div>
<div class="step"><div class="n">✓</div><h4>A fair test</h4>
    <div class="muted">Two whole mountain ranges, Silvretta and Dachstein, are kept away from
    training and used only for the final test.</div></div>
</div>""")

st.write("")
eyebrow("Results")
st.markdown("## Key results")
iou = summary["test_pooled_iou"]
clim = summary["snow_vs_temperature"]
st.html(GRID_CSS + f"""<div class="card-grid">
<div class="stat-card"><div class="num">{iou["unet"]:.2f}</div>
    <div class="lbl">U-Net accuracy (IoU) on unseen mountains, vs {iou["ndsi"]:.2f} for the
    simple snow index</div></div>
<div class="stat-card"><div class="num">{summary["median_glacier_trend_pct_per_yr"]:.1f} %</div>
    <div class="lbl">median change in glacier area per year, 2017–2025
    (inventory: {summary["median_inventory_rate_pct_per_yr"]:.1f} %)</div></div>
<div class="stat-card"><div class="num">{100 * clim["slope_per_degC"]:.0f} pp</div>
    <div class="lbl">snow left on the glaciers at the end of summer, per +1 °C of summer
    temperature (percentage points)</div></div>
</div>""")

st.write("")
eyebrow("Time-lapse")
st.markdown("## Pasterze, Austria's largest glacier")
g1, g2 = st.columns([1.15, 1])
with g1:
    st.image(str(FIGURES_DIR / "pasterze_2017_2025.gif"), width="stretch")
with g2:
    st.html(
        '<div class="muted" style="margin-top:.4rem">'
        '<span style="color:#2de1fc;font-weight:600">Cyan</span>: what the U-Net sees as glacier '
        "in that year's image.<br/>"
        '<span style="color:#ff3df2;font-weight:600">Magenta</span>: the official inventory '
        "outline from 2022.</div>"
    )
    st.markdown("""
Each frame is one late-summer satellite image. Watch the tongue of the glacier in the lower
middle: it gets thinner and shorter year by year. In some years leftover or fresh snow makes the
glacier look bigger. That's the main source of noise when you use one image per year.
""")
    if st.button("Open the glacier viewer  →", type="primary"):
        st.switch_page("views/explore.py")

st.divider()
st.caption(
    "A personal project by Kristina Krymova (TU Wien). Data: Copernicus Sentinel-2 and DEM, "
    "Austrian Glacier Inventory GI 4/GI 5 (PANGAEA), GeoSphere Austria, geoBoundaries."
)
