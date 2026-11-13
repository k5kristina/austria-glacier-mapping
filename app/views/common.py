"""Shared data loading, colours, styling and drawing helpers for the website."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

MODELS = ["ndsi", "rf", "unet"]
MODEL_LABELS = {"ndsi": "NDSI threshold", "rf": "Random Forest", "unet": "U-Net"}
COLORS = {
    "inventory": "#ff3df2",  # magenta
    "ndsi": "#ffd23f",  # yellow
    "rf": "#ff8c42",  # orange
    "unet": "#2de1fc",  # cyan
}
SPLIT_LABELS = {
    "train": "training glacier",
    "val": "validation glacier",
    "test": "test glacier (unseen region)",
}

INK = "#0a0f1c"
ACCENT = "#9fe3c9"  # soft icy mint (same as the hero)
ACCENT_DARK = "#2a9d78"  # readable mint for text and marks on white
MUTED = "#9aa5b1"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, p, li, label { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; letter-spacing: -0.03em;
  font-weight: 800 !important; color: #0a0f1c !important; }
.block-container { padding-top: 1.2rem; max-width: 1200px; }
.stMarkdown p, .stMarkdown li { line-height: 1.65; }
/* small accent label above a heading */
.eyebrow { color: #2a9d78; font-size: .78rem; font-weight: 600; letter-spacing: .14em;
  text-transform: uppercase; margin-bottom: -.4rem; }
.stButton button { border-radius: 12px; font-weight: 700; }
/* dark stat cards echo the hero */
.stat-card { border-radius: 18px; padding: 1.4rem 1.5rem; height: 100%; background: #0a0f1c;
  color: #fff; }
.stat-card .num { font-size: 2.4rem; font-weight: 800; line-height: 1.1; color: #9fe3c9;
  letter-spacing: -.03em; }
.stat-card .lbl { font-size: .95rem; color: rgba(255,255,255,.72); margin-top: .4rem;
  line-height: 1.5; }
.card-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem;
  align-items: stretch; }
@media (max-width: 800px) { .card-grid { grid-template-columns: 1fr; } }
.step { background: #f5f7fa; border: 1px solid #e6e9ef; border-radius: 18px;
  padding: 1.2rem 1.3rem; height: 100%; }
.step .n { display: inline-block; width: 28px; height: 28px; border-radius: 8px;
  background: rgba(159,227,201,.35); color: #1f7a5d; font-weight: 700; text-align: center;
  line-height: 28px; margin-bottom: .6rem; }
.step h4 { margin: 0 0 .35rem 0; font-size: 1.05rem; }
.muted { opacity: .75; font-size: .93rem; line-height: 1.55; }
.legend-chip { display: inline-flex; align-items: center; gap: .45rem; margin-right: 1.1rem;
  font-size: .9rem; }
.legend-chip span.swatch { width: 22px; height: 4px; border-radius: 2px;
  box-shadow: 0 0 0 1.5px #141414; display: inline-block; }
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def outputs_available() -> bool:
    return (PROCESSED_DIR / "app_meta.json").exists()


@st.cache_resource
def load_bundle() -> tuple[dict[str, Any], Any]:
    meta = json.loads((PROCESSED_DIR / "app_meta.json").read_text(encoding="utf-8"))
    return meta, np.load(PROCESSED_DIR / "app_bundle.npz")


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / name)


@st.cache_data
def load_summary() -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "reports" / "summary.json").read_text(encoding="utf-8"))


def glacier_ids(meta: dict[str, Any]) -> list[str]:
    return [k for k in meta if not k.startswith("_")]


def unpack(bits: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(bits)[: n * n].reshape(n, n).astype(bool)


def show_gif(path: Path) -> None:
    """Show an animated GIF scaled to its container.

    Never pass a pixel ``width`` to ``st.image`` for a GIF: Streamlit then resizes it on the
    server and keeps only the first frame. Stretching to the column is done by the browser.
    """
    st.image(str(path), width="stretch")


def eyebrow(text: str) -> None:
    """Small accent label shown above a section heading."""
    st.html(f'<div class="eyebrow">{text}</div>')


def legend_html(keys: list[str]) -> str:
    names = {"inventory": "Glacier inventory (GI 5)", **MODEL_LABELS}
    chips = "".join(
        f'<span class="legend-chip"><span class="swatch" style="background:{COLORS[k]}"></span>'
        f"{names[k]}</span>"
        for k in keys
    )
    return f"<div>{chips}</div>"


def outline(ax: Any, mask: np.ndarray, color: str, lw: float = 1.3) -> None:
    """Coloured contour with a dark halo so it stays visible on bright snow."""
    if mask.any() and not mask.all():
        cs = ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=lw)
        cs.set(path_effects=[pe.Stroke(linewidth=lw + 1.6, foreground="#141414"), pe.Normal()])


def glacier_figure(
    rgb: np.ndarray,
    layers: list[tuple[np.ndarray, str]],
    invalid: np.ndarray | None = None,
    title: str | None = None,
    size: float = 6.0,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(rgb)
    if invalid is not None and invalid.any():
        overlay = np.zeros((*invalid.shape, 4))
        overlay[invalid] = (0.1, 0.1, 0.1, 0.55)
        ax.imshow(overlay)
    for mask, color in layers:
        outline(ax, mask, color)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout(pad=0.2)
    return fig


@st.cache_data(max_entries=300, show_spinner=False)
def glacier_png(
    gid: str,
    year: int,
    layers: tuple[str, ...],
    shade_clouds: bool,
    title: str | None = None,
    size: float = 6.0,
) -> bytes:
    """Rendered glacier view as PNG, cached so revisiting a year is instant."""
    meta, bundle = load_bundle()
    n = meta[gid]["size_px"]
    rgb = np.transpose(bundle[f"{gid}_{year}_rgb"], (1, 2, 0))
    invalid = unpack(bundle[f"{gid}_{year}_invalid"], n) if shade_clouds else None
    masks = [
        (
            unpack(
                bundle[f"{gid}_label"] if k == "inventory" else bundle[f"{gid}_{year}_pred_{k}"], n
            ),
            COLORS[k],
        )
        for k in layers
    ]
    fig = glacier_figure(rgb, masks, invalid, title, size)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
