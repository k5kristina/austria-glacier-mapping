"""Animated GIFs (one frame per year) built from the saved viewer bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

INVENTORY_RGB = (255, 61, 242)  # magenta
MODEL_RGB = (45, 225, 252)  # cyan
HALO_RGB = (20, 20, 20)
# Skip a year when more than this share of the image has no satellite data (scene edge)
MAX_NO_DATA = 0.25


def no_data_fraction(rgb_chw: np.ndarray) -> float:
    """Share of pixels outside the satellite scene (all bands zero)."""
    return float((rgb_chw.max(axis=0) == 0).mean())


def _edge(mask: np.ndarray, width: int) -> np.ndarray:
    return mask & ~binary_erosion(mask, iterations=width)


def draw_outline(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    """Paint a coloured outline with a dark halo so it stays visible on bright snow."""
    if not mask.any():
        return
    halo = binary_dilation(_edge(mask, 1), iterations=1)
    rgb[halo] = HALO_RGB
    rgb[_edge(mask, 1)] = color


def _unpack(bits: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(bits)[: n * n].reshape(n, n).astype(bool)


def build_gifs(processed_dir: Path, figures_dir: Path, model: str = "unet", scale: int = 2) -> None:
    """One GIF per glacier in ``data/processed/gifs``; Pasterze is also copied to the figures."""
    meta = json.loads((processed_dir / "app_meta.json").read_text(encoding="utf-8"))
    bundle = np.load(processed_dir / "app_bundle.npz")
    out_dir = processed_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18)

    for gid, gmeta in meta.items():
        if gid.startswith("_"):
            continue
        n = gmeta["size_px"]
        label = _unpack(bundle[f"{gid}_label"], n)
        frames = []
        for yr in meta["_years"]:
            if no_data_fraction(bundle[f"{gid}_{yr}_rgb"]) > MAX_NO_DATA:
                continue
            rgb = np.transpose(bundle[f"{gid}_{yr}_rgb"], (1, 2, 0)).copy()
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
            up = np.ones((scale, scale), dtype=bool)
            draw_outline(rgb, np.kron(label, up).astype(bool), INVENTORY_RGB)
            draw_outline(
                rgb,
                np.kron(_unpack(bundle[f"{gid}_{yr}_pred_{model}"], n), up).astype(bool),
                MODEL_RGB,
            )
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 70, 28], fill=HALO_RGB)
            draw.text((8, 4), str(yr), fill=(255, 255, 255), font=font)
            frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        frames[0].save(
            out_dir / f"{gid}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=900,
            loop=0,
            optimize=True,
        )
    shutil.copy(out_dir / "AT_HT_01.gif", figures_dir / "pasterze_2017_2025.gif")
