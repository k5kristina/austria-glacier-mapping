"""Model input features: Sentinel-2 bands, spectral indices and terrain derivatives."""

from __future__ import annotations

import numpy as np

CHANNELS = [
    "B02_blue",
    "B03_green",
    "B04_red",
    "B08_nir",
    "B11_swir1",
    "B12_swir2",
    "NDSI",
    "NDWI",
    "NDVI",
    "elevation_km",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "hillshade",
]
NDSI_INDEX = CHANNELS.index("NDSI")


def compute_spectral_indices(
    b03_green: np.ndarray,
    b04_red: np.ndarray,
    b08_nir: np.ndarray,
    b11_swir1: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Compute NDSI, NDWI, and NDVI from Sentinel-2 surface reflectance bands.

    - NDSI = (Green - SWIR1) / (Green + SWIR1)  [Dozier 1989]
    - NDWI = (Green - NIR)   / (Green + NIR)
    - NDVI = (NIR - Red)     / (NIR + Red)
    """
    ndsi = (b03_green - b11_swir1) / (b03_green + b11_swir1 + eps)
    ndwi = (b03_green - b08_nir) / (b03_green + b08_nir + eps)
    ndvi = (b08_nir - b04_red) / (b08_nir + b04_red + eps)
    return {
        "NDSI": np.clip(ndsi, -1.0, 1.0).astype(np.float32),
        "NDWI": np.clip(ndwi, -1.0, 1.0).astype(np.float32),
        "NDVI": np.clip(ndvi, -1.0, 1.0).astype(np.float32),
    }


def compute_topographic_features(
    dem_m: np.ndarray,
    pixel_size_m: float = 20.0,
    sun_azimuth_deg: float = 155.0,
    sun_elevation_deg: float = 42.0,
) -> dict[str, np.ndarray]:
    """Derive elevation (km), slope (deg), aspect sin/cos, and analytical hillshade from DEM.

    Parameters
    ----------
    dem_m : np.ndarray
        2D Digital Elevation Model array in meters above sea level.
    pixel_size_m : float
        Grid spacing in meters (20 m grid, resampled from Copernicus GLO-30).
    sun_azimuth_deg : float
        Late-summer mid-morning solar azimuth in the Austrian Alps (degrees clockwise from North).
    sun_elevation_deg : float
        Solar altitude angle above horizon in degrees.
    """
    dz_dy, dz_dx = np.gradient(dem_m.astype(np.float64), pixel_size_m)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Aspect = downslope direction, clockwise from north. Rows run north -> south, so the
    # northward downhill component is +dz_dy and the eastward one is -dz_dx.
    aspect_rad = np.mod(np.arctan2(-dz_dx, dz_dy), 2.0 * np.pi)
    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)

    zenith_rad = np.radians(90.0 - sun_elevation_deg)
    azimuth_rad = np.radians(sun_azimuth_deg)

    hillshade = np.cos(zenith_rad) * np.cos(slope_rad) + np.sin(zenith_rad) * np.sin(
        slope_rad
    ) * np.cos(azimuth_rad - aspect_rad)
    hillshade = np.clip(hillshade, 0.0, 1.0).astype(np.float32)

    return {
        "Elevation_km": (dem_m / 1000.0).astype(np.float32),
        "Slope_deg": slope_deg,
        "Aspect_sin": aspect_sin,
        "Aspect_cos": aspect_cos,
        "Hillshade": hillshade,
    }


def build_feature_stack(
    bands: np.ndarray, dem_m: np.ndarray, pixel_size_m: float = 20.0
) -> np.ndarray:
    """Stack the 14 input channels from six reflectance bands (uint16, x10000) and a DEM.

    Returns an array of shape ``(14, H, W)``. Only the image and the DEM go in: no labels.
    """
    refl = bands.astype(np.float32) / 10000.0
    b02, b03, b04, b08, b11, b12 = refl
    idx = compute_spectral_indices(b03, b04, b08, b11)
    topo = compute_topographic_features(dem_m, pixel_size_m=pixel_size_m)
    return np.stack(
        [
            b02,
            b03,
            b04,
            b08,
            b11,
            b12,
            idx["NDSI"],
            idx["NDWI"],
            idx["NDVI"],
            topo["Elevation_km"],
            topo["Slope_deg"],
            topo["Aspect_sin"],
            topo["Aspect_cos"],
            topo["Hillshade"],
        ]
    ).astype(np.float32)
