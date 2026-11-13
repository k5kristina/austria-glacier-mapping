"""Download and cache the real input data.

Sources (all open, no account needed):
- Sentinel-2 L2A surface reflectance and Copernicus DEM GLO-30 from the Element84 Earth Search STAC
- Austrian Glacier Inventory outlines GI 4 (2015) and GI 5 (2021-2023) from PANGAEA
- Monthly station temperatures from the GeoSphere Austria data hub
- The Austrian border from geoBoundaries (the inventory only maps Austrian ice, so pixels across
  the border are excluded rather than treated as "no glacier")

Every glacier gets a fixed 20 m grid in UTM 32N. Each band is warped onto that grid, so imagery,
DEM and inventory masks line up pixel by pixel. Nothing in this module modifies reflectance values.
"""

from __future__ import annotations

import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyproj
import rasterio
import requests
import shapefile
import shapely
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy.ndimage import binary_dilation
from shapely.geometry import box, mapping, shape

STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
PANGAEA_GI4_URL = "https://store.pangaea.de/Publications/Buckel-etal_2018/GI_4_2015.zip"
PANGAEA_GI5_URL = "https://download.pangaea.de/dataset/991106/files/GI5_vector_files.zip"
GEOSPHERE_URL = "https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-1m"
AUSTRIA_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/AUT/ADM0/"
    "geoBoundaries-AUT-ADM0.geojson"
)

GRID_CRS = "EPSG:32632"
S2_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
# Sentinel-2 scene classification codes treated as unusable: no data, saturated, cloud shadow,
# medium/high probability cloud, thin cirrus
SCL_INVALID = (0, 1, 3, 8, 9, 10)
SCL_SNOW = 11

# Official Austrian Glacier Inventory (AGI) IDs mapping our 19 benchmark glaciers across GI4 (2015) and GI5 (2021-2023)
GLACIERS: list[dict[str, Any]] = [
    # Oetztal Alps (Train / Validation)
    {
        "id": "AT_OET_01",
        "agi_id": 7022,
        "name": "Gepatschferner",
        "rgi_id": "RGI60-11.00746",
        "region": "Oetztal",
        "region_display": "Ötztal Alps",
        "gi5_file": "Oetztaler_Alpen_GI5_proc2.geojson",
        "split_subset": "train",
        "size_class": "large",
    },
    {
        "id": "AT_OET_02",
        "agi_id": 14008,
        "name": "Mittelbergferner",
        "rgi_id": "RGI60-11.00719",
        "region": "Oetztal",
        "region_display": "Ötztal Alps",
        "gi5_file": "Oetztaler_Alpen_GI5_proc2.geojson",
        "split_subset": "train",
        "size_class": "large",
    },
    {
        "id": "AT_OET_03",
        "agi_id": 2125,
        "name": "Hintereisferner",
        "rgi_id": "RGI60-11.00897",
        "region": "Oetztal",
        "region_display": "Ötztal Alps",
        "gi5_file": "Oetztaler_Alpen_GI5_proc2.geojson",
        "split_subset": "train",
        "size_class": "large",
    },
    {
        "id": "AT_OET_04",
        "agi_id": 2133,
        "name": "Vernagtferner",
        "rgi_id": "RGI60-11.00781",
        "region": "Oetztal",
        "region_display": "Ötztal Alps",
        "gi5_file": "Oetztaler_Alpen_GI5_proc2.geojson",
        "split_subset": "val",
        "size_class": "large",
    },
    {
        "id": "AT_OET_05",
        "agi_id": 2129,
        "name": "Kesselwandferner",
        "rgi_id": "RGI60-11.00787",
        "region": "Oetztal",
        "region_display": "Ötztal Alps",
        "gi5_file": "Oetztaler_Alpen_GI5_proc2.geojson",
        "split_subset": "val",
        "size_class": "large",
    },
    # Hohe Tauern (Train / Validation)
    {
        "id": "AT_HT_01",
        "agi_id": 4027,
        "name": "Pasterze",
        "rgi_id": "RGI60-11.00106",
        "region": "Hohe_Tauern",
        "region_display": "Hohe Tauern",
        "gi5_file": "Glocknergruppe_GI5_proc2.geojson",
        "split_subset": "train",
        "size_class": "large",
    },
    {
        "id": "AT_HT_02",
        "agi_id": 5129,
        "name": "Obersulzbachkees",
        "rgi_id": "RGI60-11.00042",
        "region": "Hohe_Tauern",
        "region_display": "Hohe Tauern",
        "gi5_file": "Venedigergruppe_GI5_proc2.geojson",
        "split_subset": "train",
        "size_class": "large",
    },
    {
        "id": "AT_HT_03",
        "agi_id": 6077,
        "name": "Schlatenkees",
        "rgi_id": "RGI60-11.00068",
        "region": "Hohe_Tauern",
        "region_display": "Hohe Tauern",
        "gi5_file": "Venedigergruppe_GI5_proc2.geojson",
        "split_subset": "train",
        "size_class": "large",
    },
    {
        "id": "AT_HT_04",
        "agi_id": 5030,
        "name": "Goldbergkees",
        "rgi_id": "RGI60-11.00012",
        "region": "Hohe_Tauern",
        "region_display": "Hohe Tauern",
        "gi5_file": "Goldberggruppe_GI5_proc2.geojson",
        "split_subset": "val",
        "size_class": "small",
    },
    {
        "id": "AT_HT_05",
        "agi_id": 4036,
        "name": "Kleinfleisskees",
        "rgi_id": "RGI60-11.00015",
        "region": "Hohe_Tauern",
        "region_display": "Hohe Tauern",
        "gi5_file": "Goldberggruppe_GI5_proc2.geojson",
        "split_subset": "val",
        "size_class": "small",
    },
    # Stubai Alps (Train / Validation)
    {
        "id": "AT_STU_01",
        "agi_id": 2022,
        "name": "Sulztalferner",
        "rgi_id": "RGI60-11.00561",
        "region": "Stubai",
        "region_display": "Stubai Alps",
        "gi5_file": "Stubaier_Alpen_GI5_proc2.geojson",
        "split_subset": "train",
        "size_class": "large",
    },
    {
        "id": "AT_STU_02",
        "agi_id": 3035,
        "name": "Schaufelferner",
        "rgi_id": "RGI60-11.00593",
        "region": "Stubai",
        "region_display": "Stubai Alps",
        "gi5_file": "Stubaier_Alpen_GI5_proc2.geojson",
        "split_subset": "train",
        "size_class": "small",
    },
    {
        "id": "AT_STU_03",
        "agi_id": 3055,
        "name": "Alpeiner Ferner",
        "rgi_id": "RGI60-11.00578",
        "region": "Stubai",
        "region_display": "Stubai Alps",
        "gi5_file": "Stubaier_Alpen_GI5_proc2.geojson",
        "split_subset": "val",
        "size_class": "small",
    },
    # Silvretta (Test A: Unseen Spatial Region)
    {
        "id": "AT_SIL_01",
        "agi_id": 13019,
        "name": "Jamtalferner",
        "rgi_id": "RGI60-11.00842",
        "region": "Silvretta",
        "region_display": "Silvretta",
        "gi5_file": "Silvrettagruppe_GI5_proc2.geojson",
        "split_subset": "test_a",
        "size_class": "large",
    },
    {
        "id": "AT_SIL_02",
        "agi_id": 12008,
        "name": "Ochsentaler Gletscher",
        "rgi_id": "RGI60-11.00836",
        "region": "Silvretta",
        "region_display": "Silvretta",
        "gi5_file": "Silvrettagruppe_GI5_proc2.geojson",
        "split_subset": "test_a",
        "size_class": "large",
    },
    {
        "id": "AT_SIL_03",
        "agi_id": 12007,
        "name": "Vermuntgletscher",
        "rgi_id": "RGI60-11.00839",
        "region": "Silvretta",
        "region_display": "Silvretta",
        "gi5_file": "Silvrettagruppe_GI5_proc2.geojson",
        "split_subset": "test_a",
        "size_class": "small",
    },
    # Dachstein (Test A: Unseen Spatial Region)
    {
        "id": "AT_DAC_01",
        "agi_id": 11002,
        "name": "Hallstätter Gletscher",
        "rgi_id": "RGI60-11.00001",
        "region": "Dachstein",
        "region_display": "Dachstein",
        "gi5_file": "Dachstein_GI5_proc2.geojson",
        "split_subset": "test_a",
        "size_class": "large",
    },
    {
        "id": "AT_DAC_02",
        "agi_id": 11001,
        "name": "Schladminger Gletscher",
        "rgi_id": "RGI60-11.00002",
        "region": "Dachstein",
        "region_display": "Dachstein",
        "gi5_file": "Dachstein_GI5_proc2.geojson",
        "split_subset": "test_a",
        "size_class": "small",
    },
    {
        "id": "AT_DAC_03",
        "agi_id": 11004,
        "name": "Grosser Gosaugletscher",
        "rgi_id": "RGI60-11.00003",
        "region": "Dachstein",
        "region_display": "Dachstein",
        "gi5_file": "Dachstein_GI5_proc2.geojson",
        "split_subset": "test_a",
        "size_class": "small",
    },
]

GEOSPHERE_STATIONS: dict[str, tuple[int, str, int]] = {
    "Oetztal": (66, "Obergurgl", 1941),
    "Hohe_Tauern": (213, "Sonnblick", 3109),
    "Stubai": (196, "Patscherkofel", 2251),
    "Silvretta": (28, "Galtür", 1587),
    # Closest long-running station to the Dachstein glaciers (about 25 km away)
    "Dachstein": (205, "Feuerkogel", 1618),
}


def _download(url: str, path: Path, timeout: int = 120) -> Path:
    if not path.exists():
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


# ----------------------------------------------------------------------------------------------
# Climate
# ----------------------------------------------------------------------------------------------
def fetch_geosphere_climate(years: list[int], cache_dir: Path) -> pd.DataFrame:
    """Summer (June-September) mean temperature and its anomaly vs. 1991-2020 per region."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "geosphere_klima_v2_1m_1991_2025.json"
    if cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        resp = requests.get(
            GEOSPHERE_URL,
            params={
                "parameters": "tl_mittel",
                "station_ids": ",".join(str(v[0]) for v in GEOSPHERE_STATIONS.values()),
                "start": "1991-01-01T00:00",
                "end": "2025-12-31T23:59",
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()
        cache_file.write_text(json.dumps(raw), encoding="utf-8")

    timestamps = pd.to_datetime(raw["timestamps"])
    by_station: dict[int, pd.Series] = {}
    for feat in raw["features"]:
        values = feat["properties"]["parameters"]["tl_mittel"]["data"]
        by_station[int(feat["properties"]["station"])] = pd.Series(
            pd.to_numeric(values, errors="coerce"), index=timestamps
        )

    rows: list[dict[str, Any]] = []
    for region, (sid, name, elev) in GEOSPHERE_STATIONS.items():
        series = by_station[sid]
        summer = series[series.index.month.isin([6, 7, 8, 9])]
        # Only average complete summers (all four months present)
        counts = summer.groupby(summer.index.year).count()
        jjas = summer.groupby(summer.index.year).mean()[counts == 4]
        baseline = float(jjas.loc[1991:2020].mean())
        for yr in years:
            temp = float(jjas.get(yr, np.nan))
            rows.append(
                {
                    "region": region,
                    "station_id": sid,
                    "station_name": name,
                    "station_elev_m": elev,
                    "year": int(yr),
                    "jjas_mean_temp_c": round(temp, 2),
                    "jjas_anomaly_c": round(temp - baseline, 2),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Glacier inventory
# ----------------------------------------------------------------------------------------------
def _glacier_grid(geom_utm: Any, pixel_size_m: float) -> dict[str, Any]:
    """Square grid around a glacier with 25% padding; side length divisible by 8 for the U-Net."""
    minx, miny, maxx, maxy = geom_utm.bounds
    cx, cy = 0.5 * (minx + maxx), 0.5 * (miny + maxy)
    half = max(0.5 * 1.25 * max(maxx - minx, maxy - miny), 650.0)
    n_px = int(np.ceil(2 * half / pixel_size_m / 8.0) * 8)
    left = cx - 0.5 * n_px * pixel_size_m
    top = cy + 0.5 * n_px * pixel_size_m
    bounds = [left, top - n_px * pixel_size_m, left + n_px * pixel_size_m, top]
    to_wgs = pyproj.Transformer.from_crs(GRID_CRS, "EPSG:4326", always_xy=True)
    wgs = to_wgs.transform_bounds(*bounds)
    return {"utm_bounds": bounds, "size_px": n_px, "wgs_bbox": list(wgs)}


def load_glacier_outlines(cache_dir: Path, pixel_size_m: float) -> dict[str, dict[str, Any]]:
    """Match the study glaciers in GI 4 / GI 5 and collect every GI 5 polygon inside each grid.

    The training label is *all* GI 5 ice inside the grid (neighbouring glaciers included), so that
    correctly detected neighbouring ice is not counted as a false positive.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"glacier_outlines_v2_{int(pixel_size_m)}m.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    gi4_zip = zipfile.ZipFile(_download(PANGAEA_GI4_URL, cache_dir / "GI_4_2015.zip"))
    gi5_zip = zipfile.ZipFile(_download(PANGAEA_GI5_URL, cache_dir / "GI5_vector_files.zip"))
    to_utm = pyproj.Transformer.from_crs("EPSG:31287", GRID_CRS, always_xy=True)
    wgs_to_utm = pyproj.Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)

    def project(geom: Any, transformer: pyproj.Transformer = to_utm) -> Any:
        return shapely.transform(
            geom, lambda c: np.column_stack(transformer.transform(c[:, 0], c[:, 1]))
        )

    austria_file = _download(AUSTRIA_URL, cache_dir / "geoBoundaries-AUT-ADM0.geojson")
    austria_geojson = json.loads(austria_file.read_text(encoding="utf-8"))
    austria = project(shape(austria_geojson["features"][0]["geometry"]), wgs_to_utm)

    reader = shapefile.Reader(
        shp=io.BytesIO(gi4_zip.read("GI_4_2015/GI_4_2015.shp")),
        shx=io.BytesIO(gi4_zip.read("GI_4_2015/GI_4_2015.shx")),
        dbf=io.BytesIO(gi4_zip.read("GI_4_2015/GI_4_2015.dbf")),
        encoding="latin1",
    )
    gi4_by_id: dict[int, list[Any]] = {}
    for sr in reader.iterShapeRecords():
        gi4_by_id.setdefault(int(sr.record.as_dict()["nr"]), []).append(
            shape(sr.shape.__geo_interface__)
        )

    gi5_all: list[tuple[int, int, Any]] = []  # (agi id, year, geometry in UTM)
    for name in gi5_zip.namelist():
        if not name.endswith(".geojson"):
            continue
        for feat in json.loads(gi5_zip.read(name).decode("utf-8"))["features"]:
            props = feat["properties"]
            gid = int(float(props.get("id") or -1))
            gi5_all.append((gid, int(props.get("year") or 0), project(shape(feat["geometry"]))))
    tree = shapely.STRtree([g for _, _, g in gi5_all])

    outlines: dict[str, dict[str, Any]] = {}
    for spec in GLACIERS:
        agi_id = int(spec["agi_id"])
        own = [(yr, g) for gid, yr, g in gi5_all if gid == agi_id]
        gi5_target = shapely.unary_union([g for _, g in own])
        gi5_year = max(yr for yr, _ in own)

        gi4_geoms = gi4_by_id.get(agi_id)
        gi4_target = project(shapely.unary_union(gi4_geoms)) if gi4_geoms else None
        # Some AGI ids were renumbered between GI 4 and GI 5; only trust overlapping matches
        gi4_matched = bool(
            gi4_target is not None
            and gi4_target.intersection(gi5_target).area > 0.5 * gi5_target.area
        )

        extent = gi4_target.union(gi5_target) if gi4_matched else gi5_target
        grid = _glacier_grid(extent, pixel_size_m)
        grid_box = box(*grid["utm_bounds"])
        all_ice = shapely.unary_union(
            [gi5_all[i][2] for i in tree.query(grid_box, predicate="intersects")]
        ).intersection(grid_box)

        # Area used for this glacier's time series: its 2015 outline (or the GI 5 outline when GI 4
        # could not be matched) plus a 60 m buffer, so a shrinking glacier stays inside the region
        region = (gi4_target if gi4_matched else gi5_target).buffer(60.0)

        outlines[spec["id"]] = {
            **spec,
            "gi5_year": gi5_year,
            "gi4_matched": gi4_matched,
            "area_gi5_km2": round(gi5_target.area / 1e6, 4),
            "area_gi4_km2": round(gi4_target.area / 1e6, 4) if gi4_matched else None,
            **grid,
            "gi5_target_utm": mapping(gi5_target),
            "gi5_all_ice_utm": mapping(all_ice),
            "region_utm": mapping(region),
            "austria_utm": mapping(austria.intersection(grid_box)),
        }

    cache_file.write_text(json.dumps(outlines), encoding="utf-8")
    return outlines


def rasterize_on_grid(
    geojson: dict[str, Any], gmeta: dict[str, Any], pixel_size_m: float
) -> np.ndarray:
    """Rasterize a UTM geometry onto the glacier grid (pixel centres inside the polygon)."""
    geom = shape(geojson)
    n = gmeta["size_px"]
    if geom.is_empty:
        return np.zeros((n, n), dtype=np.uint8)
    left, _, _, top = gmeta["utm_bounds"]
    return rasterize(
        [(geom, 1)],
        out_shape=(n, n),
        transform=from_origin(left, top, pixel_size_m, pixel_size_m),
        fill=0,
        dtype=np.uint8,
    )


# ----------------------------------------------------------------------------------------------
# Raster access
# ----------------------------------------------------------------------------------------------
def _read_on_grid(
    href: str, gmeta: dict[str, Any], pixel_size_m: float, resampling: Resampling
) -> np.ndarray:
    """Warp one band of a remote Cloud-Optimized GeoTIFF onto the glacier grid."""
    n = gmeta["size_px"]
    left, _, _, top = gmeta["utm_bounds"]
    env = rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4"
    )
    with env, rasterio.open(href) as src:
        with WarpedVRT(
            src,
            crs=GRID_CRS,
            transform=from_origin(left, top, pixel_size_m, pixel_size_m),
            width=n,
            height=n,
            resampling=resampling,
            src_nodata=0,
            nodata=0,
        ) as vrt:
            return vrt.read(1)


def _stac_search(body: dict[str, Any]) -> list[dict[str, Any]]:
    resp = requests.post(STAC_SEARCH_URL, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json().get("features", [])


def fetch_dem(gmeta: dict[str, Any], pixel_size_m: float) -> np.ndarray:
    """Copernicus DEM GLO-30 on the glacier grid; merges tiles when the grid spans several."""
    items = _stac_search(
        {"collections": ["cop-dem-glo-30"], "bbox": gmeta["wgs_bbox"], "limit": 10}
    )
    dem = np.zeros((gmeta["size_px"],) * 2, dtype=np.float32)
    for item in items:
        part = _read_on_grid(
            item["assets"]["data"]["href"], gmeta, pixel_size_m, Resampling.bilinear
        )
        dem = np.where(dem > 0, dem, part.astype(np.float32))
    if np.any(dem <= 0):
        raise RuntimeError(f"DEM has gaps for {gmeta['id']}")
    return dem


def _reflectance_offset(item: dict[str, Any]) -> float:
    """Processing baseline >= 04.00 stores reflectance with a +1000 offset unless already removed."""
    props = item["properties"]
    if props.get("earthsearch:boa_offset_applied", False):
        return 0.0
    baseline = float(props.get("s2:processing_baseline", "0") or 0)
    return -1000.0 if baseline >= 4.0 else 0.0


def select_scene(
    gmeta: dict[str, Any],
    year: int,
    off_glacier: np.ndarray,
    inside: np.ndarray,
    pixel_size_m: float,
    max_candidates: int = 25,
) -> tuple[dict[str, Any], np.ndarray, dict[str, float]]:
    """Pick the late-summer scene with the fewest unusable pixels and least off-glacier snow.

    Late summer (mid-July to mid-October) is when glaciers are least covered by seasonal snow.
    Candidates are ranked by tile-level cloud cover, then the scene classification layer is read
    inside the grid (Austrian part only, ``inside``). ``off_glacier`` marks terrain well away from
    any inventory outline; snow there means seasonal snow, which makes glacier mapping harder, so
    such scenes are penalised. The search stops at the first clean scene.
    """
    items = _stac_search(
        {
            "collections": ["sentinel-2-l2a"],
            "intersects": mapping(box(*gmeta["wgs_bbox"])),
            "datetime": f"{year}-07-15T00:00:00Z/{year}-10-15T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": 60}},
            "limit": 200,
        }
    )
    if not items:
        raise RuntimeError(f"No Sentinel-2 scene for {gmeta['id']} in {year}")
    items.sort(key=lambda it: float(it["properties"].get("eo:cloud_cover", 100.0)))

    best: tuple[float, dict[str, Any], np.ndarray, dict[str, float]] | None = None
    for item in items[:max_candidates]:
        scl = _read_on_grid(item["assets"]["scl"]["href"], gmeta, pixel_size_m, Resampling.nearest)
        invalid = float(np.mean(np.isin(scl[inside], SCL_INVALID)))
        off_snow = float(np.mean(scl[off_glacier] == SCL_SNOW)) if off_glacier.any() else 0.0
        score = invalid + 0.5 * off_snow
        stats = {"invalid_fraction": round(invalid, 4), "off_glacier_snow": round(off_snow, 4)}
        if best is None or score < best[0]:
            best = (score, item, scl, stats)
        if invalid < 0.01 and off_snow < 0.02:
            break
    assert best is not None
    return best[1], best[2], best[3]


def fetch_glacier_year(
    gmeta: dict[str, Any],
    year: int,
    off_glacier: np.ndarray,
    inside: np.ndarray,
    pixel_size_m: float,
    out_file: Path,
) -> dict[str, Any]:
    """Download one glacier-year: six reflectance bands (uint16, x10000) and the SCL layer."""
    if out_file.exists():
        with np.load(out_file) as z:
            return json.loads(str(z["meta"]))

    item, scl, stats = select_scene(gmeta, year, off_glacier, inside, pixel_size_m)
    offset = _reflectance_offset(item)
    bands = []
    for key in S2_BANDS:
        dn = _read_on_grid(item["assets"][key]["href"], gmeta, pixel_size_m, Resampling.bilinear)
        dn = dn.astype(np.float32)
        refl = np.where(dn > 0, dn + offset, 0.0)
        bands.append(np.clip(refl, 0, 65535).astype(np.uint16))

    meta = {
        "glacier_id": gmeta["id"],
        "year": int(year),
        "scene_id": item["id"],
        "acquisition_date": item["properties"]["datetime"][:10],
        "tile_cloud_cover": round(float(item["properties"].get("eo:cloud_cover", np.nan)), 2),
        **stats,
    }
    np.savez_compressed(
        out_file, bands=np.stack(bands), scl=scl.astype(np.uint8), meta=json.dumps(meta)
    )
    return meta


def download_all(
    cache_dir: Path, raw_dir: Path, years: list[int], pixel_size_m: float, workers: int = 8
) -> dict[str, dict[str, Any]]:
    """Download (or reuse cached) DEM and per-year imagery for every study glacier."""
    outlines = load_glacier_outlines(cache_dir, pixel_size_m)
    (raw_dir / "s2").mkdir(parents=True, exist_ok=True)
    (raw_dir / "dem").mkdir(parents=True, exist_ok=True)

    off_glacier: dict[str, np.ndarray] = {}
    inside: dict[str, np.ndarray] = {}
    for gid, gmeta in outlines.items():
        dem_file = raw_dir / "dem" / f"{gid}.npy"
        if not dem_file.exists():
            np.save(dem_file, fetch_dem(gmeta, pixel_size_m))
        ice = rasterize_on_grid(gmeta["gi5_all_ice_utm"], gmeta, pixel_size_m).astype(bool)
        region = rasterize_on_grid(gmeta["region_utm"], gmeta, pixel_size_m).astype(bool)
        inside[gid] = rasterize_on_grid(gmeta["austria_utm"], gmeta, pixel_size_m).astype(bool)
        # In Austria and at least ~200 m away from any glacier outline
        off_glacier[gid] = inside[gid] & ~binary_dilation(
            ice | region, iterations=int(200 / pixel_size_m)
        )

    jobs = [(gid, yr) for gid in outlines for yr in years]
    px = pixel_size_m

    def run(job: tuple[str, int]) -> dict[str, Any]:
        gid, yr = job
        out = raw_dir / "s2" / f"{gid}_{yr}.npz"
        return fetch_glacier_year(outlines[gid], yr, off_glacier[gid], inside[gid], px, out)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for meta in pool.map(run, jobs):
            print(
                f"[ingest] {meta['glacier_id']} {meta['year']}: {meta['scene_id']} "
                f"invalid={meta['invalid_fraction']:.1%} off-glacier snow={meta['off_glacier_snow']:.1%}"
            )
    return outlines
