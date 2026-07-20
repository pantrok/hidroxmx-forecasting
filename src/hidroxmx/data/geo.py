"""Geospatial loaders shared by the coverage-map stage (§7 of the spec).

Design goal: keep the mapping between the pilot basins and the outer world
(HydroRIVERS, Flood Hub, GloFAS) in a single place so the coverage map (Fig. 1)
can be reproduced from either the canonical reach lists (when we have them
locally) or, as a fallback, from HydroRIVERS with the documented drainage
thresholds used by Flood Hub and GloFAS.
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Public download for the HydroRIVERS v1.0 North America regional shapefile.
# HydroSHEDS distributes it under the HydroSHEDS Terms of Use (free for
# research; attribution required). Documentation: https://www.hydrosheds.org.
HYDRORIVERS_NA_URL = (
    "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_na_shp.zip"
)

# Drainage-area thresholds (km^2) documented by the two global providers we
# want to overlay. The values below are the thresholds we use to convert
# HydroRIVERS reaches into a *proxy* Flood-Hub / GloFAS coverage layer when
# the canonical reach lists are not on hand.
#
# GloFAS: Harrigan et al., 2020 (v3.1 river network at 0.1 deg; effective
# drainage-area threshold ~ 500 km^2). Documented in
# https://doi.org/10.5194/essd-12-2043-2020
GLOFAS_UPLAND_THRESHOLD_KM2 = 500.0

# Flood Hub: Nearing et al., 2024 (Nature) — global model published with
# coverage below the historically ungauged 1000+ km^2 threshold; the
# operational service publishes forecasts on reaches with drainage areas of
# the order of a few tens of km^2. We use 25 km^2 as a conservative proxy
# lower bound.
FLOOD_HUB_UPLAND_THRESHOLD_KM2 = 25.0

# Rough Mexico bounding box in EPSG:4326 (min_lon, min_lat, max_lon, max_lat).
MEXICO_BBOX = (-118.5, 14.0, -86.0, 33.0)

# Slug → display name mapping used in the sub-basin loader.
BASIN_DISPLAY = {
    "cutzamala": "Cutzamala",
    "lerma_alto": "Lerma Alto",
    "bajio": "Bajío",
    "santiago": "Santiago",
    "panuco": "Pánuco",
    "alta_del_balsas": "Alta del Balsas",
}


@dataclass(slots=True)
class HydroRiversHandle:
    """Path plus the drainage-area column name inside the shapefile."""

    path: Path
    upland_col: str = "UPLAND_SKM"
    length_col: str = "LENGTH_KM"


# --------------------------------------------------------------------------- #
# Pilot sub-basins (produced by the hidroxai-mx dataset pipeline)
# --------------------------------------------------------------------------- #
def load_pilot_subbasins(sources: Iterable[Path]):
    """Concatenate every GeoPackage in ``sources`` into a single GeoDataFrame.

    Each source path must expose ``<basin_slug>.gpkg`` with a ``cuenca`` layer
    (as delivered by the hidroxai-mx dataset). The returned GeoDataFrame
    contains at least the columns ``basin`` (display name), ``basin_slug``,
    ``subbasin_id`` and ``geometry`` in EPSG:4326.
    """
    import geopandas as gpd
    import pandas as pd

    frames = []
    for src in sources:
        src = Path(src)
        if not src.exists():
            continue
        slug = src.stem
        gdf = gpd.read_file(src, layer="cuenca")
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        gdf = gdf.assign(
            basin=BASIN_DISPLAY.get(slug, slug),
            basin_slug=slug,
            subbasin_id=[f"{slug}_{i:03d}" for i in range(len(gdf))],
        )
        frames.append(gdf)
    if not frames:
        raise FileNotFoundError(
            f"no sub-basin GeoPackage found in the requested sources ({list(sources)})"
        )
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    keep = ["basin", "basin_slug", "subbasin_id", "geometry"]
    for extra in ("area_km2", "elevacion_media_m", "pendiente_media", "clave_estacion"):
        if extra in out.columns:
            keep.append(extra)
    return out[keep]


def default_hidroxai_gpkg_sources(root: Path | None = None) -> list[Path]:
    """Return the six GeoPackages produced by ``hidroxai-mx``.

    Search order for the ``root`` folder (first hit wins):

    1. Explicit ``root`` argument.
    2. ``HIDROXAI_MX_CUENCAS`` environment variable pointing to the
       ``.../data/processed/cuencas`` folder.
    3. ``HIDROXAI_MX_ROOT`` environment variable pointing to the repository
       root; the ``cuencas`` folder is joined automatically.
    4. Two conventional sibling paths of this repository:
       ``../hidroxai-mx/data/processed/cuencas`` and
       ``../Data/Código/hidroxai-mx/data/processed/cuencas``
       (the latter matches the layout after the reshuffle of July 2026).
    """
    if root is None:
        env_direct = os.environ.get("HIDROXAI_MX_CUENCAS")
        env_repo = os.environ.get("HIDROXAI_MX_ROOT")
        candidates: list[Path] = []
        if env_direct:
            candidates.append(Path(env_direct))
        if env_repo:
            candidates.append(Path(env_repo) / "data" / "processed" / "cuencas")
        here = Path(__file__).resolve().parents[3].parent
        candidates.extend([
            here / "hidroxai-mx" / "data" / "processed" / "cuencas",
            here.parent / "Data" / "Código" / "hidroxai-mx" / "data" / "processed" / "cuencas",
        ])
        root = next((c for c in candidates if c.exists()), candidates[-1])
    return [root / f"{slug}.gpkg" for slug in BASIN_DISPLAY]


# --------------------------------------------------------------------------- #
# HydroRIVERS (proxy source for the coverage overlay)
# --------------------------------------------------------------------------- #
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _download_via_requests(url: str, out: Path, chunk_bytes: int, retries: int) -> None:
    import time

    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": _BROWSER_UA, "Accept": "*/*"})
    tmp = out.with_suffix(out.suffix + ".part")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(out)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)
    if last_exc is not None:
        raise last_exc


def _download_via_curl(url: str, out: Path) -> None:
    """Fallback to curl. Some Python / urllib3 / OpenSSL combinations trip on
    Cloudflare handshakes where curl (OpenSSL directly) does not."""
    import shutil
    import subprocess

    if shutil.which("curl") is None:
        raise RuntimeError("curl is not on PATH; cannot use it as a download fallback")
    tmp = out.with_suffix(out.suffix + ".part")
    cmd = [
        "curl", "-fL", "--retry", "3", "--retry-delay", "2",
        "-A", _BROWSER_UA,
        "-o", str(tmp), url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"curl download failed (exit {result.returncode}): {result.stderr[-400:]}"
        )
    tmp.replace(out)


def ensure_hydrorivers(cache_dir: Path, *, url: str = HYDRORIVERS_NA_URL,
                      retries: int = 3, chunk_bytes: int = 1 << 20) -> HydroRiversHandle:
    """Download the HydroRIVERS North America shapefile if it isn't cached yet.

    HydroSHEDS is fronted by Cloudflare, so a browser-style ``User-Agent`` is
    required. Some Python / OpenSSL versions fail the TLS handshake against
    Cloudflare where system ``curl`` succeeds; therefore the function tries
    ``requests`` first and falls back to ``curl`` on any exception.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    shp_dir = cache_dir / "HydroRIVERS_v10_na"
    shp = shp_dir / "HydroRIVERS_v10_na.shp"
    if shp.exists():
        return HydroRiversHandle(path=shp)

    zip_path = cache_dir / "HydroRIVERS_v10_na_shp.zip"
    if not zip_path.exists():
        try:
            _download_via_requests(url, zip_path, chunk_bytes, retries)
        except Exception:  # noqa: BLE001
            _download_via_curl(url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(shp_dir)
    if not shp.exists():
        candidates = list(shp_dir.rglob("HydroRIVERS_v10_na.shp"))
        if not candidates:
            raise FileNotFoundError("HydroRIVERS shapefile not found after unzip")
        shp = candidates[0]
    return HydroRiversHandle(path=shp)


def load_hydrorivers_subset(handle: HydroRiversHandle, bbox=MEXICO_BBOX):
    """Read HydroRIVERS restricted to ``bbox`` (Mexico by default)."""
    import geopandas as gpd

    gdf = gpd.read_file(handle.path, bbox=bbox)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def proxy_reaches(
    rivers,
    handle: HydroRiversHandle,
    upland_threshold_km2: float,
):
    """Filter HydroRIVERS reaches by drainage-area threshold."""
    col = handle.upland_col
    if col not in rivers.columns:
        raise KeyError(
            f"HydroRIVERS shapefile does not expose the drainage-area column {col!r}; "
            f"available columns: {sorted(rivers.columns)[:15]} ..."
        )
    return rivers[rivers[col] >= upland_threshold_km2].copy()


# --------------------------------------------------------------------------- #
# Canonical reach lists (used when locally available)
# --------------------------------------------------------------------------- #
def load_flood_hub_reaches(source: Path | None):
    """Load a GeoJSON / shapefile of Flood Hub gauged reaches if provided."""
    if source is None:
        return None
    import geopandas as gpd

    gdf = gpd.read_file(source)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def load_glofas_reaches(source: Path | None):
    """Load a GeoJSON / shapefile of GloFAS reaches if provided."""
    if source is None:
        return None
    import geopandas as gpd

    gdf = gpd.read_file(source)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf
