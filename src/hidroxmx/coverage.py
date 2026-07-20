"""Coverage analysis: which pilot sub-basins are covered by Flood Hub / GloFAS?

Consumes the sub-basin GeoDataFrame from :mod:`hidroxmx.data.geo` and either
canonical reach lists or the HydroRIVERS-with-thresholds proxy. Produces:

- a per-sub-basin coverage table with the flags ``covered_flood_hub``,
  ``covered_glofas`` and a categorical ``coverage_status``;
- a shortlist of *blind tributaries* — sub-basins covered by neither
  source, with a "priority" flag for the candidate ungauged tributaries
  named in §7 of the experiment spec (Alta del Balsas headwaters
  Tlaxcala–Puebla; Pánuco upland).

The mapping between sub-basins and reach lines uses an intersection test in
EPSG:6362 (Mexican Lambert Conformal Conic) so that a small tolerance
buffer around each reach can be expressed in metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

# Sub-basins named in the brief as candidate blind tributaries.
PRIORITY_BASINS_FOR_UNGAUGED = {"alta_del_balsas", "panuco"}


@dataclass(slots=True)
class CoverageResult:
    per_subbasin: pd.DataFrame        # basin, subbasin_id, covered_flood_hub, covered_glofas, coverage_status
    blind_tributaries: pd.DataFrame   # subset with coverage_status == "none"
    summary: dict                     # counts per basin × status
    proxy_thresholds: dict            # {"flood_hub_km2": float, "glofas_km2": float, "used_proxy": bool}


def _intersects_any(subbasins, reaches, buffer_m: float = 0.0):
    """Return, for every sub-basin, ``True`` if any reach intersects its polygon."""
    import geopandas as gpd

    if reaches is None or len(reaches) == 0:
        return pd.Series(False, index=subbasins.index)

    metric_crs = "EPSG:6362"
    sb = subbasins.to_crs(metric_crs)
    rc = reaches.to_crs(metric_crs)
    if buffer_m > 0:
        rc = rc.assign(geometry=rc.buffer(buffer_m))

    joined = gpd.sjoin(sb[["geometry"]], rc[["geometry"]], predicate="intersects", how="left")
    hit = joined.groupby(level=0)["index_right"].apply(lambda s: s.notna().any())
    hit.index = subbasins.index
    return hit.astype(bool)


def compute_coverage(
    subbasins,
    flood_hub_reaches,
    glofas_reaches,
    *,
    buffer_m: float = 500.0,
    proxy_thresholds: dict | None = None,
) -> CoverageResult:
    """Assign a coverage status to every pilot sub-basin.

    Parameters
    ----------
    subbasins:
        GeoDataFrame from :func:`hidroxmx.data.geo.load_pilot_subbasins`.
    flood_hub_reaches, glofas_reaches:
        Line GeoDataFrames of reaches monitored by each provider. Either
        can be ``None`` if only one of the two sources is available.
    buffer_m:
        Metres of tolerance around every reach when testing intersection
        (default 500 m).
    proxy_thresholds:
        Optional metadata dict propagated into the summary; used by the
        driver script to record whether the proxy source was used.
    """
    covered_fh = _intersects_any(subbasins, flood_hub_reaches, buffer_m=buffer_m)
    covered_gl = _intersects_any(subbasins, glofas_reaches, buffer_m=buffer_m)

    status = []
    for fh, gl in zip(covered_fh, covered_gl):
        if fh and gl:
            status.append("both")
        elif fh:
            status.append("flood_hub_only")
        elif gl:
            status.append("glofas_only")
        else:
            status.append("none")

    per = subbasins[["basin", "basin_slug", "subbasin_id"]].copy()
    per["covered_flood_hub"] = covered_fh.values
    per["covered_glofas"] = covered_gl.values
    per["coverage_status"] = status

    blind = per[per["coverage_status"] == "none"].copy()
    blind["priority_ungauged"] = blind["basin_slug"].isin(PRIORITY_BASINS_FOR_UNGAUGED)

    summary = {
        "n_subbasins": int(len(per)),
        "by_status": per["coverage_status"].value_counts().to_dict(),
        "by_basin_status": (
            per.groupby(["basin", "coverage_status"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        ),
        "n_blind": int((per["coverage_status"] == "none").sum()),
        "n_blind_priority": int(blind["priority_ungauged"].sum()) if len(blind) else 0,
    }
    return CoverageResult(
        per_subbasin=per,
        blind_tributaries=blind,
        summary=summary,
        proxy_thresholds=dict(proxy_thresholds or {}),
    )


def plot_coverage_map(
    subbasins,
    per_subbasin: pd.DataFrame,
    out_path,
    *,
    flood_hub_reaches=None,
    glofas_reaches=None,
    title: str | None = None,
    kind: str = "combination",
    column: str = "double",
):
    """Render the coverage map (Figure 1) at J. Hydrology submission spec.

    Saves TIFF (500 dpi for combination), PDF (vector), and PNG (500 dpi
    for preview) in one pass through :func:`hidroxmx.viz.save_figure`.
    ``out_path`` is treated as a *stem* — the extension you supply is
    stripped and the helper appends ``.tif``, ``.pdf`` and ``.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from .viz import figure_size, save_figure, set_publication_defaults

    set_publication_defaults()

    # Category colours — colour-blind accessible (Wong-style pairing kept
    # here to preserve semantic mapping used in Paper 1's Fig. 1).
    palette = {
        "both":           "#2E7D5B",
        "flood_hub_only": "#1F3D5C",
        "glofas_only":    "#C9971B",
        "none":           "#B23A48",
    }

    merged = subbasins.merge(per_subbasin[["subbasin_id", "coverage_status"]], on="subbasin_id")
    merged["_color"] = merged["coverage_status"].map(palette)

    fig, ax = plt.subplots(figsize=figure_size(column=column, height_ratio=0.85))
    for status, colour in palette.items():
        piece = merged[merged["coverage_status"] == status]
        if len(piece) == 0:
            continue
        piece.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.4, alpha=0.85)

    if glofas_reaches is not None and len(glofas_reaches) > 0:
        glofas_reaches.plot(ax=ax, color="#4c72b0", linewidth=0.35, alpha=0.55)
    if flood_hub_reaches is not None and len(flood_hub_reaches) > 0:
        flood_hub_reaches.plot(ax=ax, color="#111111", linewidth=0.35, alpha=0.75)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or "Global-model coverage over the Hidro-MX pilot sub-basins")
    ax.set_aspect("equal", adjustable="datalim")

    legend_items = [
        Patch(facecolor=palette["both"], label="Covered by both providers"),
        Patch(facecolor=palette["flood_hub_only"], label="Flood Hub only"),
        Patch(facecolor=palette["glofas_only"], label="GloFAS only"),
        Patch(facecolor=palette["none"], label="Uncovered (blind)"),
    ]
    if flood_hub_reaches is not None and len(flood_hub_reaches) > 0:
        legend_items.append(Line2D([], [], color="#111111", lw=1.2, label="Flood Hub reaches"))
    if glofas_reaches is not None and len(glofas_reaches) > 0:
        legend_items.append(Line2D([], [], color="#4c72b0", lw=1.2, label="GloFAS reaches"))
    ax.legend(handles=legend_items, loc="lower left", frameon=True)

    fig.tight_layout()
    from pathlib import Path
    stem = Path(out_path).with_suffix("")
    written = save_figure(fig, stem, kind=kind)
    plt.close(fig)
    return written
