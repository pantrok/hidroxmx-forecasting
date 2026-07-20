"""Unit tests for the coverage-analysis step (Milestone 1)."""
from __future__ import annotations

import pytest

geopandas = pytest.importorskip("geopandas")
from shapely.geometry import LineString, Polygon

from hidroxmx.coverage import compute_coverage


def _synthetic_subbasins():
    """Two 1° × 1° squares in central Mexico used as sub-basins."""
    polys = [
        Polygon([(-100, 19), (-99, 19), (-99, 20), (-100, 20)]),
        Polygon([(-98, 19), (-97, 19), (-97, 20), (-98, 20)]),
    ]
    return geopandas.GeoDataFrame(
        {
            "basin": ["Cutzamala", "Alta del Balsas"],
            "basin_slug": ["cutzamala", "alta_del_balsas"],
            "subbasin_id": ["cutzamala_000", "alta_del_balsas_000"],
        },
        geometry=polys,
        crs="EPSG:4326",
    )


def test_covered_by_both_when_reach_crosses_polygon():
    sb = _synthetic_subbasins()
    fh = geopandas.GeoDataFrame(
        geometry=[LineString([(-99.9, 19.5), (-99.1, 19.5)])], crs="EPSG:4326"
    )
    gl = geopandas.GeoDataFrame(
        geometry=[LineString([(-99.8, 19.7), (-99.2, 19.4)])], crs="EPSG:4326"
    )
    res = compute_coverage(sb, fh, gl, buffer_m=0.0)
    row = res.per_subbasin.set_index("subbasin_id").loc["cutzamala_000"]
    assert bool(row["covered_flood_hub"]) is True
    assert bool(row["covered_glofas"]) is True
    assert row["coverage_status"] == "both"


def test_none_status_and_priority_flag():
    sb = _synthetic_subbasins()
    fh = geopandas.GeoDataFrame(
        geometry=[LineString([(-95, 15), (-94, 15)])], crs="EPSG:4326"
    )
    gl = fh
    res = compute_coverage(sb, fh, gl, buffer_m=0.0)
    blind = res.blind_tributaries.set_index("subbasin_id")
    assert set(blind.index) == {"cutzamala_000", "alta_del_balsas_000"}
    # Alta del Balsas is on the priority list; Cutzamala is not.
    assert bool(blind.loc["alta_del_balsas_000", "priority_ungauged"]) is True
    assert bool(blind.loc["cutzamala_000", "priority_ungauged"]) is False


def test_summary_counts_add_up():
    sb = _synthetic_subbasins()
    fh = geopandas.GeoDataFrame(
        geometry=[LineString([(-99.9, 19.5), (-99.1, 19.5)])], crs="EPSG:4326"
    )
    gl = None
    res = compute_coverage(sb, fh, gl, buffer_m=0.0)
    total = sum(res.summary["by_status"].values())
    assert total == res.summary["n_subbasins"] == len(sb)
