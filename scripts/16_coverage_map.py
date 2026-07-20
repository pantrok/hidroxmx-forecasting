#!/usr/bin/env python
"""Stage 16 — Coverage / blind-spot map (Milestone 1, §12.1 of the spec).

Overlays the reach coverage of the two global forecasting providers we compare
against (Google Flood Hub and GloFAS-CEMS) with the 123 pilot sub-basins
delivered by the Hidro-MX dataset. Produces Figure 1 (map at 300 dpi) and a
CSV listing the *blind tributaries* — sub-basins covered by neither provider,
with a priority flag for the candidate ungauged tributaries named in the
brief (Alta del Balsas headwaters, Pánuco upland).

Two input modes
---------------
1. *Canonical* — pass ``--flood-hub-reaches PATH`` and ``--glofas-reaches
   PATH`` pointing to the actual reach GeoJSON / shapefile shipped by each
   provider. Preferred for the paper's final figure.
2. *Proxy* (default) — no external file. The script downloads HydroRIVERS
   v1.0 (North America) once, caches it, and filters the reaches by the
   drainage-area thresholds documented by each provider (25 km² for Flood
   Hub, 500 km² for GloFAS). Every output CSV / JSON records that the
   proxy source was used so the reader can distinguish the two modes.

Optional upload to R2 (``--upload-to-r2``) mirrors the local artefacts
under ``{R2_PAPER2_PREFIX}/coverage/`` for reproducibility in Colab.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from hidroxmx.coverage import compute_coverage, plot_coverage_map
from hidroxmx.data.geo import (
    FLOOD_HUB_UPLAND_THRESHOLD_KM2,
    GLOFAS_UPLAND_THRESHOLD_KM2,
    default_hidroxai_gpkg_sources,
    ensure_hydrorivers,
    load_flood_hub_reaches,
    load_glofas_reaches,
    load_hydrorivers_subset,
    load_pilot_subbasins,
    proxy_reaches,
)
from hidroxmx.io import RunManifest, dump_manifest, r2_from_env, seed_everything


@click.command()
@click.option("--run-id", default="M1-coverage-map",
              help="Identifier used for R2 mirroring and the run manifest.")
@click.option("--subbasins-source",
              type=click.Path(exists=False),
              default=None,
              help="Optional path with cem/*.gpkg. Defaults to the sibling hidroxai-mx repo.")
@click.option("--flood-hub-reaches", type=click.Path(exists=True), default=None,
              help="Canonical GeoJSON / shapefile of Flood Hub gauged reaches.")
@click.option("--glofas-reaches", type=click.Path(exists=True), default=None,
              help="Canonical GeoJSON / shapefile of GloFAS reaches.")
@click.option("--out-dir", type=click.Path(), default="outputs/coverage",
              show_default=True,
              help="Local directory for Figure 1, the blind-tributary CSV and the summary JSON.")
@click.option("--cache-dir", type=click.Path(), default="cache",
              show_default=True,
              help="Local cache for HydroRIVERS (downloaded once).")
@click.option("--upload-to-r2", is_flag=True,
              help="Mirror the local artefacts under {R2_PAPER2_PREFIX}/coverage/.")
@click.option("--dpi", default=300, show_default=True, type=int)
@click.option("--buffer-m", default=500.0, show_default=True, type=float,
              help="Metres of tolerance around every reach when testing intersection.")
@click.option("--seed", default=20260606, show_default=True, type=int)
def main(
    run_id,
    subbasins_source,
    flood_hub_reaches,
    glofas_reaches,
    out_dir,
    cache_dir,
    upload_to_r2,
    dpi,
    buffer_m,
    seed,
):
    load_dotenv(override=False)
    seed_everything(seed)

    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Sub-basins
    # ------------------------------------------------------------------ #
    if subbasins_source is not None:
        source_root = Path(subbasins_source)
        sources = [source_root / f"{slug}.gpkg" for slug in (
            "cutzamala", "lerma_alto", "bajio", "santiago", "panuco", "alta_del_balsas",
        )]
    else:
        sources = default_hidroxai_gpkg_sources()

    click.echo(f"[16_coverage_map] Loading pilot sub-basins from {len(sources)} GeoPackages …")
    subbasins = load_pilot_subbasins(sources)
    click.echo(f"[16_coverage_map]   → {len(subbasins)} sub-basin polygons "
               f"across {subbasins['basin'].nunique()} basins.")

    # ------------------------------------------------------------------ #
    # 2. Reach lists (canonical if provided, HydroRIVERS proxy otherwise)
    # ------------------------------------------------------------------ #
    used_proxy = flood_hub_reaches is None or glofas_reaches is None
    fh_layer = load_flood_hub_reaches(Path(flood_hub_reaches) if flood_hub_reaches else None)
    gl_layer = load_glofas_reaches(Path(glofas_reaches) if glofas_reaches else None)

    proxy_meta = {
        "used_proxy": used_proxy,
        "flood_hub_km2": FLOOD_HUB_UPLAND_THRESHOLD_KM2,
        "glofas_km2": GLOFAS_UPLAND_THRESHOLD_KM2,
    }

    if used_proxy:
        click.echo("[16_coverage_map] No canonical reach lists — downloading HydroRIVERS …")
        handle = ensure_hydrorivers(cache_dir)
        click.echo(f"[16_coverage_map]   HydroRIVERS at {handle.path}")
        rivers = load_hydrorivers_subset(handle)
        click.echo(f"[16_coverage_map]   HydroRIVERS restricted to Mexico bbox: "
                   f"{len(rivers)} reaches.")
        if fh_layer is None:
            fh_layer = proxy_reaches(rivers, handle, FLOOD_HUB_UPLAND_THRESHOLD_KM2)
            click.echo(f"[16_coverage_map]   Flood Hub proxy ≥ {FLOOD_HUB_UPLAND_THRESHOLD_KM2} km² "
                       f"→ {len(fh_layer)} reaches.")
        if gl_layer is None:
            gl_layer = proxy_reaches(rivers, handle, GLOFAS_UPLAND_THRESHOLD_KM2)
            click.echo(f"[16_coverage_map]   GloFAS proxy ≥ {GLOFAS_UPLAND_THRESHOLD_KM2} km² "
                       f"→ {len(gl_layer)} reaches.")

    # ------------------------------------------------------------------ #
    # 3. Coverage assignment
    # ------------------------------------------------------------------ #
    click.echo(f"[16_coverage_map] Assigning coverage (buffer {buffer_m:.0f} m) …")
    result = compute_coverage(
        subbasins,
        flood_hub_reaches=fh_layer,
        glofas_reaches=gl_layer,
        buffer_m=buffer_m,
        proxy_thresholds=proxy_meta,
    )
    click.echo(f"[16_coverage_map]   status counts: {result.summary['by_status']}")
    click.echo(f"[16_coverage_map]   blind sub-basins: {result.summary['n_blind']}  "
               f"(priority ungauged: {result.summary['n_blind_priority']})")

    # ------------------------------------------------------------------ #
    # 4. Persist artefacts (Fig. 1, blind CSV, summary JSON, manifest)
    # ------------------------------------------------------------------ #
    fig_path = out_dir / "fig1_coverage_map.png"
    csv_path = out_dir / "blind_tributaries.csv"
    json_path = out_dir / "coverage_summary.json"
    per_path = out_dir / "coverage_per_subbasin.csv"
    manifest_path = out_dir / "manifest.json"

    click.echo(f"[16_coverage_map] Rendering Figure 1 → {fig_path}")
    plot_coverage_map(
        subbasins,
        result.per_subbasin,
        fig_path,
        flood_hub_reaches=fh_layer,
        glofas_reaches=gl_layer,
        dpi=dpi,
    )
    result.per_subbasin.to_csv(per_path, index=False)
    result.blind_tributaries.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "summary": result.summary,
        "proxy_thresholds": result.proxy_thresholds,
        "buffer_m": buffer_m,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = RunManifest(
        run_id=run_id,
        stage="16_coverage_map",
        config={
            "subbasins_source": str(subbasins_source or "sibling hidroxai-mx"),
            "flood_hub_reaches": str(flood_hub_reaches),
            "glofas_reaches": str(glofas_reaches),
            "buffer_m": buffer_m,
            "dpi": dpi,
            "seed": seed,
            "proxy_thresholds": result.proxy_thresholds,
        },
    ).finalise({
        "n_subbasins": result.summary["n_subbasins"],
        "n_blind": result.summary["n_blind"],
        "n_blind_priority": result.summary["n_blind_priority"],
    })
    dump_manifest(manifest, manifest_path)

    # ------------------------------------------------------------------ #
    # 5. Optional mirror to R2 for reproducibility from Colab
    # ------------------------------------------------------------------ #
    if upload_to_r2:
        click.echo("[16_coverage_map] Uploading artefacts to R2 …")
        r2 = r2_from_env()
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/coverage/{run_id}"
        for local in (fig_path, per_path, csv_path, json_path, manifest_path):
            r2.upload_file(f"{prefix}/{local.name}", local)
            click.echo(f"[16_coverage_map]   → r2://{r2.bucket}/{prefix}/{local.name}")

    click.echo("[16_coverage_map] Done.")


if __name__ == "__main__":
    main()
