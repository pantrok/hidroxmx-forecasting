#!/usr/bin/env python
"""Formalise the basin sample via explicit inclusion criteria.

The 4 basins on which the modelling experiments are executed (Alto
Lerma, Valle de México, Bajo Pánuco, Medio Balsas) are not a random
or convenience sample — they are the set of basins that satisfies
three inclusion criteria enunciated **a priori**:

1. **Statistical power for PUB leave-one-out** (``MIN_STATIONS_PUB``):
   the basin must contain at least ``MIN_STATIONS_PUB`` selected
   hydrometric stations. With fewer folds the estimate of mean NSE
   carries an unacceptably wide confidence interval (Newman et al. 2015
   recommend ``n ≥ 10`` for basin-scale leave-one-out on the CAMELS
   catalogue).

2. **Data coverage** (``MIN_MEDIAN_COVERAGE``): the median of the
   ``cobertura`` column across the basin's stations must be at least
   ``MIN_MEDIAN_COVERAGE`` of the reference window
   (2010-01-01 to 2025-12-31). The default 0.60 corresponds to
   ~9.6 valid years out of 16 — well above the ~5-year minimum
   documented by Kratzert et al. (2019) for LSTM streamflow
   forecasters. The manuscript reports a sensitivity analysis at
   the stricter 0.70 threshold (which excludes Medio Balsas) to
   confirm the main findings are not driven by the marginal basin.

3. **Exogenous drivers available** (``MIN_CLIMATE_NEIGHBOURS``): every
   station in the basin must carry a ``vecinos_clima`` entry with at
   least ``MIN_CLIMATE_NEIGHBOURS`` neighbouring climatologic stations,
   so precip / tmax / tmin exogenous features can be computed
   consistently across the basin.

Basins that fail any criterion are documented in the output table with
the specific reason for exclusion and are deferred to future work
(pooled cross-basin analysis or extended monitoring campaigns).

Output artefacts
----------------
- ``results/tables/basin_inclusion.csv`` — per-basin table with the
  three criteria and pass/fail flags.
- ``results/figures/fig_2_basin_inclusion`` — publication figure
  (TIFF + PDF + PNG) showing station count per basin colour-coded by
  inclusion status with a horizontal threshold line.

Both are mirrored to R2 under ``paper2/tables/`` and ``paper2/figures/``
when ``--upload-to-r2`` is passed.
"""
from __future__ import annotations

import os
from pathlib import Path

import click
import matplotlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hidroxmx.data.streams import layout_from_env, load_selected_stations  # noqa: E402
from hidroxmx.io import r2_from_env  # noqa: E402
from hidroxmx.viz import (  # noqa: E402
    WONG_PALETTE,
    figure_size,
    save_figure,
    set_publication_defaults,
)


MIN_STATIONS_PUB = 10
MIN_MEDIAN_COVERAGE = 0.60  # Kratzert et al. 2019 minimum for LSTM streamflow
MIN_CLIMATE_NEIGHBOURS = 1


def _summarise_basin(sub: pd.DataFrame) -> dict:
    """Compute the three criteria for one basin's subset of the manifest."""
    n = int(len(sub))
    med_cov = float(sub["cobertura"].median()) if "cobertura" in sub.columns else float("nan")
    if "vecinos_clima" in sub.columns:
        clima_counts = sub["vecinos_clima"].fillna("").astype(str).map(
            lambda s: len([v for v in s.split(",") if v.strip()])
        )
        n_with_neighbours = int((clima_counts >= MIN_CLIMATE_NEIGHBOURS).sum())
        min_clima = int(clima_counts.min()) if len(clima_counts) else 0
    else:
        n_with_neighbours = 0
        min_clima = 0
    return {
        "n_stations": n,
        "median_coverage": med_cov,
        "stations_with_clima": n_with_neighbours,
        "min_climate_neighbours": min_clima,
    }


def _apply_criteria(summary: dict) -> tuple[bool, list[str]]:
    """Return ``(included, reasons)`` for a per-basin summary dict."""
    reasons: list[str] = []
    if summary["n_stations"] < MIN_STATIONS_PUB:
        reasons.append(f"N={summary['n_stations']} < {MIN_STATIONS_PUB}")
    if not np.isfinite(summary["median_coverage"]) or \
            summary["median_coverage"] < MIN_MEDIAN_COVERAGE:
        reasons.append(f"median coverage {summary['median_coverage']:.2f} "
                       f"< {MIN_MEDIAN_COVERAGE:.2f}")
    if summary["min_climate_neighbours"] < MIN_CLIMATE_NEIGHBOURS:
        reasons.append(f"min climate neighbours "
                       f"{summary['min_climate_neighbours']} "
                       f"< {MIN_CLIMATE_NEIGHBOURS}")
    return (len(reasons) == 0, reasons)


def _build_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return one row per basin with criteria + inclusion verdict."""
    rows: list[dict] = []
    for basin, sub in manifest.groupby("cuenca"):
        summary = _summarise_basin(sub)
        included, reasons = _apply_criteria(summary)
        rows.append({
            "basin": basin,
            **summary,
            "included": included,
            "exclusion_reasons": "; ".join(reasons) if reasons else "",
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["included", "n_stations"], ascending=[False, False]).reset_index(drop=True)


def _plot_inclusion(table: pd.DataFrame, out_stem: Path) -> list[Path]:
    """Bar chart of station count per basin, coloured by inclusion status."""
    set_publication_defaults()
    fig, ax = plt.subplots(figsize=figure_size(column="double", height_ratio=0.42))

    y = np.arange(len(table))
    counts = table["n_stations"].to_numpy()
    included = table["included"].to_numpy()

    include_color = WONG_PALETTE[3]  # bluish green
    exclude_color = "#B0B0B0"        # neutral gray

    colours = [include_color if inc else exclude_color for inc in included]
    ax.barh(y, counts, color=colours, edgecolor="black", linewidth=0.5)

    for yi, count, med, inc in zip(y, counts, table["median_coverage"], included):
        text = f"n={count}, med. cov.={med:.2f}"
        ax.text(count + 0.4, yi, text, va="center", fontsize=6.5)

    ax.axvline(MIN_STATIONS_PUB, color="black", linestyle="--",
               linewidth=0.7, label=f"Threshold n ≥ {MIN_STATIONS_PUB}")
    ax.set_yticks(y)
    ax.set_yticklabels(table["basin"].tolist())
    ax.invert_yaxis()  # largest basin at the top
    ax.set_xlabel("Selected hydrometric stations")
    ax.set_title("Basin sample: hydrometric stations per basin and inclusion status",
                 loc="left", fontsize=9)
    from matplotlib.patches import Patch
    handles = [
        Patch(color=include_color, label=f"Included (n ≥ {MIN_STATIONS_PUB})"),
        Patch(color=exclude_color, label="Excluded (deferred to future work)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    ax.set_xlim(0, max(counts) * 1.35)
    fig.tight_layout()
    written = save_figure(fig, out_stem, kind="combination",
                          metadata={"Title": "Basin inclusion criteria",
                                    "Author": "Daniel Sánchez-Ruiz",
                                    "Subject": "Basin sample rationale (PUB leave-one-out)"})
    plt.close(fig)
    return written


@click.command()
@click.option("--out-table", default="results/tables/basin_inclusion.csv",
              show_default=True)
@click.option("--out-fig", default="results/figures/fig_2_basin_inclusion",
              show_default=True, help="Figure stem (no extension).")
@click.option("--min-stations", default=MIN_STATIONS_PUB, show_default=True,
              type=int, help="Override the minimum stations criterion.")
@click.option("--min-coverage", default=MIN_MEDIAN_COVERAGE, show_default=True,
              type=float,
              help="Override the median-coverage criterion. Use 0.70 to "
                   "reproduce the strict-threshold sensitivity analysis "
                   "reported in the manuscript.")
@click.option("--upload-to-r2", is_flag=True,
              help="Mirror the table (paper2/tables/) and the figure "
                   "(paper2/figures/) to R2.")
def main(out_table: str, out_fig: str, min_stations: int, min_coverage: float,
         upload_to_r2: bool):
    global MIN_STATIONS_PUB, MIN_MEDIAN_COVERAGE
    MIN_STATIONS_PUB = int(min_stations)
    MIN_MEDIAN_COVERAGE = float(min_coverage)
    load_dotenv(override=False)
    r2 = r2_from_env() if upload_to_r2 else None
    layout = layout_from_env()

    manifest = load_selected_stations(r2, layout, kind="hidro")
    click.echo(f"[22_basin] manifest rows: {len(manifest)}  "
               f"unique basins: {manifest['cuenca'].nunique()}")
    click.echo(f"[22_basin] criteria: N >= {MIN_STATIONS_PUB}, "
               f"median coverage >= {MIN_MEDIAN_COVERAGE:.2f}, "
               f"min climate neighbours >= {MIN_CLIMATE_NEIGHBOURS}")
    table = _build_table(manifest)

    click.echo("[22_basin] inclusion table:")
    for _, row in table.iterrows():
        marker = "INC" if row["included"] else "EXC"
        reason = f"  ({row['exclusion_reasons']})" if row["exclusion_reasons"] else ""
        click.echo(f"[22_basin]   {marker}  {row['basin']:<38s}  "
                   f"n={row['n_stations']:>3d}  "
                   f"med_cov={row['median_coverage']:.2f}  "
                   f"min_clima={row['min_climate_neighbours']}{reason}")

    total = len(table)
    included = int(table["included"].sum())
    click.echo(f"[22_basin] {included}/{total} basins pass all criteria.")

    out_table_path = Path(out_table)
    out_table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_table_path, index=False)
    click.echo(f"[22_basin] wrote {out_table_path.as_posix()}")

    fig_stem = Path(out_fig)
    written = _plot_inclusion(table, fig_stem)
    for p in written:
        click.echo(f"[22_basin] wrote {p.as_posix()}  "
                   f"({p.stat().st_size / 1024:.1f} KB)")

    if upload_to_r2:
        prefix_tables = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/tables"
        prefix_figs = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/figures"
        r2.upload_file(f"{prefix_tables}/{out_table_path.name}", out_table_path)
        click.echo(f"[22_basin]   -> r2://{r2.bucket}/{prefix_tables}/{out_table_path.name}")
        for p in written:
            r2.upload_file(f"{prefix_figs}/{p.name}", p)
            click.echo(f"[22_basin]   -> r2://{r2.bucket}/{prefix_figs}/{p.name}")


if __name__ == "__main__":
    main()
