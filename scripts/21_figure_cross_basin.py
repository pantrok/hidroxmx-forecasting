#!/usr/bin/env python
"""Stage 21 — Cross-basin summary figure for Milestone 3 (Fig. 4).

TODO(post-M4): visual polish pass on this figure. The current layout
suffers from (a) overlapping ±1 SD shaded bands in panel (a) that
obscure basin comparisons when three basins converge at the same y,
and (b) a legend that eats ~25% of panel (a). Options for the polish
pass: replace shaded bands with box-and-whiskers per (basin, horizon),
move the legend outside the axes, or split panel (a) into two rows
(F0-PUB on top, persistence at bottom) so lines never overlap. Batch
this with the other paper-figure polish items after Milestone 4 lands.


Assembles one figure that summarises the entire Milestone 3 sweep
across the four included basins (Alto Lerma, Valle de México,
Bajo Pánuco, Medio Balsas). The paper's headline claim — F0-PUB
outperforms persistence *at every horizon and in every basin* — is
built here.

Two panels, side by side (double-column layout):

- Panel (a): mean NSE across the effective folds of each basin,
  drawn as one solid line per basin for F0-PUB and one dashed line
  per basin for persistence. Same colour per basin so the eye
  tracks the pair (solid vs dashed) at a glance. Shaded band =
  ± 1 SD.
- Panel (b): heatmap of the mean NSE difference (F0-PUB − persist),
  rows = basins, columns = horizons. Diverging colormap centred at
  zero. Cells annotated with the numeric ∆. Green above the
  diagonal (F0-PUB wins), red below (persistence wins).

Data source: per-fold manifests live in
``paper2/runs/{run_id}/{fold}/manifest.json``. Auto-discovers folds
via ``list_objects_v2`` (same idiom the per-basin figure uses). Folds
with NSE < ``SCATTER_NSE_FLOOR`` or NaN at a given horizon are
excluded from that horizon's aggregate — same filter as the per-basin
Fig. 3 so numbers are directly comparable between the two figures.

Usage
-----
    python scripts/21_figure_cross_basin.py \\
        --run-ids "F0pub-alto-lerma-sweep-01,F0pub-valle-de-mexico-sweep-01,\\
                   F0pub-bajo-panuco-sweep-01,F0pub-medio-balsas-sweep-01" \\
        --basin-labels "Alto Lerma,Valle de México,Bajo Pánuco,Medio Balsas" \\
        --out results/figures/fig_4_cross_basin_milestone_3 \\
        --upload-to-r2
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import click
import matplotlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hidroxmx.io import r2_from_env  # noqa: E402
from hidroxmx.viz import (  # noqa: E402
    WONG_PALETTE,
    figure_size,
    save_figure,
    set_publication_defaults,
)


HORIZONS = (1, 2, 3, 5, 7)
# Same threshold as scripts/20_figure_pub_summary.py so the two figures
# are directly comparable.
SCATTER_NSE_FLOOR = -1.0


# --------------------------------------------------------------------------- #
# R2 access — mirror the pattern from stage 20
# --------------------------------------------------------------------------- #
def _list_folds(r2, run_id: str) -> list[str]:
    prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}/"
    claves: set[str] = set()
    paginator = r2._client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=r2.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            parts = rel.split("/")
            if len(parts) >= 2 and parts[-1] == "manifest.json":
                claves.add(parts[0])
    return sorted(claves)


def _load_fold(r2, run_id: str, clave: str) -> dict | None:
    key = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}/{clave}/manifest.json"
    try:
        payload = r2.get_bytes(key)
    except Exception:  # noqa: BLE001
        return None
    return json.loads(payload.decode("utf-8"))


def _load_basin_summary(r2, run_id: str) -> pd.DataFrame:
    """Return one row per fold with per-horizon F0-PUB and persist NSE."""
    claves = _list_folds(r2, run_id)
    rows = []
    for clave in claves:
        mf = _load_fold(r2, run_id, clave)
        if mf is None:
            continue
        m = mf["metrics"]
        row = {"holdout": clave}
        for h in HORIZONS:
            row[f"f0pub_h{h}"] = m.get(f"nse_h{h}", np.nan)
            row[f"persist_h{h}"] = m.get(f"persist_nse_h{h}", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _clean_agg(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Return ``{col: (mean, std)}`` after filtering below floor."""
    out: dict[str, tuple[float, float]] = {}
    for col in df.columns:
        if col == "holdout":
            continue
        s = df[col].where(df[col] >= SCATTER_NSE_FLOOR)
        out[col] = (float(np.nanmean(s)), float(np.nanstd(s, ddof=0)))
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _panel_lines(ax, agg_by_basin: dict[str, dict]):
    """Line plot: mean NSE by horizon, one line per (basin, method)."""
    basins = list(agg_by_basin.keys())
    colours = [WONG_PALETTE[i % len(WONG_PALETTE)] for i in range(len(basins))]
    x = np.array(HORIZONS, dtype=float)

    for basin, colour in zip(basins, colours):
        agg = agg_by_basin[basin]["agg"]
        n_folds = agg_by_basin[basin]["n_folds"]
        f0 = np.array([agg[f"f0pub_h{h}"][0] for h in HORIZONS])
        f0_std = np.array([agg[f"f0pub_h{h}"][1] for h in HORIZONS])
        pe = np.array([agg[f"persist_h{h}"][0] for h in HORIZONS])
        pe_std = np.array([agg[f"persist_h{h}"][1] for h in HORIZONS])

        ax.fill_between(x, f0 - f0_std, f0 + f0_std, color=colour, alpha=0.15)
        ax.plot(x, f0, color=colour, marker="o", markersize=4, linewidth=1.4,
                label=f"{basin} — F0-PUB (n={n_folds})")
        ax.plot(x, pe, color=colour, marker="s", markersize=3, linewidth=1.0,
                linestyle="--", alpha=0.9,
                label=f"{basin} — persistence")

    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xticks(list(HORIZONS))
    ax.set_xticklabels([f"{h} d" for h in HORIZONS])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("NSE (Nash–Sutcliffe efficiency)")
    ax.set_title("(a) Mean NSE by basin and horizon (± 1 SD shaded)",
                 loc="left", fontsize=8.5)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9,
              fontsize=6, ncol=2, columnspacing=1.0)


def _panel_heatmap(ax, agg_by_basin: dict[str, dict]):
    """Heatmap of Δ mean NSE (F0-PUB − persistence)."""
    basins = list(agg_by_basin.keys())
    matrix = np.zeros((len(basins), len(HORIZONS)), dtype=float)
    for i, basin in enumerate(basins):
        agg = agg_by_basin[basin]["agg"]
        for j, h in enumerate(HORIZONS):
            matrix[i, j] = agg[f"f0pub_h{h}"][0] - agg[f"persist_h{h}"][0]

    vmax = float(np.nanmax(np.abs(matrix)))
    vmax = max(vmax, 0.05)  # avoid saturation on tiny deltas
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"{h} d" for h in HORIZONS])
    ax.set_yticks(range(len(basins)))
    ax.set_yticklabels(basins)
    ax.set_xlabel("Forecast horizon")

    # Annotate every cell with the numeric Δ.
    for i in range(len(basins)):
        for j in range(len(HORIZONS)):
            val = matrix[i, j]
            colour = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i, f"{val:+.02f}", ha="center", va="center",
                    fontsize=7, color=colour)

    ax.set_title("(b) Δ mean NSE = F0-PUB − persistence",
                 loc="left", fontsize=8.5)

    # Small colourbar without stealing horizontal room.
    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label("Δ NSE", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
DEFAULT_RUN_IDS = ",".join([
    "F0pub-alto-lerma-sweep-01",
    "F0pub-valle-de-mexico-sweep-01",
    "F0pub-bajo-panuco-sweep-01",
    "F0pub-medio-balsas-sweep-01",
])
DEFAULT_LABELS = "Alto Lerma,Valle de México,Bajo Pánuco,Medio Balsas"


@click.command()
@click.option("--run-ids", default=DEFAULT_RUN_IDS, show_default=True,
              help="Comma-separated Milestone 3 run identifiers, one per basin.")
@click.option("--basin-labels", default=DEFAULT_LABELS, show_default=True,
              help="Comma-separated display names in the same order as --run-ids.")
@click.option("--out", default="results/figures/fig_4_cross_basin_milestone_3",
              show_default=True, help="Output stem (no extension).")
@click.option("--upload-to-r2", is_flag=True,
              help="Mirror the rendered figure to R2 under paper2/figures/.")
def main(run_ids: str, basin_labels: str, out: str, upload_to_r2: bool):
    load_dotenv(override=False)
    r2 = r2_from_env()

    run_id_list = [r.strip() for r in run_ids.split(",") if r.strip()]
    label_list = [l.strip() for l in basin_labels.split(",") if l.strip()]
    if len(run_id_list) != len(label_list):
        raise SystemExit(f"[21_fig] --run-ids has {len(run_id_list)} entries, "
                         f"--basin-labels has {len(label_list)} — must match.")

    click.echo(f"[21_fig] Assembling {len(run_id_list)} basins…")
    agg_by_basin: dict[str, dict] = {}
    for run_id, label in zip(run_id_list, label_list):
        click.echo(f"[21_fig]   loading {label} from run {run_id!r}…")
        df = _load_basin_summary(r2, run_id)
        if df.empty:
            click.echo(f"[21_fig]     no folds on R2 for {run_id!r} — skipping.")
            continue
        agg = _clean_agg(df)
        n_folds = len(df)
        agg_by_basin[label] = {"agg": agg, "n_folds": n_folds, "raw": df}
        click.echo(f"[21_fig]     {n_folds} folds loaded.")

    if len(agg_by_basin) < 2:
        raise SystemExit("[21_fig] Need at least 2 basins to render a cross-basin figure.")

    # Console summary — same numbers the figure encodes.
    click.echo("\n[21_fig] === Cross-basin mean NSE (post-filter) ===")
    header = "  basin".ljust(22) + " " + "".join(f"  h={h}d".rjust(10) for h in HORIZONS)
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))
    for label, entry in agg_by_basin.items():
        agg = entry["agg"]
        cells = "".join(
            f"  {agg[f'f0pub_h{h}'][0]:+.03f}".rjust(10) for h in HORIZONS
        )
        click.echo(f"  {label:<20s}  F0-PUB {cells}")
        cells = "".join(
            f"  {agg[f'persist_h{h}'][0]:+.03f}".rjust(10) for h in HORIZONS
        )
        click.echo(f"  {label:<20s}  persist{cells}")

    # Figure.
    set_publication_defaults()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=figure_size(column="double", height_ratio=0.52),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    _panel_lines(ax_a, agg_by_basin)
    _panel_heatmap(ax_b, agg_by_basin)
    fig.suptitle("F0-PUB vs persistence across four Mexican pilot basins",
                 fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(pad=1.1, w_pad=2.5, rect=(0, 0, 1, 0.96))

    stem = Path(out)
    written = save_figure(
        fig, stem, kind="combination",
        metadata={
            "Title": "Cross-basin F0-PUB summary",
            "Author": "Daniel Sánchez-Ruiz",
            "Subject": f"F0-PUB vs persistence across {len(agg_by_basin)} basins",
        },
    )
    plt.close(fig)
    for p in written:
        click.echo(f"[21_fig] wrote {p.as_posix()}  "
                   f"({p.stat().st_size / 1024:.1f} KB)")

    if upload_to_r2:
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/figures"
        for p in written:
            r2.upload_file(f"{prefix}/{p.name}", p)
            click.echo(f"[21_fig]   -> r2://{r2.bucket}/{prefix}/{p.name}")


if __name__ == "__main__":
    main()
