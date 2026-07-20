#!/usr/bin/env python
"""Render the Milestone 3 PUB summary figure at J. Hydrology spec.

Pulls the 14 per-fold manifests from R2 (authoritative source for run
artefacts) and renders a two-panel figure:

- Panel (a): grouped bar chart of NSE across forecast horizons
  {1, 2, 3, 5, 7} days. Two bars per group — persistence baseline and
  F0-PUB — with 1-sigma error bars across the 14 leave-one-out folds
  and a numeric delta annotation.
- Panel (b): per-station scatter of F0-PUB NSE against persistence NSE
  at h=1, with a y=x reference line. Points below the line are folds
  where persistence wins; points above are folds where F0-PUB wins,
  colour-coded and labelled.

The figure targets a full-page (190 mm) double-column layout because
the two panels need to sit side by side. Kind is 'combination' (500 dpi)
because it mixes filled bars (halftone) with lines/markers/text (line
art).

Usage
-----
    python scripts/20_figure_pub_summary.py \
        --run-id F0pub-alto-lerma-sweep-01 \
        --out results/figures/fig_3_pub_summary_alto_lerma
"""
from __future__ import annotations

import io
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


# Fixed station roster for the Alto Lerma basin (matches the 14 folds
# of the PUB sweep). Order preserves alphabetical listing for legibility.
ALTO_LERMA_CLAVES = (
    "ATOMX", "BRAGJ", "CALMX", "CEYGJ", "CYUGJ", "ECBGJ", "EGIMC",
    "IXCMX", "LAYMX", "SB2MX", "SL2GJ", "SLCGJ", "SLVGJ", "SMLMX",
)

HORIZONS = (1, 2, 3, 5, 7)


def _load_fold_from_r2(r2, prefix: str, clave: str) -> dict | None:
    """Fetch one fold's manifest.json from R2. Return None on missing."""
    key = f"{prefix}/{clave}/manifest.json"
    try:
        payload = r2.get_bytes(key)
    except Exception:  # noqa: BLE001
        return None
    return json.loads(payload.decode("utf-8"))


def _load_all_folds(run_id: str, claves) -> pd.DataFrame:
    import sys
    r2 = r2_from_env()
    prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}"
    click.echo(f"[20_fig] R2 endpoint: {r2.endpoint_url}  bucket: {r2.bucket}")
    click.echo(f"[20_fig] prefix    : {prefix}")
    rows = []
    for i, clave in enumerate(claves, 1):
        click.echo(f"[20_fig]   [{i:>2d}/{len(claves)}] fetching {clave}…", nl=False)
        sys.stdout.flush()
        mf = _load_fold_from_r2(r2, prefix, clave)
        if mf is None:
            click.echo(" MISSING")
            continue
        click.echo(" ok")
        m, cfg = mf["metrics"], mf["config"]
        row = {"holdout": cfg["holdout"], "name": cfg.get("holdout_name", "")}
        for h in HORIZONS:
            row[f"f0pub_nse_h{h}"] = m[f"nse_h{h}"]
            row[f"persist_nse_h{h}"] = m[f"persist_nse_h{h}"]
            row[f"f0pub_kge_h{h}"] = m[f"kge_h{h}"]
            row[f"persist_kge_h{h}"] = m[f"persist_kge_h{h}"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _panel_bars(ax, df: pd.DataFrame):
    """Grouped bar chart of persist vs F0-PUB NSE across horizons."""
    x = np.arange(len(HORIZONS))
    width = 0.36

    persist_mean = np.array([df[f"persist_nse_h{h}"].mean() for h in HORIZONS])
    persist_std = np.array([df[f"persist_nse_h{h}"].std(ddof=0) for h in HORIZONS])
    f0pub_mean = np.array([df[f"f0pub_nse_h{h}"].mean() for h in HORIZONS])
    f0pub_std = np.array([df[f"f0pub_nse_h{h}"].std(ddof=0) for h in HORIZONS])

    # Wong palette: [black, orange, sky-blue, green, yellow, blue, vermillion, purple]
    persist_color = "#7A7A7A"      # neutral gray — a naive baseline
    f0pub_color = WONG_PALETTE[5]  # blue for the model

    ax.bar(x - width / 2, persist_mean, width, yerr=persist_std,
           color=persist_color, edgecolor="black", linewidth=0.6,
           label="Persistence", capsize=2.5,
           error_kw={"elinewidth": 0.6, "ecolor": "black"})
    ax.bar(x + width / 2, f0pub_mean, width, yerr=f0pub_std,
           color=f0pub_color, edgecolor="black", linewidth=0.6,
           label="F0-PUB (multi-station)", capsize=2.5,
           error_kw={"elinewidth": 0.6, "ecolor": "black"})

    # Annotate the delta above each F0-PUB bar.
    for xi, (fm, pm) in enumerate(zip(f0pub_mean, persist_mean)):
        delta = fm - pm
        y_top = max(fm + f0pub_std[xi], pm + persist_std[xi]) + 0.02
        ax.text(xi + width / 2, y_top, f"Δ={delta:+.02f}",
                ha="center", va="bottom", fontsize=6.5,
                color=f0pub_color if delta > 0 else "black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} d" for h in HORIZONS])
    ax.set_ylabel("NSE (Nash–Sutcliffe efficiency)")
    ax.set_xlabel("Forecast horizon")
    ax.set_title("(a) PUB mean NSE across horizons (14 folds, Alto Lerma)",
                 loc="left", fontsize=8.5)
    ax.set_ylim(0, max(1.0, (f0pub_mean + f0pub_std).max() + 0.15))
    ax.axhline(0, color="black", linewidth=0.4)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)


def _panel_scatter(ax, df: pd.DataFrame, h: int = 1):
    """Per-station scatter of F0-PUB vs persistence NSE at horizon h."""
    x = df[f"persist_nse_h{h}"].to_numpy()
    y = df[f"f0pub_nse_h{h}"].to_numpy()
    wins = y > x

    ax.plot([-1, 1.05], [-1, 1.05], color="black", linewidth=0.5,
            linestyle="--", label="y = x")
    ax.scatter(x[wins], y[wins], s=32, color=WONG_PALETTE[3],
               edgecolor="black", linewidth=0.5, zorder=3,
               label=f"F0-PUB wins ({int(wins.sum())})")
    ax.scatter(x[~wins], y[~wins], s=32, color="#B0B0B0",
               edgecolor="black", linewidth=0.5, zorder=3,
               label=f"Persistence wins ({int((~wins).sum())})")

    # Label each station near its point, offsetting slightly.
    for xi, yi, name in zip(x, y, df["holdout"]):
        ax.annotate(name, (xi, yi), textcoords="offset points",
                    xytext=(4, 3), fontsize=6, color="black")

    lo = min(x.min(), y.min()) - 0.05
    hi = max(x.max(), y.max()) + 0.05
    lo = max(lo, -0.3); hi = min(hi, 1.02)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Persistence NSE (h = {h} d)")
    ax.set_ylabel(f"F0-PUB NSE (h = {h} d)")
    ax.set_title(f"(b) Per-station comparison at h = {h} d",
                 loc="left", fontsize=8.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)


@click.command()
@click.option("--run-id", default="F0pub-alto-lerma-sweep-01", show_default=True)
@click.option("--out", default="results/figures/fig_3_pub_summary_alto_lerma",
              show_default=True, help="Output path stem (no extension).")
@click.option("--horizon-scatter", default=1, show_default=True,
              help="Horizon (days) shown in the per-station scatter panel.")
def main(run_id: str, out: str, horizon_scatter: int):
    load_dotenv(override=False)
    click.echo(f"[20_fig] Loading {len(ALTO_LERMA_CLAVES)} manifests from R2 "
               f"under run-id {run_id!r}…")
    df = _load_all_folds(run_id, ALTO_LERMA_CLAVES)
    if df.empty:
        raise SystemExit("[20_fig] No manifests could be loaded; abort.")
    click.echo(f"[20_fig] Loaded {len(df)} folds.")

    # Aggregate stats echo (redundant with the on-run diagnostic but useful).
    click.echo("[20_fig] Aggregate NSE (14 folds):")
    for h in HORIZONS:
        f0 = df[f"f0pub_nse_h{h}"].mean()
        pe = df[f"persist_nse_h{h}"].mean()
        wins = int((df[f"f0pub_nse_h{h}"] > df[f"persist_nse_h{h}"]).sum())
        click.echo(f"[20_fig]   h={h}d  F0pub={f0:+.3f}  persist={pe:+.3f}  "
                   f"wins={wins}/{len(df)}")

    set_publication_defaults()
    # Double-column width, tall enough to keep the scatter square.
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=figure_size(column="double", height_ratio=0.48),
    )
    _panel_bars(ax_a, df)
    _panel_scatter(ax_b, df, h=horizon_scatter)
    fig.tight_layout(pad=1.2, w_pad=2.0)

    stem = Path(out)
    written = save_figure(
        fig, stem, kind="combination",
        metadata={
            "Title": "Milestone 3 PUB summary — Alto Lerma",
            "Author": "Daniel Sánchez-Ruiz",
            "Subject": "F0-PUB vs persistence, 14-fold leave-one-out",
        },
    )
    plt.close(fig)
    for p in written:
        click.echo(f"[20_fig] wrote {p.as_posix()}  "
                   f"({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
