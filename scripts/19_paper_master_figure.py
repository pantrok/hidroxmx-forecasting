#!/usr/bin/env python
"""Stage 19 — paper master figure with paired-bootstrap CIs.

Assembles a three-panel forest plot that summarises the paper's
central claims with the 95 % confidence intervals produced by
:mod:`scripts.18_bootstrap_analysis`:

- **Panel (a)** — Milestone 3: Δ NSE median (F0-PUB − persistence) per
  basin × horizon. Shows where F0-PUB significantly outperforms the
  naive baseline and where it is null.
- **Panel (b)** — Milestone 4: Δ NSE median per mechanism × horizon on
  Alto Lerma. Every CI straddles zero → visually confirms the null
  result of Path A.
- **Panel (c)** — Milestone 5c: Δ Value @ C/L=0.2 median (fuzzy best
  cutoff − baseline threshold) per basin × horizon. Kill-condition
  vertical line at +0.05 highlights when Path B provides operational
  value.

Reads the three CSV tables from ``results/tables/`` (or R2 with
``--from-r2``). Renders at J. Hydrology submission spec via
:mod:`hidroxmx.viz`.
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

from hidroxmx.io import r2_from_env  # noqa: E402
from hidroxmx.viz import (  # noqa: E402
    WONG_PALETTE,
    figure_size,
    save_figure,
    set_publication_defaults,
)


BASIN_ORDER = ("Alto Lerma", "Valle de México", "Bajo Pánuco", "Medio Balsas")
MECHANISM_ORDER = ("S-SIG_t1.0", "S-SIG_t0.3", "S-ATTR_t1.0")
HORIZONS = (1, 2, 3, 5, 7)
KILL_M3 = 0.05
KILL_M5c = 0.05


def _load_table(path_or_key: str, *, from_r2: bool, r2=None) -> pd.DataFrame:
    if from_r2:
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/tables"
        payload = r2.get_bytes(f"{prefix}/{path_or_key}")
        import io
        return pd.read_csv(io.BytesIO(payload))
    return pd.read_csv(path_or_key)


def _plot_forest(ax, rows: list[dict], *, x_label: str, kill_threshold: float,
                 title: str, x_pad: float = 0.05):
    """Forest plot — CI whiskers per row, dot at delta, coloured by kill/sig."""
    y_positions = list(range(len(rows)))[::-1]  # top row first

    delta_arr = np.array([r["delta"] for r in rows], dtype=float)
    lo_arr = np.array([r["ci_low"] for r in rows], dtype=float)
    hi_arr = np.array([r["ci_high"] for r in rows], dtype=float)

    for y, r in zip(y_positions, rows):
        kill = bool(r.get("kill_cleared", False))
        sig = bool(r.get("significant", False))
        if kill:
            colour = WONG_PALETTE[3]  # bluish green — strongest evidence
        elif sig and r["delta"] > 0:
            colour = WONG_PALETTE[5]  # blue — significant positive
        elif sig and r["delta"] < 0:
            colour = WONG_PALETTE[6]  # vermillion — significant negative
        else:
            colour = "#8A8A8A"        # gray — null / non-significant
        ax.hlines(y, r["ci_low"], r["ci_high"], color=colour, linewidth=1.5)
        ax.plot(r["ci_low"], y, marker="|", color=colour, markersize=6)
        ax.plot(r["ci_high"], y, marker="|", color=colour, markersize=6)
        ax.plot(r["delta"], y, marker="o", color=colour,
                markersize=5, markeredgecolor="black", markeredgewidth=0.5)

    ax.axvline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.6)
    if kill_threshold:
        ax.axvline(kill_threshold, color="#A11", linewidth=0.7,
                   linestyle="--", alpha=0.7,
                   label=f"Kill threshold (+{kill_threshold})")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel(x_label)
    ax.set_title(title, loc="left", fontsize=9)
    ax.grid(axis="x", alpha=0.3, linewidth=0.4)

    lo = min(0.0, lo_arr.min() - x_pad)
    hi = hi_arr.max() + x_pad
    ax.set_xlim(lo, hi)


def _rows_m3(df: pd.DataFrame) -> list[dict]:
    """Order: basin (outer) × horizon (inner). Median statistic only."""
    df = df[df["statistic"] == "median"].copy()
    rows = []
    for basin in BASIN_ORDER:
        for h in HORIZONS:
            m = df[(df["basin"] == basin) & (df["horizon_d"] == h)]
            if len(m) == 0:
                continue
            r = m.iloc[0].to_dict()
            r["label"] = f"{basin} · h={h}d"
            rows.append(r)
    return rows


def _rows_m4(df: pd.DataFrame) -> list[dict]:
    df = df[df["statistic"] == "median"].copy()
    rows = []
    for mech in MECHANISM_ORDER:
        for h in HORIZONS:
            m = df[(df["mechanism"] == mech) & (df["horizon_d"] == h)]
            if len(m) == 0:
                continue
            r = m.iloc[0].to_dict()
            # Neither M4 statistic reports kill_cleared (mechanism should
            # not beat lumped). Use the significance flag instead.
            r["kill_cleared"] = False
            r["label"] = f"{mech} · h={h}d"
            rows.append(r)
    return rows


def _rows_m5c(df: pd.DataFrame) -> list[dict]:
    rows = []
    for basin in BASIN_ORDER:
        for h in HORIZONS:
            m = df[(df["basin"] == basin) & (df["horizon_d"] == h)]
            if len(m) == 0:
                continue
            r = m.iloc[0].to_dict()
            r["label"] = f"{basin} · h={h}d"
            rows.append(r)
    return rows


@click.command()
@click.option("--tables-dir", default="results/tables", show_default=True,
              help="Directory containing the bootstrap CSV tables.")
@click.option("--from-r2", is_flag=True,
              help="Read the bootstrap tables from R2 (paper2/tables/) instead.")
@click.option("--out", default="results/figures/fig_5_master_bootstrap",
              show_default=True,
              help="Output stem (no extension).")
@click.option("--upload-to-r2", is_flag=True,
              help="Mirror the figure to R2 under paper2/figures/.")
def main(tables_dir: str, from_r2: bool, out: str, upload_to_r2: bool):
    load_dotenv(override=False)
    r2 = r2_from_env() if (from_r2 or upload_to_r2) else None
    tdir = Path(tables_dir)

    def load(name: str) -> pd.DataFrame:
        return _load_table(
            name if from_r2 else (tdir / name),
            from_r2=from_r2, r2=r2,
        )

    df_m3 = load("bootstrap_m3_pub.csv")
    df_m4 = load("bootstrap_m4_mechanisms.csv")
    df_m5c = load("bootstrap_m5c_alerts.csv")

    rows_m3 = _rows_m3(df_m3)
    rows_m4 = _rows_m4(df_m4)
    rows_m5c = _rows_m5c(df_m5c)

    set_publication_defaults()
    height_mm = 260  # tall enough for 3 stacked forest plots
    fig, axes = plt.subplots(
        3, 1, figsize=figure_size(column="double", height_mm=height_mm),
        gridspec_kw={"height_ratios": [len(rows_m3), len(rows_m4), len(rows_m5c)]},
    )

    _plot_forest(
        axes[0], rows_m3,
        x_label="Δ NSE median (F0-PUB − persistence), 95 % CI",
        kill_threshold=KILL_M3,
        title="(a) F0-PUB vs persistence — four basins × five forecast horizons",
    )
    _plot_forest(
        axes[1], rows_m4,
        x_label="Δ NSE median (mechanism − lumped) on Alto Lerma, 95 % CI",
        kill_threshold=0.0,  # no kill threshold on M4; we want CI-crosses-0
        title="(b) Mechanism-guided transfer vs lumped baseline (Alto Lerma); "
              "every CI straddles zero → null result",
    )
    _plot_forest(
        axes[2], rows_m5c,
        x_label="Δ Value @ C/L=0.2 median (fuzzy best-cutoff − baseline threshold), 95 % CI",
        kill_threshold=KILL_M5c,
        title="(c) Fuzzy uncertainty-aware alerting vs simple-threshold baseline",
    )
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower right", frameon=True, fontsize=6.5)

    fig.suptitle("Paired-bootstrap 95 % confidence intervals for the three core comparisons",
                 fontsize=10.5, fontweight="bold", y=0.995)
    fig.tight_layout(pad=1.1, h_pad=1.5, rect=(0, 0, 1, 0.98))

    stem = Path(out)
    written = save_figure(
        fig, stem, kind="combination",
        metadata={
            "Title": "Bootstrap CI forest plots for the three core comparisons",
            "Author": "Daniel Sánchez-Ruiz",
            "Subject": "F0-PUB vs persistence, mechanism vs lumped, fuzzy vs threshold — 95 % CI",
        },
    )
    plt.close(fig)
    for p in written:
        click.echo(f"[19_master_fig] wrote {p.as_posix()}  "
                   f"({p.stat().st_size / 1024:.1f} KB)")

    if upload_to_r2:
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/figures"
        for p in written:
            r2.upload_file(f"{prefix}/{p.name}", p)
            click.echo(f"[19_master_fig]   -> r2://{r2.bucket}/{prefix}/{p.name}")


if __name__ == "__main__":
    main()
