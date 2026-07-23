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


HORIZONS = (1, 2, 3, 5, 7)

# Folds with NSE below this threshold are excluded from the scatter panel.
# NSE < −1 means the model is worse than the training-window mean by more
# than the total variance of the test series — usually a degenerate output,
# not an honest comparison against persistence.
SCATTER_NSE_FLOOR = -1.0


def _list_folds_from_r2(r2, run_id: str) -> list[str]:
    """List every clave with a manifest.json under paper2/runs/{run_id}/.

    Basin-agnostic — the sweep script writes one folder per holdout, so
    the presence of a manifest tells us which stations are available
    regardless of which basin the sweep targets.
    """
    prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}/"
    claves: set[str] = set()
    client = r2._client()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=r2.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            parts = rel.split("/")
            if len(parts) >= 2 and parts[-1] == "manifest.json":
                claves.add(parts[0])
    return sorted(claves)


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
    if not claves:
        click.echo("[20_fig] listing available folds on R2 (auto-discover)…")
        claves = _list_folds_from_r2(r2, run_id)
        click.echo(f"[20_fig] discovered {len(claves)} folds: {claves}")
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
    """Grouped bar chart of persist vs F0-PUB NSE across horizons.

    Uses the **mean + std** (across folds) so the paper's headline
    number matches what most readers expect. Values below
    ``SCATTER_NSE_FLOOR`` (typically catastrophic model failures) are
    dropped from the mean *per horizon* so a single broken fold does
    not sink the average — same filter as the scatter panel. The
    diagnostic printed by ``main`` reports both mean and median for
    full transparency; the caption should note the number of folds
    excluded per horizon.
    """
    x = np.arange(len(HORIZONS))
    width = 0.36

    def _clean(col: str) -> pd.Series:
        s = df[col]
        return s.where(s >= SCATTER_NSE_FLOOR)

    persist_mean = np.array([np.nanmean(_clean(f"persist_nse_h{h}")) for h in HORIZONS])
    persist_std = np.array([np.nanstd(_clean(f"persist_nse_h{h}"), ddof=0) for h in HORIZONS])
    f0pub_mean = np.array([np.nanmean(_clean(f"f0pub_nse_h{h}")) for h in HORIZONS])
    f0pub_std = np.array([np.nanstd(_clean(f"f0pub_nse_h{h}"), ddof=0) for h in HORIZONS])

    persist_color = "#7A7A7A"      # neutral gray — a naive baseline
    f0pub_color = WONG_PALETTE[5]  # Wong blue for the model

    ax.bar(x - width / 2, persist_mean, width, yerr=persist_std,
           color=persist_color, edgecolor="black", linewidth=0.6,
           label="Persistence", capsize=2.5,
           error_kw={"elinewidth": 0.6, "ecolor": "black"})
    ax.bar(x + width / 2, f0pub_mean, width, yerr=f0pub_std,
           color=f0pub_color, edgecolor="black", linewidth=0.6,
           label="F0-PUB (multi-station)", capsize=2.5,
           error_kw={"elinewidth": 0.6, "ecolor": "black"})

    per_top = persist_mean + persist_std
    f0_top = f0pub_mean + f0pub_std
    per_bot = persist_mean - persist_std
    f0_bot = f0pub_mean - f0pub_std
    max_visual_top = float(np.nanmax(np.concatenate([per_top, f0_top])))
    min_visual_bot = float(np.nanmin(np.concatenate([per_bot, f0_bot])))
    y_max = min(1.45, max(1.15, max_visual_top + 0.20))
    y_min = max(-1.0, min(0.0, min_visual_bot - 0.05))

    for xi in range(len(HORIZONS)):
        delta = f0pub_mean[xi] - persist_mean[xi]
        text_y = min(f0_top[xi] + 0.03, y_max - 0.02)
        ax.text(xi + width / 2, text_y, f"Δ={delta:+.02f}",
                ha="center", va="bottom", fontsize=6.5,
                color=f0pub_color if delta > 0 else "black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} d" for h in HORIZONS])
    ax.set_ylabel("NSE (Nash–Sutcliffe efficiency)")
    ax.set_xlabel("Forecast horizon")
    n_folds = len(df)
    ax.set_title(f"(a) NSE by horizon — mean ± std ({n_folds} folds)",
                 loc="left", fontsize=8.5)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)


def _panel_scatter(ax, df: pd.DataFrame, h: int = 1):
    """Per-station scatter of F0-PUB vs persistence NSE at horizon h.

    Labelling strategy: individual labels for the seven "interesting"
    stations (outside the high-persistence cluster), placed with a
    per-station offset dictionary to avoid overlap on the mid-range
    stations that would otherwise pile up. The high-persistence cluster
    gets one boxed annotation pinned to the empty upper-left corner
    with a leader line to the cluster centroid.

    TODO(post-M4): the per-station offset dictionary is hand-tuned for
    Alto Lerma. Other basins (notably Valle de México, where 7-8
    stations cluster near persist~0.6-0.8, F0-PUB~0.6-0.75) still
    exhibit label overlap in the mid-range. Replace the dictionary with
    a collision-aware placement (adjustText, or a simple radial-sweep
    algorithm around each point that avoids other markers) once the
    modelling milestones are done and the exact station roster stops
    changing.
    """
    x_all = df[f"persist_nse_h{h}"].to_numpy()
    y_all = df[f"f0pub_nse_h{h}"].to_numpy()
    names_all = df["holdout"].tolist()

    # Exclude folds with undefined NSE (flat test series) or catastrophic
    # divergence (< NSE_FLOOR). Report them so the reader knows which
    # stations were suppressed from the scatter and can find them in the
    # per-fold manifests.
    valid = (np.isfinite(x_all) & np.isfinite(y_all)
             & (x_all >= SCATTER_NSE_FLOOR) & (y_all >= SCATTER_NSE_FLOOR))
    skipped = [names_all[i] for i in range(len(names_all)) if not valid[i]]
    if skipped:
        skipped_desc = ", ".join(
            f"{n}(persist={x_all[i]:+.2f}, F0pub={y_all[i]:+.2f})"
            for i, n in enumerate(names_all) if not valid[i]
        )
        click.echo(f"[20_fig] scatter (h={h}d): skipping {len(skipped)} folds "
                   f"with NaN or NSE<{SCATTER_NSE_FLOOR:g}: {skipped_desc}",
                   err=True)

    x = x_all[valid]
    y = y_all[valid]
    names = [names_all[i] for i in range(len(names_all)) if valid[i]]
    if len(x) == 0:
        ax.text(0.5, 0.5, f"No finite folds to plot at h = {h} d",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    wins = y > x

    # y = x reference and points
    ax.plot([-1, 1.05], [-1, 1.05], color="black", linewidth=0.5,
            linestyle="--", label="y = x")
    ax.scatter(x[wins], y[wins], s=32, color=WONG_PALETTE[3],
               edgecolor="black", linewidth=0.5, zorder=3,
               label=f"F0-PUB wins ({int(wins.sum())})")
    ax.scatter(x[~wins], y[~wins], s=32, color="#B0B0B0",
               edgecolor="black", linewidth=0.5, zorder=3,
               label=f"Persistence wins ({int((~wins).sum())})")

    # High-persistence cluster: mask points where both metrics >= 0.85.
    cluster_x_min = 0.85
    cluster_y_min = 0.85
    cluster = (x >= cluster_x_min) & (y >= cluster_y_min)
    cluster_names = [names[i] for i in range(len(names)) if cluster[i]]

    # Per-station manual offsets (dx_pt, dy_pt, ha, va) tuned by hand for
    # the Alto Lerma dataset. Any station not listed here falls back to
    # a directional heuristic — winners get a right/up label, losers a
    # left/down one — which is sensible for most datasets and can be
    # extended per basin as visual issues arise.
    manual_offsets = {
        "SLCGJ": ( 7,  4, "left",   "bottom"),
        "CALMX": ( 7, -3, "left",   "top"),
        "ECBGJ": ( 7,  4, "left",   "bottom"),
        "LAYMX": (-8,  6, "right",  "bottom"),
        "EGIMC": ( 7,  6, "left",   "bottom"),
        "SL2GJ": ( 7, -3, "left",   "top"),
        "SLVGJ": (-4, -8, "right",  "top"),
    }

    for xi, yi, name, is_cluster in zip(x, y, names, cluster):
        if is_cluster:
            continue
        if name in manual_offsets:
            dx, dy, ha, va = manual_offsets[name]
        elif yi >= xi:
            dx, dy, ha, va = 7, 3, "left", "bottom"
        else:
            dx, dy, ha, va = -7, -3, "right", "top"
        ax.annotate(name, (xi, yi), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6, color="black",
                    ha=ha, va=va)

    # Callout box for the high-persistence cluster: pinned to the empty
    # upper-left corner (where no station lands because that region
    # would require F0-PUB > 0.9 with persist < 0.5, which does not
    # occur in Alto Lerma). Arrow leads to the cluster centroid.
    if cluster_names:
        cx = float(np.mean(x[cluster]))
        cy = float(np.mean(y[cluster]))
        chunk_size = 4
        chunked = [", ".join(cluster_names[i:i + chunk_size])
                   for i in range(0, len(cluster_names), chunk_size)]
        text = ("High-persistence cluster\n"
                f"(persist NSE ≥ {cluster_x_min}, n = {len(cluster_names)}):\n"
                + "\n".join(chunked))
        ax.annotate(
            text,
            xy=(cx, cy),
            xytext=(0.02, 0.98),
            textcoords="axes fraction",
            fontsize=6, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="gray", linewidth=0.5, alpha=0.95),
            arrowprops=dict(arrowstyle="-|>", color="gray",
                            lw=0.6, mutation_scale=8,
                            connectionstyle="arc3,rad=-0.15"),
        )

    # Symmetric limits with a bit of slack; clamp so the ticks stay legible.
    lo = float(min(x.min(), y.min())) - 0.06
    hi = float(max(x.max(), y.max())) + 0.06
    lo = max(lo, SCATTER_NSE_FLOOR - 0.05)
    hi = min(hi, 1.02)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Persistence NSE (h = {h} d)")
    ax.set_ylabel(f"F0-PUB NSE (h = {h} d)")
    ax.set_title(f"(b) F0-PUB vs persistence, h = {h} d",
                 loc="left", fontsize=8.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)


@click.command()
@click.option("--run-id", default="F0pub-alto-lerma-sweep-01", show_default=True)
@click.option("--out", default="results/figures/fig_3_pub_summary_alto_lerma",
              show_default=True, help="Output path stem (no extension).")
@click.option("--horizon-scatter", default=1, show_default=True,
              help="Horizon (days) shown in the per-station scatter panel.")
@click.option("--basin-label", default="", show_default=True,
              help="Basin display name written in the panel titles "
                   "(e.g. 'Alto Lerma'). Empty leaves it out.")
@click.option("--claves", default="", show_default=True,
              help="Comma-separated station keys. Empty auto-discovers "
                   "every fold present on R2 under the run-id.")
@click.option("--upload-to-r2", is_flag=True,
              help="Mirror the rendered TIFF, PDF and PNG to R2 under "
                   "{R2_PAPER2_PREFIX}/figures/. Recommended for every "
                   "figure the paper cites so the durable source lives "
                   "on R2 alongside the run manifests it derives from.")
def main(run_id: str, out: str, horizon_scatter: int,
         basin_label: str, claves: str, upload_to_r2: bool):
    load_dotenv(override=False)
    claves_tuple: tuple[str, ...] = tuple(
        c.strip() for c in claves.split(",") if c.strip()
    )
    if claves_tuple:
        click.echo(f"[20_fig] Loading {len(claves_tuple)} manifests from R2 "
                   f"under run-id {run_id!r}…")
    else:
        click.echo(f"[20_fig] Auto-discovering folds on R2 for run-id {run_id!r}…")
    df = _load_all_folds(run_id, claves_tuple)
    if df.empty:
        raise SystemExit("[20_fig] No manifests could be loaded; abort.")
    if basin_label:
        df.attrs["basin_label"] = basin_label
    click.echo(f"[20_fig] Loaded {len(df)} folds.")

    # Aggregate stats echo. Uses nan-aware stats and reports both mean
    # and median so an outlier fold is visible in the diff between the
    # two. Flags folds with NSE < SCATTER_NSE_FLOOR at h=1 as candidates
    # for exclusion in the paper (worth investigating in their per-fold
    # manifest.json).
    click.echo(f"[20_fig] Aggregate NSE ({len(df)} folds):")
    for h in HORIZONS:
        f0_series = df[f"f0pub_nse_h{h}"]
        pe_series = df[f"persist_nse_h{h}"]
        f0 = np.nanmean(f0_series)
        f0_med = np.nanmedian(f0_series)
        pe = np.nanmean(pe_series)
        pe_med = np.nanmedian(pe_series)
        wins = int((f0_series > pe_series).sum())
        n_valid = int(f0_series.notna().sum())
        click.echo(f"[20_fig]   h={h}d  "
                   f"F0pub mean={f0:+.3f} med={f0_med:+.3f}  "
                   f"persist mean={pe:+.3f} med={pe_med:+.3f}  "
                   f"wins={wins}/{n_valid}")
    # Identify problem folds (NSE < floor at h=1 in either baseline or model).
    problem_mask = ((df["f0pub_nse_h1"] < SCATTER_NSE_FLOOR)
                    | (df["persist_nse_h1"] < SCATTER_NSE_FLOOR)
                    | df["f0pub_nse_h1"].isna()
                    | df["persist_nse_h1"].isna())
    if problem_mask.any():
        click.echo(f"[20_fig] Problem folds at h=1 (NSE<{SCATTER_NSE_FLOOR:g} or NaN):",
                   err=True)
        for _, row in df[problem_mask].iterrows():
            click.echo(f"[20_fig]   {row['holdout']:<8s}  "
                       f"persist={row['persist_nse_h1']:+.3f}  "
                       f"F0pub={row['f0pub_nse_h1']:+.3f}", err=True)

    set_publication_defaults()
    # Double-column width, tall enough to keep the scatter square.
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=figure_size(column="double", height_ratio=0.50),
    )
    _panel_bars(ax_a, df)
    _panel_scatter(ax_b, df, h=horizon_scatter)
    if basin_label:
        fig.suptitle(basin_label, fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(pad=1.2, w_pad=2.0, rect=(0, 0, 1, 0.96 if basin_label else 1.0))

    stem = Path(out)
    written = save_figure(
        fig, stem, kind="combination",
        metadata={
            "Title": f"PUB leave-one-out summary{(' — ' + basin_label) if basin_label else ''}",
            "Author": "Daniel Sánchez-Ruiz",
            "Subject": f"F0-PUB vs persistence, {len(df)}-fold leave-one-out",
        },
    )
    plt.close(fig)
    for p in written:
        click.echo(f"[20_fig] wrote {p.as_posix()}  "
                   f"({p.stat().st_size / 1024:.1f} KB)")

    if upload_to_r2:
        r2 = r2_from_env()
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/figures"
        for p in written:
            r2.upload_file(f"{prefix}/{p.name}", p)
            click.echo(f"[20_fig]   -> r2://{r2.bucket}/{prefix}/{p.name}")


if __name__ == "__main__":
    main()
