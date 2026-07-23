#!/usr/bin/env python
"""Stage 21 — paper figure for Milestone 6 (scoped predictive digital twin).

Assembles a 2×2 grid at J. Hydrology submission spec:

- **Row 1**: 1-day-ahead hydrograph slice comparing observed streamflow,
  raw F0-PUB forecast and innovation-persistence-assimilated forecast.
  One column per station (SLVGJ ‖ CMNMC by default — the two stations
  where the assimilation kill condition cleared in Milestone 6).
- **Row 2**: what-if 1-day-ahead fan chart on the same time window,
  showing the baseline forecast and the five perturbation scenarios
  (precip ±20 %, temp ±2 °C, 14-day dry-out) as coloured lines. Reveals
  the model's differential sensitivity to precipitation, temperature and
  antecedent-precipitation loss.

Data come from the ``forecast_arrays.npz`` files that stage 20 writes
per station under ``outputs/dt_demo/{run_id}/{clave}/``. Pass
``--from-r2`` to pull them instead from
``paper2/runs/{run_id}/{clave}/forecast_arrays.npz``.
"""
from __future__ import annotations

import os
from pathlib import Path

import click
import matplotlib
import numpy as np
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


SCENARIO_ORDER = ("precip_+20pct", "precip_-20pct",
                   "temp_+2C", "temp_-2C", "dry_out_14d")
SCENARIO_LABELS = {
    "precip_+20pct": "precip +20 %",
    "precip_-20pct": "precip −20 %",
    "temp_+2C":       "temp +2 °C",
    "temp_-2C":       "temp −2 °C",
    "dry_out_14d":    "dry-out 14 d",
}
SCENARIO_COLOURS = {
    "precip_+20pct": WONG_PALETTE[5],   # blue
    "precip_-20pct": WONG_PALETTE[1],   # orange
    "temp_+2C":       WONG_PALETTE[6],   # vermillion
    "temp_-2C":       WONG_PALETTE[3],   # bluish green
    "dry_out_14d":    WONG_PALETTE[7],   # reddish purple
}


def _load_npz(path_or_key: str, *, from_r2: bool, r2=None):
    if from_r2:
        import io
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/runs"
        payload = r2.get_bytes(f"{prefix}/{path_or_key}")
        return np.load(io.BytesIO(payload))
    return np.load(path_or_key)


def _panel_hydrograph(ax, data, title: str, slice_len: int = 90):
    y_true = data["y_true"][:, 0]
    y_raw = data["y_raw"][:, 0]
    y_assim = data["y_assim"][:, 0]
    n = y_true.size
    lo = max(0, n - slice_len)
    x = np.arange(lo, n)
    ax.plot(x, y_true[lo:n], color="black", linewidth=1.2,
            label="Observed", zorder=3)
    ax.plot(x, y_raw[lo:n], color="#8A8A8A", linewidth=1.0,
            linestyle="--", label="F0-PUB raw", alpha=0.85)
    ax.plot(x, y_assim[lo:n], color=WONG_PALETTE[5], linewidth=1.2,
            label="F0-PUB assimilated")
    ax.set_ylabel("Streamflow (m³/s)")
    ax.set_xlabel("Test day (last %d shown)" % slice_len)
    ax.set_title(title, loc="left", fontsize=9)
    ax.legend(loc="upper right", frameon=True, fontsize=6.5)
    ax.grid(alpha=0.3)


def _panel_whatif(ax, data, title: str, slice_len: int = 90):
    y_baseline = data["y_raw"][:, 0]
    n = y_baseline.size
    lo = max(0, n - slice_len)
    x = np.arange(lo, n)
    ax.plot(x, y_baseline[lo:n], color="black", linewidth=1.2,
            label="Baseline forecast", zorder=4)
    for name in SCENARIO_ORDER:
        key = f"whatif__{name}"
        if key not in data:
            continue
        y_scen = data[key][:, 0]
        ax.plot(x, y_scen[lo:n], color=SCENARIO_COLOURS[name],
                linewidth=0.9, alpha=0.9, label=SCENARIO_LABELS[name])
    ax.set_ylabel("Streamflow (m³/s)")
    ax.set_xlabel("Test day (last %d shown)" % slice_len)
    ax.set_title(title, loc="left", fontsize=9)
    ax.legend(loc="upper right", frameon=True, fontsize=6.0, ncol=2)
    ax.grid(alpha=0.3)


@click.command()
@click.option("--run-id", default="dt-demo-01", show_default=True)
@click.option("--stations", default="SLVGJ,CMNMC", show_default=True,
              help="Comma-separated station keys to show (max 2).")
@click.option("--basins", default="Alto Lerma,Medio Balsas", show_default=True,
              help="Basin labels for the sub-plot titles, one per station.")
@click.option("--arrays-dir", default="outputs/dt_demo", show_default=True,
              help="Directory holding per-run per-station forecast_arrays.npz.")
@click.option("--from-r2", is_flag=True,
              help="Read the npz files from R2 (paper2/runs/) instead of local.")
@click.option("--slice-len", default=90, show_default=True, type=int,
              help="Number of trailing test days plotted per panel.")
@click.option("--out", default="results/figures/fig_6_dt_demo",
              show_default=True, help="Output stem (no extension).")
@click.option("--upload-to-r2", is_flag=True,
              help="Mirror the figure to R2 under paper2/figures/.")
def main(run_id, stations, basins, arrays_dir, from_r2, slice_len, out, upload_to_r2):
    load_dotenv(override=False)
    r2 = r2_from_env() if (from_r2 or upload_to_r2) else None

    stations_list = [s.strip() for s in stations.split(",") if s.strip()]
    basins_list = [b.strip() for b in basins.split(",") if b.strip()]
    if len(stations_list) != len(basins_list):
        raise SystemExit(f"[21_dt_fig] --stations has {len(stations_list)} entries, "
                         f"--basins has {len(basins_list)} — must match.")
    if len(stations_list) > 2:
        raise SystemExit("[21_dt_fig] this figure is a 2×2 grid; pass at most 2 stations.")

    # Load npz for every requested station.
    datasets = []
    for clave in stations_list:
        if from_r2:
            rel = f"{run_id}/{clave}/forecast_arrays.npz"
            click.echo(f"[21_dt_fig] loading r2://.../{rel}")
        else:
            rel = str(Path(arrays_dir) / run_id / clave / "forecast_arrays.npz")
            click.echo(f"[21_dt_fig] loading {rel}")
        datasets.append(_load_npz(rel, from_r2=from_r2, r2=r2))

    set_publication_defaults()
    # 2 rows × N cols; keep it tall enough for the hydrograph + fan.
    n_cols = len(stations_list)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=figure_size(column="double", height_mm=160),
        squeeze=False,
    )
    for col, (clave, basin, data) in enumerate(
        zip(stations_list, basins_list, datasets)
    ):
        _panel_hydrograph(
            axes[0, col], data,
            title=f"(a{col + 1}) {basin} · {clave} — assimilation vs raw",
            slice_len=slice_len,
        )
        _panel_whatif(
            axes[1, col], data,
            title=f"(b{col + 1}) {basin} · {clave} — what-if scenarios",
            slice_len=slice_len,
        )
    fig.suptitle("Scoped predictive digital twin: assimilation and what-if scenarios",
                 fontsize=10.5, fontweight="bold", y=0.995)
    fig.tight_layout(pad=1.1, h_pad=1.5, w_pad=2.0, rect=(0, 0, 1, 0.97))

    stem = Path(out)
    written = save_figure(
        fig, stem, kind="combination",
        metadata={
            "Title": "Scoped predictive digital twin",
            "Author": "Daniel Sánchez-Ruiz",
            "Subject": (f"Assimilation and what-if scenarios on "
                        f"{', '.join(stations_list)}"),
        },
    )
    plt.close(fig)
    for p in written:
        click.echo(f"[21_dt_fig] wrote {p.as_posix()}  ({p.stat().st_size/1024:.1f} KB)")

    if upload_to_r2:
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/figures"
        for p in written:
            r2.upload_file(f"{prefix}/{p.name}", p)
            click.echo(f"[21_dt_fig]   -> r2://{r2.bucket}/{prefix}/{p.name}")


if __name__ == "__main__":
    main()
