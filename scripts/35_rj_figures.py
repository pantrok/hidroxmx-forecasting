#!/usr/bin/env python
"""Figures of the JOH revision, drawn from the tables of stages 32 and 33.

Fig. 7 is the tail diagnostic (RJ-2) and Fig. 8 the forcing-only demonstration at GANSL
(RJ-3). PDF and 600 dpi TIFF go to the
results folder; a small PNG for inspection goes to the outputs folder.
"""
from __future__ import annotations

from pathlib import Path

import click
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from hidroxmx.alert import CATEGORIES  # noqa: E402
from hidroxmx.revision import BASIN_SLUGS  # noqa: E402
from hidroxmx.viz import figure_size, save_figure, set_publication_defaults  # noqa: E402

DPI = 600
METHODS = {
    "original": ("Original score", "#0072B2", "o", "-"),
    "log": ("Log score", "#D55E00", "s", "--"),
    "aci": ("ACI", "#009E73", "^", ":"),
}
BANDS = ["<q50", "q50-q90", "q90-q95", ">=q95", ">=q99"]
BAND_LABELS = ["<q50", "q50–q90", "q90–q95", "≥q95", "≥q99"]
ALERT_COLOURS = {"GREEN": "#4DAF7C", "YELLOW": "#F0E442", "ORANGE": "#E69F00", "RED": "#C0392B"}


def write(fig, stem: Path, preview: Path, title: str) -> None:
    save_figure(fig, stem, kind="combination", formats=("tif", "pdf"), dpi=DPI,
                metadata={"Title": title, "Author": "Daniel Sánchez-Ruiz"})
    fig.savefig(preview / f"{stem.name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 7, tail diagnostic
# --------------------------------------------------------------------------- #
def tail_figure(tables: Path, out_dir: Path, stem: Path, preview: Path) -> None:
    bands = pd.read_csv(tables / "T_rj2_coverage_bands.csv")
    bands = bands[(bands["basis"] == "clipped") & (bands["h"] == 1)]
    test = pd.read_parquet(out_dir / "rj2_test.parquet")
    cal = pd.read_parquet(out_dir / "rj2_cal.parquet")
    top = pd.read_csv(tables / "T_rj2_events_top.csv")

    fig = plt.figure(figsize=figure_size("double", height_mm=150))
    grid = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.1], hspace=0.52, wspace=0.14,
                    left=0.07, right=0.985, top=0.885, bottom=0.095)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    for i, (ax, basin) in enumerate(zip(axes, BASIN_SLUGS, strict=True)):
        sub = bands[bands["basin"] == basin]
        for method, (_, colour, marker, style) in METHODS.items():
            line = sub[sub["method"] == method].set_index("band").reindex(BANDS)
            ax.plot(range(len(BANDS)), line["coverage_pct"], color=colour, marker=marker,
                    linestyle=style, linewidth=1.1, markersize=3.6)
        ax.axhline(90, color="black", linewidth=0.6, linestyle=(0, (4, 3)))
        ax.set_ylim(-4, 104)
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels(BAND_LABELS, rotation=55, ha="right", rotation_mode="anchor")
        ax.set_title(f"({'abcd'[i]}) {basin}", loc="left", fontsize=8)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Coverage of the 90% interval (%)")
    handles = [Line2D([], [], color=c, marker=m, linestyle=s, linewidth=1.1, markersize=3.6,
                      label=n)
               for n, c, m, s in METHODS.values()]
    handles.append(Line2D([], [], color="black", linewidth=0.6, linestyle=(0, (4, 3)),
                          label="90% nominal"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.0))

    ax = fig.add_subplot(grid[1, :2])
    for frame, label, colour, marker in ((cal, "Calibration 2021–2022", "#0072B2", "o"),
                                         (test, "Test 2023–2025", "#D55E00", "^")):
        f = frame[(frame["basin"] == "Bajo Pánuco") & (frame["h"] == 1)]
        err = (f["y"] - f["yhat"]).abs() / f["q95"]
        keep = (err > 0) & (f["y"] > 0)
        ax.scatter((f["y"] / f["q95"])[keep], err[keep], s=5, marker=marker, color=colour,
                   alpha=0.45, linewidths=0, label=label, rasterized=True)
    ax.axvline(1.0, color="black", linewidth=0.6, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed discharge / q95 of training")
    ax.set_ylabel("|residual| / q95 of training")
    ax.set_title("(e) Bajo Pánuco: error against discharge, h = 1 d", loc="left", fontsize=8)
    ax.legend(loc="upper left", frameon=True, fontsize=7, markerscale=2.2)

    ax = fig.add_subplot(grid[1, 2:])
    event = top[(top["basin"] == "Bajo Pánuco") & (top["h"] == 1)].iloc[0]
    peak = pd.Timestamp(event["peak_date"])
    f = test[(test["fold"] == event["fold"]) & (test["h"] == 1)
             & (test["date"].between(peak - pd.Timedelta(days=12), peak + pd.Timedelta(days=12)))]
    f = f.sort_values("date")
    ax.fill_between(f["date"], f["lo_log"], f["hi_log"], color="#D55E00", alpha=0.25,
                    linewidth=0, label="Log interval")
    ax.fill_between(f["date"], f["lo_original"].clip(lower=0), f["hi_original"], color="#0072B2",
                    alpha=0.35, linewidth=0, label="Original interval")
    ax.plot(f["date"], f["yhat"], color="#555555", linewidth=1.0, linestyle="--", label="Forecast")
    ax.plot(f["date"], f["y"], color="black", linewidth=1.2, marker="o", markersize=2.2,
            label="Observed")
    ax.axhline(event["q95"], color="black", linewidth=0.6, linestyle=(0, (4, 3)))
    ax.text(f["date"].iloc[0], event["q95"], " q95", va="bottom", fontsize=7)
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title(f"(f) {event['fold']}, largest Bajo Pánuco event, h = 1 d", loc="left", fontsize=8)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.legend(loc="upper left", frameon=True, fontsize=7)
    write(fig, stem, preview, "Tail coverage diagnostic")


# --------------------------------------------------------------------------- #
# Fig. 8, forcing-only demonstration
# --------------------------------------------------------------------------- #
def hydrograph(ax, strip, frame: pd.DataFrame, q95: float, title: str, ylabel: bool) -> None:
    f = frame.sort_values("target_date")
    x = pd.to_datetime(f["target_date"])
    ax.fill_between(x, f["lower_m3s"], f["upper_m3s"], color="#0072B2", alpha=0.25, linewidth=0,
                    label="90% interval")
    ax.plot(x, f["forecast_m3s"], color="#0072B2", linewidth=1.2, label="Forecast")
    ax.axhline(q95, color="black", linewidth=0.6, linestyle=(0, (4, 3)), label="q95 of training")
    ax.set_title(title, loc="left", fontsize=8)
    if ylabel:
        ax.set_ylabel("Discharge (m³/s)")
    ax.tick_params(labelbottom=False)
    levels = np.array([[CATEGORIES.index(v) for v in f["level_corrected"]]])
    edges = mdates.date2num(x.to_numpy())
    step = edges[1] - edges[0] if len(edges) > 1 else 1.0
    strip.imshow(levels, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
                 cmap=ListedColormap([ALERT_COLOURS[c] for c in CATEGORIES]),
                 extent=(edges[0] - step / 2, edges[-1] + step / 2, 0, 1))
    strip.set_yticks([])
    strip.grid(False)
    strip.xaxis_date()
    strip.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1, 11, 21)))
    strip.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    strip.tick_params(axis="x", labelrotation=35)
    strip.set_xlim(ax.get_xlim())


def demo_figure(tables: Path, numbers: dict, stem: Path, preview: Path) -> None:
    forecast = pd.read_csv(tables / "T_rj3_forecast.csv")
    q95 = numbers["rj3"]["q95_entrenamiento_m3s"]
    fig = plt.figure(figsize=figure_size("double", height_mm=100))
    grid = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.11], hspace=0.08, wspace=0.26,
                    left=0.075, right=0.99, top=0.83, bottom=0.17)
    panels = [("event", 1, "(a) October 2025, h = 1 d"), ("event", 3, "(b) October 2025, h = 3 d"),
              ("event", 7, "(c) October 2025, h = 7 d"), ("dry", 1, "(d) Dry reference, h = 1 d")]
    for i, (period, h, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        strip = fig.add_subplot(grid[1, i], sharex=ax)
        frame = forecast[(forecast["period"] == period) & (forecast["h"] == h)]
        hydrograph(ax, strip, frame, q95, title, ylabel=i == 0)
        if i == 0:
            strip.set_ylabel("Alert", rotation=0, ha="right", va="center", fontsize=7)
    handles = [Line2D([], [], color="#0072B2", linewidth=1.2, label="Forecast"),
               Patch(color="#0072B2", alpha=0.25, label="90% interval"),
               Line2D([], [], color="black", linewidth=0.6, linestyle=(0, (4, 3)),
                      label="q95 of training")]
    handles += [Patch(facecolor=ALERT_COLOURS[c], edgecolor="black", linewidth=0.3, label=c)
                for c in CATEGORIES]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.6)
    fig.text(0.5, 0.905, "GANSL held out, forcings only. Unverified: gauge record ends 2025-09-03",
             ha="center", va="center", fontsize=7.5, style="italic")
    write(fig, stem, preview, "Forcing-only forecast at held-out station GANSL")


@click.command()
@click.option("--figures", default="7,8", show_default=True)
@click.option("--out-dir", default="outputs/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
@click.option("--results-dir", default="results/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
def main(figures: str, out_dir: Path, results_dir: Path):
    import json

    wanted = {f.strip() for f in figures.split(",")}
    tables = results_dir / "tables"
    target = results_dir / "figures"
    target.mkdir(parents=True, exist_ok=True)
    preview = out_dir / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    set_publication_defaults()
    if "7" in wanted:
        tail_figure(tables, out_dir, target / "Fig7", preview)
    if "8" in wanted:
        numbers = json.loads((results_dir / "numbers_revision.json").read_text(encoding="utf-8"))
        demo_figure(tables, numbers, target / "Fig8", preview)


if __name__ == "__main__":
    main()
