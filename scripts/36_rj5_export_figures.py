#!/usr/bin/env python
"""RJ-5: one file per figure, in the format Elsevier asks for.

Figs. 2, 4 and 5 come from the submitted vector PDFs, kept in results/revision_joh/published,
so their content is the submitted one, rasterized at 600 dpi. Fig. 1 keeps its submitted
resolution. Fig. 3 joins its four basins in one file. Fig. 6 is redrawn with the scenario
labels of Table 4. Figs. 7 and 8 come from stage 35, which runs first.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import click
import matplotlib
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hidroxmx.revision import (  # noqa: E402
    BASIN_SLUGS,
    TYPE_DIAGNOSTIC,
    clean,
    folds_of,
    read_manifest,
    update_numbers,
)
from hidroxmx.viz import figure_size, save_figure, set_publication_defaults  # noqa: E402

SOURCE = "scripts/36_rj5_export_figures.py"
DPI = 600
FIG1_DPI = 300
HORIZONS = (1, 2, 3, 5, 7)
FROM_PDF = {"Fig2": "fig_2_basin_inclusion", "Fig4": "fig_4_cross_basin_pub",
            "Fig5": "fig_5_master_bootstrap"}
ORDER = ("Valle de México", "Alto Lerma", "Bajo Pánuco", "Medio Balsas")
MIN_FONT = 7.0


def rasterize(pdf: Path, stem: Path) -> None:
    """Render a PDF page to an LZW TIFF, through PNG because not every pdftoppm has TIFF."""
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", str(pdf), str(stem)],
                   check=True, capture_output=True)
    png = stem.with_suffix(".png")
    with Image.open(png) as page:
        page.convert("RGB").save(stem.with_suffix(".tif"), compression="tiff_lzw", dpi=(DPI, DPI))
    png.unlink()


def load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def basin_frame(mirror: Path, slug: str) -> pd.DataFrame:
    run = f"F0pub-{slug}-sweep-01"
    rows = []
    for clave in folds_of(mirror, run):
        doc = read_manifest(mirror, run, clave)
        m, cfg = doc["metrics"], doc["config"]
        row = {"holdout": cfg["holdout"], "name": cfg.get("holdout_name", "")}
        for h in HORIZONS:
            row[f"f0pub_nse_h{h}"] = m[f"nse_h{h}"]
            row[f"persist_nse_h{h}"] = m[f"persist_nse_h{h}"]
        rows.append(row)
    return pd.DataFrame(rows)


def composite_fig3(summary, mirror: Path, stem: Path, preview: Path) -> None:
    set_publication_defaults()
    fig, axes = plt.subplots(4, 2, figsize=figure_size("double", height_mm=235))
    for i, basin in enumerate(ORDER):
        df = basin_frame(mirror, BASIN_SLUGS[basin])
        ax_a, ax_b = axes[i]
        summary._panel_bars(ax_a, df)
        summary._panel_scatter(ax_b, df, h=1)
        ax_a.set_title(f"({'aceg'[i]}) {basin}: NSE by horizon, mean ± std ({len(df)} folds)",
                       loc="left", fontsize=8)
        ax_b.set_title(f"({'bdfh'[i]}) {basin}: F0-PUB vs persistence, h = 1 d",
                       loc="left", fontsize=8)
        # The square scatter of the single-basin figure crowds the station labels here.
        ax_b.set_aspect("auto")
        ax_b.legend(loc="upper left" if basin == "Bajo Pánuco" else "lower right",
                    frameon=True, framealpha=0.9)
        for ax in (ax_a, ax_b):
            for text in ax.texts:
                text.set_fontsize(max(text.get_fontsize(), MIN_FONT))
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(MIN_FONT)
    fig.tight_layout(pad=0.8, h_pad=1.0, w_pad=1.5)
    save_figure(fig, stem, kind="combination", formats=("tif", "pdf"), dpi=DPI,
                metadata={"Title": "F0-PUB versus persistence per basin",
                          "Author": "Daniel Sánchez-Ruiz"})
    fig.savefig(preview / "Fig3.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option("--out-dir", default="outputs/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
@click.option("--results-dir", default="results/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
def main(out_dir: Path, results_dir: Path):
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    preview = out_dir / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    mirror = out_dir / "r2"

    submitted = results_dir / "published" / "figures"
    for name, source in FROM_PDF.items():
        pdf = figures / f"{name}.pdf"
        shutil.copyfile(submitted / f"{source}.pdf", pdf)
        rasterize(pdf, figures / name)
        click.echo(f"[36_rj5] {name}: rasterized at {DPI} dpi")
    with Image.open(submitted / "fig_1_coverage_map.png") as page:
        page.convert("RGB").save(figures / "Fig1.tif", compression="tiff_lzw",
                                 dpi=(FIG1_DPI, FIG1_DPI))

    summary = load_script(Path("scripts/20_figure_pub_summary.py"))
    composite_fig3(summary, mirror, figures / "Fig3", preview)
    click.echo("[36_rj5] Fig3: four basins in one file")

    subprocess.run([sys.executable, "scripts/21_dt_paper_figure.py",
                    "--arrays-dir", str(mirror / "runs"), "--out", str(figures / "Fig6"),
                    "--dpi", str(DPI)], check=True)
    shutil.move(str(figures / "Fig6.png"), preview / "Fig6.png")
    click.echo("[36_rj5] Fig6: redrawn")

    inventory = {}
    for tif in sorted(figures.glob("Fig*.tif")):
        with Image.open(tif) as image:
            dpi = float(image.info["dpi"][0])
            inventory[tif.stem] = {"ancho_mm": image.size[0] / dpi * 25.4,
                                   "alto_mm": image.size[1] / dpi * 25.4, "dpi": dpi,
                                   "pdf": (figures / f"{tif.stem}.pdf").exists()}
    update_numbers(results_dir / "numbers_revision.json", "rj5", TYPE_DIAGNOSTIC, SOURCE,
                   clean({"figuras": inventory}))


if __name__ == "__main__":
    main()
