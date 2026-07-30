"""Tests for the Journal of Hydrology figure-export helpers."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from hidroxmx.viz import (  # noqa: E402
    COLUMN_WIDTHS_MM,
    FIGURE_KINDS,
    WONG_PALETTE,
    figure_size,
    save_figure,
    set_publication_defaults,
)


def _dummy_fig():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [0, 1, 0], label="a")
    ax.plot([0, 1, 2], [0.5, 0.2, 0.8], label="b")
    ax.legend()
    return fig


def test_figure_kinds_match_the_guide():
    assert FIGURE_KINDS["halftone"].dpi == 300
    assert FIGURE_KINDS["combination"].dpi == 500
    assert FIGURE_KINDS["line"].dpi == 1000
    # Pixel floors from the Elsevier guide (single column).
    assert FIGURE_KINDS["halftone"].single_col_px == 1063
    assert FIGURE_KINDS["combination"].single_col_px == 1772
    assert FIGURE_KINDS["line"].single_col_px == 3543


def test_column_widths_mm():
    assert COLUMN_WIDTHS_MM["single"] == 90.0
    assert COLUMN_WIDTHS_MM["double"] == 190.0


def test_figure_size_derives_width_from_column():
    w, h = figure_size(column="single")
    assert abs(w - 90.0 / 25.4) < 1e-6
    assert h > 0
    w2, _ = figure_size(column="double")
    assert w2 > w


def test_figure_size_rejects_unknown_column():
    with pytest.raises(ValueError):
        figure_size(column="triple")


def test_wong_palette_has_eight_hex_colours():
    assert len(WONG_PALETTE) == 8
    for c in WONG_PALETTE:
        assert c.startswith("#") and len(c) == 7


def test_set_publication_defaults_is_idempotent_and_touches_rcparams():
    set_publication_defaults()
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42
    # A second call should not raise or change the final state meaningfully.
    set_publication_defaults()
    assert plt.rcParams["pdf.fonttype"] == 42


def test_save_figure_writes_tif_pdf_png_at_correct_dpi(tmp_path):
    fig = _dummy_fig()
    written = save_figure(fig, tmp_path / "fig", kind="combination")
    plt.close(fig)
    names = [p.suffix for p in written]
    assert names == [".tif", ".pdf", ".png"]
    for p in written:
        assert p.exists()
        assert p.stat().st_size > 0
    # TIFF must carry the requested dpi (Elsevier reads this metadata).
    with Image.open(tmp_path / "fig.tif") as im:
        dpi_x, dpi_y = im.info.get("dpi", (0, 0))
        assert round(dpi_x) == 500
        assert round(dpi_y) == 500


def test_save_figure_respects_kind_dpi(tmp_path):
    fig = _dummy_fig()
    save_figure(fig, tmp_path / "line", kind="line", formats=("tif",))
    save_figure(fig, tmp_path / "half", kind="halftone", formats=("tif",))
    plt.close(fig)
    with Image.open(tmp_path / "line.tif") as im:
        assert round(im.info["dpi"][0]) == 1000
    with Image.open(tmp_path / "half.tif") as im:
        assert round(im.info["dpi"][0]) == 300


def test_save_figure_rejects_unknown_kind(tmp_path):
    fig = _dummy_fig()
    with pytest.raises(ValueError):
        save_figure(fig, tmp_path / "fig", kind="bogus")
    plt.close(fig)


def test_save_figure_rejects_unknown_format(tmp_path):
    fig = _dummy_fig()
    with pytest.raises(ValueError):
        save_figure(fig, tmp_path / "fig", formats=("gif",))
    plt.close(fig)
