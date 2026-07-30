"""Figure export at Journal of Hydrology / Elsevier submission spec.

The guide (fetched from
https://www.sciencedirect.com/journal/journal-of-hydrology/publish/guide-for-authors)
mandates the following raster resolutions when submitting non-vector
artwork:

- **Halftones** — colour or grayscale photographs, choropleths, heatmaps,
  anything with continuous-tone fill. TIFF / JPG / PNG, **≥ 300 dpi**.
  Single-column pixel floor: 1063 px. Full-page: 2244 px.
- **Bitmapped line drawings** — line art with no continuous-tone fill.
  TIFF / JPG / PNG, **≥ 1000 dpi**. Single-column floor: 3543 px. Full-
  page: 7480 px.
- **Combinations** — line art on top of a halftone (most model plots,
  most maps). TIFF / JPG / PNG, **≥ 500 dpi**. Single-column floor:
  1772 px. Full-page: 3740 px.

Vector formats (EPS, PDF) are preferred for line and combination figures
because they scale losslessly. We therefore save both a raster TIFF at
the correct dpi *and* a vector PDF for every figure, plus a raster PNG
at the same dpi for git preview / README embedding.

Physical widths (from the pixel counts above at 300 dpi):

- Single column: ~90 mm (3.54 in)
- 1.5 column:    ~140 mm (5.51 in) — Elsevier's intermediate width
- Full page:     ~190 mm (7.48 in)

Font choice: sans-serif (Arial / Helvetica / DejaVu Sans) at 8 pt so
the figure remains readable after column-width reduction. ``pdf.fonttype``
and ``ps.fonttype`` are forced to 42 (TrueType) so text is editable
downstream in Illustrator / Inkscape and gets properly embedded in the
vector output.

Colour palette: Wong (2011) 8-colour categorical palette
(colourblind-accessible), plus ``viridis`` for sequential and
``cividis`` for print-safe sequential, per the guide's accessibility
mandate.

Every figure produced through this module is derived from data via
reproducible analytical / statistical methods, so it falls under the
GenAI-permitted category of the Elsevier policy. The paper's figure
captions must nevertheless include the disclosure sentence when AI
tools were involved in generating the underlying visualisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Elsevier dpi table (verbatim from the guide)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FigureKindSpec:
    dpi: int
    single_col_px: int
    full_page_px: int
    label: str


FIGURE_KINDS: dict[str, FigureKindSpec] = {
    "halftone":    FigureKindSpec(dpi=300,  single_col_px=1063, full_page_px=2244,
                                  label="halftone (photos, heatmaps, choropleths)"),
    "combination": FigureKindSpec(dpi=500,  single_col_px=1772, full_page_px=3740,
                                  label="combination (line + halftone)"),
    "line":        FigureKindSpec(dpi=1000, single_col_px=3543, full_page_px=7480,
                                  label="bitmapped line drawing"),
}


COLUMN_WIDTHS_MM = {
    "single": 90.0,   # 1063 px @ 300 dpi
    "1.5":    140.0,  # Elsevier's intermediate width
    "double": 190.0,  # 2244 px @ 300 dpi
}


# --------------------------------------------------------------------------- #
# Colour palette — Wong (2011), 8 colours, colour-blind accessible
# --------------------------------------------------------------------------- #
WONG_PALETTE: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)


# --------------------------------------------------------------------------- #
# Matplotlib defaults
# --------------------------------------------------------------------------- #
def set_publication_defaults() -> None:
    """Apply matplotlib rcParams for J. Hydrology submission figures.

    Safe to call multiple times. Preserves the user's backend choice.
    """
    plt.rcParams.update({
        # Fonts sized for readability after column-width reduction.
        "font.family":   "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":       8,
        "axes.labelsize":  8,
        "axes.titlesize":  9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,

        # Thin lines so print resolution does not smear them.
        "lines.linewidth":     1.0,
        "lines.markersize":    4,
        "axes.linewidth":      0.6,
        "grid.linewidth":      0.4,
        "xtick.major.width":   0.6,
        "ytick.major.width":   0.6,
        "xtick.direction":     "out",
        "ytick.direction":     "out",

        # Grid on by default at low opacity so it does not dominate.
        "axes.grid":  True,
        "grid.alpha": 0.3,

        # Text must remain editable in the vector output (TrueType, not paths).
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
        "svg.fonttype": "none",

        # Prefer the Wong palette for categorical colouring.
        "axes.prop_cycle": plt.cycler(color=WONG_PALETTE),
    })


# --------------------------------------------------------------------------- #
# Figure-size helpers
# --------------------------------------------------------------------------- #
def figure_size(column: str = "single",
                height_ratio: float = 0.72,
                height_mm: float | None = None) -> tuple[float, float]:
    """Return ``(width_in, height_in)`` for ``plt.figure(figsize=…)``.

    ``column`` is one of the keys in :data:`COLUMN_WIDTHS_MM`
    (``'single'``, ``'1.5'``, ``'double'``). Height is derived from
    ``height_ratio`` unless ``height_mm`` is supplied.
    """
    if column not in COLUMN_WIDTHS_MM:
        raise ValueError(f"unknown column={column!r}; choose from {sorted(COLUMN_WIDTHS_MM)}")
    width_mm = COLUMN_WIDTHS_MM[column]
    width_in = width_mm / 25.4
    if height_mm is not None:
        height_in = height_mm / 25.4
    else:
        height_in = width_in * height_ratio
    return (width_in, height_in)


# --------------------------------------------------------------------------- #
# save_figure — write TIFF + PDF + PNG at the correct spec
# --------------------------------------------------------------------------- #
DEFAULT_FORMATS: tuple[str, ...] = ("tif", "pdf", "png")


def save_figure(fig: "plt.Figure", path_stem: str | Path, *,
                kind: str = "combination",
                formats: Iterable[str] = DEFAULT_FORMATS,
                metadata: dict | None = None) -> list[Path]:
    """Save ``fig`` at the resolution the journal requires for ``kind``.

    Parameters
    ----------
    fig : matplotlib Figure.
    path_stem : path *without* extension; the function appends ``.tif``,
        ``.pdf``, ``.png`` etc. Parent directories are created.
    kind : one of ``'halftone'`` (300 dpi), ``'combination'`` (500 dpi)
        or ``'line'`` (1000 dpi). See :data:`FIGURE_KINDS`.
    formats : iterable of extensions to write. Defaults to TIFF + PDF +
        PNG so every submission requirement is met in one call.
    metadata : optional dict passed to matplotlib as figure metadata
        (respected for PDF; ignored for other formats).

    Returns the list of written paths.
    """
    if kind not in FIGURE_KINDS:
        raise ValueError(f"unknown kind={kind!r}; choose from {sorted(FIGURE_KINDS)}")
    dpi = FIGURE_KINDS[kind].dpi
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fmt in formats:
        fmt = fmt.lstrip(".").lower()
        out = stem.with_suffix(f".{fmt}")
        save_kwargs: dict = {"bbox_inches": "tight"}
        if metadata and fmt in ("pdf", "svg"):
            save_kwargs["metadata"] = metadata
        if fmt == "tif" or fmt == "tiff":
            # Elsevier accepts LZW-compressed TIFF and it shrinks the file
            # to a fraction of the uncompressed size without quality loss.
            fig.savefig(out, dpi=dpi,
                        pil_kwargs={"compression": "tiff_lzw"},
                        **save_kwargs)
        elif fmt == "pdf":
            fig.savefig(out, format="pdf", **save_kwargs)
        elif fmt in ("png", "jpg", "jpeg", "eps", "svg"):
            fig.savefig(out, dpi=dpi, **save_kwargs)
        else:
            raise ValueError(f"unsupported format {fmt!r}")
        written.append(out)
    return written
