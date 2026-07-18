#!/usr/bin/env python
"""Stage 16 — Coverage / blind-spot map (§7 of the experiment spec).

Milestone 1 of the execution order (§12.1). First non-stub script of the
repository — the goal of Milestone 1 is:

- Load the delineated sub-basin polygons streamed from R2
  (``processed/cuencas/*.gpkg``).
- Load Google Flood Hub reach coverage (``reaches.geojson``) and GloFAS-CEMS
  coverage layers (public FTP / API). Both live under R2 as cached copies.
- Compute per sub-basin: is its outlet reach covered by Flood Hub? by
  GloFAS? At what confidence? Aggregate to a coverage flag with a threshold
  co-designed for the paper.
- Output Figure 1 (map, ~300 dpi) and a table of the internal ungauged
  tributaries below the threshold (candidates: Alta del Balsas headwaters
  Tlaxcala–Puebla; Pánuco upland sub-basins).

Because Fase 0 focuses on the repository skeleton, the current file is a
click stub. The implementation lands as the first substantive commit of
Milestone 1.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--out-prefix", default="paper2/coverage", show_default=True)
@click.option("--dpi", default=300, show_default=True)
def main(out_prefix: str, dpi: int) -> None:
    raise SystemExit(
        "scripts/16_coverage_map.py — stub. First substantive implementation "
        "lands in Milestone 1."
    )


if __name__ == "__main__":
    main()
