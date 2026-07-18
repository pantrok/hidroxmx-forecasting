#!/usr/bin/env python
"""Stage 10 — Build response-kernel and static-attribute vectors per sub-basin.

Milestone: §12.2 (backbone) — feeds Stage 11 (parameter head) and Stage 13
(donor matching). This script is a stub in Fase 0 and will be filled in when
the backbone forecaster is trained.

Contract
--------
- Reads the canonical hydrometric Parquet and the sub-basin GeoPackages
  streamed from R2 (see conf/experiments/r2.yaml).
- Computes per sub-basin: recession constant, lag-to-peak, baseflow index,
  runoff ratio, flow-duration-curve shape.
- Computes per sub-basin the static attribute vector a_b (area, slope,
  hypsometry, drainage density, channel length / sinuosity, aspect, IDW
  climatology).
- Persists both as compact Parquet under {paper2.signatures}/ in R2.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--config", default="conf/experiments/splits.yaml", show_default=True)
@click.option("--out-prefix", default="paper2/signatures", show_default=True)
def main(config: str, out_prefix: str) -> None:
    raise SystemExit(
        "scripts/10_build_signatures.py — stub. Implementation lands with "
        "Milestone 2 of the experiment spec (§12.2)."
    )


if __name__ == "__main__":
    main()
