#!/usr/bin/env python
"""Stage 14 — Uncertainty quantification (conformal / CQR / Bayesian / ensemble).

Milestone: §12.5 (Path B calibration diagnostics). Stub in Fase 0.
"""
from __future__ import annotations

import click


@click.command()
@click.option(
    "--method",
    type=click.Choice(["conformal", "cqr", "adaptive_conformal", "bayes", "ensemble"]),
    required=True,
)
@click.option("--run-id", required=True)
def main(method: str, run_id: str) -> None:
    raise SystemExit(
        "scripts/14_uq.py — stub. Implementation lands with Milestone 5."
    )


if __name__ == "__main__":
    main()
