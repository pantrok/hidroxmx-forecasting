#!/usr/bin/env python
"""Stage 13 — Donor matching under four selection criteria.

Milestone: §12.3 (load-bearing test) → §12.4 (main transfer experiment). Stub
in Fase 0.
"""
from __future__ import annotations

import click


@click.command()
@click.option(
    "--criterion",
    type=click.Choice(["attr", "perf", "signature", "invariance"]),
    required=True,
)
@click.option("--run-id", required=True)
def main(criterion: str, run_id: str) -> None:
    raise SystemExit(
        "scripts/13_donor_matching.py — stub. Implementation lands with "
        "Milestone 3–4."
    )


if __name__ == "__main__":
    main()
