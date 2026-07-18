#!/usr/bin/env python
"""Stage 12 — Parameter head H : a_b → θ_b.

Milestone: §12.2 (backbone). Feeds F1 and Stage 13 (donor matching). Stub in
Fase 0.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--config", default="conf/experiments/training.yaml", show_default=True)
@click.option("--run-id", required=True)
def main(config: str, run_id: str) -> None:
    raise SystemExit(
        "scripts/12_param_head.py — stub. Implementation lands with Milestone 2."
    )


if __name__ == "__main__":
    main()
