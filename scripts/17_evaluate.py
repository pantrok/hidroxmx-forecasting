#!/usr/bin/env python
"""Stage 17 — Evaluation (metrics, paired bootstrap, figures).

Milestone: §12.7. Consumes ``hidroxmx.eval.METRICS`` and the alert thresholds
declared in ``conf/experiments/metrics.yaml``. Stub in Fase 0.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--config", default="conf/experiments/metrics.yaml", show_default=True)
@click.option("--run-id", required=True)
def main(config: str, run_id: str) -> None:
    raise SystemExit(
        "scripts/17_evaluate.py — stub. Implementation lands with Milestone 7."
    )


if __name__ == "__main__":
    main()
