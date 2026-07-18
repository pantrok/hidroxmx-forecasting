#!/usr/bin/env python
"""Stage 11 — Train the forecaster F0 (baseline) or F1 (physics-guided).

Milestone: §12.2 (backbone) — checkpoint-resumable via
``hidroxmx.io.CheckpointStore``. Reads training config from
``conf/experiments/training.yaml``. This script is a stub in Fase 0.

Contract
--------
- Loads ``feature_table.parquet`` lazily by basin/year partitions from R2.
- Builds an iterable windowing dataset (look-back L; horizons 1..7 days).
- Trains F0 (encoder–decoder LSTM) or F1 (F0 + soft physics constraints)
  under mixed precision with gradient accumulation.
- Persists ``last.ckpt`` and ``best.ckpt`` to R2 every N steps and every epoch.
- Writes a run manifest (config hash + git SHA + metrics) to R2.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--config", default="conf/experiments/training.yaml", show_default=True)
@click.option("--run-id", required=True, help="Stable identifier used for checkpoints.")
@click.option("--physics", is_flag=True, help="Train F1 with soft physics constraints.")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Restore last.ckpt from R2 if present.")
def main(config: str, run_id: str, physics: bool, resume: bool) -> None:
    raise SystemExit(
        "scripts/11_train_forecaster.py — stub. Implementation lands with "
        "Milestone 2 of the experiment spec (§12.2)."
    )


if __name__ == "__main__":
    main()
