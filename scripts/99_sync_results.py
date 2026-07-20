#!/usr/bin/env python
"""Sync run artefacts from R2 into the local git-tracked ``results/`` tree.

Colab runs write manifest.json / history.json / folds_summary.csv to
``outputs/`` inside the Colab VM and mirror them to R2 (with
``--upload-to-r2``). The VM is ephemeral. Running this script locally
after a Colab run pulls those small artefacts from R2 and places them
under ``results/{stage}/{run_id}[/{fold}]/`` — the same layout the
drivers produce when run locally — so you can commit them alongside a
matching code SHA.

Usage:
    python scripts/99_sync_results.py --stage 12_train_multistation --run-id F0pub-alto-lerma-gpu-01
    python scripts/99_sync_results.py --stage 11_train_forecaster   --run-id F0-alto-lerma-gpu-03

Passing ``--commit`` will also stage and commit the pulled files with a
canonical message. It never pushes; do that yourself when you have
reviewed the diff.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click
from dotenv import load_dotenv

from hidroxmx.io import publish_results, r2_from_env, results_dir


# Filenames that stages 11 / 12 write to the run folder. Small text-only
# artefacts only. Checkpoints stay on R2.
KNOWN_ARTEFACTS = ("manifest.json", "history.json", "folds_summary.csv")


def _download_if_exists(r2, key: str, dst: Path) -> bool:
    """Try to download ``key`` to ``dst`` — silently skip if missing."""
    try:
        payload = r2.get_bytes(key)
    except Exception:  # noqa: BLE001
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(payload)
    return True


@click.command()
@click.option("--stage", required=True,
              help="Stage name — matches the folder under results/ (e.g. 11_train_forecaster).")
@click.option("--run-id", required=True, help="Run identifier.")
@click.option("--folds", default="", show_default=True,
              help="Comma-separated list of fold subpaths to sync (only for multi-fold stages). "
                   "Empty pulls the top-level artefacts only.")
@click.option("--commit", is_flag=True,
              help="Stage and commit the new files under results/ with a canonical message.")
def main(stage: str, run_id: str, folds: str, commit: bool):
    load_dotenv(override=False)
    r2 = r2_from_env()
    prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}"

    fold_list = [f.strip() for f in folds.split(",") if f.strip()]
    tmp_root = Path(".sync_tmp") / stage / run_id
    if tmp_root.exists():
        # Wipe stale downloads so a re-sync does not carry old artefacts forward.
        for p in tmp_root.rglob("*"):
            if p.is_file():
                p.unlink()

    pulled_any = False

    # Top-level artefacts (single-fold stages produce these directly).
    for name in KNOWN_ARTEFACTS:
        if _download_if_exists(r2, f"{prefix}/{name}", tmp_root / name):
            pulled_any = True
    if any((tmp_root / n).exists() for n in KNOWN_ARTEFACTS):
        written = publish_results(
            [tmp_root / n for n in KNOWN_ARTEFACTS], stage=stage, run_id=run_id,
        )
        for p in written:
            click.echo(f"  -> {p.as_posix()}")

    # Per-fold artefacts (stage 12 with --holdout X or *).
    for fold in fold_list:
        fold_prefix = f"{prefix}/{fold}"
        pulled_fold = False
        for name in KNOWN_ARTEFACTS:
            if _download_if_exists(r2, f"{fold_prefix}/{name}",
                                    tmp_root / fold / name):
                pulled_fold = True
                pulled_any = True
        if pulled_fold:
            written = publish_results(
                [tmp_root / fold / n for n in KNOWN_ARTEFACTS],
                stage=stage, run_id=run_id, subpath=fold,
            )
            for p in written:
                click.echo(f"  -> {p.as_posix()}")

    if not pulled_any:
        raise SystemExit(
            f"No artefacts found under r2://{r2.bucket}/{prefix}[/…]. "
            f"Did the run finish with --upload-to-r2?"
        )

    if commit:
        target = results_dir(stage, run_id)
        subprocess.run(["git", "add", str(target)], check=True)
        msg = f"results({stage}): add manifests for {run_id}"
        subprocess.run(["git", "commit", "-m", msg], check=True)
        click.echo(f"Committed: {msg}")
        click.echo("Push with:  git push")


if __name__ == "__main__":
    main()
