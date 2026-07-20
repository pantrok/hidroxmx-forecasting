"""Persistent, git-tracked results directory for paper-writing artefacts.

Runs generate two flavours of artefact:

- **Heavy binaries** (model checkpoints, feature dumps): stored under
  ``outputs/`` locally and mirrored to Cloudflare R2. Never committed
  to git — they can be many MB per run and their content is not
  paper-legible.
- **Text-only artefacts** (run manifests with config and metrics,
  per-epoch loss histories, aggregate fold summaries): copied into
  ``results/{stage}/{run_id}/`` in the repository so every experiment
  the paper references has a durable, browsable, git-tracked record.
  These are small (KB per run).

This module exposes one helper — :func:`publish_results` — that copies
a list of files into the canonical location. Both stage 11 and stage 12
call it at the end of a run.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

RESULTS_ROOT = Path("results")


def results_dir(stage: str, run_id: str, *, subpath: str | None = None,
                root: Path | None = None) -> Path:
    """Return the results directory for ``run_id`` of ``stage``.

    ``subpath`` is appended after ``run_id`` (used by multi-fold stages
    to keep per-fold artefacts separate). ``root`` overrides the default
    ``results/`` prefix — mainly for testing.
    """
    base = (root or RESULTS_ROOT) / stage / run_id
    if subpath:
        base = base / subpath
    return base


def publish_results(files: Iterable[Path], *, stage: str, run_id: str,
                    subpath: str | None = None,
                    root: Path | None = None) -> list[Path]:
    """Copy each existing file in ``files`` into ``results/{stage}/{run_id}[/subpath]``.

    Missing files are silently skipped so a driver can call this with a
    fixed set of paths regardless of whether every artefact was produced
    on this run. Returns the list of paths actually written.
    """
    target_dir = results_dir(stage, run_id, subpath=subpath, root=root)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for src in files:
        src = Path(src)
        if not src.exists():
            continue
        dst = target_dir / src.name
        shutil.copy2(src, dst)
        written.append(dst)
    return written
