"""Atomic checkpoint save/restore, local + R2 mirror.

A checkpoint bundle is a compressed ``.pt`` written atomically (temp file →
rename). It contains the model state dict, optimiser, scheduler, epoch/step,
best metric, RNG states (Python / numpy / torch / cuda), the AMP scaler and a
config hash. Save is idempotent; restore is best-effort (falls back to a fresh
run when no ``last.ckpt`` exists yet).
"""
from __future__ import annotations

import io
import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .r2 import R2Client


def atomic_write(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically (temp file in the same directory → rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp",
        delete=False,
    ) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
        tmp = Path(fh.name)
    tmp.replace(path)
    return path


@dataclass(slots=True)
class CheckpointStore:
    """Local + R2 mirrored checkpoint store scoped to a run id."""

    run_id: str
    local_dir: Path
    r2: R2Client | None = None
    r2_prefix: str = "paper2/runs"
    keep_last_k: int = 3

    def _local(self, name: str) -> Path:
        return self.local_dir / self.run_id / name

    def _remote(self, name: str) -> str:
        return f"{self.r2_prefix}/{self.run_id}/{name}"

    # -------- save -----------------------------------------------------------
    def save(self, state: dict[str, Any], *, name: str = "last.ckpt") -> Path:
        import torch  # imported lazily so `pip install` without CUDA still succeeds

        buf = io.BytesIO()
        torch.save(state, buf)
        local = atomic_write(self._local(name), buf.getvalue())
        if self.r2 is not None:
            self.r2.upload_file(self._remote(name), local)
        return local

    # -------- restore --------------------------------------------------------
    def restore(self, *, name: str = "last.ckpt") -> dict[str, Any] | None:
        import torch

        local = self._local(name)
        if not local.exists() and self.r2 is not None and self.r2.exists(self._remote(name)):
            self.r2.download_file(self._remote(name), local)
        if not local.exists():
            return None
        return torch.load(str(local), map_location="cpu", weights_only=False)

    # -------- rolling history + done marker ---------------------------------
    def mark_done(self, stage: str) -> None:
        marker = f".done.{stage}"
        atomic_write(self._local(marker), b"1")
        if self.r2 is not None:
            self.r2.put_bytes(self._remote(marker), b"1")

    def is_done(self, stage: str) -> bool:
        marker = f".done.{stage}"
        if self._local(marker).exists():
            return True
        if self.r2 is not None and self.r2.exists(self._remote(marker)):
            return True
        return False


def collect_rng_state() -> dict[str, Any]:
    """Capture Python / numpy / torch (+CUDA) RNG state for reproducibility."""
    import numpy as np

    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch

        state["torch"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.random.get_rng_state_all()
    except Exception:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    import numpy as np

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch" in state:
            torch.random.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.random.set_rng_state_all(state["cuda"])
    except Exception:
        pass
