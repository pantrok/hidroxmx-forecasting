"""Run manifest: config hash + git SHA + metrics, written next to the checkpoint."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoint import atomic_write


def _git_sha(cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        return None


def config_hash(cfg: dict[str, Any]) -> str:
    """Deterministic short hash of a JSON-serialisable config."""
    data = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass(slots=True)
class RunManifest:
    """A single row of provenance written next to the run outputs."""

    run_id: str
    stage: str
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    started_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at_utc: str | None = None
    git_sha: str | None = None
    config_sha16: str | None = None

    def finalise(self, metrics: dict[str, float]) -> "RunManifest":
        self.metrics = dict(metrics)
        self.ended_at_utc = datetime.now(timezone.utc).isoformat()
        if self.git_sha is None:
            self.git_sha = _git_sha()
        if self.config_sha16 is None:
            self.config_sha16 = config_hash(self.config)
        return self


def dump_manifest(manifest: RunManifest, path: Path) -> Path:
    payload = json.dumps(asdict(manifest), indent=2, ensure_ascii=False).encode("utf-8")
    return atomic_write(path, payload)
