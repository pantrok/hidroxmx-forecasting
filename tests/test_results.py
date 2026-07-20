"""Tests for the results-publishing helper."""
from __future__ import annotations

import json

from hidroxmx.io import publish_results, results_dir


def test_publish_results_copies_only_existing_files(tmp_path):
    src_a = tmp_path / "a.json"
    src_a.write_text('{"metric": 1}')
    src_missing = tmp_path / "missing.json"
    written = publish_results(
        [src_a, src_missing],
        stage="stg", run_id="run1", root=tmp_path / "results",
    )
    assert len(written) == 1
    assert written[0].name == "a.json"
    assert (tmp_path / "results" / "stg" / "run1" / "a.json").exists()
    assert json.loads(written[0].read_text())["metric"] == 1


def test_publish_results_handles_subpath(tmp_path):
    src = tmp_path / "manifest.json"
    src.write_text('{}')
    written = publish_results(
        [src], stage="stg", run_id="run1", subpath="fold-A",
        root=tmp_path / "results",
    )
    assert (tmp_path / "results" / "stg" / "run1" / "fold-A" / "manifest.json").exists()
    assert written[0].parent.name == "fold-A"


def test_results_dir_composition():
    assert results_dir("stg", "r1").as_posix() == "results/stg/r1"
    assert results_dir("stg", "r1", subpath="k").as_posix() == "results/stg/r1/k"
