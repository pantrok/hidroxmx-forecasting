"""Tests for the paired bootstrap engine."""
from __future__ import annotations

import numpy as np
import pytest

from hidroxmx.eval import (
    PairedBootstrapResult,
    paired_bootstrap,
    paired_bootstrap_kill_check,
)


rng = np.random.default_rng(20260721)


def test_paired_bootstrap_returns_populated_result():
    a = rng.normal(1.0, 0.5, size=50)
    b = rng.normal(0.0, 0.5, size=50)
    r = paired_bootstrap(a, b, n_boot=1000, seed=1)
    assert isinstance(r, PairedBootstrapResult)
    assert r.n == 50
    assert np.isfinite(r.delta)
    assert r.ci_low < r.ci_high
    assert r.ci_level == 0.95
    assert r.statistic == "median"


def test_paired_bootstrap_significant_when_effect_is_large():
    a = np.linspace(0.5, 1.5, 40)   # median 1.0
    b = np.linspace(0.0, 1.0, 40)   # median 0.5
    r = paired_bootstrap(a, b, n_boot=2000, seed=2)
    # Effect ~ +0.5; CI should exclude 0.
    assert r.ci_low > 0.0
    assert r.significant is True


def test_paired_bootstrap_not_significant_when_no_effect():
    a = rng.normal(0.0, 1.0, size=200)
    b = rng.normal(0.0, 1.0, size=200)
    r = paired_bootstrap(a, b, n_boot=2000, seed=3)
    # CI should straddle zero for iid same-distribution samples.
    assert r.ci_low <= 0.0 <= r.ci_high
    assert r.significant is False


def test_paired_bootstrap_uses_mean_statistic_when_requested():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])   # outlier tail
    b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    r_mean = paired_bootstrap(a, b, statistic="mean", n_boot=1000, seed=4)
    r_med = paired_bootstrap(a, b, statistic="median", n_boot=1000, seed=4)
    # Mean is dragged up by the outlier; median is not.
    assert r_mean.delta > r_med.delta


def test_paired_bootstrap_pairwise_nan_filter():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    b = np.array([0.5, np.nan, 2.5, 3.5, 4.5])
    r = paired_bootstrap(a, b, n_boot=500, seed=5)
    # Two rows have NaN in one column each → three effective pairs.
    assert r.n == 3


def test_paired_bootstrap_empty_after_filter():
    a = np.full(5, np.nan)
    b = np.arange(5, dtype=float)
    r = paired_bootstrap(a, b, n_boot=500, seed=6)
    assert r.n == 0
    assert np.isnan(r.delta)
    assert r.significant is False


def test_paired_bootstrap_rejects_bad_shapes_and_ci():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(5), np.zeros(6))
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(5), np.zeros(5), ci=0.0)
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(5), np.zeros(5), ci=1.5)
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(5), np.zeros(5), statistic="mode")


def test_paired_bootstrap_kill_check_clears_when_threshold_below_ci_low():
    # Big positive effect: median delta well above +0.05.
    a = np.full(30, 1.0)
    b = np.full(30, 0.5)
    result, kill = paired_bootstrap_kill_check(
        a, b, threshold=0.05, n_boot=500, seed=7,
    )
    assert result.delta == pytest.approx(0.5)
    assert kill is True


def test_paired_bootstrap_kill_check_fails_when_ci_low_below_threshold():
    a = rng.normal(0.0, 1.0, size=50)
    b = rng.normal(0.0, 1.0, size=50)
    result, kill = paired_bootstrap_kill_check(
        a, b, threshold=0.05, n_boot=500, seed=8,
    )
    # Same distribution → CI-low well below +0.05.
    assert kill is False
    assert result.ci_low < 0.05


def test_paired_bootstrap_seed_reproducibility():
    a = rng.normal(1, 0.5, 30)
    b = rng.normal(0, 0.5, 30)
    r1 = paired_bootstrap(a, b, n_boot=500, seed=1234)
    r2 = paired_bootstrap(a, b, n_boot=500, seed=1234)
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high


def test_paired_bootstrap_result_as_row_has_expected_keys():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 1.5, 2.5])
    r = paired_bootstrap(a, b, n_boot=500, seed=9)
    row = r.as_row()
    for k in ("n", "delta", "ci_low", "ci_high", "ci_level",
              "statistic", "significant"):
        assert k in row
