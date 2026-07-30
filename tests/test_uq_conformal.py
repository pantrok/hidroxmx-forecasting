"""Tests for split conformal prediction."""
from __future__ import annotations

import numpy as np
import pytest

from hidroxmx.uq import SplitConformal, coverage_and_width, tail_coverage


rng = np.random.default_rng(20260721)


def _make_calibration_and_test(n_cal=500, n_test=500, n_h=3, sigma=1.0, bias=0.0):
    """Synthetic regression: yhat = y + Gaussian noise (heteroscedastic if sigma is scalar)."""
    y_cal = rng.normal(0, 2, size=(n_cal, n_h))
    yhat_cal = y_cal + rng.normal(bias, sigma, size=y_cal.shape)
    y_test = rng.normal(0, 2, size=(n_test, n_h))
    yhat_test = y_test + rng.normal(bias, sigma, size=y_test.shape)
    return y_cal, yhat_cal, y_test, yhat_test


def test_split_conformal_marginal_coverage_within_bounds():
    """Coverage on a large iid test set should hit (1 − α) ± noise."""
    y_cal, yhat_cal, y_test, yhat_test = _make_calibration_and_test(
        n_cal=1000, n_test=2000, sigma=1.0)
    sc = SplitConformal(alpha=0.1).fit(y_cal, yhat_cal)
    lower, upper = sc.predict_interval(yhat_test)
    result = coverage_and_width(y_test, lower, upper)
    # With n_cal = 1000, empirical coverage should be very close to 0.90.
    for h in range(y_test.shape[1]):
        assert 0.85 <= result["coverage"][h] <= 0.95, (
            f"h={h}: coverage {result['coverage'][h]:.3f} out of 90 % band"
        )


def test_split_conformal_quantile_grows_with_noise():
    """Larger residuals → larger interval width."""
    y_cal, yhat_cal, _, _ = _make_calibration_and_test(sigma=0.5)
    q_small = SplitConformal(alpha=0.1).fit(y_cal, yhat_cal).quantiles_
    y_cal, yhat_cal, _, _ = _make_calibration_and_test(sigma=2.0)
    q_big = SplitConformal(alpha=0.1).fit(y_cal, yhat_cal).quantiles_
    assert np.all(q_big > q_small)


def test_split_conformal_alpha_bounds_rejected():
    y = rng.normal(size=(50, 3))
    with pytest.raises(ValueError):
        SplitConformal(alpha=0.0).fit(y, y)
    with pytest.raises(ValueError):
        SplitConformal(alpha=1.0).fit(y, y)


def test_split_conformal_shape_mismatch_rejected():
    y = rng.normal(size=(50, 3))
    yhat = rng.normal(size=(50, 4))
    with pytest.raises(ValueError):
        SplitConformal().fit(y, yhat)


def test_split_conformal_handles_nan_calibration():
    """A NaN row in one horizon should not blow up the quantile of the others."""
    y = rng.normal(size=(100, 3))
    yhat = y + rng.normal(size=y.shape)
    yhat[10, 0] = np.nan
    sc = SplitConformal(alpha=0.1).fit(y, yhat)
    assert sc.quantiles_.shape == (3,)
    assert np.all(np.isfinite(sc.quantiles_))


def test_predict_interval_before_fit_raises():
    with pytest.raises(RuntimeError):
        SplitConformal().predict_interval(np.zeros((5, 3)))


def test_coverage_and_width_matches_manual_check():
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    lower = np.array([[0.5, 1.5], [2.5, 3.5], [10.0, 10.0]])
    upper = np.array([[1.5, 2.5], [3.5, 4.5], [12.0, 12.0]])
    r = coverage_and_width(y, lower, upper)
    # First two rows are inside, the third is outside for both horizons.
    assert r["coverage"][0] == pytest.approx(2 / 3)
    assert r["coverage"][1] == pytest.approx(2 / 3)
    # Widths: horizon 0: (1 + 1 + 2)/3, horizon 1: (1 + 1 + 2)/3
    assert r["mean_interval_width"][0] == pytest.approx(4 / 3)
    assert r["mean_interval_width"][1] == pytest.approx(4 / 3)


def test_tail_coverage_uses_upper_tail_only():
    n = 500
    y = rng.normal(0, 1, size=(n, 2))
    lower = y - 1.0
    upper = y + 1.0
    # Poison the tail so tail coverage < 1.
    tail_mask = y > np.quantile(y, 0.95, axis=0)
    # For a few extreme rows, push the true value outside the interval.
    corrupt_rows = np.where(tail_mask[:, 0])[0][:3]
    y_corrupt = y.copy()
    y_corrupt[corrupt_rows, 0] += 5.0
    tc = tail_coverage(y_corrupt, lower, upper, threshold_quantile=0.95)
    assert 0.0 <= tc[0] <= 1.0
    assert tc[1] == pytest.approx(1.0)  # untouched horizon
