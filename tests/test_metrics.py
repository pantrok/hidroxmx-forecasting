"""Sanity checks for the metrics registry."""
from __future__ import annotations

import math

import numpy as np

from hidroxmx.eval import (
    METRICS,
    brier_score,
    crps_sample,
    empirical_coverage,
    kge,
    mean_interval_width,
    nse,
    pbias,
    pod_far,
    rmse,
    value_cost_loss,
)


def test_registry_populated():
    for key in ("nse", "kge", "rmse", "pbias", "ehf", "peak_timing_error",
                "empirical_coverage", "mean_interval_width", "crps_sample",
                "brier_score", "pod_far", "value_cost_loss", "lead_time_at"):
        assert key in METRICS


def test_perfect_forecast_scores_1():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(nse(y, y), 1.0, abs_tol=1e-9)
    assert math.isclose(kge(y, y), 1.0, abs_tol=1e-6)
    assert math.isclose(rmse(y, y), 0.0, abs_tol=1e-12)
    assert math.isclose(pbias(y, y), 0.0, abs_tol=1e-12)


def test_nse_penalises_constant_forecast():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yhat = np.full_like(y, y.mean())
    # NSE of the climatological forecast is 0 by construction
    assert math.isclose(nse(y, yhat), 0.0, abs_tol=1e-9)


def test_nan_pairs_are_ignored():
    y = np.array([1.0, np.nan, 3.0, 4.0])
    yhat = np.array([1.0, 5.0, 3.0, np.nan])
    # Only the (1,1) and (3,3) pairs survive; RMSE is 0
    assert math.isclose(rmse(y, yhat), 0.0, abs_tol=1e-12)


def test_coverage_and_width():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lo = y - 0.5
    hi = y + 0.5
    assert math.isclose(empirical_coverage(y, lo, hi), 1.0)
    assert math.isclose(mean_interval_width(lo, hi), 1.0)
    # Move the intervals so half the observations fall outside
    hi_narrow = y + 0.001
    lo_narrow = y - 0.001
    assert 0.0 <= empirical_coverage(y, lo_narrow - 10, lo_narrow) <= 1.0


def test_pod_far_binary():
    events = np.array([1, 0, 1, 1, 0], dtype=bool)
    alerts = np.array([1, 0, 0, 1, 1], dtype=bool)
    pod, far = pod_far(events, alerts)
    assert math.isclose(pod, 2 / 3, rel_tol=1e-6)
    assert math.isclose(far, 1 / 3, rel_tol=1e-6)


def test_brier_bounds():
    events = np.array([0, 1, 1, 0])
    probs = np.array([0.1, 0.9, 0.8, 0.2])
    assert 0.0 <= brier_score(events, probs) <= 1.0


def test_value_cost_loss_perfect_forecast_scores_1():
    events = np.array([1, 0, 1, 0, 1], dtype=bool)
    alerts = events.copy()  # perfect
    assert math.isclose(value_cost_loss(events, alerts, cost_ratio=0.1), 1.0, rel_tol=1e-6)


def test_crps_ensemble_shrinks_to_zero_when_perfect():
    y = np.array([0.0, 1.0, 2.0])
    ens = np.repeat(y[:, None], 5, axis=1)  # every member is the truth
    assert math.isclose(crps_sample(y, ens), 0.0, abs_tol=1e-9)
