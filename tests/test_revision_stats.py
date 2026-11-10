"""Tests for the tools added for the JOH revision."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from hidroxmx.alert import (
    ShoulderTrapezoidalMF,
    TrapezoidalMF,
    build_alert_fis,
    score_to_category,
)
from hidroxmx.eval import BCaResult, paired_bca, paired_bootstrap
from hidroxmx.revision import CLEARED, NOT_DEFINED, classify, sign_test
from hidroxmx.uq import (
    adaptive_interval,
    adjacent_miss_pairs,
    log_score_interval,
    miss_run_lengths,
    order_statistic_rank,
    tail_events,
)


def _days(*offsets: int) -> np.ndarray:
    return np.datetime64("2024-01-01") + np.asarray(offsets).astype("timedelta64[D]")


# --- BCa ------------------------------------------------------------------------
def test_bca_percentile_limits_match_paired_bootstrap():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.4, 0.2, 5), rng.normal(0.1, 0.2, 5)
    ref = paired_bootstrap(a, b, n_boot=4000, seed=7)
    res = paired_bca(a, b, n_boot=4000, seed=7)
    assert res.estimate == pytest.approx(ref.delta)
    assert res.pct_low == ref.ci_low
    assert res.pct_high == ref.ci_high


def test_bca_agrees_with_scipy_on_a_smooth_sample():
    rng = np.random.default_rng(2)
    a, b = rng.gamma(2.0, 1.0, 60), rng.gamma(2.0, 0.8, 60)
    res = paired_bca(a, b, n_boot=10_000, seed=3)
    ref = stats.bootstrap(
        (a, b), lambda x, y, axis=-1: np.median(x, axis=axis) - np.median(y, axis=axis),
        paired=True, vectorized=True, n_resamples=10_000, method="BCa", random_state=3,
    )
    assert res.defined
    assert res.bca_low == pytest.approx(ref.confidence_interval.low, abs=0.08)
    assert res.bca_high == pytest.approx(ref.confidence_interval.high, abs=0.08)


def test_bca_jackknife_uses_the_paired_estimator():
    a = np.array([0.1, 0.5, 0.9, 1.3, 2.1])
    b = np.array([0.9, 0.0, 0.7, 0.1, 0.3])
    primary = paired_bca(a, b, n_boot=2000, seed=4)
    secondary = paired_bca(a, b, estimator="median_of_differences", n_boot=2000, seed=4)
    assert primary.estimate == pytest.approx(np.median(a) - np.median(b))
    assert secondary.estimate == pytest.approx(np.median(a - b))
    assert primary.estimate != pytest.approx(secondary.estimate)


@pytest.mark.parametrize("a,b,reason", [
    ([0.2, 0.3], [0.1, 0.1], "n < 3"),
    ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], "jackknife"),
])
def test_bca_undefined_cases_are_flagged(a, b, reason):
    res = paired_bca(np.array(a), np.array(b), n_boot=500, seed=5)
    assert not res.defined
    assert reason in res.reason
    assert np.isnan(res.bca_low)


def test_bca_drops_non_finite_pairs():
    a = np.array([0.1, np.nan, 0.5, 0.6, 0.9])
    b = np.array([0.0, 0.2, np.nan, 0.1, 0.3])
    assert paired_bca(a, b, n_boot=500, seed=6).n == 3


def test_bca_rejects_unknown_estimator():
    with pytest.raises(ValueError):
        paired_bca(np.ones(4), np.ones(4), estimator="mean")


# --- Membership functions ----------------------------------------------------------
def test_submitted_trapezoid_drops_to_zero_at_the_shoulder():
    assert float(TrapezoidalMF(1.2, 1.7, 3.0, 3.0)(3.0)) == 0.0


@pytest.mark.parametrize("x,expected", [(3.0, 1.0), (2.0, 1.0), (1.45, 0.5), (1.0, 0.0)])
def test_corrected_trapezoid_right_shoulder(x, expected):
    assert float(ShoulderTrapezoidalMF(1.2, 1.7, 3.0, 3.0)(x)) == pytest.approx(expected)


def test_corrected_trapezoid_left_shoulder():
    mf = ShoulderTrapezoidalMF(0.0, 0.0, 0.4, 0.8)
    assert float(mf(0.0)) == 1.0
    assert float(mf(0.6)) == pytest.approx(0.5)


def test_forecast_at_the_edge_is_green_only_with_the_submitted_function():
    inputs = {"flow_ratio": 3.0, "width_ratio": 0.1}
    assert str(score_to_category(build_alert_fis().infer(inputs))) == "GREEN"
    assert str(score_to_category(build_alert_fis(corrected=True).infer(inputs))) == "RED"


def test_default_fis_matches_the_submitted_one_inside_the_universe():
    a, b = build_alert_fis(), build_alert_fis(corrected=False)
    for flow in (0.2, 0.9, 1.4, 2.5):
        assert a.infer({"flow_ratio": flow, "width_ratio": 0.4}) == \
            b.infer({"flow_ratio": flow, "width_ratio": 0.4})


# --- Conformal diagnostics ----------------------------------------------------------
def test_rank_is_clamped_and_flagged():
    assert order_statistic_rank(99, 0.1) == (90, False)
    assert order_statistic_rank(10, 0.0) == (10, True)
    assert order_statistic_rank(10, 1.2) == (1, True)


def test_log_interval_is_wider_for_larger_forecasts_and_non_negative():
    y = np.array([[1.0], [10.0], [100.0], [1000.0]] * 5)
    yhat = y * 1.2
    lower, upper, q = log_score_interval(y, yhat, np.array([[5.0], [500.0]]), alpha=0.1)
    assert q.shape == (1,)
    assert (upper - lower)[1, 0] > (upper - lower)[0, 0]
    assert (lower >= 0).all()


def test_aci_widens_after_misses_and_only_with_past_outcomes():
    scores = np.linspace(0.1, 10.0, 200)
    m = 300
    origins = _days(*range(m))
    targets = origins + np.timedelta64(3, "D")
    y = np.full(m, 100.0)
    yhat = np.zeros(m)                       # every forecast misses
    res = adaptive_interval(scores, y, yhat, origins, targets, alpha=0.1, gamma=0.02)
    # Horizon 3: the first outcome becomes known on its target date, the fourth origin.
    assert (res.alpha_path[:3] == 0.1).all()
    assert res.alpha_path[3] < 0.1
    assert res.upper[-1] > res.upper[0]
    assert (np.diff(res.alpha_path) <= 0).all()


def test_aci_stays_at_the_nominal_level_when_every_forecast_is_covered():
    scores = np.linspace(1.0, 2.0, 100)
    origins = _days(*range(50))
    res = adaptive_interval(scores, np.zeros(50), np.zeros(50), origins,
                            origins + np.timedelta64(1, "D"), alpha=0.1, gamma=0.005)
    assert (np.diff(res.alpha_path) > 0).all()     # narrower after covered days
    assert res.clamped_share == 0.0


def test_adjacent_pairs_skip_calendar_gaps():
    first, second = adjacent_miss_pairs(_days(0, 1, 2, 10, 11), np.array([1, 0, 1, 1, 1], bool))
    assert first.tolist() == [True, False, True]
    assert second.tolist() == [False, True, True]


def test_miss_runs_break_on_covered_days_and_gaps():
    dates = _days(0, 1, 2, 3, 9, 10, 20)
    misses = np.array([1, 1, 0, 1, 1, 1, 1], bool)
    assert miss_run_lengths(dates, misses).tolist() == [2, 1, 2, 1]


def test_events_merge_gaps_under_the_separation():
    events = tail_events(_days(0, 1, 3, 6, 10, 14), min_separation_days=3)
    assert [e.tolist() for e in events] == [[0, 1, 2, 3], [4], [5]]


def test_events_of_an_empty_tail():
    assert tail_events(_days()) == []


# --- Classes of RJ-1 ----------------------------------------------------------------
def _res(low: float, high: float, estimate: float = 0.1, defined: bool = True) -> BCaResult:
    nan = float("nan")
    return BCaResult(5, estimate, low, high, low if defined else nan, high if defined else nan,
                     0.0, 0.0, defined, "" if defined else "all jackknife values are equal")


def test_classify_uses_the_lower_limit_and_flags_undefined():
    assert classify(_res(0.06, 0.3)) == CLEARED
    assert classify(_res(0.0, 0.0, defined=False)) == NOT_DEFINED


def test_sign_test_drops_ties():
    assert sign_test(np.array([0.1, 0.2, 0.0, np.nan]), 0.0)[:2] == (2, 2)
