"""Tests for the scoped predictive digital twin (assimilation + scenarios)."""
from __future__ import annotations

import numpy as np
import pytest

from hidroxmx.twin import (
    SCENARIO_LIBRARY,
    InnovationPersistence,
    apply_scenario,
    assimilate_forecasts,
    dry_out,
    perturb_precip,
    perturb_temp,
    residuals_from_history,
)


# --------------------------------------------------------------------------- #
# Innovation persistence
# --------------------------------------------------------------------------- #
def test_innovation_persistence_rejects_bad_hyperparams():
    with pytest.raises(ValueError):
        InnovationPersistence(decay=0.0)
    with pytest.raises(ValueError):
        InnovationPersistence(decay=1.5)
    with pytest.raises(ValueError):
        InnovationPersistence(history_days=0)


def test_innovation_scalar_from_single_residual():
    ip = InnovationPersistence(decay=0.5, history_days=1)
    corr = ip.correction_per_horizon(np.array([3.0]), [1, 3, 7])
    # d^1=0.5, d^3=0.125, d^7=0.0078125
    assert corr == pytest.approx([1.5, 0.375, 3.0 * 0.5 ** 7])


def test_innovation_ewma_weights_most_recent_more_heavily():
    ip = InnovationPersistence(decay=0.6, history_days=5)
    residuals = np.array([1.0, 1.0, 1.0, 1.0, 5.0])   # spike at the end
    corr = ip.correction_per_horizon(residuals, [1])
    # EWMA gives most weight to the last element → correction dominated by 5.
    assert corr[0] > 1.5   # if it were equal weighting the answer would be ~1.8


def test_innovation_handles_nan_history():
    ip = InnovationPersistence(decay=0.7, history_days=5)
    residuals = np.array([np.nan, np.nan, 2.0])
    corr = ip.correction_per_horizon(residuals, [1, 3])
    assert np.all(np.isfinite(corr))


def test_assimilate_forecasts_matches_scalar_case():
    raw = np.array([[10.0, 12.0, 14.0]])   # one window, three horizons
    residuals = np.array([[2.0]])          # innovation = 2
    horizons = [1, 3, 7]
    out = assimilate_forecasts(raw, residuals, horizons, decay=0.5, history_days=1)
    expected = raw[0] + 2.0 * np.array([0.5, 0.125, 0.5 ** 7])
    assert np.allclose(out[0], expected)


def test_residuals_from_history_shape_check():
    with pytest.raises(ValueError):
        residuals_from_history(np.zeros(5), np.zeros(6))


def test_assimilate_forecasts_rejects_bad_shapes():
    with pytest.raises(ValueError):
        assimilate_forecasts(np.zeros(5), np.zeros((5, 3)), [1, 3, 7])
    with pytest.raises(ValueError):
        assimilate_forecasts(np.zeros((5, 3)), np.zeros((4, 3)), [1, 3, 7])
    with pytest.raises(ValueError):
        assimilate_forecasts(np.zeros((5, 3)), np.zeros((5, 3)), [1, 3])


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
def _make_window():
    """Two features: precip and tmax, standardised by hand for the test."""
    feature_cols = ["precip_mm", "tmax_c"]
    stats = {"precip_mm": (10.0, 5.0), "tmax_c": (25.0, 4.0)}
    # 3-step lookback in physical space then standardise.
    raw = np.array([[10.0, 25.0],
                    [12.0, 26.0],
                    [8.0, 24.0]], dtype=float)
    std = np.zeros_like(raw)
    std[:, 0] = (raw[:, 0] - 10.0) / 5.0
    std[:, 1] = (raw[:, 1] - 25.0) / 4.0
    return std, feature_cols, stats


def test_perturb_precip_multiplies_in_physical_space():
    x, cols, stats = _make_window()
    y = perturb_precip(x, cols, delta_pct=+20.0, stats=stats)
    # Raw precip [10, 12, 8] → [12, 14.4, 9.6]; check first entry only.
    raw_y = y[0, 0] * 5.0 + 10.0
    assert raw_y == pytest.approx(12.0)


def test_perturb_precip_never_negative():
    x, cols, stats = _make_window()
    y = perturb_precip(x, cols, delta_pct=-500.0, stats=stats)
    raw_y = y[..., 0] * 5.0 + 10.0
    assert np.all(raw_y >= 0.0)


def test_perturb_temp_shifts_additively():
    x, cols, stats = _make_window()
    y = perturb_temp(x, cols, delta_c=+2.0, stats=stats)
    raw_y = y[0, 1] * 4.0 + 25.0
    assert raw_y == pytest.approx(27.0)


def test_dry_out_zeroes_last_days():
    x, cols, stats = _make_window()   # 3-step lookback
    y = dry_out(x, cols, days=2, stats=stats)
    raw_precip = y[:, 0] * 5.0 + 10.0
    # First step untouched, last two should be zero.
    assert raw_precip[0] == pytest.approx(10.0)
    assert raw_precip[1] == pytest.approx(0.0)
    assert raw_precip[2] == pytest.approx(0.0)


def test_apply_scenario_from_library_by_name():
    x, cols, stats = _make_window()
    for name in SCENARIO_LIBRARY:
        y = apply_scenario(name, x, cols, stats)
        assert y.shape == x.shape


def test_apply_scenario_rejects_unknown_name():
    x, cols, stats = _make_window()
    with pytest.raises(ValueError):
        apply_scenario("bogus", x, cols, stats)


def test_perturbation_does_not_mutate_input():
    x, cols, stats = _make_window()
    before = x.copy()
    _ = perturb_precip(x, cols, +20.0, stats)
    _ = perturb_temp(x, cols, +2.0, stats)
    _ = dry_out(x, cols, 2, stats)
    assert np.array_equal(x, before)
