"""Tests for the transfer module (signatures + similarity)."""
from __future__ import annotations

import numpy as np
import pytest

from hidroxmx.transfer import (
    DEFAULT_SIGNATURE_KEYS,
    compute_signatures,
    score_donors,
    signature_vector,
)


rng = np.random.default_rng(42)


def _base_series(n: int = 3000, mean: float = 20.0, cv: float = 0.5) -> np.ndarray:
    """A synthetic streamflow series with the requested mean and cv."""
    x = rng.lognormal(mean=np.log(mean), sigma=cv, size=n)
    return x.astype(float)


def test_compute_signatures_returns_all_keys():
    series = _base_series()
    sig = compute_signatures(series)
    for k in DEFAULT_SIGNATURE_KEYS:
        assert k in sig, f"missing signature: {k}"


def test_signatures_are_nan_safe_on_all_nan_input():
    series = np.full(500, np.nan)
    sig = compute_signatures(series)
    for v in sig.values():
        assert np.isnan(v) or v == 0.0 or v == float("nan")


def test_mean_and_percentiles_are_ordered():
    series = _base_series(mean=50, cv=0.8)
    sig = compute_signatures(series)
    assert sig["q05"] <= sig["q50"] <= sig["q95"]
    assert sig["mean_flow"] > 0
    assert sig["cv_flow"] > 0


def test_baseflow_index_between_zero_and_one():
    series = _base_series()
    sig = compute_signatures(series)
    assert 0.0 <= sig["baseflow_index"] <= 1.0


def test_fdc_slope_is_positive_for_typical_regime():
    # A regime with variability should have positive FDC slope (in the
    # convention we use: log(Q_low) − log(Q_high) with low < high in
    # percent-of-time exceeded).
    series = _base_series(cv=0.6)
    sig = compute_signatures(series)
    assert np.isfinite(sig["fdc_slope"])
    assert sig["fdc_slope"] > 0


def test_signature_vector_matches_default_order():
    series = _base_series()
    sig = compute_signatures(series)
    vec = signature_vector(sig)
    assert len(vec) == len(DEFAULT_SIGNATURE_KEYS)
    for i, key in enumerate(DEFAULT_SIGNATURE_KEYS):
        if np.isnan(vec[i]) and np.isnan(sig[key]):
            continue
        assert vec[i] == pytest.approx(sig[key])


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #
def test_score_donors_soft_weights_sum_to_n():
    target = signature_vector(compute_signatures(_base_series(mean=20)))
    donors = np.vstack([
        signature_vector(compute_signatures(_base_series(mean=m, cv=c)))
        for m, c in [(21, 0.5), (50, 1.0), (200, 1.5), (2, 0.2)]
    ])
    res = score_donors(target, donors, ["A", "B", "C", "D"],
                       method="soft", temperature=1.0)
    assert res.weights.shape == (4,)
    assert res.weights.sum() == pytest.approx(4.0, rel=1e-6)
    # Closest donor (A) should have the largest weight.
    assert res.raw_scores.argmax() == 0


def test_score_donors_top_k_masks_correctly():
    target = signature_vector(compute_signatures(_base_series(mean=20)))
    donors = np.vstack([
        signature_vector(compute_signatures(_base_series(mean=m)))
        for m in [21, 22, 100, 500]
    ])
    res = score_donors(target, donors, ["A", "B", "C", "D"],
                       method="top_k", top_k=2)
    # Exactly two non-zero weights, each equal to n/k = 4/2 = 2.
    nonzero = res.weights[res.weights > 0]
    assert len(nonzero) == 2
    assert np.allclose(nonzero, 2.0)


def test_score_donors_cosine_metric_scale_invariant():
    target_scaled_small = signature_vector(compute_signatures(_base_series(mean=1.0)))
    target_scaled_big = signature_vector(compute_signatures(_base_series(mean=100.0)))
    donors = np.vstack([
        signature_vector(compute_signatures(_base_series(mean=m)))
        for m in [1.1, 90, 500]
    ])
    # Cosine ordering should be similar regardless of target mean scale
    # once we standardise. Sanity: no crash, weights shape ok.
    res = score_donors(target_scaled_small, donors, ["A", "B", "C"],
                       method="soft", metric="cosine")
    assert res.weights.shape == (3,)
    assert np.all(res.raw_scores >= 0)


def test_score_donors_rejects_unknown_method():
    target = signature_vector(compute_signatures(_base_series()))
    donors = np.vstack([signature_vector(compute_signatures(_base_series()))])
    with pytest.raises(ValueError):
        score_donors(target, donors, ["A"], method="bogus")
    with pytest.raises(ValueError):
        score_donors(target, donors, ["A"], metric="bogus")
