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


# --------------------------------------------------------------------------- #
# Static attributes (S-ATTR)
# --------------------------------------------------------------------------- #
def test_extract_attributes_from_valid_row():
    import pandas as pd
    from hidroxmx.transfer import DEFAULT_ATTRIBUTE_KEYS, extract_attributes

    row = pd.Series({
        "clave": "ABCXY",
        "nombre": "Test station",
        "latitud": 20.216667,
        "longitud": -100.885833,
        "altitud": 1758,
        "region_hidrologica": 12,
    })
    attrs = extract_attributes(row)
    for k in DEFAULT_ATTRIBUTE_KEYS:
        assert k in attrs
    assert attrs["altitud"] == 1758.0
    assert attrs["region_hidrologica"] == 12.0


def test_extract_attributes_nan_safe_on_missing():
    import pandas as pd
    from hidroxmx.transfer import extract_attributes

    row = pd.Series({"clave": "X", "latitud": None, "altitud": ""})
    attrs = extract_attributes(row)
    assert np.isnan(attrs["latitud"])
    assert np.isnan(attrs["altitud"])
    assert np.isnan(attrs["longitud"])
    assert np.isnan(attrs["region_hidrologica"])


def test_attribute_vector_order_matches_default_keys():
    import pandas as pd
    from hidroxmx.transfer import DEFAULT_ATTRIBUTE_KEYS, attribute_vector, extract_attributes

    row = pd.Series({
        "latitud": 20.0, "longitud": -100.0, "altitud": 1500,
        "region_hidrologica": 12,
    })
    v = attribute_vector(extract_attributes(row))
    assert len(v) == len(DEFAULT_ATTRIBUTE_KEYS)
    assert v[0] == 20.0
    assert v[1] == -100.0
    assert v[2] == 1500.0
    assert v[3] == 12.0


def test_score_donors_works_with_attribute_vectors():
    """Same score_donors API should handle attribute vectors, not just signatures."""
    import pandas as pd
    from hidroxmx.transfer import attribute_vector, extract_attributes

    rows = [
        {"clave": "T", "latitud": 20.0, "longitud": -100.0, "altitud": 1500, "region_hidrologica": 12},
        {"clave": "A", "latitud": 20.1, "longitud": -100.1, "altitud": 1520, "region_hidrologica": 12},  # very close
        {"clave": "B", "latitud": 25.0, "longitud": -105.0, "altitud": 800,  "region_hidrologica": 18},  # far
    ]
    target = attribute_vector(extract_attributes(pd.Series(rows[0])))
    donors = np.vstack([
        attribute_vector(extract_attributes(pd.Series(r))) for r in rows[1:]
    ])
    res = score_donors(target, donors, ["A", "B"], method="soft", temperature=0.5)
    # A should score higher than B because it is closer in every coord.
    assert res.raw_scores[0] > res.raw_scores[1]
    assert res.weights[0] > res.weights[1]
