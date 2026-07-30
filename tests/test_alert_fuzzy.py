"""Tests for the Mamdani fuzzy alert FIS."""
from __future__ import annotations

import numpy as np
import pytest

from hidroxmx.alert import (
    CATEGORIES,
    FuzzyVariable,
    MamdaniFIS,
    MamdaniRule,
    TrapezoidalMF,
    TriangularMF,
    build_alert_fis,
    category_to_index,
    score_to_category,
)


# --------------------------------------------------------------------------- #
# Membership functions
# --------------------------------------------------------------------------- #
def test_triangular_mf_peaks_and_shoulders():
    mf = TriangularMF(0.0, 1.0, 2.0)
    assert mf(1.0) == pytest.approx(1.0)
    assert mf(0.5) == pytest.approx(0.5)
    assert mf(1.5) == pytest.approx(0.5)
    assert mf(-0.5) == pytest.approx(0.0)
    assert mf(2.5) == pytest.approx(0.0)


def test_trapezoidal_mf_is_flat_across_top():
    mf = TrapezoidalMF(0.0, 1.0, 2.0, 3.0)
    assert mf(1.0) == pytest.approx(1.0)
    assert mf(1.5) == pytest.approx(1.0)
    assert mf(2.0) == pytest.approx(1.0)
    assert mf(0.5) == pytest.approx(0.5)
    assert mf(2.5) == pytest.approx(0.5)
    assert mf(-1) == pytest.approx(0.0)


def test_membership_functions_vectorised():
    mf = TriangularMF(0.0, 1.0, 2.0)
    x = np.linspace(-1, 3, 41)
    y = mf(x)
    assert y.shape == x.shape
    assert np.all((y >= 0.0) & (y <= 1.0))


# --------------------------------------------------------------------------- #
# Mamdani engine — tiny hand-checkable FIS
# --------------------------------------------------------------------------- #
def _toy_fis():
    """3-class toy FIS whose MFs cover the whole universe so every x
    fires at least one rule; used for monotonicity checks."""
    x = FuzzyVariable("x", 0, 10, {
        "LOW":  TrapezoidalMF(0, 0, 3, 5),
        "MID":  TriangularMF(3, 5, 7),
        "HIGH": TrapezoidalMF(5, 7, 10, 10),
    })
    y = FuzzyVariable("y", 0, 10, {
        "SMALL":  TrapezoidalMF(0, 0, 2, 4),
        "MEDIUM": TriangularMF(3, 5, 7),
        "BIG":    TrapezoidalMF(6, 8, 10, 10),
    })
    rules = [
        MamdaniRule({"x": "LOW"}, ("y", "SMALL")),
        MamdaniRule({"x": "MID"}, ("y", "MEDIUM")),
        MamdaniRule({"x": "HIGH"}, ("y", "BIG")),
    ]
    return MamdaniFIS({"x": x}, y, rules)


def test_mamdani_fis_monotonic_on_toy_example():
    fis = _toy_fis()
    y_low = fis.infer({"x": 1.0})
    y_mid = fis.infer({"x": 5.0})
    y_high = fis.infer({"x": 9.0})
    assert y_low < y_mid < y_high


def test_mamdani_fis_returns_universe_min_when_no_rule_fires():
    """A FIS with a hole in its coverage falls back to universe_min."""
    x = FuzzyVariable("x", 0, 10, {
        "LOW":  TrapezoidalMF(0, 0, 1, 2),
        "HIGH": TrapezoidalMF(8, 9, 10, 10),
    })
    y = FuzzyVariable("y", 0, 10, {"SMALL": TrapezoidalMF(0, 0, 2, 4),
                                   "BIG":   TrapezoidalMF(6, 8, 10, 10)})
    rules = [MamdaniRule({"x": "LOW"}, ("y", "SMALL")),
             MamdaniRule({"x": "HIGH"}, ("y", "BIG"))]
    fis = MamdaniFIS({"x": x}, y, rules)
    val = fis.infer({"x": 5.0})  # no rule fires
    assert val == pytest.approx(0.0)


def test_infer_batch_matches_infer_pointwise():
    fis = _toy_fis()
    xs = np.linspace(0, 10, 21)
    batch = fis.infer_batch({"x": xs})
    point = np.array([fis.infer({"x": x}) for x in xs])
    assert np.allclose(batch, point)


def test_infer_batch_length_mismatch_raises():
    fis = build_alert_fis()
    with pytest.raises(ValueError):
        fis.infer_batch({"flow_ratio": np.zeros(5),
                         "width_ratio": np.zeros(6)})


# --------------------------------------------------------------------------- #
# Default alert FIS — sanity of qualitative behaviour
# --------------------------------------------------------------------------- #
def test_alert_fis_high_flow_narrow_width_returns_red():
    fis = build_alert_fis()
    score = fis.infer({"flow_ratio": 2.0, "width_ratio": 0.1})
    cat = str(score_to_category(score))
    assert cat == "RED"


def test_alert_fis_low_flow_narrow_width_returns_green():
    fis = build_alert_fis()
    score = fis.infer({"flow_ratio": 0.1, "width_ratio": 0.1})
    cat = str(score_to_category(score))
    assert cat == "GREEN"


def test_alert_fis_high_flow_wide_width_returns_orange_not_red():
    fis = build_alert_fis()
    score_narrow = fis.infer({"flow_ratio": 2.0, "width_ratio": 0.1})
    score_wide = fis.infer({"flow_ratio": 2.0, "width_ratio": 1.5})
    # High flow with wide uncertainty should be less alarming than
    # high flow with narrow (confident) intervals.
    assert score_wide < score_narrow


def test_alert_fis_mid_flow_wide_width_returns_orange():
    fis = build_alert_fis()
    score = fis.infer({"flow_ratio": 1.0, "width_ratio": 1.5})
    cat = str(score_to_category(score))
    # Uncertain mid-range flow → escalate to ORANGE per rule R4.
    assert cat in ("YELLOW", "ORANGE")


def test_score_to_category_bins_correctly():
    scores = np.array([0.0, 0.9, 1.5, 2.7, 3.5])
    cats = score_to_category(scores)
    assert list(cats) == ["GREEN", "GREEN", "YELLOW", "ORANGE", "RED"]


def test_category_index_roundtrip():
    scores = np.array([0.5, 1.5, 2.5, 3.5])
    cats = score_to_category(scores)
    idx = category_to_index(cats)
    assert list(idx) == [0, 1, 2, 3]


def test_rules_summary_is_readable():
    fis = build_alert_fis()
    summary = fis.rules_summary()
    assert "IF flow_ratio is HIGH" in summary
    assert "THEN alert_level is RED" in summary
    # Five rules total.
    assert summary.count("\n") == 4


def test_alert_categories_are_the_expected_four():
    assert CATEGORIES == ("GREEN", "YELLOW", "ORANGE", "RED")
