"""Split conformal prediction — post-hoc, model-agnostic UQ (Path B step 1).

Split conformal (Vovk et al. 2005; Angelopoulos & Bates 2023) wraps any
point-forecast model with prediction intervals that carry a finite-sample
marginal-coverage guarantee under the mild assumption of *exchangeability*
between the calibration and test sets — no distributional assumption on
the residuals, no re-training of the base model.

Given calibration predictions ``yhat_val`` and the corresponding truths
``y_val`` (in physical units), we compute the empirical (1 − α) quantile
of the absolute residuals ``|y − yhat|`` per forecast horizon and use it
as the interval half-width around every future point forecast. The
resulting intervals ``[yhat − q, yhat + q]`` cover the true value with
probability at least

    1 − α − 1 / (n + 1)

on any exchangeable test point, per Lei et al. (2018).

The Milestone-3 F0-PUB validation window is the natural calibration set:
it never entered training, sits between train and test in time, and
already has denormalised predictions available through the same
inference pipeline that produced the test-set metrics.

This module is intentionally 40 lines of NumPy — the paper's
contribution is not the UQ engine itself but its downstream use by the
Mamdani fuzzy alert layer (Milestone 5b).

References
----------
- Vovk, Gammerman & Shafer (2005). Algorithmic Learning in a Random
  World. Springer.
- Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018). Distribution-free
  predictive inference for regression. JASA 113(523).
- Angelopoulos & Bates (2023). A gentle introduction to conformal
  prediction and distribution-free uncertainty quantification.
  Foundations and Trends in ML 16(4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class SplitConformal:
    """Absolute-residual split conformal wrapper for multi-horizon regression.

    Parameters
    ----------
    alpha : float in (0, 1)
        Target miscoverage rate. ``alpha = 0.1`` yields nominal 90 %
        coverage intervals.

    Attributes
    ----------
    quantiles_ : ndarray of shape (H,), populated after ``fit``.
        Per-horizon interval half-width in physical units.
    n_calibration_ : int
        Number of calibration examples that produced ``quantiles_``.
    """
    alpha: float = 0.1
    quantiles_: np.ndarray = field(default=None, init=False, repr=False)
    n_calibration_: int = field(default=0, init=False)

    def fit(self, y_val: np.ndarray, yhat_val: np.ndarray) -> "SplitConformal":
        """Calibrate on ``|y_val − yhat_val|`` per horizon.

        Inputs are shape ``[N, H]`` in physical units. NaN entries are
        dropped from the per-horizon quantile computation so a partially
        missing validation set does not corrupt the calibration.
        """
        y_val = np.asarray(y_val, dtype=float)
        yhat_val = np.asarray(yhat_val, dtype=float)
        if y_val.shape != yhat_val.shape:
            raise ValueError(f"shape mismatch: y {y_val.shape} vs yhat {yhat_val.shape}")
        if y_val.ndim != 2:
            raise ValueError(f"expected 2-D arrays [N, H], got {y_val.shape}")
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

        residuals = np.abs(y_val - yhat_val)  # [N, H]
        n = residuals.shape[0]
        # Finite-sample correction: use rank = ceil((n+1)(1-alpha)).
        rank = int(np.ceil((n + 1) * (1 - self.alpha)))
        rank = min(max(rank, 1), n)
        # Per-horizon nan-safe quantile from the sorted residuals.
        q = np.empty(residuals.shape[1], dtype=float)
        for h in range(residuals.shape[1]):
            col = residuals[:, h]
            col = col[np.isfinite(col)]
            if col.size == 0:
                q[h] = np.nan
                continue
            k = min(rank, col.size)
            q[h] = float(np.sort(col)[k - 1])
        self.quantiles_ = q
        self.n_calibration_ = n
        return self

    def predict_interval(self, yhat: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` prediction intervals around ``yhat``."""
        if self.quantiles_ is None:
            raise RuntimeError("call fit() before predict_interval()")
        yhat = np.asarray(yhat, dtype=float)
        return yhat - self.quantiles_, yhat + self.quantiles_


def coverage_and_width(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
                       ) -> dict[str, np.ndarray]:
    """Per-horizon empirical coverage and mean interval width.

    All three inputs are shape ``[N, H]`` in physical units. Returns a
    dict with ``coverage`` (fraction of tests within the interval) and
    ``mean_interval_width`` (average of ``upper − lower``), each a
    length-``H`` array.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (y_true.shape == lower.shape == upper.shape):
        raise ValueError("y_true, lower, upper must share the same shape [N, H]")
    coverage = np.zeros(y_true.shape[1], dtype=float)
    width = np.zeros(y_true.shape[1], dtype=float)
    for h in range(y_true.shape[1]):
        mask = np.isfinite(y_true[:, h]) & np.isfinite(lower[:, h]) & np.isfinite(upper[:, h])
        if not mask.any():
            coverage[h] = np.nan
            width[h] = np.nan
            continue
        inside = (y_true[mask, h] >= lower[mask, h]) & (y_true[mask, h] <= upper[mask, h])
        coverage[h] = float(inside.mean())
        width[h] = float((upper[mask, h] - lower[mask, h]).mean())
    return {"coverage": coverage, "mean_interval_width": width}


def tail_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  threshold_quantile: float = 0.95) -> np.ndarray:
    """Per-horizon coverage restricted to the upper tail of ``y_true``.

    ``threshold_quantile`` picks the tail bound from the *observed* test
    values (Q95 by default). Reveals whether the interval calibrated
    globally still covers extreme events — the setting where a fuzzy
    alert layer earns its keep.
    """
    y_true = np.asarray(y_true, dtype=float)
    out = np.zeros(y_true.shape[1], dtype=float)
    for h in range(y_true.shape[1]):
        col = y_true[:, h]
        col_lo = lower[:, h]
        col_hi = upper[:, h]
        mask_finite = np.isfinite(col) & np.isfinite(col_lo) & np.isfinite(col_hi)
        if not mask_finite.any():
            out[h] = np.nan
            continue
        threshold = float(np.quantile(col[mask_finite], threshold_quantile))
        mask = mask_finite & (col >= threshold)
        if not mask.any():
            out[h] = np.nan
            continue
        inside = (col[mask] >= col_lo[mask]) & (col[mask] <= col_hi[mask])
        out[h] = float(inside.mean())
    return out
