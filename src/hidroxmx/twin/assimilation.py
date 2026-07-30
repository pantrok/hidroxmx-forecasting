"""Retrospective residual assimilation for the F0-PUB forecaster.

Implements the classical **innovation-persistence** (a.k.a. autoregressive
residual correction) update that operational hydrological forecasters
apply as the entry-level data-assimilation step (Reggiani & Weerts
2008; Cane et al. 2020). No model retraining is required: we treat
the F0-PUB point forecast as fixed and post-hoc adjust each new
horizon by a decayed function of the most recent observed
model-vs-truth error.

Concept
-------
At time ``t`` the model has issued a 1-day-ahead forecast ``ŷ(t-1)``
based on information up to ``t-2``. The corresponding observation
``y(t-1)`` is now available, so the *innovation* (observed minus
predicted) is

    e_1 = y(t-1) − ŷ(t-1)

Assuming the error autocorrelation decays exponentially with lead
time, the assimilated forecast at horizon ``h`` becomes

    ŷ_assim(t+h) = ŷ(t+h) + decay^h · e_1

with ``decay ∈ (0, 1)``. Typical values in flood forecasting are
0.6–0.9. Higher ``decay`` = stronger persistence, better at short
horizons but noisier at long horizons.

Extension to a K-day history uses an EWMA of the K most recent
one-day residuals as the innovation, weighted by ``decay``.

This module is deliberately compact and pure NumPy. The paper's
predictive-twin claim rests on the assimilation demonstrably
reducing test-window RMSE by ≥ 10 % vs the un-assimilated F0-PUB
baseline (kill condition of §6 of the brief).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class InnovationPersistence:
    """Innovation-persistence residual assimilator.

    Parameters
    ----------
    decay : ``d`` in ``(0, 1)``. The correction applied at horizon ``h``
        is ``d^h · innovation``.
    history_days : number of trailing one-day residuals to average into
        the innovation. ``1`` uses the most recent error only; larger
        values apply EWMA smoothing with weight ``d``.
    """
    decay: float = 0.7
    history_days: int = 5

    def __post_init__(self):
        if not (0.0 < self.decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {self.decay}")
        if self.history_days < 1:
            raise ValueError(f"history_days must be >= 1, got {self.history_days}")

    def _innovation(self, recent_residuals: np.ndarray) -> float:
        """Reduce a residual history to one scalar innovation (EWMA)."""
        r = np.asarray(recent_residuals, dtype=float)
        r = r[np.isfinite(r)]
        if r.size == 0:
            return 0.0
        r = r[-self.history_days:]
        if r.size == 1:
            return float(r[0])
        weights = self.decay ** np.arange(r.size - 1, -1, -1)
        return float(np.average(r, weights=weights))

    def correction_per_horizon(self, recent_residuals: np.ndarray,
                                horizons: np.ndarray | list[int]) -> np.ndarray:
        """Return a length-``H`` array of additive corrections."""
        innov = self._innovation(recent_residuals)
        h = np.asarray(horizons, dtype=float)
        return innov * (self.decay ** h)

    def assimilate(self, raw_forecast: np.ndarray,
                    recent_residuals: np.ndarray,
                    horizons: np.ndarray | list[int]) -> np.ndarray:
        """Apply innovation persistence to one raw multi-horizon forecast."""
        correction = self.correction_per_horizon(recent_residuals, horizons)
        return np.asarray(raw_forecast, dtype=float) + correction


def residuals_from_history(y_true_hist: np.ndarray,
                            y_pred_hist: np.ndarray) -> np.ndarray:
    """Return the trailing residual series ``y_true - y_pred`` from history.

    Both inputs are 1-D arrays of daily values ordered by time. NaN entries
    are preserved so the assimilator's own filter can drop them.
    """
    y_true_hist = np.asarray(y_true_hist, dtype=float)
    y_pred_hist = np.asarray(y_pred_hist, dtype=float)
    if y_true_hist.shape != y_pred_hist.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true_hist.shape} vs y_pred {y_pred_hist.shape}"
        )
    return y_true_hist - y_pred_hist


def assimilate_forecasts(raw_forecasts: np.ndarray,
                          residual_history_per_window: np.ndarray,
                          horizons: np.ndarray | list[int],
                          decay: float = 0.7,
                          history_days: int = 5) -> np.ndarray:
    """Vectorised assimilation across all test windows.

    Parameters
    ----------
    raw_forecasts : shape ``[N, H]`` — raw F0-PUB forecasts for each
        test window and horizon.
    residual_history_per_window : shape ``[N, K]`` — for each test
        window, the ``K`` trailing 1-day residuals observed up to but
        excluding the forecast issue time. NaN entries are dropped.
    horizons : the ``H`` horizon indices (days).
    decay, history_days : forwarded to :class:`InnovationPersistence`.
    """
    raw_forecasts = np.asarray(raw_forecasts, dtype=float)
    residual_history_per_window = np.asarray(residual_history_per_window, dtype=float)
    if raw_forecasts.ndim != 2:
        raise ValueError(f"raw_forecasts must be 2-D, got {raw_forecasts.shape}")
    n, h_dim = raw_forecasts.shape
    if residual_history_per_window.shape[0] != n:
        raise ValueError(
            f"residual_history N mismatch: {residual_history_per_window.shape[0]} vs {n}"
        )
    if h_dim != len(horizons):
        raise ValueError(
            f"horizons length {len(horizons)} != raw_forecasts.shape[1] {h_dim}"
        )
    engine = InnovationPersistence(decay=decay, history_days=history_days)
    out = np.empty_like(raw_forecasts)
    for i in range(n):
        out[i] = engine.assimilate(raw_forecasts[i],
                                    residual_history_per_window[i],
                                    horizons)
    return out
