"""Single source of truth for evaluation metrics.

All metrics operate on ``numpy`` arrays (observed ``y``, predicted ``yhat``) and
return a Python ``float`` (or a small tuple). Every function silently ignores
pairs where either value is NaN, so callers can mix gauged and gap-flagged
observations without pre-masking.

The dictionary :data:`METRICS` lists every metric under a short, stable key so
it can be selected from a YAML experiment config.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pair(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite-value intersection of two arrays."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs yhat {yhat.shape}")
    mask = np.isfinite(y) & np.isfinite(yhat)
    return y[mask], yhat[mask]


# --------------------------------------------------------------------------- #
# Deterministic backbone metrics
# --------------------------------------------------------------------------- #
def nse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Nash–Sutcliffe efficiency."""
    y, yhat = _pair(y, yhat)
    if y.size < 2:
        return float("nan")
    denom = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / denom) if denom > 0 else float("nan")


def kge(y: np.ndarray, yhat: np.ndarray) -> float:
    """Kling–Gupta efficiency (Gupta et al., 2009)."""
    y, yhat = _pair(y, yhat)
    if y.size < 2 or y.std() == 0 or yhat.std() == 0:
        return float("nan")
    r = float(np.corrcoef(y, yhat)[0, 1])
    alpha = yhat.std() / y.std()
    beta = yhat.mean() / y.mean() if y.mean() != 0 else float("nan")
    if not np.isfinite(beta):
        return float("nan")
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error."""
    y, yhat = _pair(y, yhat)
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def pbias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Percent bias (positive = model over-predicts)."""
    y, yhat = _pair(y, yhat)
    total = np.sum(y)
    if total == 0:
        return float("nan")
    return float(100.0 * np.sum(yhat - y) / total)


def ehf(y: np.ndarray, yhat: np.ndarray, quantile: float = 0.95) -> float:
    """High-flow error (relative mean bias above the ``quantile``-th percentile of y)."""
    y, yhat = _pair(y, yhat)
    if y.size == 0:
        return float("nan")
    threshold = np.quantile(y, quantile)
    mask = y >= threshold
    if not mask.any():
        return float("nan")
    return float(np.mean(yhat[mask] - y[mask]) / (np.mean(y[mask]) or float("nan")))


def peak_timing_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """Difference in argmax location (in units of index, positive = predicted peak late)."""
    y, yhat = _pair(y, yhat)
    if y.size == 0:
        return float("nan")
    return float(int(np.argmax(yhat)) - int(np.argmax(y)))


# --------------------------------------------------------------------------- #
# Probabilistic / interval metrics
# --------------------------------------------------------------------------- #
def empirical_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of observations that fall inside ``[lower, upper]``."""
    y = np.asarray(y, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    mask = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if not mask.any():
        return float("nan")
    inside = (y[mask] >= lower[mask]) & (y[mask] <= upper[mask])
    return float(inside.mean())


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Sharpness — mean width of the prediction interval."""
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    mask = np.isfinite(lower) & np.isfinite(upper)
    if not mask.any():
        return float("nan")
    return float(np.mean(upper[mask] - lower[mask]))


def crps_sample(y: np.ndarray, ensemble: np.ndarray) -> float:
    """CRPS estimated from an ensemble (shape ``[N, M]`` for N obs, M members)."""
    y = np.asarray(y, float)
    ens = np.asarray(ensemble, float)
    if ens.ndim != 2 or ens.shape[0] != y.shape[0]:
        raise ValueError("ensemble must have shape [N_obs, N_members] matching y")
    mask = np.isfinite(y) & np.all(np.isfinite(ens), axis=1)
    if not mask.any():
        return float("nan")
    y = y[mask]
    ens = ens[mask]
    n_members = ens.shape[1]
    term1 = np.mean(np.abs(ens - y[:, None]), axis=1)
    diff = np.abs(ens[:, :, None] - ens[:, None, :]).mean(axis=(1, 2))
    return float(np.mean(term1 - 0.5 * diff * (n_members / (n_members - 1) if n_members > 1 else 1.0)))


# --------------------------------------------------------------------------- #
# Decision-frontier metrics
# --------------------------------------------------------------------------- #
def brier_score(events: np.ndarray, probs: np.ndarray) -> float:
    """Brier score for a binary event indicator vs. its predicted probability."""
    events = np.asarray(events, float)
    probs = np.asarray(probs, float)
    mask = np.isfinite(events) & np.isfinite(probs)
    if not mask.any():
        return float("nan")
    return float(np.mean((probs[mask] - events[mask]) ** 2))


def pod_far(events: np.ndarray, alerts: np.ndarray) -> tuple[float, float]:
    """Probability of detection and false alarm ratio for binary alerts."""
    events = np.asarray(events, bool)
    alerts = np.asarray(alerts, bool)
    tp = int((events & alerts).sum())
    fn = int((events & ~alerts).sum())
    fp = int((~events & alerts).sum())
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (fp + tp) if (fp + tp) else float("nan")
    return float(pod), float(far)


def value_cost_loss(events: np.ndarray, alerts: np.ndarray, cost_ratio: float) -> float:
    """Richardson (2000) cost-loss value score for a fixed cost/loss ratio.

    Parameters
    ----------
    events : bool array of realised events (loss L if occurs).
    alerts : bool array of forecast-based alert actions (cost C always incurred).
    cost_ratio : float in (0, 1), representing C / L.
    """
    events = np.asarray(events, bool)
    alerts = np.asarray(alerts, bool)
    if events.size == 0 or events.size != alerts.size:
        return float("nan")
    hit = float((events & alerts).mean())
    fa = float((~events & alerts).mean())
    miss = float((events & ~alerts).mean())
    base_rate = float(events.mean())
    ec_forecast = cost_ratio * (hit + fa) + miss
    ec_climate = min(cost_ratio, base_rate)
    ec_perfect = cost_ratio * base_rate
    if ec_climate == ec_perfect:
        return float("nan")
    return float((ec_climate - ec_forecast) / (ec_climate - ec_perfect))


def lead_time_at(pod_curve: Iterable[tuple[int, float]], target_pod: float) -> int:
    """Maximum lead time (days) at which the achieved POD is still ≥ target_pod.

    ``pod_curve`` is an iterable of ``(lead_days, achieved_pod)`` pairs.
    """
    valid = [lead for lead, val in pod_curve if val >= target_pod]
    return int(max(valid)) if valid else -1


# --------------------------------------------------------------------------- #
# Registry consumed by conf/experiments/metrics.yaml
# --------------------------------------------------------------------------- #
METRICS: dict[str, Callable] = {
    "nse": nse,
    "kge": kge,
    "rmse": rmse,
    "pbias": pbias,
    "ehf": ehf,
    "peak_timing_error": peak_timing_error,
    "empirical_coverage": empirical_coverage,
    "mean_interval_width": mean_interval_width,
    "crps_sample": crps_sample,
    "brier_score": brier_score,
    "pod_far": pod_far,
    "value_cost_loss": value_cost_loss,
    "lead_time_at": lead_time_at,
}
