"""Hydrological signatures for donor-similarity computation (S-SIG).

A hydrological signature is a scalar summary of a stream's flow regime
that captures a physically-meaningful aspect (magnitude, variability,
timing, low-flow persistence). Two streams whose signatures are close
in a suitably scaled space tend to share catchment dynamics and
therefore make good donor/target pairs for transfer learning under
Path A (Kratzert et al. 2019, Newman et al. 2015, McMillan 2020).

The signature set implemented here is deliberately compact — nine
scalars that between them describe the flow duration curve, the
peak-flow regime, the low-flow regime and the baseflow contribution.
Every signature is nan-safe and returns ``nan`` if the input series
lacks enough finite observations to estimate it robustly.

References
----------
- Kratzert et al. (2019). Towards learning universal, regional, and
  local hydrological behaviors via machine learning applied to
  large-sample datasets. HESS 23(11).
- McMillan (2020). Linking hydrologic signatures to hydrologic
  processes: A review. Hydrological Processes 34(6).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


DEFAULT_SIGNATURE_KEYS: tuple[str, ...] = (
    "mean_flow",
    "cv_flow",
    "q05",
    "q50",
    "q95",
    "fdc_slope",
    "baseflow_index",
    "high_flow_freq",
    "low_flow_freq",
)


# --------------------------------------------------------------------------- #
# Individual signatures
# --------------------------------------------------------------------------- #
def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def mean_flow(series: np.ndarray) -> float:
    x = _finite(series)
    return float(np.mean(x)) if x.size else float("nan")


def cv_flow(series: np.ndarray) -> float:
    """Coefficient of variation of daily streamflow (std / mean)."""
    x = _finite(series)
    if x.size < 2:
        return float("nan")
    mu = float(np.mean(x))
    if mu <= 0:
        return float("nan")
    return float(np.std(x, ddof=0) / mu)


def flow_percentile(series: np.ndarray, q: float) -> float:
    """The ``q``-th percentile of the flow series (0 ≤ q ≤ 100)."""
    x = _finite(series)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def fdc_slope(series: np.ndarray, low_q: float = 33.0, high_q: float = 66.0) -> float:
    """Slope of the flow-duration curve between two exceedance percentiles.

    Defined in log space, per Yilmaz et al. (2008). Steep FDC → flashy
    regime; flat FDC → damped / baseflow-dominated regime. Returns nan
    when either flow is non-positive (undefined in log space).
    """
    x = _finite(series)
    if x.size < 10:
        return float("nan")
    q_low = float(np.percentile(x, 100 - low_q))    # exceeded low_q% of time
    q_high = float(np.percentile(x, 100 - high_q))  # exceeded high_q% of time
    if q_low <= 0 or q_high <= 0:
        return float("nan")
    return float((np.log(q_low) - np.log(q_high)) / ((high_q - low_q) / 100.0))


def baseflow_index(series: np.ndarray, alpha: float = 0.925) -> float:
    """Baseflow index via the Lyne-Hollick single-pass digital filter.

    Ratio of estimated baseflow volume to total streamflow volume, in
    the range (0, 1]. High BFI → groundwater-dominated regime. The
    alpha=0.925 default is the value most widely used in the UK/US
    literature (Ladson et al. 2013).
    """
    x = _finite(series)
    if x.size < 30:
        return float("nan")
    x = np.clip(x, 0.0, None)  # negatives are meaningless for baseflow
    q_direct = np.zeros_like(x)
    q_direct[0] = 0.0
    for t in range(1, len(x)):
        candidate = alpha * q_direct[t - 1] + (1.0 + alpha) / 2.0 * (x[t] - x[t - 1])
        q_direct[t] = max(0.0, min(candidate, x[t]))
    q_base = x - q_direct
    total = float(np.sum(x))
    if total <= 0:
        return float("nan")
    return float(np.sum(q_base) / total)


def high_flow_frequency(series: np.ndarray, threshold_quantile: float = 0.9) -> float:
    """Fraction of days above the ``threshold_quantile`` of the record."""
    x = _finite(series)
    if x.size == 0:
        return float("nan")
    thr = float(np.percentile(x, threshold_quantile * 100))
    return float(np.mean(x >= thr))


def low_flow_frequency(series: np.ndarray, threshold_quantile: float = 0.1) -> float:
    """Fraction of days below the ``threshold_quantile`` of the record."""
    x = _finite(series)
    if x.size == 0:
        return float("nan")
    thr = float(np.percentile(x, threshold_quantile * 100))
    return float(np.mean(x <= thr))


# --------------------------------------------------------------------------- #
# Registry + convenience API
# --------------------------------------------------------------------------- #
def compute_signatures(series: np.ndarray) -> dict[str, float]:
    """Return the full signature dict for one station's raw daily series."""
    return {
        "mean_flow": mean_flow(series),
        "cv_flow": cv_flow(series),
        "q05": flow_percentile(series, 5),
        "q50": flow_percentile(series, 50),
        "q95": flow_percentile(series, 95),
        "fdc_slope": fdc_slope(series),
        "baseflow_index": baseflow_index(series),
        "high_flow_freq": high_flow_frequency(series, 0.9),
        "low_flow_freq": low_flow_frequency(series, 0.1),
    }


def signature_vector(sig: dict[str, float],
                     keys: Sequence[str] = DEFAULT_SIGNATURE_KEYS) -> np.ndarray:
    """Return the signature dict as an ordered numeric vector."""
    return np.array([sig.get(k, float("nan")) for k in keys], dtype=float)
