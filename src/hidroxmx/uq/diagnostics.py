"""Diagnostics of conformal intervals in the high-flow tail.

Two alternative intervals and the helpers that describe how misses cluster in
time. None of this replaces the split-conformal interval of
:mod:`hidroxmx.uq.conformal`: it exists to separate two explanations of the tail
collapse, a homoscedastic score (:func:`log_score_interval`) and drift between
calibration and test (:func:`adaptive_interval`).

References
----------
- Gibbs & Candès (2021). Adaptive conformal inference under distribution shift.
  NeurIPS 34.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def order_statistic_rank(n: int, alpha: float) -> tuple[int, bool]:
    """Return the split-conformal rank ``ceil((n + 1)(1 - alpha))`` clamped to ``[1, n]``.

    The flag tells whether the clamp acted, that is, whether ``alpha`` asked for an
    order statistic beyond the calibration sample.
    """
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    clamped = min(max(rank, 1), n)
    return clamped, clamped != rank


def log_score_interval(y_cal: np.ndarray, yhat_cal: np.ndarray, yhat: np.ndarray,
                       alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split conformal with the score ``|log1p(y) - log1p(yhat)|``, per horizon.

    Inputs are ``[N, H]`` in m³/s. The interval is built in log space and inverted,
    so its width in m³/s grows with the forecast. Returns ``(lower, upper, q)`` with
    the lower limit floored at 0.
    """
    scores = np.abs(np.log1p(np.asarray(y_cal, float)) - np.log1p(np.asarray(yhat_cal, float)))
    n = scores.shape[0]
    rank, _ = order_statistic_rank(n, alpha)
    q = np.sort(scores, axis=0)[rank - 1]
    centre = np.log1p(np.asarray(yhat, float))
    lower = np.maximum(np.expm1(centre - q), 0.0)
    upper = np.expm1(centre + q)
    return lower, upper, q


@dataclass(slots=True, frozen=True)
class AdaptiveResult:
    """Output of :func:`adaptive_interval` for one horizon."""
    lower: np.ndarray
    upper: np.ndarray
    alpha_path: np.ndarray       # alpha used at each origin
    clamped_share: float         # share of origins that hit the end of the sample


def adaptive_interval(scores_cal: np.ndarray, y: np.ndarray, yhat: np.ndarray,
                      origins: np.ndarray, targets: np.ndarray, *,
                      alpha: float = 0.1, gamma: float = 0.005) -> AdaptiveResult:
    """Adaptive conformal inference with delayed feedback, for one horizon.

    Calibration scores stay fixed. At each origin the miscoverage level follows
    ``alpha_t+1 = alpha_t + gamma (alpha - err_t)``, and a forecast's ``err`` enters
    only when its target date has been reached, so each interval uses past
    information alone. ``origins`` and ``targets`` are sorted date arrays aligned with
    ``y`` and ``yhat``.
    """
    scores = np.sort(np.asarray(scores_cal, float))
    n = scores.size
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    origins = np.asarray(origins, dtype="datetime64[D]")
    targets = np.asarray(targets, dtype="datetime64[D]")
    m = y.size
    lower = np.empty(m)
    upper = np.empty(m)
    path = np.empty(m)
    misses = np.empty(m, dtype=bool)
    level = float(alpha)
    applied = 0
    clamped = 0
    for i in range(m):
        while applied < i and targets[applied] <= origins[i]:
            level += gamma * (alpha - float(misses[applied]))
            applied += 1
        rank, hit_end = order_statistic_rank(n, level)
        clamped += int(hit_end)
        q = scores[rank - 1]
        lower[i] = yhat[i] - q
        upper[i] = yhat[i] + q
        path[i] = level
        misses[i] = not (lower[i] <= y[i] <= upper[i])
    return AdaptiveResult(lower, upper, path, clamped / m if m else float("nan"))


def adjacent_miss_pairs(dates: np.ndarray, misses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return miss indicators on tail days and on the tail day right after them.

    ``dates`` are the sorted tail days of one station. Only pairs one calendar day
    apart count, so a gap in the tail never reads as temporal dependence.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    misses = np.asarray(misses, dtype=bool)
    if dates.size < 2:
        return np.empty(0, bool), np.empty(0, bool)
    adjacent = (dates[1:] - dates[:-1]) == np.timedelta64(1, "D")
    return misses[:-1][adjacent], misses[1:][adjacent]


def miss_run_lengths(dates: np.ndarray, misses: np.ndarray) -> np.ndarray:
    """Lengths of the runs of consecutive-day misses among tail days."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    misses = np.asarray(misses, dtype=bool)
    runs: list[int] = []
    current = 0
    for i in range(dates.size):
        if misses[i]:
            contiguous = i > 0 and misses[i - 1] and \
                (dates[i] - dates[i - 1]) == np.timedelta64(1, "D")
            current = current + 1 if contiguous else 1
            nxt = i + 1
            ends = nxt == dates.size or not misses[nxt] or \
                (dates[nxt] - dates[i]) != np.timedelta64(1, "D")
            if ends:
                runs.append(current)
                current = 0
    return np.asarray(runs, dtype=int)


def tail_events(dates: np.ndarray, min_separation_days: int = 3) -> list[np.ndarray]:
    """Group sorted tail days into events, as index arrays.

    Two tail days belong to separate events when at least ``min_separation_days``
    non-tail days lie between them.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    if dates.size == 0:
        return []
    gaps = (dates[1:] - dates[:-1]).astype(int) - 1
    cuts = np.flatnonzero(gaps >= min_separation_days) + 1
    return np.split(np.arange(dates.size), cuts)
