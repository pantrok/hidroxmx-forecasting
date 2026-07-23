"""What-if scenario perturbations on the F0-PUB feature window.

The scoped predictive digital twin exposes four operator-facing
scenarios that answer questions of the form "what would the forecast
look like if …". Each perturbs one or more of the exogenous feature
columns and leaves the streamflow-lag family alone (those are the
target-derived state and must reflect the model's own dynamics).

Available perturbations
-----------------------
- ``perturb_precip(feature_window, delta_pct)`` — multiplicative shift
  on the precip column(s) by ``(1 + delta_pct / 100)``.
- ``perturb_temp(feature_window, delta_c)`` — additive shift on the
  tmax_c and tmin_c columns by ``delta_c`` degrees.
- ``dry_out(feature_window, days=N)`` — set precip to zero for the last
  N days of the lookback window (a drought-lead scenario).

All perturbations operate on the *standardised* feature window that
the LSTM consumes. This means the caller passes standardisation
statistics so the shift is re-scaled back to the same standardised
space the model was trained on.

Perturbations always return a fresh array — the caller's window is
never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


PRECIP_COLS = ("precip_mm", "precip_mm_ma7")
TEMP_COLS = ("tmax_c", "tmax_c_ma7", "tmin_c", "tmin_c_ma7")


@dataclass(slots=True, frozen=True)
class ScenarioResult:
    """Container for a single scenario evaluation."""
    name: str
    perturbed_forecast: np.ndarray   # [N, H] in physical units
    baseline_forecast: np.ndarray    # [N, H] in physical units
    delta: np.ndarray                # perturbed − baseline, [N, H]

    def summary(self) -> dict[str, float]:
        """Aggregate delta statistics for a compact report."""
        delta = self.delta[np.isfinite(self.delta)]
        if delta.size == 0:
            return {"mean_delta": float("nan"), "median_delta": float("nan"),
                    "max_abs_delta": float("nan")}
        return {
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
            "max_abs_delta": float(np.max(np.abs(delta))),
        }


def _column_indices(feature_cols: Sequence[str],
                     target: Sequence[str]) -> list[int]:
    return [i for i, c in enumerate(feature_cols) if c in target]


def perturb_precip(feature_window: np.ndarray,
                    feature_cols: Sequence[str],
                    delta_pct: float,
                    stats: dict[str, tuple[float, float]]) -> np.ndarray:
    """Multiplicative precip perturbation in physical space.

    Parameters
    ----------
    feature_window : ``[L, F]`` (or ``[N, L, F]``) standardised features.
    feature_cols : the ``F`` column names in order.
    delta_pct : e.g. ``20.0`` → precip × 1.20.
    stats : ``{col: (mu, sigma)}`` from
        :func:`hidroxmx.data.features.standardise`.
    """
    x = np.array(feature_window, dtype=float, copy=True)
    idx = _column_indices(feature_cols, PRECIP_COLS)
    factor = 1.0 + delta_pct / 100.0
    for i in idx:
        col = feature_cols[i]
        mu, sigma = stats.get(col, (0.0, 1.0))
        # x is standardised → recover raw, scale, re-standardise.
        raw = x[..., i] * sigma + mu
        raw = np.clip(raw * factor, 0.0, None)   # precip cannot be negative
        x[..., i] = (raw - mu) / (sigma if sigma > 0 else 1.0)
    return x


def perturb_temp(feature_window: np.ndarray,
                  feature_cols: Sequence[str],
                  delta_c: float,
                  stats: dict[str, tuple[float, float]]) -> np.ndarray:
    """Additive temperature perturbation in physical space."""
    x = np.array(feature_window, dtype=float, copy=True)
    idx = _column_indices(feature_cols, TEMP_COLS)
    for i in idx:
        col = feature_cols[i]
        mu, sigma = stats.get(col, (0.0, 1.0))
        raw = x[..., i] * sigma + mu + delta_c
        x[..., i] = (raw - mu) / (sigma if sigma > 0 else 1.0)
    return x


def dry_out(feature_window: np.ndarray,
             feature_cols: Sequence[str],
             days: int,
             stats: dict[str, tuple[float, float]]) -> np.ndarray:
    """Set the last ``days`` days of precip to zero (drought-lead scenario)."""
    x = np.array(feature_window, dtype=float, copy=True)
    idx = _column_indices(feature_cols, PRECIP_COLS)
    lookback = x.shape[-2] if x.ndim >= 2 else x.shape[-1]
    start = max(0, lookback - days)
    for i in idx:
        col = feature_cols[i]
        mu, sigma = stats.get(col, (0.0, 1.0))
        zero_std = (0.0 - mu) / (sigma if sigma > 0 else 1.0)
        if x.ndim == 3:
            x[:, start:, i] = zero_std
        else:
            x[start:, i] = zero_std
    return x


# --------------------------------------------------------------------------- #
# Library of canonical scenarios the paper reports
# --------------------------------------------------------------------------- #
SCENARIO_LIBRARY: dict[str, Callable[[np.ndarray, Sequence[str], dict], np.ndarray]] = {
    "precip_+20pct": lambda x, cols, s: perturb_precip(x, cols, +20.0, s),
    "precip_-20pct": lambda x, cols, s: perturb_precip(x, cols, -20.0, s),
    "temp_+2C":       lambda x, cols, s: perturb_temp(x, cols, +2.0, s),
    "temp_-2C":       lambda x, cols, s: perturb_temp(x, cols, -2.0, s),
    "dry_out_14d":    lambda x, cols, s: dry_out(x, cols, 14, s),
}


def apply_scenario(name: str,
                    feature_window: np.ndarray,
                    feature_cols: Sequence[str],
                    stats: dict[str, tuple[float, float]]) -> np.ndarray:
    """Apply a named scenario from :data:`SCENARIO_LIBRARY`."""
    if name not in SCENARIO_LIBRARY:
        raise ValueError(f"unknown scenario {name!r}; "
                         f"choose from {sorted(SCENARIO_LIBRARY)}")
    return SCENARIO_LIBRARY[name](feature_window, feature_cols, stats)
