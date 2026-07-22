"""Paired bootstrap confidence intervals for hydrological model comparisons.

Milestone 7 consolidates the paper's central comparisons (M3 F0-PUB vs
persistence, M4 mechanism vs lumped, M5c fuzzy vs point-threshold
baseline) with **paired non-parametric bootstrap** confidence intervals.
Pairing is critical here: the two methods are evaluated on the *same*
folds and horizons, so a paired resample controls fold-level nuisance
variation that would otherwise inflate the interval width.

The engine here is intentionally 40 lines of NumPy — the paper's
contribution is not the resampling recipe but its downstream use in
the kill-condition tables (Milestones 3, 4, 5c).

Statistical notes
-----------------
- Default statistic is the **median**, matching the aggregate reporting
  we settled on in Milestone 5c after outlier folds distorted the mean.
  ``mean`` is also supported for tables where the manuscript prefers
  it (e.g., NSE aggregates in Milestone 3).
- The confidence interval is the **percentile** interval at level
  ``ci`` — simplest and cleanest for a paper. BCa can be added later
  as an ablation if a reviewer asks.
- The pairing is applied by resampling **fold indices** (not
  observations), so both metrics ``a[i]`` and ``b[i]`` move together
  in each bootstrap replicate.
- The default RNG seed is 20260721 — the same seed used throughout
  the codebase, so ``paired_bootstrap`` is bit-reproducible across runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_N_BOOT = 10_000
DEFAULT_CI = 0.95
DEFAULT_SEED = 20260721


@dataclass(slots=True, frozen=True)
class PairedBootstrapResult:
    """Central estimate + confidence interval + significance flag."""
    n: int                       # number of folds after nan filtering
    delta: float                 # a - b at the requested statistic
    ci_low: float
    ci_high: float
    ci_level: float              # e.g. 0.95
    statistic: str               # 'median' or 'mean'
    significant: bool            # CI excludes 0

    def as_row(self) -> dict:
        return {
            "n": self.n,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "statistic": self.statistic,
            "significant": self.significant,
        }


def _stat(values: np.ndarray, statistic: str) -> float:
    if statistic == "median":
        return float(np.nanmedian(values))
    if statistic == "mean":
        return float(np.nanmean(values))
    raise ValueError(f"unknown statistic {statistic!r}")


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *,
                     n_boot: int = DEFAULT_N_BOOT,
                     ci: float = DEFAULT_CI,
                     statistic: str = "median",
                     seed: int = DEFAULT_SEED
                     ) -> PairedBootstrapResult:
    """Paired bootstrap CI for the difference between two metric arrays.

    Parameters
    ----------
    a, b : arrays of the same length. NaN entries in either are
        dropped pair-wise before resampling.
    n_boot : number of bootstrap replicates.
    ci : confidence level in ``(0, 1)``.
    statistic : ``'median'`` (default) or ``'mean'``.
    seed : RNG seed for reproducibility.

    Returns
    -------
    PairedBootstrapResult with ``delta`` and ``ci`` in the same units as
    the inputs, and ``significant`` set when the CI excludes zero.
    """
    if not (0 < ci < 1):
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a {a.shape} vs b {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got {a.shape}")

    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    n = int(a.size)
    if n < 2:
        # Can't bootstrap a single point; report the raw delta if any.
        if n == 0:
            return PairedBootstrapResult(0, float("nan"), float("nan"),
                                         float("nan"), ci, statistic, False)
        delta = _stat(a, statistic) - _stat(b, statistic)
        return PairedBootstrapResult(n, delta, float("nan"),
                                     float("nan"), ci, statistic, False)

    delta_observed = _stat(a, statistic) - _stat(b, statistic)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        sample = idx[k]
        deltas[k] = _stat(a[sample], statistic) - _stat(b[sample], statistic)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    ci_low = float(np.quantile(deltas, lo_q))
    ci_high = float(np.quantile(deltas, hi_q))
    significant = (ci_low > 0.0) or (ci_high < 0.0)

    return PairedBootstrapResult(
        n=n, delta=float(delta_observed),
        ci_low=ci_low, ci_high=ci_high,
        ci_level=ci, statistic=statistic,
        significant=significant,
    )


def paired_bootstrap_kill_check(a: np.ndarray, b: np.ndarray, *,
                                 threshold: float = 0.05,
                                 statistic: str = "median",
                                 n_boot: int = DEFAULT_N_BOOT,
                                 ci: float = DEFAULT_CI,
                                 seed: int = DEFAULT_SEED) -> tuple[PairedBootstrapResult, bool]:
    """Paired bootstrap + a kill-condition check ``delta ≥ threshold``.

    Returns ``(result, kill_cleared)`` where ``kill_cleared`` is True
    when the ``lower CI end`` of ``a − b`` is at or above ``threshold``.
    Stricter than mere significance: for Milestones 3 and 5c the paper
    reports whether the ≥ +0.05 improvement is *guaranteed* (CI-lower
    above threshold) rather than merely *plausible* (point estimate).
    """
    result = paired_bootstrap(a, b, n_boot=n_boot, ci=ci,
                              statistic=statistic, seed=seed)
    kill_cleared = bool(np.isfinite(result.ci_low) and result.ci_low >= threshold)
    return result, kill_cleared
