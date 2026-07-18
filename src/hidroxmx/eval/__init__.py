"""Evaluation utilities: metrics registry, decision-frontier tools, figures."""
from .metrics import (
    METRICS,
    brier_score,
    crps_sample,
    empirical_coverage,
    ehf,
    kge,
    lead_time_at,
    mean_interval_width,
    nse,
    pbias,
    peak_timing_error,
    pod_far,
    rmse,
    value_cost_loss,
)

__all__ = [
    "METRICS",
    "brier_score",
    "crps_sample",
    "empirical_coverage",
    "ehf",
    "kge",
    "lead_time_at",
    "mean_interval_width",
    "nse",
    "pbias",
    "peak_timing_error",
    "pod_far",
    "rmse",
    "value_cost_loss",
]
