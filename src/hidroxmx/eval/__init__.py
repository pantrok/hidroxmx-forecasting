"""Evaluation utilities: metrics registry, decision-frontier tools, figures."""
from .bootstrap import (
    PairedBootstrapResult,
    paired_bootstrap,
    paired_bootstrap_kill_check,
)
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
    "PairedBootstrapResult",
    "brier_score",
    "crps_sample",
    "empirical_coverage",
    "ehf",
    "kge",
    "lead_time_at",
    "mean_interval_width",
    "nse",
    "paired_bootstrap",
    "paired_bootstrap_kill_check",
    "pbias",
    "peak_timing_error",
    "pod_far",
    "rmse",
    "value_cost_loss",
]
