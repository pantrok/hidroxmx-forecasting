"""Uncertainty quantification for Path B."""
from .conformal import SplitConformal, coverage_and_width, tail_coverage
from .diagnostics import (
    AdaptiveResult,
    adaptive_interval,
    adjacent_miss_pairs,
    log_score_interval,
    miss_run_lengths,
    order_statistic_rank,
    tail_events,
)

__all__ = [
    "AdaptiveResult",
    "SplitConformal",
    "adaptive_interval",
    "adjacent_miss_pairs",
    "coverage_and_width",
    "log_score_interval",
    "miss_run_lengths",
    "order_statistic_rank",
    "tail_coverage",
    "tail_events",
]
