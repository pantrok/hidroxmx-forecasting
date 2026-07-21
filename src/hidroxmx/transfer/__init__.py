"""Donor-matching mechanisms for Path A (Milestone 4)."""
from .signatures import (
    DEFAULT_SIGNATURE_KEYS,
    baseflow_index,
    compute_signatures,
    cv_flow,
    fdc_slope,
    flow_percentile,
    high_flow_frequency,
    low_flow_frequency,
    mean_flow,
    signature_vector,
)
from .similarity import SimilarityResult, score_donors

__all__ = [
    "DEFAULT_SIGNATURE_KEYS",
    "SimilarityResult",
    "baseflow_index",
    "compute_signatures",
    "cv_flow",
    "fdc_slope",
    "flow_percentile",
    "high_flow_frequency",
    "low_flow_frequency",
    "mean_flow",
    "score_donors",
    "signature_vector",
]
