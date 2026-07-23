"""Scoped predictive digital twin: retrospective assimilation + what-if scenarios."""
from .assimilation import (
    InnovationPersistence,
    assimilate_forecasts,
    residuals_from_history,
)
from .scenarios import (
    SCENARIO_LIBRARY,
    ScenarioResult,
    apply_scenario,
    dry_out,
    perturb_precip,
    perturb_temp,
)

__all__ = [
    "InnovationPersistence",
    "SCENARIO_LIBRARY",
    "ScenarioResult",
    "apply_scenario",
    "assimilate_forecasts",
    "dry_out",
    "perturb_precip",
    "perturb_temp",
    "residuals_from_history",
]
