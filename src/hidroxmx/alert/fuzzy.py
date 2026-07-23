"""Mamdani fuzzy inference system for flood-alert level assignment.

Path B step 2. Takes a *point* streamflow forecast from F0-PUB plus an
*interval width* from the conformal UQ layer and returns a crisp alert
level in ``[0, 4]`` that discretises into the four operational
categories

    GREEN  ≤ 1 < YELLOW ≤ 2 < ORANGE ≤ 3 < RED

The rule base is deliberately small (five rules) and hydrologically
interpretable so it can be exported verbatim into the paper's rule
table without needing further explanation to a reviewer.

Design decisions
----------------
- **Inputs are ratios**, not absolute quantities: ``flow_ratio`` is the
  point forecast divided by the target's training-window Q95, and
  ``width_ratio`` is the conformal interval width divided by the same
  reference. This makes the FIS **basin-agnostic**: identical rule
  base + membership functions apply to every station.
- **Mamdani style** (min-max composition, centroid defuzzification)
  rather than Sugeno so the output surface is monotonic and the fired
  rule → alert-level mapping is directly explainable per case.
- **NumPy only, no external fuzzy library**. The engine is 60 lines,
  auditable in one screen, and easy to unit-test.

The default FIS (``build_alert_fis``) is what the paper reports; users
can pass their own :class:`FuzzyVariable` and :class:`MamdaniRule`
lists to explore alternative rule bases in the ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Membership functions
# --------------------------------------------------------------------------- #
class MembershipFunction:
    """Base class — subclasses override ``__call__`` on scalar or array input."""

    def __call__(self, x):  # pragma: no cover
        raise NotImplementedError


@dataclass(slots=True, frozen=True)
class TriangularMF(MembershipFunction):
    """Triangular membership function with vertices ``a < b < c``."""
    a: float
    b: float
    c: float

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        left = (x - self.a) / max(self.b - self.a, 1e-12)
        right = (self.c - x) / max(self.c - self.b, 1e-12)
        return np.clip(np.minimum(left, right), 0.0, 1.0)


@dataclass(slots=True, frozen=True)
class TrapezoidalMF(MembershipFunction):
    """Trapezoidal membership function with vertices ``a ≤ b ≤ c ≤ d``."""
    a: float
    b: float
    c: float
    d: float

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        left = (x - self.a) / max(self.b - self.a, 1e-12)
        flat = np.ones_like(x)
        right = (self.d - x) / max(self.d - self.c, 1e-12)
        return np.clip(np.minimum(np.minimum(left, flat), right), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Fuzzy variables + rules
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class FuzzyVariable:
    """Named variable with a universe and one or more membership functions."""
    name: str
    universe_min: float
    universe_max: float
    mfs: dict[str, MembershipFunction] = field(default_factory=dict)


@dataclass(slots=True)
class MamdaniRule:
    """One Mamdani rule.

    ``antecedent`` maps input-variable name → membership-function name;
    entries are ANDed via the ``min`` t-norm. ``consequent`` picks the
    output MF whose activation the rule contributes to the aggregation.
    ``weight`` scales the activation before aggregation.
    """
    antecedent: dict[str, str]
    consequent: tuple[str, str]  # (output_var_name, output_mf_name)
    weight: float = 1.0

    def activation(self, mu_inputs: dict[str, dict[str, float]]) -> float:
        acts = [mu_inputs[var][mf] for var, mf in self.antecedent.items()]
        return float(self.weight * min(acts)) if acts else 0.0


# --------------------------------------------------------------------------- #
# Mamdani inference engine
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class MamdaniFIS:
    inputs: dict[str, FuzzyVariable]
    output: FuzzyVariable
    rules: list[MamdaniRule]
    resolution: int = 201

    def _fuzzify(self, values: dict[str, float]) -> dict[str, dict[str, float]]:
        mu = {}
        for name, x in values.items():
            var = self.inputs[name]
            mu[name] = {mf_name: float(mf(x)) for mf_name, mf in var.mfs.items()}
        return mu

    def infer(self, values: dict[str, float]) -> float:
        """Single-observation inference; returns the crisp centroid output."""
        mu_inputs = self._fuzzify(values)
        activations = [rule.activation(mu_inputs) for rule in self.rules]

        universe = np.linspace(self.output.universe_min,
                               self.output.universe_max,
                               self.resolution)
        aggregated = np.zeros_like(universe)
        for act, rule in zip(activations, self.rules):
            _, out_mf_name = rule.consequent
            out_mf = self.output.mfs[out_mf_name]
            clipped = np.minimum(out_mf(universe), act)
            aggregated = np.maximum(aggregated, clipped)

        total = aggregated.sum()
        if total <= 0:
            return float(self.output.universe_min)
        return float((universe * aggregated).sum() / total)

    def infer_batch(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        """Point-wise inference over 1-D arrays (all inputs same length)."""
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        for k in keys:
            if len(batch[k]) != n:
                raise ValueError(f"input {k!r} length {len(batch[k])} != {n}")
        out = np.empty(n, dtype=float)
        for i in range(n):
            values = {k: float(batch[k][i]) for k in keys}
            out[i] = self.infer(values)
        return out

    def rules_summary(self) -> str:
        """Human-readable rule table for direct paper export."""
        lines = []
        for i, r in enumerate(self.rules, 1):
            ant = " AND ".join(f"{v} is {m}" for v, m in r.antecedent.items())
            cons = f"{r.consequent[0]} is {r.consequent[1]}"
            lines.append(f"R{i}: IF {ant} THEN {cons}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Default alert FIS (the one the manuscript reports)
# --------------------------------------------------------------------------- #
GREEN_MAX = 1.0
YELLOW_MAX = 2.0
ORANGE_MAX = 3.0
# RED > ORANGE_MAX


def build_alert_fis() -> MamdaniFIS:
    """Return the flood-alert FIS the paper reports (basin-agnostic).

    Inputs
    ------
    ``flow_ratio``  : point forecast divided by target's training-window Q95.
    ``width_ratio`` : conformal interval width divided by the same reference.

    Output
    ------
    ``alert_level`` : crisp score in ``[0, 4]`` — GREEN/YELLOW/ORANGE/RED
    partition thresholds at 1, 2, 3.
    """
    flow = FuzzyVariable(
        name="flow_ratio", universe_min=0.0, universe_max=3.0,
        mfs={
            "LOW":  TrapezoidalMF(0.0, 0.0, 0.4, 0.8),
            "MID":  TriangularMF(0.5, 1.0, 1.5),
            "HIGH": TrapezoidalMF(1.2, 1.7, 3.0, 3.0),
        },
    )
    width = FuzzyVariable(
        name="width_ratio", universe_min=0.0, universe_max=2.0,
        mfs={
            "NARROW": TrapezoidalMF(0.0, 0.0, 0.2, 0.5),
            "WIDE":   TrapezoidalMF(0.3, 0.6, 2.0, 2.0),
        },
    )
    alert = FuzzyVariable(
        name="alert_level", universe_min=0.0, universe_max=4.0,
        mfs={
            "GREEN":  TrapezoidalMF(0.0, 0.0, 0.5, 1.2),
            "YELLOW": TriangularMF(0.8, 1.5, 2.2),
            "ORANGE": TriangularMF(1.8, 2.5, 3.2),
            "RED":    TrapezoidalMF(2.8, 3.5, 4.0, 4.0),
        },
    )
    rules = [
        MamdaniRule({"flow_ratio": "HIGH", "width_ratio": "NARROW"},
                    ("alert_level", "RED")),
        MamdaniRule({"flow_ratio": "HIGH", "width_ratio": "WIDE"},
                    ("alert_level", "ORANGE")),
        MamdaniRule({"flow_ratio": "MID", "width_ratio": "NARROW"},
                    ("alert_level", "YELLOW")),
        MamdaniRule({"flow_ratio": "MID", "width_ratio": "WIDE"},
                    ("alert_level", "ORANGE")),
        MamdaniRule({"flow_ratio": "LOW"}, ("alert_level", "GREEN")),
    ]
    return MamdaniFIS(inputs={"flow_ratio": flow, "width_ratio": width},
                      output=alert, rules=rules)


CATEGORIES = ("GREEN", "YELLOW", "ORANGE", "RED")
_THRESHOLDS = np.array([GREEN_MAX, YELLOW_MAX, ORANGE_MAX])


def score_to_category(score: float | np.ndarray) -> np.ndarray:
    """Discretise the crisp alert score in ``[0, 4]`` to a category array.

    Returns a NumPy array of strings from :data:`CATEGORIES`.
    Vectorised — works for scalars and arrays alike.
    """
    idx = np.searchsorted(_THRESHOLDS, np.asarray(score, dtype=float),
                          side="right")
    idx = np.clip(idx, 0, 3)
    return np.array(CATEGORIES)[idx]


_CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}


def category_to_index(category) -> np.ndarray:
    """Inverse of :func:`score_to_category` — GREEN → 0, RED → 3.

    Accepts a scalar string or an array-like of strings and returns an
    integer array of the same shape. Uses a plain dict lookup rather
    than ``searchsorted`` because :data:`CATEGORIES` is ordered by
    severity, not alphabetically.
    """
    arr = np.asarray(category)
    if arr.ndim == 0:
        return np.asarray(_CATEGORY_TO_IDX[str(arr)])
    flat = np.array([_CATEGORY_TO_IDX[str(c)] for c in arr.flat],
                    dtype=int)
    return flat.reshape(arr.shape)
