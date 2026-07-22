"""Static-attribute vectors for donor-similarity (S-ATTR).

Complements the hydrological-signature route (:mod:`hidroxmx.transfer.signatures`)
with a fundamentally different information source: **physical, geographic
and administrative** attributes of the station catchment, rather than
behavioural signatures derived from streamflow. Two catchments that
share latitude, elevation and hydrologic-region membership are, by
climatological priors, more likely to share dynamics — even before we
look at their flow records (Newman et al., 2015; Addor et al., 2018).

The attributes exposed here come from the CONAGUA selected-stations
manifest that this project ships as :file:`estaciones_seleccionadas_hidrometricas.csv`:

- ``latitud``, ``longitud`` — geographic centroid of the gauge.
- ``altitud``      — elevation at the gauge (proxy for
  temperature and precipitation regime via lapse rates).
- ``region_hidrologica`` — CONAGUA hydrologic-region code (37 regions,
  each a coherent hydrographic unit; sharing a region is a strong
  categorical prior).

Extra attributes (drainage area, mean slope, land cover) are not in
the CONAGUA catalogue and would require joining with INEGI or
HydroBASINS. They are TODO for a Milestone-4b extension when Path A
needs an even stronger physical prior.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_ATTRIBUTE_KEYS: tuple[str, ...] = (
    "latitud",
    "longitud",
    "altitud",
    "region_hidrologica",
)


def _to_float(value) -> float:
    """Best-effort numeric coercion. Returns NaN when the column is a
    string label that cannot be parsed, so downstream standardisation
    silently drops the coordinate."""
    try:
        if value is None:
            return float("nan")
        if isinstance(value, str) and not value.strip():
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def extract_attributes(station_row: pd.Series) -> dict[str, float]:
    """Pull the static attributes for one station off its manifest row.

    ``station_row`` is one row of :func:`hidroxmx.data.streams.load_selected_stations`.
    Missing columns / non-numeric cells become NaN so the standardisation
    step in :mod:`hidroxmx.transfer.similarity` treats them as a
    neutral (zero-after-scaling) coordinate rather than crashing.
    """
    return {key: _to_float(station_row.get(key)) for key in DEFAULT_ATTRIBUTE_KEYS}


def attribute_vector(attrs: dict[str, float],
                     keys: Sequence[str] = DEFAULT_ATTRIBUTE_KEYS) -> np.ndarray:
    """Return the attribute dict as an ordered numeric vector."""
    return np.array([attrs.get(k, float("nan")) for k in keys], dtype=float)
