"""Temporal and spatial folds (see docs/experiment-spec.md §2).

Temporal split (frozen, reported verbatim in the paper):

- train : 2010-01-01 → 2020-12-31
- val   : 2021-01-01 → 2022-12-31
- test  : 2023-01-01 → 2025-12-31

The 2024 Cutzamala drought and the October-2025 floods fall inside the test
window, providing built-in OOD stress. Never leak test into calibration.

Spatial splits:

- PUB (pseudo-ungauged basin): leave-one-station-out inside a basin. Fold
  ``k`` withholds station ``k`` and trains on the rest of the basin's stations.
- PUR (pure ungauged region): withhold every station whose hydrological
  region code matches one of the target regions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Frozen temporal split
# --------------------------------------------------------------------------- #
DEFAULT_SPLITS = {
    "train": (pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31")),
    "val":   (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")),
    "test":  (pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31")),
}


@dataclass(slots=True, frozen=True)
class TemporalSplit:
    train: tuple[pd.Timestamp, pd.Timestamp]
    val:   tuple[pd.Timestamp, pd.Timestamp]
    test:  tuple[pd.Timestamp, pd.Timestamp]

    @classmethod
    def default(cls) -> "TemporalSplit":
        return cls(**DEFAULT_SPLITS)

    def mask(self, dates: pd.Series, fold: str) -> pd.Series:
        start, end = getattr(self, fold)
        return (dates >= start) & (dates <= end)

    def apply(self, df: pd.DataFrame, *, date_col: str = "fecha") -> dict[str, pd.DataFrame]:
        return {fold: df.loc[self.mask(df[date_col], fold)].copy()
                for fold in ("train", "val", "test")}


# --------------------------------------------------------------------------- #
# Extreme-event stratification
# --------------------------------------------------------------------------- #
def extreme_mask(df: pd.DataFrame, target: str,
                 quantile: float = 0.95,
                 per_station: bool = True,
                 station_col: str = "clave_estacion") -> pd.Series:
    """Boolean mask of rows whose ``target`` is above the per-station q-quantile."""
    if per_station and station_col in df.columns:
        thr = df.groupby(station_col)[target].transform(lambda s: s.quantile(quantile))
    else:
        thr = df[target].quantile(quantile)
    return df[target] >= thr


# --------------------------------------------------------------------------- #
# Spatial folds
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SpatialFold:
    name: str            # human-readable identifier for logs / manifests
    holdout: list[str]   # station keys held out (never used for training)
    train:   list[str]   # station keys used for training


def pub_leave_one_out(stations: pd.DataFrame, *, basin_col: str = "cuenca",
                       key_col: str = "clave") -> Iterator[SpatialFold]:
    """Pseudo-ungauged folds inside every basin."""
    for basin, sub in stations.groupby(basin_col):
        keys = sub[key_col].tolist()
        for key in keys:
            train_keys = [k for k in keys if k != key]
            if not train_keys:
                continue
            yield SpatialFold(name=f"pub/{basin}/{key}", holdout=[key], train=train_keys)


def pur_by_region(stations: pd.DataFrame, *, region_col: str = "region_hidrologica",
                  key_col: str = "clave") -> Iterator[SpatialFold]:
    """Pure ungauged region folds — hold out every station of one region."""
    stations = stations.copy()
    stations[region_col] = stations[region_col].astype(str)
    for region, sub in stations.groupby(region_col):
        keys_out = sub[key_col].tolist()
        keys_in = stations.loc[stations[region_col] != region, key_col].tolist()
        if not keys_in:
            continue
        yield SpatialFold(name=f"pur/{region}", holdout=keys_out, train=keys_in)
