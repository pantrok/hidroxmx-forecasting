"""Tests for the data layer added in Milestone 2 (splits and windows).

Streams tests hit R2 and are exercised in the smoke script of Stage 11 rather
than in the unit suite; here we only test what runs without network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hidroxmx.data.splits import (
    DEFAULT_SPLITS,
    SpatialFold,
    TemporalSplit,
    extreme_mask,
    pub_leave_one_out,
    pur_by_region,
)
from hidroxmx.data.windows import WindowSpec, collect_windows, iter_windows


# --------------------------------------------------------------------------- #
# Temporal split
# --------------------------------------------------------------------------- #
def _series(days: int) -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "fecha": idx,
        "clave_estacion": "STATION",
        "gasto_medio_m3s": np.arange(days, dtype=float),
    })


def test_default_temporal_split_boundaries_are_stable():
    split = TemporalSplit.default()
    assert split.train == DEFAULT_SPLITS["train"]
    assert split.val == DEFAULT_SPLITS["val"]
    assert split.test == DEFAULT_SPLITS["test"]


def test_apply_partitions_disjointly_and_covers_all_years():
    df = _series(days=int((pd.Timestamp("2025-12-31") - pd.Timestamp("2010-01-01")).days) + 1)
    split = TemporalSplit.default()
    parts = split.apply(df)
    total = sum(len(v) for v in parts.values())
    assert total == len(df)  # no overlap, no leftover
    assert parts["train"]["fecha"].max() < parts["val"]["fecha"].min()
    assert parts["val"]["fecha"].max() < parts["test"]["fecha"].min()


def test_extreme_mask_uses_per_station_quantile_when_column_present():
    df = pd.DataFrame({
        "clave_estacion": ["A"] * 100 + ["B"] * 100,
        "gasto_medio_m3s": np.concatenate([np.arange(100), np.arange(1000, 1100)]),
    })
    mask = extreme_mask(df, "gasto_medio_m3s", quantile=0.9)
    assert mask.iloc[:100].sum() == mask.iloc[100:].sum()  # symmetric per station


# --------------------------------------------------------------------------- #
# Spatial folds
# --------------------------------------------------------------------------- #
def _stations():
    return pd.DataFrame({
        "clave": ["a1", "a2", "a3", "b1", "b2"],
        "cuenca": ["Alpha", "Alpha", "Alpha", "Beta", "Beta"],
        "region_hidrologica": [12, 12, 12, 18, 18],
    })


def test_pub_leave_one_out_yields_one_fold_per_station_per_basin():
    folds = list(pub_leave_one_out(_stations()))
    assert len(folds) == 5
    for fold in folds:
        assert isinstance(fold, SpatialFold)
        assert len(fold.holdout) == 1
        assert fold.holdout[0] not in fold.train


def test_pur_by_region_holds_out_full_regions():
    folds = list(pur_by_region(_stations()))
    assert len(folds) == 2
    for fold in folds:
        assert set(fold.holdout).isdisjoint(fold.train)


# --------------------------------------------------------------------------- #
# Sliding windows
# --------------------------------------------------------------------------- #
def test_iter_windows_yields_expected_count_and_shapes():
    df = _series(days=200)
    spec = WindowSpec(lookback=30, horizons=[1, 3, 7],
                      feature_cols=["gasto_medio_m3s"], target_col="gasto_medio_m3s")
    wins = list(iter_windows(df, spec))
    # iter_windows uses `for t in range(L - 1, len(series) - hmax)` →
    # (len − hmax) − (L − 1) = 200 − 7 − 29 = 164 windows.
    assert len(wins) == len(df) - spec.horizon_max - (spec.lookback - 1)
    assert wins[0]["x"].shape == (spec.lookback, len(spec.feature_cols))
    assert wins[0]["y"].shape == (len(spec.horizons),)


def test_iter_windows_skips_windows_with_nan():
    df = _series(days=200)
    df.loc[50, "gasto_medio_m3s"] = np.nan
    spec = WindowSpec(lookback=10, horizons=[1],
                      feature_cols=["gasto_medio_m3s"], target_col="gasto_medio_m3s")
    wins = list(iter_windows(df, spec))
    # windows that contain index 50 in x or y are skipped
    assert len(wins) < 200 - 10


def test_collect_windows_stacks_arrays():
    df = _series(days=100)
    spec = WindowSpec(lookback=10, horizons=[1, 2],
                      feature_cols=["gasto_medio_m3s"], target_col="gasto_medio_m3s")
    stacked = collect_windows(df, spec)
    assert stacked["x"].ndim == 3
    assert stacked["y"].ndim == 2
    assert stacked["x"].shape[0] == stacked["y"].shape[0]


# --------------------------------------------------------------------------- #
# Model smoke test (only if torch is present)
# --------------------------------------------------------------------------- #
def test_forecaster_forward_shape_when_torch_available():
    torch = pytest.importorskip("torch")
    from hidroxmx.models.forecaster import LSTMEncDecConfig, LSTMEncoderDecoder

    cfg = LSTMEncDecConfig(input_dim=4, hidden_dim=16, num_layers=1, horizons=3)
    model = LSTMEncoderDecoder(cfg)
    x = torch.zeros(8, 30, 4)
    y = model(x)
    assert y.shape == (8, 3)
