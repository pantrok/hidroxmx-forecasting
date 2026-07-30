"""Tests for the shared feature-engineering module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hidroxmx.data.features import (
    CLIMA_COLS_RAW,
    TARGET_COL,
    TARGET_LOG_COL,
    build_features,
    reindex_daily,
    standardise,
    target_stats_m3s,
)


def _target(days: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=days, freq="D")
    base = 10 + 5 * np.sin(np.arange(days) * 2 * np.pi / 365)
    noise = rng.normal(0, 1, days)
    return pd.DataFrame({
        "fecha": idx,
        "clave_estacion": "TESTA",
        TARGET_COL: (base + noise).clip(min=0),
        "nivel_m": 1.0 + 0.1 * noise,
    })


def _clima(days: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "fecha": idx,
        "precip_mm": rng.exponential(2.0, days),
        "tmax_c": 25 + 5 * np.sin(np.arange(days) * 2 * np.pi / 365) + rng.normal(0, 2, days),
        "tmin_c": 10 + 5 * np.sin(np.arange(days) * 2 * np.pi / 365) + rng.normal(0, 2, days),
    })


def test_reindex_daily_fills_gaps():
    df = pd.DataFrame({"fecha": pd.to_datetime(["2010-01-01", "2010-01-05"]),
                       "gasto_medio_m3s": [1.0, 2.0]})
    out = reindex_daily(df)
    assert len(out) == 5
    assert out["fecha"].dt.day.tolist() == [1, 2, 3, 4, 5]
    assert pd.isna(out.loc[out["fecha"] == pd.Timestamp("2010-01-03"), "gasto_medio_m3s"]).all()


def test_build_features_no_clima_produces_lags_and_log_column():
    tgt = _target()
    train_mask = tgt["fecha"] < pd.Timestamp("2010-11-01")
    series, feats, clip_upper = build_features(tgt, pd.DataFrame(), use_clima=False,
                                                train_mask=train_mask, clip_upper=None)
    assert TARGET_LOG_COL in series.columns
    assert f"{TARGET_LOG_COL}_lag30" in series.columns
    assert f"{TARGET_LOG_COL}_ma7" in series.columns
    # First 30 rows must have NaN in lag30 (shift 30 days).
    assert series[f"{TARGET_LOG_COL}_lag30"].iloc[:30].isna().all()
    assert TARGET_LOG_COL in feats
    assert clip_upper is not None
    assert clip_upper > 0


def test_build_features_with_clima_adds_climate_columns():
    tgt = _target()
    clima = _clima()
    train_mask = tgt["fecha"] < pd.Timestamp("2010-11-01")
    _, feats, _ = build_features(tgt, clima, use_clima=True,
                                 train_mask=train_mask, clip_upper=None)
    for col in CLIMA_COLS_RAW:
        assert col in feats
        assert f"{col}_ma7" in feats


def test_build_features_honours_supplied_clip_upper():
    tgt = _target()
    tgt.loc[10, TARGET_COL] = 999.0  # outlier
    train_mask = tgt["fecha"] < pd.Timestamp("2010-11-01")
    series, _, clip_upper = build_features(tgt, pd.DataFrame(), use_clima=False,
                                            train_mask=train_mask, clip_upper=25.0)
    assert clip_upper == 25.0
    assert float(series[TARGET_COL].max()) <= 25.0


def test_standardise_uses_train_only_stats():
    tgt = _target()
    train_mask = tgt["fecha"] < pd.Timestamp("2010-11-01")
    series, feats, _ = build_features(tgt, pd.DataFrame(), use_clima=False,
                                       train_mask=train_mask, clip_upper=None)
    # Drop rows with NaN so mean/std are well-defined on the train slice.
    valid = series.dropna(subset=[TARGET_LOG_COL, *feats])
    train_mask2 = valid["fecha"] < pd.Timestamp("2010-11-01")
    out, stats = standardise(valid, train_mask2, feats, TARGET_LOG_COL)
    train_slice = out.loc[train_mask2, TARGET_LOG_COL]
    assert abs(float(train_slice.mean())) < 1e-8
    assert abs(float(train_slice.std(ddof=0)) - 1.0) < 1e-8
    assert TARGET_LOG_COL in stats
    mu, sigma = stats[TARGET_LOG_COL]
    assert sigma > 0


def test_target_stats_m3s_reports_expected_summaries():
    tgt = _target()
    train_mask = tgt["fecha"] < pd.Timestamp("2010-11-01")
    series, _, clip_upper = build_features(tgt, pd.DataFrame(), use_clima=False,
                                            train_mask=train_mask, clip_upper=None)
    ts = target_stats_m3s(series, train_mask, clip_upper)
    assert ts["train_mean_m3s"] is not None
    assert ts["train_p50_m3s"] > 0
    assert ts["train_p95_m3s"] > ts["train_p50_m3s"]
    assert ts["clip_upper_m3s"] == clip_upper
