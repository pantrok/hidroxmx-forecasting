"""Feature engineering shared by every forecaster stage.

Everything in this module is single-station: build a modelling table for
one clave from its raw hydrometric series and (optionally) the mean of
its climatological neighbours. Multi-station pipelines call these
helpers once per station and concatenate the resulting windows.

The transforms applied here are the same as those documented in stage
11:

- Upper-clip streamflow at the training-window ``TARGET_UPPER_CLIP_QUANTILE``
  to neutralise CONAGUA capture-error spikes.
- ``log1p`` on the streamflow feature family (target + lags + moving
  averages) so low- and high-flow errors weigh comparably.
- Lags at {1, 3, 7, 14, 30} days and moving averages at {7, 30} days
  (shifted so the average excludes the current day and does not leak).
- Optional climate features: mean of the neighbouring climatologic
  stations' precip_mm / tmax_c / tmin_c, each with a 7-day trailing MA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COL = "gasto_medio_m3s"
TARGET_LOG_COL = f"{TARGET_COL}_log"
LAG_DAYS: tuple[int, ...] = (1, 3, 7, 14, 30)
MA_DAYS: tuple[int, ...] = (7, 30)
CLIMA_COLS_RAW: tuple[str, ...] = ("precip_mm", "tmax_c", "tmin_c")
TARGET_UPPER_CLIP_QUANTILE = 0.999


def reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` reindexed on a contiguous daily calendar.

    Duplicate ``fecha`` rows are collapsed with ``keep="last"`` before
    reindexing — SIH occasionally re-issues a corrected observation for
    the same day, and the later record is treated as authoritative.
    """
    if df.empty:
        return df
    df = df.sort_values("fecha").drop_duplicates(subset="fecha", keep="last")
    df = df.set_index("fecha")
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    df.index.name = "fecha"
    return df.reset_index()


def build_features(target: pd.DataFrame,
                   clima: pd.DataFrame,
                   *,
                   use_clima: bool,
                   train_mask: pd.Series,
                   clip_upper: float | None = None,
                   ) -> tuple[pd.DataFrame, list[str], float | None]:
    """Build the modelling table with log1p-transformed streamflow features.

    Parameters mirror the stage-11 driver. If ``clip_upper`` is ``None``,
    it is computed from the training portion of ``target[TARGET_COL]``.
    The value used is returned so downstream callers can log it or share
    it across folds.
    """
    df = target[["fecha", TARGET_COL, "nivel_m"]].copy()

    if clip_upper is None:
        train_target = df.loc[train_mask, TARGET_COL].dropna()
        if len(train_target):
            clip_upper = float(train_target.clip(lower=0.0).quantile(TARGET_UPPER_CLIP_QUANTILE))
    if clip_upper is not None:
        df[TARGET_COL] = df[TARGET_COL].clip(lower=0.0, upper=clip_upper)

    df[TARGET_LOG_COL] = np.log1p(df[TARGET_COL])
    for lag in LAG_DAYS:
        df[f"{TARGET_LOG_COL}_lag{lag}"] = df[TARGET_LOG_COL].shift(lag)
    for w in MA_DAYS:
        df[f"{TARGET_LOG_COL}_ma{w}"] = df[TARGET_LOG_COL].shift(1).rolling(w, min_periods=w).mean()

    feature_cols: list[str] = [
        TARGET_LOG_COL,
        *(f"{TARGET_LOG_COL}_lag{lag}" for lag in LAG_DAYS),
        *(f"{TARGET_LOG_COL}_ma{w}" for w in MA_DAYS),
    ]

    if use_clima and not clima.empty:
        df = df.merge(clima, on="fecha", how="left")
        for col in CLIMA_COLS_RAW:
            if col in df.columns:
                df[col] = df[col].astype(float)
                df[f"{col}_ma7"] = df[col].shift(1).rolling(7, min_periods=7).mean()
                feature_cols.extend([col, f"{col}_ma7"])
    return df, feature_cols, clip_upper


def standardise(series: pd.DataFrame, train_mask: pd.Series,
                feature_cols: list[str], target_col: str
                ) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Z-score every feature and the target using train-only mean/std."""
    stats: dict[str, tuple[float, float]] = {}
    out = series.copy()
    cols_to_norm = list(dict.fromkeys([*feature_cols, target_col]))
    for col in cols_to_norm:
        mu = float(out.loc[train_mask, col].mean())
        sigma = float(out.loc[train_mask, col].std(ddof=0)) or 1.0
        out[col] = (out[col] - mu) / sigma
        stats[col] = (mu, sigma)
    return out, stats


def target_stats_m3s(series: pd.DataFrame, train_mask: pd.Series,
                     clip_upper: float | None) -> dict[str, float | None]:
    """Descriptive stats of the streamflow target on the training window (in m³/s)."""
    train_target = series.loc[train_mask, TARGET_COL].dropna().to_numpy(dtype=np.float64)
    if train_target.size == 0:
        return {"train_mean_m3s": None, "train_std_m3s": None,
                "train_min_m3s": None, "train_p50_m3s": None,
                "train_p95_m3s": None, "train_max_m3s": None,
                "clip_upper_m3s": clip_upper}
    return {
        "train_mean_m3s": float(train_target.mean()),
        "train_std_m3s": float(train_target.std(ddof=0)),
        "train_min_m3s": float(train_target.min()),
        "train_p50_m3s": float(np.median(train_target)),
        "train_p95_m3s": float(np.quantile(train_target, 0.95)),
        "train_max_m3s": float(train_target.max()),
        "clip_upper_m3s": float(clip_upper) if clip_upper is not None else None,
    }
