"""Sliding-window dataset over a station's daily series.

Design decisions:

- **Iterable** over materialised: we deliberately do not stack the whole
  window tensor in RAM. The dataset can be exhausted many times per epoch
  by re-iterating; consumers plug it into a ``DataLoader`` with a batch
  sampler if they want minibatching.
- **NaN awareness**: a window is skipped entirely if any lookback feature
  or any horizon target is missing. Downstream models never see NaNs.
- **Torch-optional**: the dataset works with a raw Python iterator API so
  the metrics registry can smoke-test it without importing torch. If
  torch is available we expose a subclass of ``torch.utils.data.IterableDataset``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class WindowSpec:
    lookback: int                   # number of past days fed to the encoder
    horizons: Sequence[int]         # forecast horizons in days (e.g. [1, 2, 3, 5, 7])
    feature_cols: Sequence[str]     # columns used as encoder features (x_{t-L..t})
    target_col: str                 # column predicted at t + h days

    @property
    def horizon_max(self) -> int:
        return int(max(self.horizons))


def iter_windows(series: pd.DataFrame, spec: WindowSpec) -> Iterator[dict]:
    """Yield windows of a single-station DataFrame as dicts.

    The DataFrame must already be sorted by ``fecha`` (ascending) with a
    complete daily index; the caller is responsible for reindexing.
    """
    if len(series) < spec.lookback + spec.horizon_max:
        return
    feats = series[list(spec.feature_cols)].to_numpy(dtype=np.float32)
    tgt = series[spec.target_col].to_numpy(dtype=np.float32)
    dates = series["fecha"].to_numpy()

    L, hmax = spec.lookback, spec.horizon_max
    horizons = np.asarray(spec.horizons, dtype=np.int64)
    for t in range(L - 1, len(series) - hmax):
        window = feats[t - L + 1: t + 1]           # shape [L, F]
        targets = tgt[t + horizons]                # shape [H]
        if np.isnan(window).any() or np.isnan(targets).any():
            continue
        yield {
            "x": window,
            "y": targets,
            "t0": dates[t],
            "targets_at": dates[t + horizons],
        }


def collect_windows(series: pd.DataFrame, spec: WindowSpec) -> dict[str, np.ndarray]:
    """Materialise every window into stacked arrays (helper for small experiments)."""
    xs, ys, t0s, taus = [], [], [], []
    for win in iter_windows(series, spec):
        xs.append(win["x"])
        ys.append(win["y"])
        t0s.append(win["t0"])
        taus.append(win["targets_at"])
    if not xs:
        return {"x": np.empty((0, spec.lookback, len(spec.feature_cols)), dtype=np.float32),
                "y": np.empty((0, len(spec.horizons)), dtype=np.float32),
                "t0": np.array([], dtype="datetime64[ns]"),
                "targets_at": np.empty((0, len(spec.horizons)), dtype="datetime64[ns]")}
    return {
        "x": np.stack(xs, axis=0),
        "y": np.stack(ys, axis=0),
        "t0": np.array(t0s),
        "targets_at": np.stack(taus, axis=0),
    }


try:
    import torch
    from torch.utils.data import IterableDataset

    class WindowIterableDataset(IterableDataset):
        """Torch-flavoured wrapper of :func:`iter_windows` for a single station."""

        def __init__(self, series: pd.DataFrame, spec: WindowSpec):
            super().__init__()
            self.series = series
            self.spec = spec

        def __iter__(self):
            for win in iter_windows(self.series, self.spec):
                yield (torch.from_numpy(win["x"]),
                       torch.from_numpy(win["y"]))
except ImportError:  # torch not installed in this environment
    WindowIterableDataset = None  # type: ignore
