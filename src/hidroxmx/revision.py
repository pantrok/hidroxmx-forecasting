"""Shared plumbing of the JOH revision analyses (see docs/revision_log.md).

Everything here serves ``scripts/3*_rj*.py``: the frozen snapshot, reading R2
through curl, and rebuilding one fold exactly as ``scripts/15_conformal_uq.py``
does. Predictions were not stored for the reported runs, so they are regenerated
from each fold's ``best.ckpt`` and checked against its manifest.
"""
from __future__ import annotations

import json
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .alert import category_to_index, score_to_category
from .data.features import (
    CLIMA_COLS_RAW,
    TARGET_COL,
    TARGET_LOG_COL,
    build_features,
    reindex_daily,
    standardise,
    target_stats_m3s,
)
from .data.splits import TemporalSplit
from .data.streams import LocalRoots, load_multi_station_daily, load_station_daily
from .data.windows import WindowSpec, collect_windows
from .eval import BCaResult, paired_bca
from .io.r2 import R2Client

HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 7)
LOOKBACK = 90
ALPHA = 0.1
BOOT_SEED = 20260721
N_BOOT = 10_000
THRESHOLD = 0.05
COST_LOSS = 0.2
CUTOFFS: tuple[str, ...] = ("YELLOW", "ORANGE", "RED")
MIN_TRAIN_ROWS = 500

BASIN_SLUGS: dict[str, str] = {
    "Alto Lerma": "alto-lerma",
    "Valle de México": "valle-de-mexico",
    "Bajo Pánuco": "bajo-panuco",
    "Medio Balsas": "medio-balsas",
}

# Files and bytes of the v2026.06 snapshot, from dvc.lock at d5e0544 of hidroxai-mx.
SNAPSHOT_SERIES: dict[str, tuple[int, int]] = {
    "series_hidrometricas.parquet": (197, 142_891_675),
    "series_climatologicas.parquet": (150, 836_473_785),
}
SNAPSHOT_CATALOGS = (
    "estaciones_seleccionadas_hidrometricas.csv",
    "estaciones_seleccionadas_climatologicas.csv",
)

WATCHED = (("Bajo Pánuco", 3), ("Bajo Pánuco", 5), ("Bajo Pánuco", 7), ("Alto Lerma", 5))
CLEARED = "Cleared, BCa-robust"
ENCOURAGING = "Encouraging but under-powered signal"
NOT_CLEARED = "Not cleared"
NOT_DEFINED = "Not defined"
PRIMARY_ESTIMATOR = "difference_of_medians"
SECONDARY_ESTIMATOR = "median_of_differences"

TYPE_PREREGISTERED = "revisión (preregistrado, intervalo alternativo)"
TYPE_DIAGNOSTIC = "revisión (diagnóstico)"


# --------------------------------------------------------------------------- #
# Frozen snapshot
# --------------------------------------------------------------------------- #
def extract_snapshot(zip_path: Path, root: Path) -> LocalRoots:
    """Unpack the series and catalogs of the v2026.06 archive under ``root``."""
    roots = LocalRoots(root)
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.endswith("/") or not name.startswith("processed/"):
                continue
            relative = name.split("processed/", 1)[1]
            wanted = (relative.split("/", 1)[0] in SNAPSHOT_SERIES
                      or relative in SNAPSHOT_CATALOGS or relative.startswith("cuencas/"))
            target = roots.processed / relative
            if wanted and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    return roots


def check_snapshot(roots: LocalRoots) -> dict[str, dict[str, int]]:
    """Raise unless the series on disk match the frozen file counts and bytes."""
    found: dict[str, dict[str, int]] = {}
    for folder, (n_expected, bytes_expected) in SNAPSHOT_SERIES.items():
        files = [p for p in (roots.processed / folder).rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files)
        found[folder] = {"files": len(files), "bytes": size}
        if (len(files), size) != (n_expected, bytes_expected):
            raise SystemExit(
                f"snapshot mismatch in {folder}: {len(files)} files, {size} bytes; "
                f"expected {n_expected} and {bytes_expected}"
            )
    return found


# --------------------------------------------------------------------------- #
# R2 through curl (botocore's TLS handshake fails on this platform)
# --------------------------------------------------------------------------- #
def list_r2_keys(r2: R2Client, prefix: str) -> list[str]:
    """List every key under ``prefix`` with presigned URLs and curl."""
    import subprocess

    client = r2._client()
    keys: list[str] = []
    token = None
    while True:
        params: dict[str, Any] = {"Bucket": r2.bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            params["ContinuationToken"] = token
        url = client.generate_presigned_url("list_objects_v2", Params=params, ExpiresIn=600)
        out = subprocess.run(["curl", "-fsS", "--max-time", "60", url],
                             capture_output=True, check=True).stdout
        root = ET.fromstring(out)
        ns = {"s": root.tag.split("}")[0].strip("{")}
        keys += [urllib.parse.unquote(c.find("s:Key", ns).text)
                 for c in root.findall("s:Contents", ns)]
        if root.find("s:IsTruncated", ns).text != "true":
            return keys
        token = root.find("s:NextContinuationToken", ns).text


def mirror_r2(r2: R2Client, keys: list[str], folder: Path) -> None:
    """Copy ``keys`` under ``folder``, keeping their path below ``paper2/``."""
    for key in keys:
        target = folder / key.split("/", 1)[1]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            r2._curl_get(key, target)


def read_manifest(mirror: Path, run_id: str, clave: str) -> dict:
    return json.loads((mirror / "runs" / run_id / clave / "manifest.json")
                      .read_text(encoding="utf-8"))


def folds_of(mirror: Path, run_id: str) -> list[str]:
    return sorted(p.parent.name for p in (mirror / "runs" / run_id).glob("*/manifest.json"))


# --------------------------------------------------------------------------- #
# One fold, as in scripts/15_conformal_uq.py
# --------------------------------------------------------------------------- #
def neighbours_of(row: pd.Series) -> list[str]:
    text = str(row.get("vecinos_clima", "") or "")
    return [v.strip() for v in text.split(",") if v.strip()]


def load_target(roots: LocalRoots, clave: str) -> pd.DataFrame:
    df = load_station_daily(
        None, "", clave=clave,
        columns=["clave_estacion", "fecha", TARGET_COL, "nivel_m", "calidad"],
        years=range(2010, 2026), local_path=roots.series_hidro,
    )
    return reindex_daily(df)


def load_climate(roots: LocalRoots, neighbours: list[str]) -> pd.DataFrame:
    df = load_multi_station_daily(
        None, "", claves=neighbours, columns=["clave_estacion", "fecha", *CLIMA_COLS_RAW],
        years=range(2010, 2026), local_path=roots.series_clima,
    )
    if df.empty:
        return pd.DataFrame(columns=["fecha", *CLIMA_COLS_RAW])
    return reindex_daily(df.groupby("fecha", as_index=False)[list(CLIMA_COLS_RAW)].mean())


@dataclass(slots=True)
class FoldData:
    clave: str
    feature_cols: list[str]
    mu: float
    sigma: float
    target_stats: dict
    train_q: dict[str, float]
    windows_val: dict[str, np.ndarray]
    windows_test: dict[str, np.ndarray]
    raw: pd.Series                       # discharge floored at 0, not clipped above


def train_quantiles(series: pd.DataFrame, mask_train: pd.Series) -> dict[str, float]:
    q = series.loc[mask_train, TARGET_COL].dropna().to_numpy(dtype=np.float64)
    return {f"q{int(p * 100)}": float(np.quantile(q, p)) for p in (0.50, 0.90, 0.95, 0.99)}


def prepare_fold(roots: LocalRoots, row: pd.Series, split: TemporalSplit) -> FoldData | None:
    clave = str(row["clave"])
    target_raw = load_target(roots, clave)
    if target_raw.empty:
        return None
    clima = load_climate(roots, neighbours_of(row))
    series, feature_cols, clip_upper = build_features(
        target_raw, clima, use_clima=True,
        train_mask=split.mask(target_raw["fecha"], "train"), clip_upper=None,
    )
    mask_train = split.mask(series["fecha"], "train")
    if int(mask_train.sum()) < MIN_TRAIN_ROWS:
        return None
    series_std, stats = standardise(series, mask_train, feature_cols, TARGET_LOG_COL)
    spec = WindowSpec(lookback=LOOKBACK, horizons=HORIZONS, feature_cols=feature_cols,
                      target_col=TARGET_LOG_COL)
    mu, sigma = stats[TARGET_LOG_COL]
    return FoldData(
        clave=clave, feature_cols=feature_cols, mu=mu, sigma=sigma,
        target_stats=target_stats_m3s(series, mask_train, clip_upper),
        train_q=train_quantiles(series, mask_train),
        windows_val=collect_windows(series_std.loc[split.mask(series["fecha"], "val")], spec),
        windows_test=collect_windows(series_std.loc[split.mask(series["fecha"], "test")], spec),
        raw=target_raw.set_index("fecha")[TARGET_COL].clip(lower=0.0),
    )


def raw_at(raw: pd.Series, dates: np.ndarray) -> np.ndarray:
    """Raw discharge on an ``[N, H]`` array of target dates."""
    flat = raw.reindex(pd.to_datetime(dates.ravel())).to_numpy(dtype=float)
    return flat.reshape(dates.shape)


def alert_levels(fis, yhat: np.ndarray, width: np.ndarray, q95: float) -> np.ndarray:
    """Alert index per window and horizon, as in scripts/17_evaluate_alerts.py.

    ``width`` has one value per horizon, or one per window and horizon.
    """
    flow = np.clip(yhat / max(q95, 1e-6), 0.0, 3.0)
    wide = np.clip(np.broadcast_to(width, yhat.shape) / max(q95, 1e-6), 0.0, 2.0)
    scores = np.column_stack([
        fis.infer_batch({"flow_ratio": flow[:, j], "width_ratio": wide[:, j]})
        for j in range(flow.shape[1])
    ])
    return category_to_index(score_to_category(scores))


# --------------------------------------------------------------------------- #
# Comparison of alert variants
# --------------------------------------------------------------------------- #
def classify(res: BCaResult) -> str:
    """Class of a cell against the +0.05 criterion of docs/revision_log.md."""
    if not res.defined:
        return NOT_DEFINED
    if res.bca_low >= THRESHOLD:
        return CLEARED
    return ENCOURAGING if res.estimate > THRESHOLD else NOT_CLEARED


def bca(a: np.ndarray, b: np.ndarray, estimator: str = PRIMARY_ESTIMATOR) -> BCaResult:
    return paired_bca(a, b, estimator=estimator, n_boot=N_BOOT, ci=0.95, seed=BOOT_SEED)


def sign_test(delta: np.ndarray, reference: float) -> tuple[int, int, float]:
    """Exact one-sided sign test; folds exactly at the reference are dropped."""
    shifted = delta[np.isfinite(delta)] - reference
    shifted = shifted[np.abs(shifted) > 1e-12]
    wins = int((shifted > 0).sum())
    n = int(shifted.size)
    p = float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else float("nan")
    return wins, n, p


def leave_one_out(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Smallest and largest difference of medians after dropping one fold."""
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2:
        return float("nan"), float("nan")
    values = [np.median(np.delete(a, i)) - np.median(np.delete(b, i)) for i in range(a.size)]
    return float(min(values)), float(max(values))


# --------------------------------------------------------------------------- #
# Numbers file
# --------------------------------------------------------------------------- #
def update_numbers(path: Path, block: str, kind: str, source: str, payload: dict) -> None:
    """Replace one block of the numbers file, stamped with its source and type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[block] = {"fuente": source, "tipo": kind, **payload}
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def clean(value: Any) -> Any:
    """Turn numpy scalars and non-finite floats into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value
