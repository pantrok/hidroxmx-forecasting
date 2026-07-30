"""Lazy readers for the Hidro-MX Parquet snapshot.

Two backends, one API:

- **Local** — reads the parquets and CSVs from a hidrated hidroxai-mx
  checkout on disk. Fast for iteration on Windows / Linux and mandatory
  while the dataset lives in DVC's content-addressable layout inside R2
  (i.e. files under ``dvcstore/{ab}/cdef…`` addressed by their MD5, not
  by their logical path).
- **R2 streaming** — same API, streamed from a *hierarchical* R2 prefix
  (defaults to ``R2_DATASET_PREFIX``). Use this when the dataset is
  re-published to R2 under a stable, logical layout (e.g. mirrored into
  a stable ``inputs/processed/…`` mirror) or in Colab once a small
  bootstrap fetches the snapshot outside DVC.

The backend is chosen at call time by :func:`resolve_source`:

1. Explicit ``local_root`` argument.
2. ``HIDROXAI_MX_ROOT`` environment variable (repo root).
3. Fallback: R2 streaming with the supplied :class:`R2Client`.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ..io.r2 import R2Client


DATASET_PREFIX_ENV = "R2_DATASET_PREFIX"
DEFAULT_DATASET_PREFIX = "dvcstore"
LOCAL_ROOT_ENV = "HIDROXAI_MX_ROOT"


@dataclass(slots=True)
class DatasetLayout:
    """Location of every deliverable of the ``hidroxai-mx`` snapshot inside R2.

    Paths are relative to the bucket root; ``prefix`` is the DVC store prefix
    (see ``conf/experiments/r2.yaml``). If the snapshot moves, override the
    ``prefix`` at construction time.
    """

    prefix: str = DEFAULT_DATASET_PREFIX

    @property
    def series_hidro_key(self) -> str:
        return f"{self.prefix}/processed/series_hidrometricas.parquet"

    @property
    def series_clima_key(self) -> str:
        return f"{self.prefix}/processed/series_climatologicas.parquet"

    @property
    def feature_table_key(self) -> str:
        return f"{self.prefix}/features/feature_table.parquet"

    def selected_hidro_key(self) -> str:
        return f"{self.prefix}/processed/estaciones_seleccionadas_hidrometricas.csv"

    def selected_clima_key(self) -> str:
        return f"{self.prefix}/processed/estaciones_seleccionadas_climatologicas.csv"


def layout_from_env() -> DatasetLayout:
    """Read the dataset prefix from the environment."""
    return DatasetLayout(prefix=os.environ.get(DATASET_PREFIX_ENV, DEFAULT_DATASET_PREFIX))


@dataclass(slots=True)
class LocalRoots:
    """Local mirror of the ``hidroxai-mx`` data folder."""

    repo_root: Path

    @property
    def processed(self) -> Path:
        return self.repo_root / "data" / "processed"

    @property
    def features(self) -> Path:
        return self.repo_root / "data" / "features"

    @property
    def series_hidro(self) -> Path:
        return self.processed / "series_hidrometricas.parquet"

    @property
    def series_clima(self) -> Path:
        return self.processed / "series_climatologicas.parquet"

    @property
    def feature_table(self) -> Path:
        return self.features / "feature_table.parquet"

    @property
    def selected_hidro_csv(self) -> Path:
        return self.processed / "estaciones_seleccionadas_hidrometricas.csv"

    @property
    def selected_clima_csv(self) -> Path:
        return self.processed / "estaciones_seleccionadas_climatologicas.csv"


def local_roots_from_env() -> LocalRoots | None:
    """Return a :class:`LocalRoots` if ``HIDROXAI_MX_ROOT`` is set to a folder that exists."""
    val = os.environ.get(LOCAL_ROOT_ENV)
    if not val:
        return None
    root = Path(val)
    return LocalRoots(root) if root.exists() else None


# --------------------------------------------------------------------------- #
# CSV manifests (station metadata)
# --------------------------------------------------------------------------- #
def load_selected_stations(r2: R2Client | None, layout: DatasetLayout, *,
                           kind: str = "hidro",
                           local: LocalRoots | None = None) -> pd.DataFrame:
    """Fetch the CSV of selected stations, preferring the local hidrated copy."""
    local = local or local_roots_from_env()
    if local is not None:
        path = local.selected_hidro_csv if kind == "hidro" else local.selected_clima_csv
        if path.exists():
            return pd.read_csv(path, dtype={"clave": str})
    if r2 is None:
        raise RuntimeError(
            "no local root and no R2 client provided; set HIDROXAI_MX_ROOT or pass r2="
        )
    key = layout.selected_hidro_key() if kind == "hidro" else layout.selected_clima_key()
    payload = r2.get_bytes(key)
    return pd.read_csv(io.BytesIO(payload), dtype={"clave": str})


# --------------------------------------------------------------------------- #
# Parquet time-series streaming
# --------------------------------------------------------------------------- #
def _open_r2_filesystem(r2: R2Client):
    """Return a ``pyarrow.fs.S3FileSystem`` configured for the R2 endpoint."""
    import pyarrow.fs as pafs

    endpoint = r2.endpoint_url
    if endpoint.startswith("https://"):
        endpoint = endpoint[len("https://"):]
    elif endpoint.startswith("http://"):
        endpoint = endpoint[len("http://"):]
    return pafs.S3FileSystem(
        endpoint_override=endpoint,
        access_key=r2.access_key_id,
        secret_key=r2.secret_access_key,
        region=r2.region,
        scheme="https",
    )


def open_partitioned_dataset(source, *, r2: R2Client | None = None):
    """Return a ``pyarrow.dataset.Dataset``, either from a local path or from R2.

    ``source`` may be:

    - a ``Path`` / ``str`` pointing at a local ``.parquet`` directory
      (Hive-partitioned by ``anio``); or
    - an ``str`` prefix inside the R2 bucket referenced by ``r2``.
    """
    import pyarrow.dataset as pads

    if isinstance(source, Path) or (isinstance(source, str) and os.sep in source):
        return pads.dataset(str(source), format="parquet", partitioning="hive")
    if r2 is None:
        raise RuntimeError("R2 source requested but no R2Client passed")
    fs = _open_r2_filesystem(r2)
    return pads.dataset(f"{r2.bucket}/{source}", filesystem=fs,
                        format="parquet", partitioning="hive")


def _resolve_dataset(local_path: Path | None, r2: R2Client | None,
                     r2_key: str):
    """Prefer the local Parquet directory when it exists; otherwise stream R2."""
    if local_path is not None and Path(local_path).exists():
        return open_partitioned_dataset(Path(local_path))
    if r2 is None:
        raise RuntimeError(
            f"local path {local_path!r} not found and no R2 client provided"
        )
    return open_partitioned_dataset(r2_key, r2=r2)


def load_station_daily(
    r2: R2Client | None,
    r2_key: str,
    *,
    clave: str,
    columns: Sequence[str] | None = None,
    years: Iterable[int] | None = None,
    local_path: Path | None = None,
) -> pd.DataFrame:
    """Load the daily record of a single station.

    Reads from ``local_path`` when it exists; otherwise streams the same
    dataset from R2 under ``r2_key``.
    """
    import pyarrow.compute as pc

    ds = _resolve_dataset(local_path, r2, r2_key)
    columns = list(columns) if columns else None
    if columns and "clave_estacion" not in columns:
        columns = list(columns) + ["clave_estacion"]
    if columns and "fecha" not in columns:
        columns = list(columns) + ["fecha"]

    import pyarrow as pa

    filt = pc.field("clave_estacion") == clave
    if years is not None:
        years = list(years)
        filt = filt & pc.is_in(
            pc.field("anio"), value_set=pa.array([int(y) for y in years], type=pa.int32())
        )

    table = ds.to_table(columns=columns, filter=filt)
    df = table.to_pandas().sort_values("fecha").reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def load_multi_station_daily(
    r2: R2Client | None,
    r2_key: str,
    *,
    claves: Iterable[str],
    columns: Sequence[str] | None = None,
    years: Iterable[int] | None = None,
    local_path: Path | None = None,
) -> pd.DataFrame:
    """Load daily records of many stations at once, one round-trip."""
    import pyarrow.compute as pc

    ds = _resolve_dataset(local_path, r2, r2_key)
    claves = list(claves)
    columns = list(columns) if columns else None
    if columns and "clave_estacion" not in columns:
        columns = columns + ["clave_estacion"]
    if columns and "fecha" not in columns:
        columns = columns + ["fecha"]

    import pyarrow as pa

    filt = pc.is_in(pc.field("clave_estacion"), value_set=pa.array(claves, type=pa.string()))
    if years is not None:
        filt = filt & pc.is_in(
            pc.field("anio"), value_set=pa.array([int(y) for y in years], type=pa.int32())
        )

    table = ds.to_table(columns=columns, filter=filt)
    df = table.to_pandas().sort_values(["clave_estacion", "fecha"]).reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def coverage_of_station(df: pd.DataFrame, target: str,
                        start: str = "2010-01-01",
                        end: str = "2025-12-31") -> float:
    """Fraction of non-missing daily observations of ``target`` in ``[start, end]``."""
    if df.empty or target not in df.columns:
        return 0.0
    mask = (df["fecha"] >= pd.Timestamp(start)) & (df["fecha"] <= pd.Timestamp(end))
    slice_ = df.loc[mask]
    if slice_.empty:
        return 0.0
    n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    valid = int(slice_[target].notna().sum())
    return float(valid) / n_days
