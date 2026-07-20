#!/usr/bin/env python
"""Stage 11 — Train the F0 forecaster on one station of the pilot basin.

Milestone 2 slice: end-to-end train / val / test loop that exercises every
layer the later paths reuse:

- streaming from R2 (or a local mirror via ``HIDROXAI_MX_ROOT``);
- reindexing to a continuous daily calendar and building lags / moving
  averages on the raw ``gasto_medio_m3s`` series in m³/s;
- joining the mean of the station's ``vecinos_clima`` (precip, tmax,
  tmin) as exogenous drivers;
- frozen temporal split, train-only standardisation, sliding windows;
- LSTM encoder-decoder (F0), checkpoint-resumable training;
- naive baselines (persistence + climatology) reported alongside the
  model for a plain-English "did we beat doing nothing" test;
- test-set metrics *in physical units* (NSE, KGE, RMSE in m³/s) plus
  target descriptive statistics on the run manifest.

Defaults target a Colab GPU session budget: ``hidden=64``, ``epochs=30``,
``batch_size=64``, ``lookback=90``. A full run finishes in a few minutes
on a T4 and in tens of minutes on CPU.

The station is auto-selected as the station in ``--basin`` with the
highest observed target coverage inside the reference window
(2010-01-01 through 2025-12-31); override with ``--clave``. ``--basin``
matches the ``cuenca`` column of the selected-stations manifest (case-
insensitive substring). Cutzamala is not a labelled basin in the CONAGUA
selection — Alto Lerma is the default because it exposes 14 stations for
Milestone-3 PUB leave-one-out.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from hidroxmx.data.splits import TemporalSplit
from hidroxmx.data.streams import (
    DatasetLayout,
    coverage_of_station,
    layout_from_env,
    load_multi_station_daily,
    load_selected_stations,
    load_station_daily,
    local_roots_from_env,
)
from hidroxmx.data.windows import WindowSpec, collect_windows
from hidroxmx.eval import kge, nse, pbias, rmse
from hidroxmx.io import (
    CheckpointStore,
    RunManifest,
    dump_manifest,
    r2_from_env,
    seed_everything,
)
from hidroxmx.io.checkpoint import collect_rng_state, restore_rng_state


TARGET_COL = "gasto_medio_m3s"
LAG_DAYS = (1, 3, 7, 14, 30)
MA_DAYS = (7, 30)
CLIMA_COLS_RAW = ("precip_mm", "tmax_c", "tmin_c")
DEFAULT_BASIN = "Alto Lerma"
TARGET_UPPER_CLIP_QUANTILE = 0.999  # trims CONAGUA capture-error spikes on the train window only


# --------------------------------------------------------------------------- #
# Station selection
# --------------------------------------------------------------------------- #
def _select_station(r2, layout: DatasetLayout, basin: str) -> tuple[str, pd.Series]:
    """Return ``(clave, manifest_row)`` for the highest-coverage station in ``basin``."""
    manifest = load_selected_stations(r2, layout, kind="hidro")
    if "cuenca" in manifest.columns:
        pool = manifest[manifest["cuenca"].astype(str).str.contains(basin,
                                                                    case=False,
                                                                    na=False)]
    else:
        pool = manifest.iloc[0:0]
    if len(pool) == 0:
        raise SystemExit(
            f"[11_train] No station in the selected manifest matches basin={basin!r}. "
            f"Available basins: {sorted(set(manifest['cuenca'].dropna().astype(str)))}"
        )
    if "cobertura" in pool.columns:
        pool = pool.sort_values("cobertura", ascending=False)
    row = pool.iloc[0]
    return str(row["clave"]), row


# --------------------------------------------------------------------------- #
# Series loaders
# --------------------------------------------------------------------------- #
def _reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.set_index("fecha").sort_index()
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    df.index.name = "fecha"
    return df.reset_index()


def _load_target_series(r2, layout: DatasetLayout, clave: str) -> pd.DataFrame:
    """Load the raw hydrometric series (gasto_medio_m3s in m³/s) for one station."""
    local = local_roots_from_env()
    df = load_station_daily(
        r2,
        layout.series_hidro_key,
        clave=clave,
        columns=["clave_estacion", "fecha", TARGET_COL, "nivel_m", "calidad"],
        years=range(2010, 2026),
        local_path=(local.series_hidro if local else None),
    )
    return _reindex_daily(df)


def _load_climate_neighbours(r2, layout: DatasetLayout,
                             vecinos: list[str]) -> pd.DataFrame:
    """Load raw daily climatology for ``vecinos`` and average across stations."""
    if not vecinos:
        return pd.DataFrame(columns=["fecha", *CLIMA_COLS_RAW])
    local = local_roots_from_env()
    df = load_multi_station_daily(
        r2,
        layout.series_clima_key,
        claves=vecinos,
        columns=["clave_estacion", "fecha", *CLIMA_COLS_RAW],
        years=range(2010, 2026),
        local_path=(local.series_clima if local else None),
    )
    if df.empty:
        return pd.DataFrame(columns=["fecha", *CLIMA_COLS_RAW])
    daily = df.groupby("fecha", as_index=False)[list(CLIMA_COLS_RAW)].mean()
    return _reindex_daily(daily)


def _build_features(target: pd.DataFrame, clima: pd.DataFrame,
                    use_clima: bool, train_mask: pd.Series,
                    clip_upper: float | None) -> tuple[pd.DataFrame, list[str], float | None]:
    """Build the modelling table with log-transformed streamflow + optional clima features.

    The raw ``gasto_medio_m3s`` column occasionally contains CONAGUA capture
    errors (e.g. displaced decimal points that leave five-digit spikes on
    stations whose true regime never exceeds three digits). We (i) clip the
    upper tail of the *training* portion at ``TARGET_UPPER_CLIP_QUANTILE``
    to freeze a robust ceiling, (ii) apply ``clip_upper`` to all folds so
    that a single day's error cannot dominate the loss, and (iii) fit the
    model on ``log1p`` of streamflow — the standard hydrological transform
    that stabilises variance and makes low-flow errors weigh comparably to
    high-flow errors. The returned ``clip_upper`` is stored on the manifest
    so downstream evaluation can undo the clip if needed.
    """
    df = target[["fecha", TARGET_COL, "nivel_m"]].copy()

    if clip_upper is None:
        train_target = df.loc[train_mask, TARGET_COL].dropna()
        if len(train_target):
            clip_upper = float(train_target.clip(lower=0.0).quantile(TARGET_UPPER_CLIP_QUANTILE))
    if clip_upper is not None:
        df[TARGET_COL] = df[TARGET_COL].clip(lower=0.0, upper=clip_upper)

    # Log-transform the streamflow feature family (target + lags + moving averages).
    log_col = f"{TARGET_COL}_log"
    df[log_col] = np.log1p(df[TARGET_COL])
    for lag in LAG_DAYS:
        df[f"{log_col}_lag{lag}"] = df[log_col].shift(lag)
    for w in MA_DAYS:
        df[f"{log_col}_ma{w}"] = df[log_col].shift(1).rolling(w, min_periods=w).mean()

    feature_cols = [
        log_col,
        *(f"{log_col}_lag{lag}" for lag in LAG_DAYS),
        *(f"{log_col}_ma{w}" for w in MA_DAYS),
    ]

    if use_clima and not clima.empty:
        df = df.merge(clima, on="fecha", how="left")
        for col in CLIMA_COLS_RAW:
            if col in df.columns:
                df[col] = df[col].astype(float)
                df[f"{col}_ma7"] = df[col].shift(1).rolling(7, min_periods=7).mean()
                feature_cols.extend([col, f"{col}_ma7"])
    return df, feature_cols, clip_upper


# --------------------------------------------------------------------------- #
# Standardisation (train-only)
# --------------------------------------------------------------------------- #
def _standardise(series: pd.DataFrame, train_mask: pd.Series,
                 feature_cols: list[str],
                 target_col: str) -> tuple[pd.DataFrame,
                                            dict[str, tuple[float, float]]]:
    stats: dict[str, tuple[float, float]] = {}
    out = series.copy()
    cols_to_norm = list(dict.fromkeys([*feature_cols, target_col]))
    for col in cols_to_norm:
        mu = float(out.loc[train_mask, col].mean())
        sigma = float(out.loc[train_mask, col].std(ddof=0)) or 1.0
        out[col] = (out[col] - mu) / sigma
        stats[col] = (mu, sigma)
    return out, stats


# --------------------------------------------------------------------------- #
# Naive baselines
# --------------------------------------------------------------------------- #
def _persistence_forecast(t0_values: np.ndarray, horizons: list[int]) -> np.ndarray:
    """Persistence: predict ``y[t0]`` for every horizon."""
    return np.repeat(t0_values.reshape(-1, 1), len(horizons), axis=1)


def _climatology_forecast(train_mean: float, n_windows: int,
                          n_horizons: int) -> np.ndarray:
    """Climatology: predict the training mean for every window and horizon."""
    return np.full((n_windows, n_horizons), train_mean, dtype=np.float64)


def _baseline_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray,
                      horizons: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, h in enumerate(horizons):
        out[f"{label}_nse_h{h}"] = float(nse(y_true[:, i], y_pred[:, i]))
        out[f"{label}_kge_h{h}"] = float(kge(y_true[:, i], y_pred[:, i]))
        out[f"{label}_rmse_h{h}_m3s"] = float(rmse(y_true[:, i], y_pred[:, i]))
    return out


# --------------------------------------------------------------------------- #
# Batches + eval
# --------------------------------------------------------------------------- #
def _to_batches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True,
                seed: int = 0):
    import torch

    n = len(x)
    order = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield (torch.from_numpy(x[idx]), torch.from_numpy(y[idx]))


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  horizons: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, h in enumerate(horizons):
        out[f"nse_h{h}"] = float(nse(y_true[:, i], y_pred[:, i]))
        out[f"kge_h{h}"] = float(kge(y_true[:, i], y_pred[:, i]))
        out[f"rmse_h{h}_m3s"] = float(rmse(y_true[:, i], y_pred[:, i]))
        out[f"pbias_h{h}_pct"] = float(pbias(y_true[:, i], y_pred[:, i]))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--run-id", default="F0-alto-lerma-topcov", show_default=True)
@click.option("--basin", default=DEFAULT_BASIN, show_default=True,
              help="Substring matched against the 'cuenca' column of the manifest.")
@click.option("--clave", default=None,
              help="Override station key; defaults to top-coverage station in the basin.")
@click.option("--lookback", default=90, show_default=True)
@click.option("--horizons", default="1,2,3,5,7", show_default=True,
              help="Comma-separated forecast horizons in days.")
@click.option("--hidden", default=64, show_default=True)
@click.option("--layers", default=1, show_default=True)
@click.option("--dropout", default=0.0, show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--epochs", default=30, show_default=True)
@click.option("--lr", default=5e-4, show_default=True)
@click.option("--patience", default=5, show_default=True,
              help="Early-stopping patience (in epochs) on val loss. 0 disables it.")
@click.option("--use-clima/--no-clima", default=True, show_default=True,
              help="Join the mean of the station's climatological neighbours as exogenous features.")
@click.option("--seed", default=20260101, show_default=True)
@click.option("--out-dir", default="outputs/f0", show_default=True)
@click.option("--upload-to-r2", is_flag=True,
              help="Mirror checkpoints and manifest under {R2_PAPER2_PREFIX}/runs/{run_id}/.")
def main(
    run_id, basin, clave, lookback, horizons, hidden, layers, dropout,
    batch_size, epochs, lr, patience, use_clima, seed, out_dir, upload_to_r2,
):
    load_dotenv(override=False)
    seed_everything(seed)

    horizons_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]
    out_dir = Path(out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    r2 = r2_from_env()
    layout = layout_from_env()

    # 1. Station -----------------------------------------------------------
    if clave is None:
        clave, station_row = _select_station(r2, layout, basin=basin)
    else:
        manifest = load_selected_stations(r2, layout, kind="hidro")
        matches = manifest[manifest["clave"].astype(str) == clave]
        if matches.empty:
            raise SystemExit(f"[11_train] Station {clave!r} not in the selected manifest.")
        station_row = matches.iloc[0]

    vecinos_str = str(station_row.get("vecinos_clima", "") or "")
    vecinos = [v.strip() for v in vecinos_str.split(",") if v.strip()] if vecinos_str else []
    click.echo(f"[11_train] Station: {clave}  ({station_row.get('nombre', '?')})")
    click.echo(f"[11_train] Basin : {station_row.get('cuenca', '?')}  "
               f"Region: {station_row.get('region_hidrologica', '?')}")
    click.echo(f"[11_train] Climate neighbours: {vecinos or '(none)'}")

    # 2. Series ------------------------------------------------------------
    target_raw = _load_target_series(r2, layout, clave)
    if target_raw.empty:
        raise SystemExit(f"[11_train] Empty target series for station {clave}; abort.")
    cov = coverage_of_station(target_raw, TARGET_COL)
    click.echo(f"[11_train] Rows: {len(target_raw):,}  coverage(target)={cov*100:.1f}%")

    clima = _load_climate_neighbours(r2, layout, vecinos) if use_clima else pd.DataFrame()
    if use_clima:
        click.echo(f"[11_train] Climate rows: {len(clima):,}")

    # 3. Temporal split (needed BEFORE feature engineering so clipping uses train only)
    split = TemporalSplit.default()
    mask_train_raw = split.mask(target_raw["fecha"], "train")

    series, feature_cols, clip_upper = _build_features(
        target_raw, clima, use_clima=use_clima,
        train_mask=mask_train_raw, clip_upper=None,
    )
    target_log_col = f"{TARGET_COL}_log"
    click.echo(f"[11_train] Feature columns ({len(feature_cols)}): {feature_cols}")
    click.echo(f"[11_train] Target upper-clip (train p{TARGET_UPPER_CLIP_QUANTILE*100:g}): "
               f"{clip_upper:.2f} m3/s")

    mask_train = split.mask(series["fecha"], "train")
    mask_val = split.mask(series["fecha"], "val")
    mask_test = split.mask(series["fecha"], "test")
    if mask_train.sum() < 500:
        raise SystemExit(
            f"[11_train] Too few train rows ({int(mask_train.sum())}); "
            f"pick another station or lower the coverage bar.")

    train_target_m3s = series.loc[mask_train, TARGET_COL].dropna().to_numpy(dtype=np.float64)
    if train_target_m3s.size < 100:
        raise SystemExit("[11_train] Training target has too few observations to estimate stats.")
    target_stats = {
        "train_mean_m3s": float(train_target_m3s.mean()),
        "train_std_m3s": float(train_target_m3s.std(ddof=0)),
        "train_min_m3s": float(train_target_m3s.min()),
        "train_p50_m3s": float(np.median(train_target_m3s)),
        "train_p95_m3s": float(np.quantile(train_target_m3s, 0.95)),
        "train_max_m3s": float(train_target_m3s.max()),
        "clip_upper_m3s": float(clip_upper) if clip_upper is not None else None,
    }
    click.echo(f"[11_train] Target train stats (m3/s, post-clip): "
               f"mean={target_stats['train_mean_m3s']:.2f}  "
               f"p50={target_stats['train_p50_m3s']:.2f}  "
               f"p95={target_stats['train_p95_m3s']:.2f}  "
               f"max={target_stats['train_max_m3s']:.2f}")

    series_std, stats = _standardise(series, mask_train, feature_cols, target_log_col)

    spec = WindowSpec(
        lookback=lookback,
        horizons=horizons_list,
        feature_cols=feature_cols,
        target_col=target_log_col,
    )
    windows_train = collect_windows(series_std.loc[mask_train], spec)
    windows_val = collect_windows(series_std.loc[mask_val], spec)
    windows_test = collect_windows(series_std.loc[mask_test], spec)
    click.echo(f"[11_train] Windows train/val/test: "
               f"{len(windows_train['x'])}/{len(windows_val['x'])}/{len(windows_test['x'])}")
    if len(windows_train["x"]) < 128:
        raise SystemExit("[11_train] Not enough training windows to fit F0; abort.")

    # 4. Model + optimiser --------------------------------------------------
    import torch
    from torch import nn

    from hidroxmx.models.forecaster import LSTMEncDecConfig, LSTMEncoderDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"[11_train] Device: {device}")
    cfg_model = LSTMEncDecConfig(
        input_dim=len(feature_cols),
        hidden_dim=hidden,
        num_layers=layers,
        horizons=len(horizons_list),
        dropout=dropout,
    )
    model = LSTMEncoderDecoder(cfg_model).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    # 5. Checkpoint store (mirrors under paper2/runs/{run_id}) --------------
    ckpt_store = CheckpointStore(
        run_id=run_id,
        local_dir=out_dir / "ckpts",
        r2=r2 if upload_to_r2 else None,
        r2_prefix=os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/runs",
    )
    resumed = ckpt_store.restore(name="last.ckpt")
    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    if resumed is not None:
        try:
            model.load_state_dict(resumed["model"])
            optimiser.load_state_dict(resumed["optimizer"])
            start_epoch = int(resumed.get("epoch", 0)) + 1
            best_val = float(resumed.get("best_val", best_val))
            best_epoch = int(resumed.get("best_epoch", -1))
            restore_rng_state(resumed.get("rng", {}))
            click.echo(f"[11_train] Resumed from epoch {start_epoch}; "
                       f"best_val={best_val:.4f} at epoch {best_epoch}")
        except Exception as exc:  # noqa: BLE001
            click.echo(f"[11_train] Failed to resume ({exc}); starting fresh.")

    # 6. Training loop ------------------------------------------------------
    val_x = torch.from_numpy(windows_val["x"]).to(device)
    val_y = torch.from_numpy(windows_val["y"]).to(device)
    test_x = torch.from_numpy(windows_test["x"]).to(device)

    history = []
    epochs_since_improved = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for xb, yb in _to_batches(
                windows_train["x"], windows_train["y"],
                batch_size=batch_size, shuffle=True, seed=seed + epoch):
            xb = xb.to(device); yb = yb.to(device)
            optimiser.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            epoch_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)
        train_loss = epoch_loss / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).item())
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(entry)
        improved = val_loss < best_val
        marker = " *" if improved else ""
        click.echo(f"[11_train] epoch {epoch:03d} "
                   f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")

        state = {
            "model": model.state_dict(),
            "optimizer": optimiser.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val_loss),
            "best_epoch": best_epoch if not improved else epoch,
            "config": {"hidden": hidden, "layers": layers, "dropout": dropout,
                       "lookback": lookback, "horizons": horizons_list,
                       "lr": lr, "batch_size": batch_size, "seed": seed,
                       "clave": clave, "use_clima": use_clima,
                       "feature_cols": feature_cols},
            "stats": stats,
            "target_stats": target_stats,
            "rng": collect_rng_state(),
        }
        ckpt_store.save(state, name="last.ckpt")
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_since_improved = 0
            ckpt_store.save(state, name="best.ckpt")
        else:
            epochs_since_improved += 1
            if patience and epochs_since_improved >= patience:
                click.echo(f"[11_train] Early stopping at epoch {epoch:03d} "
                           f"(no val improvement for {patience} epochs; best epoch {best_epoch:03d} "
                           f"with val_loss={best_val:.4f}).")
                break

    # Restore best checkpoint for evaluation so we report the model that
    # actually generalises rather than the last-epoch overfit.
    best_ckpt = ckpt_store.restore(name="best.ckpt")
    if best_ckpt is not None:
        model.load_state_dict(best_ckpt["model"])
        click.echo(f"[11_train] Restored best.ckpt (epoch {int(best_ckpt.get('epoch', -1)):03d}, "
                   f"val_loss={float(best_ckpt.get('best_val', float('nan'))):.4f}) for test evaluation.")

    # 7. Test-set predictions in physical units ----------------------------
    #    The model was trained on standardised log1p(gasto). Undo in two
    #    steps: (a) reverse the z-score using the train mu/sigma of the log
    #    column, (b) invert log1p to recover m3/s.
    mu_t, sigma_t = stats[target_log_col]

    def denorm_to_m3s(z: np.ndarray) -> np.ndarray:
        return np.expm1(z * sigma_t + mu_t)

    model.eval()
    with torch.no_grad():
        test_pred = model(test_x).cpu().numpy()
    test_pred_m3s = denorm_to_m3s(test_pred)
    test_y_m3s = denorm_to_m3s(windows_test["y"])

    metrics = _eval_metrics(test_y_m3s, test_pred_m3s, horizons_list)

    # 8. Naive baselines on the SAME test windows --------------------------
    #    Recover y[t0] (persistence anchor) in physical units for each window.
    #    windows_test["x"][:, -1, 0] is the standardised value of feature #0,
    #    which is target_log_col by construction (see _build_features).
    persistence_x_std = windows_test["x"][:, -1, 0].astype(np.float64)
    persistence_m3s = denorm_to_m3s(persistence_x_std)
    persistence_pred = _persistence_forecast(persistence_m3s, horizons_list)
    metrics.update(_baseline_metrics("persist", test_y_m3s, persistence_pred, horizons_list))

    clim_pred = _climatology_forecast(target_stats["train_mean_m3s"],
                                      len(test_y_m3s), len(horizons_list))
    metrics.update(_baseline_metrics("clim", test_y_m3s, clim_pred, horizons_list))

    metrics["train_windows"] = len(windows_train["x"])
    metrics["val_windows"] = len(windows_val["x"])
    metrics["test_windows"] = len(windows_test["x"])
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_val_loss"] = float(best_val)
    click.echo("[11_train] === Test metrics (physical units, best.ckpt) ===")
    for h in horizons_list:
        click.echo(
            f"[11_train]   h={h:>2d}d  "
            f"F0 NSE={metrics[f'nse_h{h}']:+.3f} KGE={metrics[f'kge_h{h}']:+.3f} "
            f"RMSE={metrics[f'rmse_h{h}_m3s']:6.2f} m3/s  |  "
            f"persist NSE={metrics[f'persist_nse_h{h}']:+.3f} KGE={metrics[f'persist_kge_h{h}']:+.3f}  |  "
            f"clim NSE={metrics[f'clim_nse_h{h}']:+.3f}"
        )

    # 9. Persist manifest and history --------------------------------------
    manifest = RunManifest(
        run_id=run_id,
        stage="11_train_forecaster",
        config={
            "clave": clave, "station_name": str(station_row.get("nombre", "")),
            "basin": str(station_row.get("cuenca", basin)),
            "region_hidrologica": str(station_row.get("region_hidrologica", "")),
            "vecinos_clima": vecinos,
            "use_clima": use_clima,
            "lookback": lookback,
            "horizons": horizons_list, "hidden": hidden, "layers": layers,
            "dropout": dropout, "batch_size": batch_size, "epochs": epochs,
            "lr": lr, "seed": seed,
            "coverage_of_station": cov,
            "feature_cols": feature_cols,
            "target_stats_m3s": target_stats,
        },
    ).finalise({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    dump_manifest(manifest, out_dir / "manifest.json")

    if upload_to_r2:
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}"
        for local in (out_dir / "history.json", out_dir / "manifest.json"):
            r2.upload_file(f"{prefix}/{local.name}", local)
            click.echo(f"[11_train]   -> r2://{r2.bucket}/{prefix}/{local.name}")

    del model, optimiser
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    click.echo("[11_train] Done.")


if __name__ == "__main__":
    main()
