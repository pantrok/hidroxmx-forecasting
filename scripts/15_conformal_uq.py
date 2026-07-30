#!/usr/bin/env python
"""Post-hoc split-conformal uncertainty quantification over F0-PUB checkpoints.

Wraps the F0-PUB point forecaster with prediction intervals that carry
a finite-sample marginal-coverage guarantee. For each PUB fold produced
by ``scripts/12_train_multistation.py``:

1. Restore the ``best.ckpt`` from R2 (or the local outputs/ mirror).
2. Reconstruct the exact validation and test windows (same data
   pipeline as the training script, no re-training).
3. Generate denormalised point forecasts on val and test.
4. Fit :class:`hidroxmx.uq.SplitConformal` on val absolute residuals
   at ``alpha=0.1`` (90 % nominal coverage), one quantile per horizon.
5. Emit interval bounds on test, then evaluate:
   - marginal coverage (should be ≥ 0.90 − 1/(n+1)),
   - mean interval width (sharpness proxy),
   - tail coverage restricted to test-window Q95 (does the interval
     still cover extremes?).
6. Save the per-fold manifest with all the UQ metrics and mirror to
   R2 / results/.

Reuses the training-script helpers to keep the data pipeline identical:
the UQ layer must operate on exactly the model the point-forecast tables
report, otherwise the reported coverage number cannot be trusted.
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

from hidroxmx.data.features import (
    CLIMA_COLS_RAW,
    TARGET_COL,
    TARGET_LOG_COL,
    build_features,
    reindex_daily,
    standardise,
    target_stats_m3s,
)
from hidroxmx.data.splits import TemporalSplit
from hidroxmx.data.streams import (
    coverage_of_station,
    layout_from_env,
    load_multi_station_daily,
    load_selected_stations,
    load_station_daily,
    local_roots_from_env,
)
from hidroxmx.data.windows import WindowSpec, collect_windows
from hidroxmx.io import (
    CheckpointStore,
    RunManifest,
    dump_manifest,
    publish_results,
    r2_from_env,
    seed_everything,
)
from hidroxmx.uq import SplitConformal, coverage_and_width, tail_coverage


DEFAULT_ALPHA = 0.1
MIN_TRAIN_ROWS = 500


# --------------------------------------------------------------------------- #
# Data / prep helpers (mirror scripts/12)
# --------------------------------------------------------------------------- #
def _load_basin_stations(r2, layout, basin):
    manifest = load_selected_stations(r2, layout, kind="hidro")
    pool = manifest[manifest["cuenca"].astype(str).str.contains(basin, case=False, na=False)]
    if pool.empty:
        raise SystemExit(f"[15_uq] No stations match basin={basin!r}.")
    return pool.copy()


def _load_target_series(r2, layout, clave):
    local = local_roots_from_env()
    df = load_station_daily(
        r2, layout.series_hidro_key, clave=clave,
        columns=["clave_estacion", "fecha", TARGET_COL, "nivel_m", "calidad"],
        years=range(2010, 2026),
        local_path=(local.series_hidro if local else None),
    )
    return reindex_daily(df)


def _load_climate_neighbours(r2, layout, vecinos):
    if not vecinos:
        return pd.DataFrame(columns=["fecha", *CLIMA_COLS_RAW])
    local = local_roots_from_env()
    df = load_multi_station_daily(
        r2, layout.series_clima_key, claves=vecinos,
        columns=["clave_estacion", "fecha", *CLIMA_COLS_RAW],
        years=range(2010, 2026),
        local_path=(local.series_clima if local else None),
    )
    if df.empty:
        return pd.DataFrame(columns=["fecha", *CLIMA_COLS_RAW])
    daily = df.groupby("fecha", as_index=False)[list(CLIMA_COLS_RAW)].mean()
    return reindex_daily(daily)


def _prepare_station(r2, layout, station_row, use_clima, split, spec, feature_cols_ref):
    clave = str(station_row["clave"])
    vecinos_str = str(station_row.get("vecinos_clima", "") or "")
    vecinos = [v.strip() for v in vecinos_str.split(",") if v.strip()] if vecinos_str else []
    target_raw = _load_target_series(r2, layout, clave)
    if target_raw.empty:
        return None
    cov = coverage_of_station(target_raw, TARGET_COL)
    clima = _load_climate_neighbours(r2, layout, vecinos) if use_clima else pd.DataFrame()
    mask_train_raw = split.mask(target_raw["fecha"], "train")
    series, feature_cols, clip_upper = build_features(
        target_raw, clima, use_clima=use_clima,
        train_mask=mask_train_raw, clip_upper=None,
    )
    mask_train = split.mask(series["fecha"], "train")
    mask_val = split.mask(series["fecha"], "val")
    mask_test = split.mask(series["fecha"], "test")
    if int(mask_train.sum()) < MIN_TRAIN_ROWS:
        return None
    series_std, stats = standardise(series, mask_train, feature_cols, TARGET_LOG_COL)
    tgt_stats = target_stats_m3s(series, mask_train, clip_upper)
    cols = feature_cols_ref if feature_cols_ref is not None else feature_cols
    for c in cols:
        if c not in series_std.columns:
            series_std[c] = 0.0
    aligned_spec = WindowSpec(
        lookback=spec.lookback, horizons=spec.horizons,
        feature_cols=cols, target_col=TARGET_LOG_COL,
    )
    return {
        "clave": clave,
        "name": str(station_row.get("nombre", "")),
        "coverage": float(cov),
        "feature_cols": cols,
        "stats": stats,
        "target_stats": tgt_stats,
        "windows_val": collect_windows(series_std.loc[mask_val], aligned_spec),
        "windows_test": collect_windows(series_std.loc[mask_test], aligned_spec),
    }


# --------------------------------------------------------------------------- #
# One fold of UQ evaluation
# --------------------------------------------------------------------------- #
def _run_conformal_fold(target_clave, stations_pool, *,
                        r2, layout, split, base_run_id, alpha,
                        lookback, horizons, use_clima, hidden, layers, dropout,
                        seed, run_id, out_root, upload_to_r2):
    click.echo(f"\n[15_uq] === Fold: holdout={target_clave} ===")

    target_row = stations_pool[stations_pool["clave"].astype(str) == target_clave]
    if target_row.empty:
        raise SystemExit(f"[15_uq] holdout {target_clave!r} not in basin pool")
    target_row = target_row.iloc[0]

    ref_spec = WindowSpec(lookback=lookback, horizons=horizons,
                          feature_cols=[], target_col=TARGET_LOG_COL)
    target_prep = _prepare_station(r2, layout, target_row, use_clima, split,
                                   ref_spec, feature_cols_ref=None)
    if target_prep is None:
        click.echo(f"[15_uq] holdout {target_clave!r} has too little data — skipping.")
        return {"target_clave": target_clave, "metrics": {}, "skipped": True}
    ref_features = target_prep["feature_cols"]
    n_val = len(target_prep["windows_val"]["x"])
    n_test = len(target_prep["windows_test"]["x"])
    click.echo(f"[15_uq] Target {target_clave} ({target_prep['name']}): "
               f"windows val/test={n_val}/{n_test}")
    if n_val == 0 or n_test == 0:
        click.echo(f"[15_uq] Insufficient val ({n_val}) or test ({n_test}) windows — skipping.")
        return {"target_clave": target_clave, "metrics": {}, "skipped": True}

    # --- Restore the F0-PUB checkpoint for this fold. -----------------------
    import torch
    from hidroxmx.models.forecaster import LSTMEncDecConfig, LSTMEncoderDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_model = LSTMEncDecConfig(
        input_dim=len(ref_features), hidden_dim=hidden, num_layers=layers,
        horizons=len(horizons), dropout=dropout,
    )
    model = LSTMEncoderDecoder(cfg_model).to(device)

    base_fold_id = f"{base_run_id}/{target_clave}"
    ckpt_prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/runs"
    ckpt_store = CheckpointStore(
        run_id=base_fold_id,
        local_dir=out_root / "ckpt_cache" / target_clave,
        r2=r2, r2_prefix=ckpt_prefix,
    )
    resumed = ckpt_store.restore(name="best.ckpt")
    if resumed is None:
        click.echo(f"[15_uq] Missing best.ckpt under {ckpt_prefix}/{base_fold_id}/ — skipping.")
        return {"target_clave": target_clave, "metrics": {}, "skipped": True}
    model.load_state_dict(resumed["model"])
    model.eval()

    mu_t, sigma_t = target_prep["stats"][TARGET_LOG_COL]

    def denorm_to_m3s(z):
        return np.expm1(z * sigma_t + mu_t)

    val_x = torch.from_numpy(target_prep["windows_val"]["x"]).to(device)
    test_x = torch.from_numpy(target_prep["windows_test"]["x"]).to(device)
    with torch.no_grad():
        yhat_val = model(val_x).cpu().numpy()
        yhat_test = model(test_x).cpu().numpy()

    y_val_m3s = denorm_to_m3s(target_prep["windows_val"]["y"])
    y_test_m3s = denorm_to_m3s(target_prep["windows_test"]["y"])
    yhat_val_m3s = denorm_to_m3s(yhat_val)
    yhat_test_m3s = denorm_to_m3s(yhat_test)

    # --- Split conformal calibration + evaluation. -------------------------
    sc = SplitConformal(alpha=alpha).fit(y_val_m3s, yhat_val_m3s)
    lower, upper = sc.predict_interval(yhat_test_m3s)
    cw = coverage_and_width(y_test_m3s, lower, upper)
    tc = tail_coverage(y_test_m3s, lower, upper, threshold_quantile=0.95)

    metrics: dict[str, float] = {
        "alpha": float(alpha),
        "n_calibration": int(sc.n_calibration_),
        "n_test": int(n_test),
    }
    click.echo(f"[15_uq] === Conformal UQ (alpha={alpha}, "
               f"nominal coverage={1 - alpha:.2f}) on {target_clave} ===")
    for i, h in enumerate(horizons):
        cov = float(cw["coverage"][i])
        miw = float(cw["mean_interval_width"][i])
        tail = float(tc[i])
        q = float(sc.quantiles_[i])
        metrics[f"coverage_h{h}"] = cov
        metrics[f"mean_interval_width_h{h}_m3s"] = miw
        metrics[f"tail_coverage_q95_h{h}"] = tail
        metrics[f"conformal_quantile_h{h}_m3s"] = q
        click.echo(f"[15_uq]   h={h:>2d}d  coverage={cov*100:5.1f}%  "
                   f"width={miw:6.2f} m3/s  tail_cov(Q95)={tail*100:5.1f}%  "
                   f"q_hat={q:.2f}")

    # --- Persist artefacts. ------------------------------------------------
    fold_id = f"{run_id}/{target_clave}"
    out_dir = out_root / target_clave
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_manifest = RunManifest(
        run_id=fold_id, stage="15_conformal_uq",
        config={
            "base_run_id": base_run_id,
            "holdout": target_clave,
            "holdout_name": target_prep["name"],
            "alpha": float(alpha),
            "lookback": lookback,
            "horizons": list(horizons),
            "hidden": hidden, "layers": layers, "dropout": dropout,
            "use_clima": use_clima, "seed": seed,
            "feature_cols": ref_features,
            "target_stats_m3s": target_prep["target_stats"],
        },
    ).finalise({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    dump_manifest(fold_manifest, out_dir / "manifest.json")
    if upload_to_r2:
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{fold_id}"
        r2.upload_file(f"{prefix}/manifest.json", out_dir / "manifest.json")
        click.echo(f"[15_uq]   -> r2://{r2.bucket}/{prefix}/manifest.json")
    published = publish_results(
        [out_dir / "manifest.json"],
        stage="15_conformal_uq", run_id=run_id, subpath=target_clave,
    )
    for p in published:
        click.echo(f"[15_uq]   -> git: {p.as_posix()}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"target_clave": target_clave, "metrics": metrics}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--run-id", default="UQ-conformal-alto-lerma-sweep-01", show_default=True)
@click.option("--base-run-id", default="F0pub-alto-lerma-sweep-01", show_default=True,
              help="F0-PUB sweep whose checkpoints supply the point forecast.")
@click.option("--basin", default="Alto Lerma", show_default=True)
@click.option("--holdout", default="*", show_default=True)
@click.option("--alpha", default=DEFAULT_ALPHA, show_default=True, type=float,
              help="Target miscoverage rate — 0.1 → nominal 90 % coverage.")
@click.option("--lookback", default=90, show_default=True)
@click.option("--horizons", default="1,2,3,5,7", show_default=True)
@click.option("--hidden", default=64, show_default=True)
@click.option("--layers", default=1, show_default=True)
@click.option("--dropout", default=0.0, show_default=True)
@click.option("--use-clima/--no-clima", default=True, show_default=True)
@click.option("--seed", default=20260101, show_default=True)
@click.option("--out-dir", default="outputs/uq_conformal", show_default=True)
@click.option("--upload-to-r2", is_flag=True)
def main(run_id, base_run_id, basin, holdout, alpha, lookback, horizons,
         hidden, layers, dropout, use_clima, seed, out_dir, upload_to_r2):
    load_dotenv(override=False)
    seed_everything(seed)

    horizons_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]
    out_root = Path(out_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    r2 = r2_from_env()
    layout = layout_from_env()
    split = TemporalSplit.default()

    stations_pool = _load_basin_stations(r2, layout, basin)
    click.echo(f"[15_uq] Basin: {basin}  stations: {len(stations_pool)}  "
               f"base run: {base_run_id}  alpha: {alpha}")

    all_claves = stations_pool["clave"].astype(str).tolist()
    holdouts = all_claves if holdout == "*" else [holdout]

    fold_summaries = []
    for target_clave in holdouts:
        result = _run_conformal_fold(
            target_clave, stations_pool,
            r2=r2, layout=layout, split=split,
            base_run_id=base_run_id, alpha=alpha,
            lookback=lookback, horizons=horizons_list, use_clima=use_clima,
            hidden=hidden, layers=layers, dropout=dropout,
            seed=seed, run_id=run_id, out_root=out_root,
            upload_to_r2=upload_to_r2,
        )
        fold_summaries.append(result)

    completed = [fs for fs in fold_summaries if not fs.get("skipped")]
    if len(completed) > 1:
        rows = []
        for fs in completed:
            row = {"holdout": fs["target_clave"]}
            for h in horizons_list:
                row[f"coverage_h{h}"] = fs["metrics"][f"coverage_h{h}"]
                row[f"mean_interval_width_h{h}_m3s"] = fs["metrics"][f"mean_interval_width_h{h}_m3s"]
                row[f"tail_coverage_q95_h{h}"] = fs["metrics"][f"tail_coverage_q95_h{h}"]
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(out_root / "folds_summary.csv", index=False)
        click.echo("\n[15_uq] === Aggregate conformal UQ summary ===")
        for h in horizons_list:
            avg_cov = float(summary_df[f"coverage_h{h}"].mean())
            avg_miw = float(summary_df[f"mean_interval_width_h{h}_m3s"].mean())
            avg_tail = float(summary_df[f"tail_coverage_q95_h{h}"].mean())
            click.echo(f"[15_uq]   h={h:>2d}d  "
                       f"mean coverage={avg_cov*100:5.1f}%  "
                       f"mean width={avg_miw:6.2f} m3/s  "
                       f"mean tail_cov(Q95)={avg_tail*100:5.1f}%")
        if upload_to_r2:
            prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}"
            r2.upload_file(f"{prefix}/folds_summary.csv", out_root / "folds_summary.csv")
            click.echo(f"[15_uq]   -> r2://{r2.bucket}/{prefix}/folds_summary.csv")
        published = publish_results(
            [out_root / "folds_summary.csv"],
            stage="15_conformal_uq", run_id=run_id,
        )
        for p in published:
            click.echo(f"[15_uq]   -> git: {p.as_posix()}")

    click.echo("[15_uq] Done.")


if __name__ == "__main__":
    main()
