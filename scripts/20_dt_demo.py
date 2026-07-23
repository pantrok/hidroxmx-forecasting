#!/usr/bin/env python
"""Stage 20 — scoped predictive digital twin demonstrator (Milestone 6).

For one representative station per basin, restore the M3 F0-PUB
checkpoint, regenerate the test-window forecasts and evaluate two
digital-twin capabilities on top:

1. **Retrospective assimilation** via
   :class:`hidroxmx.twin.InnovationPersistence`. Compare test-window
   RMSE before and after applying the innovation-persistence
   correction. Kill condition of §6: reduction ≥ 10 % at some
   horizon.
2. **What-if scenarios** — apply the five perturbations in
   :data:`hidroxmx.twin.SCENARIO_LIBRARY` (precip ±20 %, temp ±2 °C,
   14-day dry-out) to the test feature windows and report the
   mean/median forecast shift per horizon.

Representative stations (chosen so both event-rich and event-sparse
regimes are covered):
- Alto Lerma      → SLVGJ  (Salvatierra, high autocorrelation)
- Valle de México → TTLMX  (Totolica, event-rich)
- Bajo Pánuco     → ATCHD  (Atlapexco, event-rich)
- Medio Balsas    → CMNMC  (Caimanera, large-flow event-rich)

Reuses the M3/M5 data pipeline (same features, standardisation and
window builder) so every number here is directly comparable to the
manuscript's Table 3 / Figure 5.
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
from hidroxmx.eval import rmse
from hidroxmx.io import (
    CheckpointStore,
    RunManifest,
    dump_manifest,
    publish_results,
    r2_from_env,
    seed_everything,
)
from hidroxmx.twin import (
    SCENARIO_LIBRARY,
    apply_scenario,
    assimilate_forecasts,
)


MIN_TRAIN_ROWS = 500

REPRESENTATIVE_STATIONS = {
    "Alto Lerma":      ("SLVGJ", "F0pub-alto-lerma-sweep-01"),
    "Valle de México": ("TTLMX", "F0pub-valle-de-mexico-sweep-01"),
    "Bajo Pánuco":     ("ATCHD", "F0pub-bajo-panuco-sweep-01"),
    "Medio Balsas":    ("CMNMC", "F0pub-medio-balsas-sweep-01"),
}


# --------------------------------------------------------------------------- #
# Data pipeline (mirrors stage 15/17)
# --------------------------------------------------------------------------- #
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


def _prepare_station(r2, layout, station_row, split, spec):
    clave = str(station_row["clave"])
    vecinos_str = str(station_row.get("vecinos_clima", "") or "")
    vecinos = [v.strip() for v in vecinos_str.split(",") if v.strip()] if vecinos_str else []
    target_raw = _load_target_series(r2, layout, clave)
    if target_raw.empty:
        return None
    clima = _load_climate_neighbours(r2, layout, vecinos)
    mask_train_raw = split.mask(target_raw["fecha"], "train")
    series, feature_cols, clip_upper = build_features(
        target_raw, clima, use_clima=True,
        train_mask=mask_train_raw, clip_upper=None,
    )
    mask_train = split.mask(series["fecha"], "train")
    mask_test = split.mask(series["fecha"], "test")
    if int(mask_train.sum()) < MIN_TRAIN_ROWS:
        return None
    series_std, stats = standardise(series, mask_train, feature_cols, TARGET_LOG_COL)
    tgt_stats = target_stats_m3s(series, mask_train, clip_upper)
    aligned_spec = WindowSpec(
        lookback=spec.lookback, horizons=spec.horizons,
        feature_cols=feature_cols, target_col=TARGET_LOG_COL,
    )
    return {
        "clave": clave,
        "name": str(station_row.get("nombre", "")),
        "feature_cols": feature_cols,
        "stats": stats,
        "target_stats": tgt_stats,
        "windows_test": collect_windows(series_std.loc[mask_test], aligned_spec),
        "series_std_test": series_std.loc[mask_test],
    }


# --------------------------------------------------------------------------- #
# Twin driver — one station
# --------------------------------------------------------------------------- #
def _run_station(basin, clave, base_run_id, *,
                  r2, layout, split, lookback, horizons,
                  hidden, layers, dropout, seed,
                  decay, history_days, run_id, out_root, upload_to_r2):
    click.echo(f"\n[20_dt] === Station: {clave} ({basin}) ===")
    manifest = load_selected_stations(r2, layout, kind="hidro")
    row = manifest[manifest["clave"].astype(str) == clave]
    if row.empty:
        raise SystemExit(f"[20_dt] station {clave!r} not in manifest")
    row = row.iloc[0]
    ref_spec = WindowSpec(lookback=lookback, horizons=horizons,
                          feature_cols=[], target_col=TARGET_LOG_COL)
    prep = _prepare_station(r2, layout, row, split, ref_spec)
    if prep is None or len(prep["windows_test"]["x"]) == 0:
        click.echo(f"[20_dt] {clave}: insufficient data, skipping.")
        return {"basin": basin, "clave": clave, "skipped": True}

    n_test = len(prep["windows_test"]["x"])
    click.echo(f"[20_dt] Loaded {clave} ({prep['name']}), test windows={n_test}")

    # Restore checkpoint
    import torch
    from hidroxmx.models.forecaster import LSTMEncDecConfig, LSTMEncoderDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_model = LSTMEncDecConfig(
        input_dim=len(prep["feature_cols"]), hidden_dim=hidden,
        num_layers=layers, horizons=len(horizons), dropout=dropout,
    )
    model = LSTMEncoderDecoder(cfg_model).to(device)
    ckpt_prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/runs"
    ckpt_store = CheckpointStore(
        run_id=f"{base_run_id}/{clave}",
        local_dir=out_root / "ckpt_cache" / clave,
        r2=r2, r2_prefix=ckpt_prefix,
    )
    resumed = ckpt_store.restore(name="best.ckpt")
    if resumed is None:
        click.echo(f"[20_dt] missing best.ckpt for {base_run_id}/{clave} — skipping.")
        return {"basin": basin, "clave": clave, "skipped": True}
    model.load_state_dict(resumed["model"])
    model.eval()

    mu_t, sigma_t = prep["stats"][TARGET_LOG_COL]

    def denorm(z):
        return np.expm1(z * sigma_t + mu_t)

    # 1. Raw forecast on test windows
    test_x = torch.from_numpy(prep["windows_test"]["x"]).to(device)
    with torch.no_grad():
        raw_pred = model(test_x).cpu().numpy()
    y_true_m3s = denorm(prep["windows_test"]["y"])
    y_pred_m3s = denorm(raw_pred)

    # 2. Recent residuals — use the model's own 1-day-ahead residuals from
    #    earlier test windows. The 1-day forecast of test_window[k] targets
    #    day k+1 of the test period, so residuals_1d[k] = y_true[k, 0] − ŷ[k, 0]
    #    is known BEFORE test_window[k + 1] is issued and can inform its
    #    correction. This is the standard operational DA setup — no data
    #    leakage because the residual at time t is only used to correct
    #    forecasts issued after t.
    residuals_1d = y_true_m3s[:, 0] - y_pred_m3s[:, 0]   # shape [N]
    residuals_history = np.zeros((n_test, history_days), dtype=float)
    for i in range(n_test):
        start = max(0, i - history_days)
        hist = residuals_1d[start:i]
        if hist.size < history_days:
            hist = np.concatenate([np.zeros(history_days - hist.size), hist])
        residuals_history[i] = hist

    # 3. Assimilated forecast
    assim_pred = assimilate_forecasts(
        y_pred_m3s, residuals_history, horizons,
        decay=decay, history_days=history_days,
    )

    # 4. RMSE comparison per horizon
    metrics = {"clave": clave, "basin": basin, "n_test": int(n_test),
               "decay": float(decay), "history_days": int(history_days)}
    click.echo(f"[20_dt] Assimilation RMSE comparison:")
    click.echo(f"[20_dt]   {'h':>3s}  {'raw':>8s}  {'assim':>8s}  {'Δ':>7s}  {'red%':>6s}")
    for i, h in enumerate(horizons):
        r_raw = float(rmse(y_true_m3s[:, i], y_pred_m3s[:, i]))
        r_asm = float(rmse(y_true_m3s[:, i], assim_pred[:, i]))
        red_pct = float(100.0 * (r_raw - r_asm) / r_raw) if r_raw > 0 else float("nan")
        metrics[f"rmse_raw_h{h}_m3s"] = r_raw
        metrics[f"rmse_assim_h{h}_m3s"] = r_asm
        metrics[f"rmse_reduction_pct_h{h}"] = red_pct
        marker = " *" if red_pct >= 10.0 else ""
        click.echo(f"[20_dt]   {h:>2d}d  {r_raw:>8.2f}  {r_asm:>8.2f}  "
                   f"{r_raw - r_asm:+7.2f}  {red_pct:>5.1f}%{marker}")

    kill_cleared = any(metrics[f"rmse_reduction_pct_h{h}"] >= 10.0 for h in horizons)
    metrics["assimilation_kill_cleared"] = bool(kill_cleared)

    # 5. What-if scenarios — apply each perturbation, run model, compare.
    scenario_summary: dict[str, dict[str, float]] = {}
    click.echo(f"[20_dt] What-if scenarios (mean forecast shift, m3/s):")
    for scen_name in SCENARIO_LIBRARY:
        perturbed_x = apply_scenario(
            scen_name, prep["windows_test"]["x"],
            prep["feature_cols"], prep["stats"],
        )
        # scenarios return float64 from NumPy ops; the model weights are
        # float32 — cast before passing to the LSTM.
        perturbed_x = perturbed_x.astype(np.float32)
        with torch.no_grad():
            per_pred = model(torch.from_numpy(perturbed_x).to(device)).cpu().numpy()
        per_pred_m3s = denorm(per_pred)
        delta = per_pred_m3s - y_pred_m3s
        row_summary = {}
        parts = []
        for i, h in enumerate(horizons):
            m = float(np.nanmean(delta[:, i]))
            row_summary[f"mean_delta_h{h}_m3s"] = m
            parts.append(f"h={h}:{m:+.2f}")
        scenario_summary[scen_name] = row_summary
        click.echo(f"[20_dt]   {scen_name:<14s}  " + "  ".join(parts))

    # 6. Persist manifest
    fold_id = f"{run_id}/{clave}"
    out_dir = out_root / clave
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_manifest = RunManifest(
        run_id=fold_id, stage="20_dt_demo",
        config={
            "basin": basin, "clave": clave,
            "base_run_id": base_run_id,
            "lookback": lookback, "horizons": list(horizons),
            "hidden": hidden, "layers": layers, "dropout": dropout,
            "seed": seed, "decay": decay, "history_days": history_days,
            "scenarios": list(SCENARIO_LIBRARY.keys()),
            "target_stats_m3s": prep["target_stats"],
        },
    ).finalise({**{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                **{f"{scen}__{k}": float(v)
                   for scen, d in scenario_summary.items()
                   for k, v in d.items()},
                "assimilation_kill_cleared": float(kill_cleared)})
    dump_manifest(fold_manifest, out_dir / "manifest.json")
    if upload_to_r2:
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{fold_id}"
        r2.upload_file(f"{prefix}/manifest.json", out_dir / "manifest.json")
        click.echo(f"[20_dt]   -> r2://{r2.bucket}/{prefix}/manifest.json")
    published = publish_results(
        [out_dir / "manifest.json"],
        stage="20_dt_demo", run_id=run_id, subpath=clave,
    )
    for p in published:
        click.echo(f"[20_dt]   -> git: {p.as_posix()}")

    # Save raw arrays for the paper's Fig. 6 (hydrograph + fan)
    arrays_path = out_dir / "forecast_arrays.npz"
    np.savez(arrays_path,
             y_true=y_true_m3s, y_raw=y_pred_m3s, y_assim=assim_pred,
             horizons=np.array(horizons))
    click.echo(f"[20_dt]   -> arrays: {arrays_path.as_posix()}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"basin": basin, "clave": clave, "metrics": metrics,
            "scenarios": scenario_summary, "kill_cleared": kill_cleared}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--run-id", default="dt-demo-01", show_default=True)
@click.option("--lookback", default=90, show_default=True)
@click.option("--horizons", default="1,2,3,5,7", show_default=True)
@click.option("--hidden", default=64, show_default=True)
@click.option("--layers", default=1, show_default=True)
@click.option("--dropout", default=0.0, show_default=True)
@click.option("--decay", default=0.7, show_default=True, type=float,
              help="Innovation-persistence decay in (0, 1).")
@click.option("--history-days", default=5, show_default=True, type=int,
              help="Number of trailing 1-day residuals for the EWMA innovation.")
@click.option("--seed", default=20260101, show_default=True)
@click.option("--out-dir", default="outputs/dt_demo", show_default=True)
@click.option("--upload-to-r2", is_flag=True)
def main(run_id, lookback, horizons, hidden, layers, dropout,
         decay, history_days, seed, out_dir, upload_to_r2):
    load_dotenv(override=False)
    seed_everything(seed)

    horizons_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]
    out_root = Path(out_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    r2 = r2_from_env()
    layout = layout_from_env()
    split = TemporalSplit.default()

    results = []
    for basin, (clave, base_run_id) in REPRESENTATIVE_STATIONS.items():
        r = _run_station(
            basin, clave, base_run_id,
            r2=r2, layout=layout, split=split,
            lookback=lookback, horizons=horizons_list,
            hidden=hidden, layers=layers, dropout=dropout, seed=seed,
            decay=decay, history_days=history_days,
            run_id=run_id, out_root=out_root, upload_to_r2=upload_to_r2,
        )
        results.append(r)

    # Consolidated summary
    completed = [r for r in results if not r.get("skipped")]
    click.echo("\n[20_dt] === DT summary ===")
    for r in completed:
        cleared = "KILL CLEARED" if r["kill_cleared"] else "kill not cleared"
        best_red = max(r["metrics"][f"rmse_reduction_pct_h{h}"] for h in horizons_list)
        click.echo(f"[20_dt]   {r['basin']:<18s} {r['clave']:<7s}  "
                   f"best RMSE reduction={best_red:+.1f}%  ({cleared})")

    if completed:
        summary_df = pd.DataFrame([
            {"basin": r["basin"], "clave": r["clave"],
             **{k: v for k, v in r["metrics"].items() if isinstance(v, (int, float, bool))}}
            for r in completed
        ])
        summary_csv = out_root / "dt_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        click.echo(f"[20_dt] wrote {summary_csv.as_posix()}")
        if upload_to_r2:
            prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}"
            r2.upload_file(f"{prefix}/dt_summary.csv", summary_csv)
            click.echo(f"[20_dt]   -> r2://{r2.bucket}/{prefix}/dt_summary.csv")
        published = publish_results(
            [summary_csv], stage="20_dt_demo", run_id=run_id,
        )
        for p in published:
            click.echo(f"[20_dt]   -> git: {p.as_posix()}")

    click.echo("[20_dt] Done.")


if __name__ == "__main__":
    main()
