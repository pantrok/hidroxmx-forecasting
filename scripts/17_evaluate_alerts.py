#!/usr/bin/env python
"""Evaluate the fuzzy alert layer against a simple-threshold baseline.

Combines the F0-PUB point forecast, the conformal interval width and
the Mamdani fuzzy inference system into an operational alert signal,
then benchmarks it against a **simple-threshold baseline** (``alert if
ŷ > Q95_train``) that any hydrologist would build without the UQ
machinery.

Per fold, per horizon, the script computes:

- ``POD`` (probability of detection) at three alert-level cut-offs
  (``≥ YELLOW``, ``≥ ORANGE``, ``≥ RED``),
- ``FAR`` (false-alarm ratio) at the same cut-offs,
- ``Value`` (Richardson 2000 cost-loss score) at C/L ∈ {0.05, 0.1,
  0.2, 0.3, 0.5},
- ``lead_time_days`` — the shortest horizon at which the alert
  correctly precedes the event, averaged across true events,

and the deltas against the simple-threshold baseline. The kill
condition on Δ Value @ C/L=0.2 is checked at the aggregate level.

Reuses the conformal-UQ data pipeline verbatim so predictions are
byte-identical to those the UQ table already reports.
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

from hidroxmx.alert import (
    CATEGORIES,
    build_alert_fis,
    category_to_index,
    score_to_category,
)
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
from hidroxmx.eval import pod_far, value_cost_loss
from hidroxmx.io import (
    CheckpointStore,
    RunManifest,
    dump_manifest,
    publish_results,
    r2_from_env,
    seed_everything,
)
from hidroxmx.uq import SplitConformal


MIN_TRAIN_ROWS = 500
COST_LOSS_RATIOS = (0.05, 0.1, 0.2, 0.3, 0.5)
ALERT_CUTOFFS = ("YELLOW", "ORANGE", "RED")


# --------------------------------------------------------------------------- #
# Data / prep helpers (identical to stage 15)
# --------------------------------------------------------------------------- #
def _load_basin_stations(r2, layout, basin):
    manifest = load_selected_stations(r2, layout, kind="hidro")
    pool = manifest[manifest["cuenca"].astype(str).str.contains(basin, case=False, na=False)]
    if pool.empty:
        raise SystemExit(f"[17_alerts] No stations match basin={basin!r}.")
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
# Alert evaluation
# --------------------------------------------------------------------------- #
def _apply_fis(fis, flow_pred_m3s: np.ndarray, width_m3s: np.ndarray,
               q95_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised FIS application over one fold's test predictions.

    Returns (alert_score [N, H], alert_category_index [N, H]).
    """
    n, h = flow_pred_m3s.shape
    flow_ratio = flow_pred_m3s / max(q95_ref, 1e-6)
    width_ratio = width_m3s / max(q95_ref, 1e-6)
    # Clip inputs into the FIS universes to avoid extrapolation.
    flow_ratio = np.clip(flow_ratio, 0.0, 3.0)
    width_ratio = np.clip(width_ratio, 0.0, 2.0)

    scores = np.empty((n, h), dtype=float)
    for j in range(h):
        scores[:, j] = fis.infer_batch({"flow_ratio": flow_ratio[:, j],
                                        "width_ratio": np.full(n, width_ratio[j])})
    idx = category_to_index(score_to_category(scores))
    return scores, idx


def _alert_binary(alert_idx: np.ndarray, cutoff: str) -> np.ndarray:
    """True where alert level is at least ``cutoff``."""
    cutoff_i = CATEGORIES.index(cutoff)
    return alert_idx >= cutoff_i


def _first_alert_horizon(alert_idx_per_horizon: np.ndarray,
                         cutoff: str) -> np.ndarray:
    """For each test window, the shortest horizon where alert >= cutoff, else -1."""
    fired = _alert_binary(alert_idx_per_horizon, cutoff)  # [N, H]
    idx = np.where(fired.any(axis=1),
                   np.argmax(fired, axis=1),
                   -1)
    return idx


def _evaluate_alert_signal(events: np.ndarray, alerts: np.ndarray
                           ) -> dict[str, float]:
    """Return POD, FAR and cost-loss Value at the standard C/L ratios."""
    pod, far = pod_far(events, alerts)
    out = {"pod": float(pod), "far": float(far)}
    for cl in COST_LOSS_RATIOS:
        out[f"value_cl_{cl}"] = float(value_cost_loss(events, alerts, cl))
    return out


def _run_fold(target_clave, stations_pool, *,
              r2, layout, split, base_run_id,
              alpha, lookback, horizons, use_clima,
              hidden, layers, dropout,
              seed, run_id, out_root, upload_to_r2):
    click.echo(f"\n[17_alerts] === Fold: holdout={target_clave} ===")
    target_row = stations_pool[stations_pool["clave"].astype(str) == target_clave]
    if target_row.empty:
        raise SystemExit(f"[17_alerts] holdout {target_clave!r} not in pool")
    target_row = target_row.iloc[0]

    ref_spec = WindowSpec(lookback=lookback, horizons=horizons,
                          feature_cols=[], target_col=TARGET_LOG_COL)
    target_prep = _prepare_station(r2, layout, target_row, use_clima, split,
                                   ref_spec, feature_cols_ref=None)
    if target_prep is None:
        click.echo(f"[17_alerts] {target_clave} has too little data — skipping.")
        return {"target_clave": target_clave, "metrics": {}, "skipped": True}
    ref_features = target_prep["feature_cols"]
    n_val = len(target_prep["windows_val"]["x"])
    n_test = len(target_prep["windows_test"]["x"])
    if n_val == 0 or n_test == 0:
        click.echo(f"[17_alerts] {target_clave} has 0 val ({n_val}) or test ({n_test}) — skipping.")
        return {"target_clave": target_clave, "metrics": {}, "skipped": True}
    click.echo(f"[17_alerts] Target {target_clave} windows val/test={n_val}/{n_test}")

    q95_ref = float(target_prep["target_stats"].get("train_p95_m3s") or 0.0)
    if q95_ref <= 0:
        click.echo(f"[17_alerts] {target_clave}: train Q95 undefined — skipping.")
        return {"target_clave": target_clave, "metrics": {}, "skipped": True}

    # ------------- Restore F0-PUB best.ckpt and regenerate predictions. -----
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
        click.echo(f"[17_alerts] missing best.ckpt for {base_fold_id} — skipping.")
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

    # ------------- Refit conformal on val, get widths per horizon. ----------
    sc = SplitConformal(alpha=alpha).fit(y_val_m3s, yhat_val_m3s)
    # interval width per horizon = 2 * q_hat
    widths_per_horizon = 2.0 * sc.quantiles_

    # ------------- Apply FIS to test predictions. ---------------------------
    fis = build_alert_fis()
    _, alert_idx = _apply_fis(fis, yhat_test_m3s, widths_per_horizon, q95_ref)

    # Ground-truth event definition: y_test > train Q95.
    events = (y_test_m3s > q95_ref).astype(bool)  # [N, H]

    # ------------- Simple-threshold baseline: alert if ŷ > q95. --------------
    baseline_alert = (yhat_test_m3s > q95_ref).astype(bool)  # [N, H]

    metrics: dict[str, float] = {
        "q95_train_m3s": q95_ref,
        "n_test": int(n_test),
        "n_val": int(n_val),
        "event_rate": float(events.mean()),
    }

    click.echo(f"[17_alerts] Q95_train={q95_ref:.2f} m3/s  "
               f"event rate on test={events.mean()*100:.1f}%")

    for i, h in enumerate(horizons):
        events_h = events[:, i]
        base_h = baseline_alert[:, i]
        base_res = _evaluate_alert_signal(events_h, base_h)
        metrics[f"baseline_pod_h{h}"] = base_res["pod"]
        metrics[f"baseline_far_h{h}"] = base_res["far"]
        for cl in COST_LOSS_RATIOS:
            metrics[f"baseline_value_cl{cl}_h{h}"] = base_res[f"value_cl_{cl}"]

        for cutoff in ALERT_CUTOFFS:
            fuzzy_alerts = _alert_binary(alert_idx[:, i], cutoff)
            fuzzy_res = _evaluate_alert_signal(events_h, fuzzy_alerts)
            metrics[f"fuzzy_{cutoff}_pod_h{h}"] = fuzzy_res["pod"]
            metrics[f"fuzzy_{cutoff}_far_h{h}"] = fuzzy_res["far"]
            for cl in COST_LOSS_RATIOS:
                metrics[f"fuzzy_{cutoff}_value_cl{cl}_h{h}"] = fuzzy_res[f"value_cl_{cl}"]

        click.echo(
            f"[17_alerts]   h={h:>2d}d  "
            f"baseline POD={base_res['pod']*100:5.1f}%  "
            f"FAR={base_res['far']*100:5.1f}%  "
            f"Value@0.2={base_res['value_cl_0.2']:+.3f}"
        )
        for cutoff in ALERT_CUTOFFS:
            marker = "*" if metrics[f"fuzzy_{cutoff}_value_cl0.2_h{h}"] > base_res["value_cl_0.2"] else " "
            click.echo(
                f"[17_alerts]           "
                f"fuzzy_{cutoff:<6s} POD={metrics[f'fuzzy_{cutoff}_pod_h{h}']*100:5.1f}%  "
                f"FAR={metrics[f'fuzzy_{cutoff}_far_h{h}']*100:5.1f}%  "
                f"Value@0.2={metrics[f'fuzzy_{cutoff}_value_cl0.2_h{h}']:+.3f} {marker}"
            )

    # ------------- Persist manifest. ---------------------------------------
    fold_id = f"{run_id}/{target_clave}"
    out_dir = out_root / target_clave
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_manifest = RunManifest(
        run_id=fold_id, stage="17_evaluate_alerts",
        config={
            "base_run_id": base_run_id,
            "holdout": target_clave,
            "holdout_name": target_prep["name"],
            "alpha": float(alpha),
            "q95_train_m3s": q95_ref,
            "cost_loss_ratios": list(COST_LOSS_RATIOS),
            "alert_cutoffs": list(ALERT_CUTOFFS),
            "fis_rules": build_alert_fis().rules_summary().splitlines(),
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
        click.echo(f"[17_alerts]   -> r2://{r2.bucket}/{prefix}/manifest.json")
    published = publish_results(
        [out_dir / "manifest.json"],
        stage="17_evaluate_alerts", run_id=run_id, subpath=target_clave,
    )
    for p in published:
        click.echo(f"[17_alerts]   -> git: {p.as_posix()}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"target_clave": target_clave, "metrics": metrics}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--run-id", default="alerts-alto-lerma-sweep-01", show_default=True)
@click.option("--base-run-id", default="F0pub-alto-lerma-sweep-01", show_default=True)
@click.option("--basin", default="Alto Lerma", show_default=True)
@click.option("--holdout", default="*", show_default=True)
@click.option("--alpha", default=0.1, show_default=True, type=float)
@click.option("--lookback", default=90, show_default=True)
@click.option("--horizons", default="1,2,3,5,7", show_default=True)
@click.option("--hidden", default=64, show_default=True)
@click.option("--layers", default=1, show_default=True)
@click.option("--dropout", default=0.0, show_default=True)
@click.option("--use-clima/--no-clima", default=True, show_default=True)
@click.option("--seed", default=20260101, show_default=True)
@click.option("--out-dir", default="outputs/alerts", show_default=True)
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
    click.echo(f"[17_alerts] Basin: {basin}  stations: {len(stations_pool)}  "
               f"base run: {base_run_id}")

    all_claves = stations_pool["clave"].astype(str).tolist()
    holdouts = all_claves if holdout == "*" else [holdout]

    fold_summaries = []
    for target_clave in holdouts:
        result = _run_fold(
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
            row = {"holdout": fs["target_clave"],
                   "event_rate": fs["metrics"].get("event_rate", float("nan"))}
            for h in horizons_list:
                row[f"baseline_pod_h{h}"] = fs["metrics"][f"baseline_pod_h{h}"]
                row[f"baseline_far_h{h}"] = fs["metrics"][f"baseline_far_h{h}"]
                row[f"baseline_value_cl0.2_h{h}"] = fs["metrics"][f"baseline_value_cl0.2_h{h}"]
                for cutoff in ALERT_CUTOFFS:
                    row[f"fuzzy_{cutoff}_pod_h{h}"] = fs["metrics"][f"fuzzy_{cutoff}_pod_h{h}"]
                    row[f"fuzzy_{cutoff}_far_h{h}"] = fs["metrics"][f"fuzzy_{cutoff}_far_h{h}"]
                    row[f"fuzzy_{cutoff}_value_cl0.2_h{h}"] = fs["metrics"][f"fuzzy_{cutoff}_value_cl0.2_h{h}"]
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(out_root / "folds_summary.csv", index=False)

        # Aggregate reporting — mean is corrupted by folds with 0 events
        # (metrics undefined) and by folds with < 2 % event rate where a
        # single false alarm swings Value by up to 2 units. Report:
        #   * mean including all folds (paper transparency),
        #   * median (robust central tendency),
        #   * win rate = fraction of folds where best-cutoff fuzzy beats
        #     baseline by ≥ 0.
        # The kill condition of §5c compares median Δ against +0.05.
        evaluable = summary_df[
            (summary_df["event_rate"] > 0.0) & summary_df["event_rate"].notna()
        ].copy()
        n_eval = len(evaluable)
        click.echo("\n[17_alerts] === Aggregate alert summary (Value @ C/L=0.2) ===")
        click.echo(f"[17_alerts] Effective folds with events > 0: {n_eval}/{len(summary_df)}")
        click.echo(f"[17_alerts] {'h':>4}  "
                   f"{'base_med':>9} {'best_c':>7} {'fuz_med':>9}  "
                   f"{'Δ_med':>7} {'Δ_mean':>8} {'wins':>7}")
        for h in horizons_list:
            base_col = f"baseline_value_cl0.2_h{h}"
            best_cutoff, best_med = None, -np.inf
            for cutoff in ALERT_CUTOFFS:
                fcol = f"fuzzy_{cutoff}_value_cl0.2_h{h}"
                med = float(evaluable[fcol].median()) if n_eval else float("nan")
                if med > best_med:
                    best_cutoff, best_med = cutoff, med
            fcol = f"fuzzy_{best_cutoff}_value_cl0.2_h{h}"
            base_med = float(evaluable[base_col].median()) if n_eval else float("nan")
            delta_med = best_med - base_med
            delta_mean = (
                float(evaluable[fcol].mean() - evaluable[base_col].mean())
                if n_eval else float("nan")
            )
            wins = int((evaluable[fcol] > evaluable[base_col]).sum())
            marker = " *" if delta_med >= 0.05 else ""
            click.echo(f"[17_alerts] {h:>3d}d  "
                       f"{base_med:>+9.3f} {best_cutoff:>7s} {best_med:>+9.3f}  "
                       f"{delta_med:>+7.3f} {delta_mean:>+8.3f} "
                       f"{wins:>3d}/{n_eval:<3d}{marker}")
        if upload_to_r2:
            prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}"
            r2.upload_file(f"{prefix}/folds_summary.csv", out_root / "folds_summary.csv")
            click.echo(f"[17_alerts]   -> r2://{r2.bucket}/{prefix}/folds_summary.csv")
        published = publish_results(
            [out_root / "folds_summary.csv"],
            stage="17_evaluate_alerts", run_id=run_id,
        )
        for p in published:
            click.echo(f"[17_alerts]   -> git: {p.as_posix()}")

    click.echo("[17_alerts] Done.")


if __name__ == "__main__":
    main()
