#!/usr/bin/env python
"""Regenerate the predictions of the 52 conformal folds and check them.

The reported runs kept checkpoints and manifests, not predictions. This stage
rebuilds every fold of the v2026.06 snapshot, restores its ``best.ckpt`` and writes
the arrays the revision analyses read. It then applies the reproduction gate of
docs/revision_log.md and stops if any fold leaves the tolerance.
"""
from __future__ import annotations

import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from hidroxmx.alert import CATEGORIES, build_alert_fis
from hidroxmx.data.splits import TemporalSplit
from hidroxmx.eval import nse, value_cost_loss
from hidroxmx.io import r2_from_env
from hidroxmx.models.forecaster import LSTMEncDecConfig, LSTMEncoderDecoder
from hidroxmx.revision import (
    ALPHA,
    BASIN_SLUGS,
    COST_LOSS,
    CUTOFFS,
    HORIZONS,
    alert_levels,
    check_snapshot,
    extract_snapshot,
    folds_of,
    list_r2_keys,
    mirror_r2,
    prepare_fold,
    raw_at,
    read_manifest,
)
from hidroxmx.uq import SplitConformal, coverage_and_width, tail_coverage

TOL_NSE = 1e-5
TOL_Q_REL = 1e-4
TOL_VALUE = 1e-6


def same(a: float, b: float, tol: float) -> bool:
    if not (np.isfinite(a) or np.isfinite(b)):
        return True
    return bool(abs(a - b) <= tol)


def run_fold(basin: str, slug: str, clave: str, cat: pd.DataFrame, roots, split, mirror: Path,
             preds_dir: Path, fis) -> dict:
    row = cat[cat["clave"] == clave].iloc[0]
    fold = prepare_fold(roots, row, split)
    if fold is None:
        raise SystemExit(f"{clave}: no data in the snapshot")
    ckpt = mirror / "runs" / f"F0pub-{slug}-sweep-01" / clave / "best.ckpt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = LSTMEncoderDecoder(LSTMEncDecConfig(
        input_dim=len(fold.feature_cols), hidden_dim=64, num_layers=1,
        horizons=len(HORIZONS), dropout=0.0,
    ))
    model.load_state_dict(state["model"])
    model.eval()

    def to_m3s(z: np.ndarray) -> np.ndarray:
        return np.expm1(z * fold.sigma + fold.mu)

    with torch.no_grad():
        pred_val = model(torch.from_numpy(fold.windows_val["x"])).numpy()
        pred_test = model(torch.from_numpy(fold.windows_test["x"])).numpy()
    y_val, y_test = to_m3s(fold.windows_val["y"]), to_m3s(fold.windows_test["y"])
    yhat_val, yhat_test = to_m3s(pred_val), to_m3s(pred_test)

    conformal = SplitConformal(alpha=ALPHA).fit(y_val, yhat_val)
    lower, upper = conformal.predict_interval(yhat_test)
    cover = coverage_and_width(y_test, lower, upper)
    tail = tail_coverage(y_test, lower, upper, threshold_quantile=0.95)

    q95 = float(fold.target_stats["train_p95_m3s"])
    levels = alert_levels(fis, yhat_test, 2.0 * conformal.quantiles_, q95)
    events = y_test > q95

    f0 = read_manifest(mirror, f"F0pub-{slug}-sweep-01", clave)["metrics"]
    uq = read_manifest(mirror, f"UQ-conformal-{slug}-sweep-01", clave)["metrics"]
    alerts = read_manifest(mirror, f"alerts-{slug}-sweep-01", clave)["metrics"]
    n_test = len(y_test)
    failures: list[str] = []
    worst = {"nse": 0.0, "q_rel": 0.0, "coverage": 0.0, "value": 0.0}
    for j, h in enumerate(HORIZONS):
        nse_now = nse(y_test[:, j], yhat_test[:, j])
        nse_ref = f0[f"nse_h{h}"]
        scale = max(1.0, abs(nse_ref)) if np.isfinite(nse_ref) else 1.0
        d_nse = abs(nse_now - nse_ref) / scale
        if np.isfinite(d_nse):
            worst["nse"] = max(worst["nse"], d_nse)
        if not same(nse_now, nse_ref, TOL_NSE * scale):
            failures.append(f"nse_h{h}")
        q_ref = uq[f"conformal_quantile_h{h}_m3s"]
        q_rel = abs(conformal.quantiles_[j] - q_ref) / max(abs(q_ref), 1e-9)
        worst["q_rel"] = max(worst["q_rel"], q_rel)
        if q_rel > TOL_Q_REL:
            failures.append(f"q_h{h}")
        for name, now, ref in (("coverage", cover["coverage"][j], uq[f"coverage_h{h}"]),
                               ("tail", tail[j], uq[f"tail_coverage_q95_h{h}"])):
            if np.isfinite(now) and np.isfinite(ref):
                worst["coverage"] = max(worst["coverage"], abs(now - ref))
            if not same(now, ref, 1.0 / n_test + 1e-12):
                failures.append(f"{name}_h{h}")
        pairs = [("baseline", yhat_test[:, j] > q95)]
        pairs += [(f"fuzzy_{c}", levels[:, j] >= CATEGORIES.index(c)) for c in CUTOFFS]
        for name, alert in pairs:
            now = value_cost_loss(events[:, j], alert, COST_LOSS)
            ref = alerts[f"{name}_value_cl{COST_LOSS}_h{h}"]
            if np.isfinite(now) and np.isfinite(ref):
                worst["value"] = max(worst["value"], abs(now - ref))
            if not same(now, ref, TOL_VALUE):
                failures.append(f"{name}_h{h}")

    np.savez_compressed(
        preds_dir / f"{clave}.npz",
        y_val=y_val, yhat_val=yhat_val, y_test=y_test, yhat_test=yhat_test,
        raw_val=raw_at(fold.raw, fold.windows_val["targets_at"]),
        raw_test=raw_at(fold.raw, fold.windows_test["targets_at"]),
        t0_val=fold.windows_val["t0"].astype("datetime64[D]"),
        t0_test=fold.windows_test["t0"].astype("datetime64[D]"),
        at_val=fold.windows_val["targets_at"].astype("datetime64[D]"),
        at_test=fold.windows_test["targets_at"].astype("datetime64[D]"),
        q_conformal=conformal.quantiles_, horizons=np.array(HORIZONS),
        q50=fold.train_q["q50"], q90=fold.train_q["q90"], q95=q95, q99=fold.train_q["q99"],
        clip_upper=float(fold.target_stats["clip_upper_m3s"]),
        event_rate=float(alerts.get("event_rate", np.nan)),
    )
    return {"basin": basin, "fold": clave, "n_val": len(y_val), "n_test": n_test,
            "max_scaled_nse": worst["nse"], "max_rel_q": worst["q_rel"],
            "max_abs_coverage": worst["coverage"], "max_abs_value": worst["value"],
            "failures": ",".join(failures), "passed": not failures}


@click.command()
@click.option("--snapshot-zip", type=click.Path(exists=True, path_type=Path), required=True,
              help="Archive HidroXAI-MX-v2026.06.zip.")
@click.option("--snapshot-root", type=click.Path(path_type=Path), required=True,
              help="Folder for the unpacked snapshot, outside any synced drive.")
@click.option("--out-dir", default="outputs/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
@click.option("--results-dir", default="results/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
def main(snapshot_zip: Path, snapshot_root: Path, out_dir: Path, results_dir: Path):
    load_dotenv(override=False)
    roots = extract_snapshot(snapshot_zip, snapshot_root)
    found = check_snapshot(roots)
    click.echo(f"[30_rj] snapshot verified: {found}")

    r2 = r2_from_env()
    prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2")
    mirror = out_dir / "r2"
    keys = list_r2_keys(r2, f"{prefix}/runs/")
    suffixes = ("/manifest.json", "/best.ckpt", "forecast_arrays.npz")
    wanted = [k for k in keys if k.endswith(suffixes)]
    mirror_r2(r2, wanted, mirror)
    click.echo(f"[30_rj] R2 mirror ready: {len(wanted)} objects")

    cat = pd.read_csv(roots.selected_hidro_csv, dtype={"clave": str})
    split = TemporalSplit.default()
    preds = out_dir / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    fis = build_alert_fis()

    rows = []
    for basin, slug in BASIN_SLUGS.items():
        for clave in folds_of(mirror, f"UQ-conformal-{slug}-sweep-01"):
            rows.append(run_fold(basin, slug, clave, cat, roots, split, mirror, preds, fis))
            r = rows[-1]
            click.echo(f"[30_rj] {basin:16s} {clave:6s} val={r['n_val']:4d} test={r['n_test']:4d} "
                       f"nse={r['max_scaled_nse']:.1e} q={r['max_rel_q']:.1e} "
                       f"cov={r['max_abs_coverage']:.1e} value={r['max_abs_value']:.1e} "
                       f"{'ok' if r['passed'] else 'FAIL ' + r['failures']}")

    gate = pd.DataFrame(rows)
    tables = results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    gate.to_csv(tables / "T_rj_gate.csv", index=False)
    failed = gate.loc[~gate["passed"], "fold"].tolist()
    if failed:
        raise SystemExit(f"[30_rj] reproduction gate failed for: {failed}")
    click.echo(f"[30_rj] gate passed for {len(gate)} folds")


if __name__ == "__main__":
    main()
