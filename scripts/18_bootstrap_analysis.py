#!/usr/bin/env python
"""Stage 18 — paired-bootstrap consolidation of Milestones 3, 4, 5c.

Reads every fold manifest from R2 for the four PUB sweeps (M3 lumped),
the two mechanism sweeps (M4 S-SIG, S-ATTR on Alto Lerma) and the
four M5c alert sweeps, then computes a **paired non-parametric
bootstrap** with 95 % confidence intervals for the paper's central
comparisons:

- **M3**: F0-PUB mean NSE − persistence mean NSE, per (basin, horizon).
- **M4**: F0-txfr mean NSE − F0-PUB lumped mean NSE, per (mechanism,
  horizon) on Alto Lerma.
- **M5c**: fuzzy Value @ C/L=0.2 (best cutoff per fold) − baseline
  Value, per (basin, horizon), on folds with test event_rate > 0.

Every table is written to ``results/tables/`` (git-tracked) so the
manuscript can cite the CI on numeric estimates directly.

Output layout
-------------
``results/tables/bootstrap_m3_pub.csv``
    Δ mean NSE and 95 % CI for the four PUB sweeps × 5 horizons.

``results/tables/bootstrap_m4_mechanisms.csv``
    Δ mean NSE, one row per mechanism × horizon on Alto Lerma. The
    null result is defended by the CI straddling zero.

``results/tables/bootstrap_m5c_alerts.csv``
    Δ median Value and 95 % CI for the four alert sweeps × 5
    horizons; ``kill_cleared`` flags rows whose CI-lower is at or
    above the +0.05 threshold from §5c of the brief.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from hidroxmx.eval import paired_bootstrap, paired_bootstrap_kill_check
from hidroxmx.io import publish_results, r2_from_env


HORIZONS = (1, 2, 3, 5, 7)
ALERT_CUTOFFS = ("YELLOW", "ORANGE", "RED")
CI_LEVEL = 0.95
N_BOOT = 10_000
SEED = 20260721
KILL_THRESHOLD_M3 = 0.05
KILL_THRESHOLD_M5c = 0.05


# --------------------------------------------------------------------------- #
# R2 helpers (same idiom as stages 15/17)
# --------------------------------------------------------------------------- #
def _list_folds(r2, run_id: str) -> list[str]:
    prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}/"
    paginator = r2._client().get_paginator("list_objects_v2")
    claves: set[str] = set()
    for page in paginator.paginate(Bucket=r2.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            parts = rel.split("/")
            if len(parts) >= 2 and parts[-1] == "manifest.json":
                claves.add(parts[0])
    return sorted(claves)


def _load_fold(r2, run_id: str, clave: str) -> dict | None:
    key = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}/{clave}/manifest.json"
    try:
        payload = r2.get_bytes(key)
    except Exception:  # noqa: BLE001
        return None
    return json.loads(payload.decode("utf-8"))


def _fold_metrics(r2, run_id: str) -> list[dict]:
    """Return a list of fold-metric dicts for every completed fold on ``run_id``."""
    out = []
    for clave in _list_folds(r2, run_id):
        mf = _load_fold(r2, run_id, clave)
        if mf is None:
            continue
        row = {"holdout": clave}
        row.update(mf.get("metrics", {}))
        row["_config"] = mf.get("config", {})
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Milestone 3 — F0-PUB vs persistence
# --------------------------------------------------------------------------- #
M3_RUNS = {
    "Alto Lerma":      "F0pub-alto-lerma-sweep-01",
    "Valle de México": "F0pub-valle-de-mexico-sweep-01",
    "Bajo Pánuco":     "F0pub-bajo-panuco-sweep-01",
    "Medio Balsas":    "F0pub-medio-balsas-sweep-01",
}


def _bootstrap_m3(r2) -> pd.DataFrame:
    """Compute mean and median bootstrap for M3.

    Mean is the traditional reporting statistic for NSE aggregates
    (matches Kratzert et al. 2019). Median is robust to catastrophic
    folds (F0-PUB NSE < −1) that appear in Valle de México and Bajo
    Pánuco — the paper reports both so the reader can see the
    outlier sensitivity directly.
    """
    rows = []
    for basin, run_id in M3_RUNS.items():
        click.echo(f"[18_boot] M3 loading {basin} ({run_id})…")
        folds = _fold_metrics(r2, run_id)
        click.echo(f"[18_boot]   {len(folds)} folds")
        for h in HORIZONS:
            a = np.array([f.get(f"nse_h{h}", np.nan) for f in folds], dtype=float)
            b = np.array([f.get(f"persist_nse_h{h}", np.nan) for f in folds], dtype=float)
            for statistic in ("mean", "median"):
                res, kill = paired_bootstrap_kill_check(
                    a, b, threshold=KILL_THRESHOLD_M3,
                    statistic=statistic, n_boot=N_BOOT, ci=CI_LEVEL, seed=SEED,
                )
                rows.append({
                    "basin": basin, "horizon_d": h, "metric": "NSE",
                    "statistic": statistic,
                    "n_folds": res.n,
                    "delta": res.delta, "ci_low": res.ci_low, "ci_high": res.ci_high,
                    "significant": res.significant,
                    "kill_cleared": kill,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Milestone 4 — mechanism vs lumped (Alto Lerma)
# --------------------------------------------------------------------------- #
M4_RUNS = {
    "S-SIG_t1.0": "F0txfr-sig-alto-lerma-sweep-01",
    "S-SIG_t0.3": "F0txfr-sig-alto-lerma-temp03-01",
    "S-ATTR_t1.0": "F0txfr-attr-alto-lerma-sweep-01",
}
M4_BASELINE_RUN = "F0pub-alto-lerma-sweep-01"


def _bootstrap_m4(r2) -> pd.DataFrame:
    rows = []
    baseline = {f["holdout"]: f for f in _fold_metrics(r2, M4_BASELINE_RUN)}
    for mechanism, run_id in M4_RUNS.items():
        click.echo(f"[18_boot] M4 loading {mechanism} ({run_id})…")
        folds = _fold_metrics(r2, run_id)
        paired = [(f, baseline.get(f["holdout"])) for f in folds
                  if baseline.get(f["holdout"]) is not None]
        click.echo(f"[18_boot]   {len(paired)} paired folds")
        for h in HORIZONS:
            a = np.array([f.get(f"nse_h{h}", np.nan) for f, _ in paired], dtype=float)
            b = np.array([lump.get(f"nse_h{h}", np.nan) for _, lump in paired], dtype=float)
            for statistic in ("mean", "median"):
                res = paired_bootstrap(a, b, statistic=statistic,
                                        n_boot=N_BOOT, ci=CI_LEVEL, seed=SEED)
                rows.append({
                    "mechanism": mechanism, "horizon_d": h, "metric": "NSE",
                    "statistic": statistic,
                    "n_folds": res.n,
                    "delta": res.delta, "ci_low": res.ci_low, "ci_high": res.ci_high,
                    "significant": res.significant,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Milestone 5c — fuzzy vs baseline threshold
# --------------------------------------------------------------------------- #
M5C_RUNS = {
    "Alto Lerma":      "alerts-alto-lerma-sweep-01",
    "Valle de México": "alerts-valle-de-mexico-sweep-01",
    "Bajo Pánuco":     "alerts-bajo-panuco-sweep-01",
    "Medio Balsas":    "alerts-medio-balsas-sweep-01",
}


def _best_fuzzy_value(fold: dict, h: int) -> float:
    """Return the fuzzy Value @ C/L=0.2 with the cutoff that maximises it."""
    best = -np.inf
    for cutoff in ALERT_CUTOFFS:
        v = fold.get(f"fuzzy_{cutoff}_value_cl0.2_h{h}", np.nan)
        if np.isfinite(v) and v > best:
            best = v
    return float(best) if np.isfinite(best) else float("nan")


def _bootstrap_m5c(r2) -> pd.DataFrame:
    rows = []
    for basin, run_id in M5C_RUNS.items():
        click.echo(f"[18_boot] M5c loading {basin} ({run_id})…")
        folds = _fold_metrics(r2, run_id)
        # Restrict to folds with events on test (else Value is degenerate).
        folds = [f for f in folds
                 if float(f.get("event_rate", 0.0)) > 0.0
                 and np.isfinite(f.get("event_rate", np.nan))]
        click.echo(f"[18_boot]   {len(folds)} evaluable folds (event_rate > 0)")
        for h in HORIZONS:
            a = np.array([_best_fuzzy_value(f, h) for f in folds], dtype=float)
            b = np.array([f.get(f"baseline_value_cl0.2_h{h}", np.nan) for f in folds],
                         dtype=float)
            res, kill = paired_bootstrap_kill_check(
                a, b, threshold=KILL_THRESHOLD_M5c,
                statistic="median", n_boot=N_BOOT, ci=CI_LEVEL, seed=SEED,
            )
            rows.append({
                "basin": basin, "horizon_d": h, "metric": "Value_CL0.2",
                "statistic": "median",
                "n_folds": res.n,
                "delta": res.delta, "ci_low": res.ci_low, "ci_high": res.ci_high,
                "significant": res.significant,
                "kill_cleared": kill,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--out-dir", default="results/tables", show_default=True)
@click.option("--upload-to-r2", is_flag=True,
              help="Mirror the CSV tables to R2 under paper2/tables/.")
def main(out_dir: str, upload_to_r2: bool):
    load_dotenv(override=False)
    r2 = r2_from_env()
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # M3 -----
    click.echo("\n[18_boot] === Milestone 3 — F0-PUB vs persistence ===")
    df_m3 = _bootstrap_m3(r2)
    m3_path = out_root / "bootstrap_m3_pub.csv"
    df_m3.to_csv(m3_path, index=False)
    click.echo(f"[18_boot] wrote {m3_path.as_posix()}")
    for statistic in ("mean", "median"):
        click.echo(f"[18_boot] --- statistic: {statistic} ---")
        for basin in df_m3["basin"].unique():
            sub = df_m3[(df_m3["basin"] == basin) & (df_m3["statistic"] == statistic)]
            click.echo(f"[18_boot]   {basin}:")
            for _, r in sub.iterrows():
                marker = " *" if r["kill_cleared"] else ""
                click.echo(f"[18_boot]     h={int(r['horizon_d']):>2d}d  "
                           f"ΔNSE={r['delta']:+.3f} "
                           f"CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                           f"  n={int(r['n_folds'])}{marker}")

    # M4 -----
    click.echo("\n[18_boot] === Milestone 4 — mechanism vs lumped (Alto Lerma) ===")
    df_m4 = _bootstrap_m4(r2)
    m4_path = out_root / "bootstrap_m4_mechanisms.csv"
    df_m4.to_csv(m4_path, index=False)
    click.echo(f"[18_boot] wrote {m4_path.as_posix()}")
    for statistic in ("mean", "median"):
        click.echo(f"[18_boot] --- statistic: {statistic} ---")
        for mech in df_m4["mechanism"].unique():
            sub = df_m4[(df_m4["mechanism"] == mech) & (df_m4["statistic"] == statistic)]
            click.echo(f"[18_boot]   {mech}:")
            for _, r in sub.iterrows():
                marker = " sig" if r["significant"] else ""
                click.echo(f"[18_boot]     h={int(r['horizon_d']):>2d}d  "
                           f"ΔNSE={r['delta']:+.4f} "
                           f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                           f"  n={int(r['n_folds'])}{marker}")

    # M5c -----
    click.echo("\n[18_boot] === Milestone 5c — fuzzy vs baseline (Value @ C/L=0.2) ===")
    df_m5c = _bootstrap_m5c(r2)
    m5c_path = out_root / "bootstrap_m5c_alerts.csv"
    df_m5c.to_csv(m5c_path, index=False)
    click.echo(f"[18_boot] wrote {m5c_path.as_posix()}")
    for basin in df_m5c["basin"].unique():
        sub = df_m5c[df_m5c["basin"] == basin]
        click.echo(f"[18_boot]   {basin}:")
        for _, r in sub.iterrows():
            marker = " *" if r["kill_cleared"] else ""
            click.echo(f"[18_boot]     h={int(r['horizon_d']):>2d}d  "
                       f"ΔVal={r['delta']:+.3f} "
                       f"CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                       f"  n={int(r['n_folds'])}{marker}")

    if upload_to_r2:
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + "/tables"
        for p in (m3_path, m4_path, m5c_path):
            r2.upload_file(f"{prefix}/{p.name}", p)
            click.echo(f"[18_boot]   -> r2://{r2.bucket}/{prefix}/{p.name}")
    published = publish_results(
        [m3_path, m4_path, m5c_path],
        stage="18_bootstrap", run_id="paper2-bootstrap",
    )
    for p in published:
        click.echo(f"[18_boot]   -> git: {p.as_posix()}")


if __name__ == "__main__":
    main()
