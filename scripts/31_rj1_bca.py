#!/usr/bin/env python
"""RJ-1: BCa intervals for Path B and the sensitivities of Table 8.

Reads the fold manifests and the regenerated predictions of stage 30. Every class
follows docs/revision_log.md. The percentile limits are checked against the published
tables before anything is written, so the intervals compare like with like.
"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

from hidroxmx.alert import CATEGORIES, build_alert_fis
from hidroxmx.eval import value_cost_loss
from hidroxmx.revision import (
    BASIN_SLUGS,
    CLEARED,
    COST_LOSS,
    CUTOFFS,
    HORIZONS,
    SECONDARY_ESTIMATOR,
    THRESHOLD,
    TYPE_DIAGNOSTIC,
    TYPE_PREREGISTERED,
    WATCHED,
    alert_levels,
    bca,
    classify,
    clean,
    folds_of,
    leave_one_out,
    read_manifest,
    sign_test,
    update_numbers,
)

SOURCE = "scripts/31_rj1_bca.py"
TABLE5_CELLS = {
    "Alto Lerma": (5, 7), "Valle de México": (3, 5, 7),
    "Bajo Pánuco": (5, 7), "Medio Balsas": (5, 7),
}
def fold_values(mirror: Path, preds: Path) -> pd.DataFrame:
    """Per-fold values at C/L = 0.2 for both membership functions and all cutoffs."""
    fis = build_alert_fis(corrected=True)
    rows = []
    for basin, slug in BASIN_SLUGS.items():
        for clave in folds_of(mirror, f"alerts-{slug}-sweep-01"):
            m = read_manifest(mirror, f"alerts-{slug}-sweep-01", clave)["metrics"]
            rate = m.get("event_rate", np.nan)
            if not (np.isfinite(rate) and rate > 0):
                continue
            d = np.load(preds / f"{clave}.npz")
            q95 = float(d["q95"])
            y, yhat = d["y_test"], d["yhat_test"]
            levels = alert_levels(fis, yhat, 2.0 * d["q_conformal"], q95)
            events = y > q95
            for j, h in enumerate(HORIZONS):
                row = {"basin": basin, "fold": clave, "h": h, "event_rate": rate,
                       "threshold": m[f"baseline_value_cl{COST_LOSS}_h{h}"]}
                for c in CUTOFFS:
                    row[f"submitted_{c}"] = m[f"fuzzy_{c}_value_cl{COST_LOSS}_h{h}"]
                    row[f"corrected_{c}"] = value_cost_loss(
                        events[:, j], levels[:, j] >= CATEGORIES.index(c), COST_LOSS)
                rows.append(row)
    return pd.DataFrame(rows)


def best_of(frame: pd.DataFrame, prefix: str) -> tuple[np.ndarray, list[str]]:
    """Largest finite value over the cutoffs, as ``_best_fuzzy_value`` of stage 18."""
    cols = [f"{prefix}_{c}" for c in CUTOFFS]
    values = frame[cols].to_numpy(dtype=float)
    finite = np.isfinite(values).any(axis=1)
    best = np.full(len(frame), np.nan)
    best[finite] = np.nanmax(values[finite], axis=1)
    names = [CUTOFFS[int(np.nanargmax(v))] if f else ""
             for v, f in zip(values, finite, strict=True)]
    return best, names


def main_table(values: pd.DataFrame, published: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, folds = [], []
    worst = 0.0
    for (basin, h), frame in values.groupby(["basin", "h"], sort=False):
        a, names = best_of(frame, "submitted")
        b = frame["threshold"].to_numpy(dtype=float)
        res = bca(a, b)
        alt = bca(a, b, SECONDARY_ESTIMATOR)
        ref = published[(published.basin == basin) & (published.horizon_d == h)].iloc[0]
        worst = max(worst, abs(res.estimate - ref.delta), abs(res.pct_low - ref.ci_low),
                    abs(res.pct_high - ref.ci_high))
        delta = a - b
        pos = sign_test(delta, 0.0)
        over = sign_test(delta, THRESHOLD)
        low, high = leave_one_out(a, b)
        rows.append({
            "basin": basin, "h": h, "n": res.n, "T": res.estimate,
            "pct_low": res.pct_low, "pct_high": res.pct_high,
            "bca_low": res.bca_low, "bca_high": res.bca_high,
            "z0": res.z0, "acceleration": res.acceleration, "reason": res.reason,
            "class": classify(res),
            "T_secondary": alt.estimate, "bca_low_secondary": alt.bca_low,
            "bca_high_secondary": alt.bca_high, "class_secondary": classify(alt),
            "sign_pos_k": pos[0], "sign_pos_n": pos[1], "sign_pos_p": pos[2],
            "sign_over_k": over[0], "sign_over_n": over[1], "sign_over_p": over[2],
            "lofo_min": low, "lofo_max": high,
        })
        for clave, va, vb, name, rate in zip(frame["fold"], a, b, names, frame["event_rate"],
                                             strict=True):
            folds.append({"basin": basin, "h": h, "fold": clave, "event_rate": rate,
                          "fuzzy": va, "best_cutoff": name, "threshold": vb, "delta_v": va - vb})
    if worst > 1e-9:
        raise SystemExit(f"[31_rj1] percentile limits do not reproduce Table 8: {worst:.3e}")
    click.echo(f"[31_rj1] Table 8 reproduced, largest difference {worst:.1e}")
    return pd.DataFrame(rows), pd.DataFrame(folds)


def sensitivity(values: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    variants: list[tuple[str, str, str | None]] = [
        ("Main", "submitted", None), ("S1 corrected", "corrected", None)]
    for c in CUTOFFS:
        variants.append((f"S2 {c}", "submitted", c))
        variants.append((f"S1+S2 {c}", "corrected", c))
    base = main.set_index(["basin", "h"])["class"]
    rows = []
    for (basin, h), frame in values.groupby(["basin", "h"], sort=False):
        b = frame["threshold"].to_numpy(dtype=float)
        for name, prefix, cutoff in variants:
            a = frame[f"{prefix}_{cutoff}"].to_numpy(dtype=float) if cutoff \
                else best_of(frame, prefix)[0]
            res = bca(a, b)
            cls = classify(res)
            rows.append({"variant": name, "basin": basin, "h": h, "n": res.n, "T": res.estimate,
                         "bca_low": res.bca_low, "bca_high": res.bca_high, "class": cls,
                         "main_class": base[(basin, h)], "changed": cls != base[(basin, h)]})
    return pd.DataFrame(rows)


def table5(mirror: Path, published: pd.DataFrame) -> pd.DataFrame:
    rows = []
    worst = 0.0
    for basin, horizons in TABLE5_CELLS.items():
        run = f"F0pub-{BASIN_SLUGS[basin]}-sweep-01"
        metrics = [read_manifest(mirror, run, c)["metrics"] for c in folds_of(mirror, run)]
        for h in horizons:
            a = np.array([m.get(f"nse_h{h}", np.nan) for m in metrics], dtype=float)
            b = np.array([m.get(f"persist_nse_h{h}", np.nan) for m in metrics], dtype=float)
            res = bca(a, b)
            ref = published[(published.basin == basin) & (published.horizon_d == h)
                            & (published.statistic == "median")].iloc[0]
            worst = max(worst, abs(res.pct_low - ref.ci_low), abs(res.pct_high - ref.ci_high))
            rows.append({"basin": basin, "h": h, "n": res.n, "T": res.estimate,
                         "pct_low": res.pct_low, "pct_high": res.pct_high,
                         "bca_low": res.bca_low, "bca_high": res.bca_high,
                         "class": classify(res), "reason": res.reason})
    if worst > 1e-9:
        raise SystemExit(f"[31_rj1] percentile limits do not reproduce Table 5: {worst:.3e}")
    click.echo(f"[31_rj1] Table 5 reproduced, largest difference {worst:.1e}")
    return pd.DataFrame(rows)


def by_cell(frame: pd.DataFrame, fields: list[str]) -> dict:
    out: dict = {}
    for _, r in frame.iterrows():
        slug = BASIN_SLUGS[r["basin"]].replace("-", "_")
        out.setdefault(slug, {})[f"h{int(r['h'])}"] = {f: r[f] for f in fields}
    return out


@click.command()
@click.option("--out-dir", default="outputs/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
@click.option("--results-dir", default="results/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
@click.option("--published-dir", default="results/revision_joh/published", show_default=True,
              type=click.Path(path_type=Path))
def main(out_dir: Path, results_dir: Path, published_dir: Path):
    mirror, preds = out_dir / "r2", out_dir / "preds"
    tables = results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    numbers = results_dir / "numbers_revision.json"

    values = fold_values(mirror, preds)
    main_df, folds = main_table(values, pd.read_csv(published_dir / "bootstrap_m5c_alerts.csv"))
    sens = sensitivity(values, main_df)
    t5 = table5(mirror, pd.read_csv(published_dir / "bootstrap_m3_pub.csv"))

    main_df.to_csv(tables / "T_rj1_bca.csv", index=False)
    folds.to_csv(tables / "T_rj1_folds.csv", index=False)
    sens.to_csv(tables / "T_rj1_sensitivity.csv", index=False)
    t5.to_csv(tables / "T_rj1_table5_bca.csv", index=False)

    watched = main_df[[(b, h) in WATCHED for b, h in zip(main_df.basin, main_df.h, strict=True)]]
    changed = watched[watched["class"] != CLEARED]
    disagree = main_df[main_df["class"] != main_df["class_secondary"]]
    fields = ["n", "T", "pct_low", "pct_high", "bca_low", "bca_high", "z0", "acceleration",
              "class", "T_secondary", "bca_low_secondary", "bca_high_secondary",
              "class_secondary", "sign_pos_k", "sign_pos_n", "sign_pos_p", "sign_over_k",
              "sign_over_n", "sign_over_p", "lofo_min", "lofo_max"]
    update_numbers(numbers, "rj1", TYPE_PREREGISTERED, SOURCE, clean({
        "estadistico_principal": "median(fuzzy) - median(threshold), as in Table 8",
        "celdas": by_cell(main_df, fields),
        "cambios_en_celdas_vigiladas": [f"{r.basin} h{r.h}: {r['class']}"
                                        for _, r in changed.iterrows()],
        "discrepancias_con_secundario": [f"{r.basin} h{r.h}: {r['class']} / {r.class_secondary}"
                                         for _, r in disagree.iterrows()],
    }))
    update_numbers(numbers, "rj1_tabla5", TYPE_DIAGNOSTIC, SOURCE, clean({
        "celdas": by_cell(t5, ["n", "T", "pct_low", "pct_high", "bca_low", "bca_high", "class"]),
    }))
    payload: dict = {}
    for name, frame in sens.groupby("variant", sort=False):
        payload[name] = by_cell(frame, ["n", "T", "bca_low", "bca_high", "class", "changed"])
    update_numbers(numbers, "rj1_sensibilidad", TYPE_DIAGNOSTIC, SOURCE, clean(payload))

    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
    click.echo(main_df[["basin", "h", "n", "T", "pct_low", "pct_high", "bca_low", "bca_high",
                        "class", "class_secondary", "sign_over_k", "sign_over_n", "sign_over_p",
                        "lofo_min", "lofo_max"]].round(3).to_string(index=False))
    click.echo("\nundefined reasons: " + str(main_df.loc[main_df["reason"] != "", "reason"]
                                             .value_counts().to_dict()))
    click.echo("\nTable 5:\n" + t5.round(3).to_string(index=False))
    click.echo("\nSensitivity, watched cells:")
    w = sens[[(b, h) in WATCHED for b, h in zip(sens.basin, sens.h, strict=True)]]
    click.echo(w.pivot_table(index=["basin", "h"], columns="variant", values="class",
                             aggfunc="first", sort=False).to_string())
    click.echo(w.pivot_table(index=["basin", "h"], columns="variant", values="T",
                             aggfunc="first", sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
