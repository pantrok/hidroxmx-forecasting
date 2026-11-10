#!/usr/bin/env python
"""RJ-2: why conformal tail coverage collapses.

Works on the regenerated predictions of stage 30, for the 52 conformal folds and all
horizons, with the definitions frozen in docs/revision_log.md. Nothing here is
retrained: the two variants change the conformal score only. Tail means Q >= q95 of
the training window; clipped discharge is primary and raw discharge a sensitivity.
"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from hidroxmx.revision import (
    ALPHA,
    BASIN_SLUGS,
    HORIZONS,
    TYPE_DIAGNOSTIC,
    clean,
    folds_of,
    update_numbers,
)
from hidroxmx.uq import (
    adaptive_interval,
    adjacent_miss_pairs,
    log_score_interval,
    miss_run_lengths,
    tail_events,
)

SOURCE = "scripts/32_rj2_tail_diagnostics.py"
GAMMA = 0.005
RISE_PP = 20.0
MIN_Q99_DAYS = 20
MIN_KS_DAYS = 10
MIN_FIT_POINTS = 10
METHODS = ("original", "log", "aci")
REPORTED_H = (1, 7)


def load_long(preds: Path, mirror: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One row per fold, horizon and target day, for calibration and for test."""
    tests, cals, aci_rows = [], [], []
    for basin, slug in BASIN_SLUGS.items():
        for clave in folds_of(mirror, f"UQ-conformal-{slug}-sweep-01"):
            d = np.load(preds / f"{clave}.npz")
            y_val, yh_val, y, yh = d["y_val"], d["yhat_val"], d["y_test"], d["yhat_test"]
            log_lo, log_hi, _ = log_score_interval(y_val, yh_val, yh, ALPHA)
            base = {"basin": basin, "fold": clave, "q50": float(d["q50"]), "q90": float(d["q90"]),
                    "q95": float(d["q95"]), "q99": float(d["q99"])}
            for j, h in enumerate(HORIZONS):
                q = float(d["q_conformal"][j])
                aci = adaptive_interval(np.abs(y_val[:, j] - yh_val[:, j]), y[:, j], yh[:, j],
                                        d["t0_test"], d["at_test"][:, j],
                                        alpha=ALPHA, gamma=GAMMA)
                tests.append(pd.DataFrame({
                    **base, "h": h, "date": d["at_test"][:, j].astype("datetime64[ns]"),
                    "y": y[:, j], "raw": d["raw_test"][:, j], "yhat": yh[:, j],
                    "lo_original": yh[:, j] - q, "hi_original": yh[:, j] + q,
                    "lo_log": log_lo[:, j], "hi_log": log_hi[:, j],
                    "lo_aci": aci.lower, "hi_aci": aci.upper,
                }))
                cals.append(pd.DataFrame({
                    **base, "h": h, "date": d["at_val"][:, j].astype("datetime64[ns]"),
                    "y": y_val[:, j], "raw": d["raw_val"][:, j], "yhat": yh_val[:, j],
                }))
                aci_rows.append({"basin": basin, "fold": clave, "h": h,
                                 "clamped_share": aci.clamped_share,
                                 "alpha_end": float(aci.alpha_path[-1])})
    return pd.concat(tests, ignore_index=True), pd.concat(cals, ignore_index=True), \
        pd.DataFrame(aci_rows)


def covered(frame: pd.DataFrame, method: str, column: str = "y") -> pd.Series:
    return (frame[column] >= frame[f"lo_{method}"]) & (frame[column] <= frame[f"hi_{method}"])


def tail_counts(test: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, frame in (("calibration", cal), ("test", test)):
        tail = frame["y"] >= frame["q95"]
        per_fold = tail.groupby([frame["basin"], frame["h"], frame["fold"]]).sum()
        for (basin, h), sub in frame.groupby(["basin", "h"], sort=False):
            n_tail = int(tail.loc[sub.index].sum())
            counts = per_fold.loc[basin, h]
            rows.append({"basin": basin, "h": h, "window": window, "days": len(sub),
                         "tail_days": n_tail, "tail_share_pct": 100 * n_tail / len(sub),
                         "folds": int(counts.size), "folds_without_tail": int((counts == 0).sum())})
    return pd.DataFrame(rows)


def error_scale(test: pd.DataFrame, cal: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    deciles, slopes, tails = [], [], []
    ks_station = []
    for window, frame in (("calibration", cal), ("test", test)):
        f = frame.assign(r=frame["y"] - frame["yhat"])
        f["abs_r"] = f["r"].abs()
        f["abs_rn"] = f["abs_r"] / f["q95"]
        grouped = f.groupby(["basin", "fold", "h"], sort=False)
        f["decile"] = np.ceil(10 * grouped["y"].rank(method="first", pct=True)).astype(int)
        deciles.append(f.groupby(["basin", "h", "decile"], sort=False)["abs_rn"]
                       .median().rename("median_abs_r_over_q95").reset_index()
                       .assign(window=window))
        for (basin, fold, h), sub in grouped:
            nz = sub[sub["abs_r"] > 0]
            x = np.log1p(nz["y"].to_numpy())
            if len(nz) >= MIN_FIT_POINTS and np.ptp(x) > 0:
                slope = np.polyfit(x, np.log(nz["abs_r"].to_numpy()), 1)[0]
                slopes.append({"basin": basin, "fold": fold, "h": h, "window": window,
                               "slope": float(slope), "n": len(nz)})
        tail = f[f["y"] >= f["q95"]].assign(rn=lambda t: t["r"] / t["q95"])
        for (basin, h), sub in tail.groupby(["basin", "h"], sort=False):
            tails.append({"basin": basin, "h": h, "window": window, "tail_days": len(sub),
                          "abs_r_q50": sub["abs_rn"].quantile(0.5),
                          "abs_r_q90": sub["abs_rn"].quantile(0.9),
                          "signed_r_median": sub["rn"].median()})
    tails_df = pd.DataFrame(tails)

    def tail_of(frame: pd.DataFrame) -> pd.DataFrame:
        t = frame[frame["y"] >= frame["q95"]]
        return t.assign(rn=(t["y"] - t["yhat"]) / t["q95"])

    tc, tt = tail_of(cal), tail_of(test)
    pooled = []
    for (basin, h), sub_t in tt.groupby(["basin", "h"], sort=False):
        sub_c = tc[(tc["basin"] == basin) & (tc["h"] == h)]
        if len(sub_c) and len(sub_t):
            res = ks_2samp(sub_c["rn"], sub_t["rn"])
            pooled.append({"basin": basin, "h": h, "n_cal": len(sub_c), "n_test": len(sub_t),
                           "ks_d": res.statistic, "ks_p": res.pvalue})
        for fold, ft in sub_t.groupby("fold"):
            fc = sub_c[sub_c["fold"] == fold]
            if len(fc) >= MIN_KS_DAYS and len(ft) >= MIN_KS_DAYS:
                res = ks_2samp(fc["rn"], ft["rn"])
                ks_station.append({"basin": basin, "h": h, "fold": fold, "n_cal": len(fc),
                                   "n_test": len(ft), "ks_d": res.statistic, "ks_p": res.pvalue})
    slopes_df = pd.DataFrame(slopes)
    slope_summary = (slopes_df.groupby(["basin", "h", "window"], sort=False)["slope"]
                     .agg(median_slope="median", folds="count").reset_index())
    return (pd.concat(deciles, ignore_index=True), slope_summary, tails_df,
            pd.DataFrame(pooled), pd.DataFrame(ks_station))


def band_of(frame: pd.DataFrame, column: str) -> pd.Series:
    v = frame[column]
    out = pd.Series("q90-q95", index=frame.index)
    out[v < frame["q50"]] = "<q50"
    out[(v >= frame["q50"]) & (v < frame["q90"])] = "q50-q90"
    out[v >= frame["q95"]] = ">=q95"
    return out


def coverage_bands(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for basis in ("y", "raw"):
        frame = test[test[basis].notna()]
        bands = band_of(frame, basis)
        top = frame[basis] >= frame["q99"]
        for method in METHODS:
            hit = covered(frame, method, basis)
            for band in ("<q50", "q50-q90", "q90-q95", ">=q95", ">=q99"):
                mask = top if band == ">=q99" else bands == band
                per_fold = hit[mask].groupby([frame.loc[mask, "basin"], frame.loc[mask, "h"],
                                              frame.loc[mask, "fold"]]).agg(["mean", "size"])
                for (basin, h), sub in per_fold.groupby(level=[0, 1], sort=False):
                    days = int(sub["size"].sum())
                    if band == ">=q99" and days < MIN_Q99_DAYS:
                        continue
                    rows.append({"basis": "clipped" if basis == "y" else "raw", "method": method,
                                 "band": band, "basin": basin, "h": h,
                                 "coverage_pct": 100 * sub["mean"].mean(),
                                 "folds": len(sub), "days": days})
    return pd.DataFrame(rows)


def clustering(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, event_rows = [], []
    tail = test[test["y"] >= test["q95"]]
    for (basin, h), sub in tail.groupby(["basin", "h"], sort=False):
        for method in METHODS:
            first, second, runs = [], [], []
            n_events = peak_cov = any_cov = 0
            for fold, f in sub.groupby("fold", sort=False):
                f = f.sort_values("date")
                dates = f["date"].to_numpy().astype("datetime64[D]")
                hit = covered(f, method).to_numpy()
                miss = ~hit
                a, b = adjacent_miss_pairs(dates, miss)
                first.append(a)
                second.append(b)
                runs.append(miss_run_lengths(dates, miss))
                for idx in tail_events(dates):
                    y = f["y"].to_numpy()[idx]
                    peak = idx[int(np.argmax(y))]
                    n_events += 1
                    peak_cov += int(hit[peak])
                    any_cov += int(hit[idx].any())
                    if method == "original":
                        rec = f.iloc[peak]
                        event_rows.append({
                            "basin": basin, "h": h, "fold": fold,
                            "start": str(dates[idx[0]]), "end": str(dates[idx[-1]]),
                            "peak_date": str(dates[peak]), "tail_days": len(idx),
                            "peak_m3s": rec["y"], "raw_peak_m3s": rec["raw"], "q95": rec["q95"],
                            "peak_over_q95": rec["y"] / rec["q95"],
                            **{f"peak_covered_{m}": bool(covered(f.iloc[[peak]], m).iloc[0])
                               for m in METHODS},
                            **{f"any_covered_{m}": bool(covered(f.iloc[idx], m).any())
                               for m in METHODS},
                        })
            a = np.concatenate(first) if first else np.empty(0, bool)
            b = np.concatenate(second) if second else np.empty(0, bool)
            r = np.concatenate(runs) if runs else np.empty(0, int)
            varied = a.size > 1 and a.std() > 0 and b.std() > 0
            rows.append({
                "basin": basin, "h": h, "method": method, "tail_days": len(sub),
                "miss_pct": 100 * float((~covered(sub, method)).mean()),
                "adjacent_pairs": int(a.size),
                "lag1_autocorrelation": float(np.corrcoef(a, b)[0, 1]) if varied else np.nan,
                "p_miss_after_miss": float(b[a].mean()) if a.any() else np.nan,
                "p_miss_after_cover": float(b[~a].mean()) if (~a).any() else np.nan,
                "runs": int(r.size), "run_mean": float(r.mean()) if r.size else np.nan,
                "run_max": int(r.max()) if r.size else 0,
                "runs_3_plus_pct": 100 * float((r >= 3).mean()) if r.size else np.nan,
                "events": n_events,
                "peak_covered_pct": 100 * peak_cov / n_events if n_events else np.nan,
                "any_covered_pct": 100 * any_cov / n_events if n_events else np.nan,
            })
    events = pd.DataFrame(event_rows)
    top = (events.sort_values("peak_over_q95", ascending=False)
           .groupby(["basin", "h"], sort=False).head(3)
           .sort_values(["h", "basin", "peak_over_q95"], ascending=[True, True, False]))
    return pd.DataFrame(rows), top, events


def variants(test: pd.DataFrame, aci: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tail = test["y"] >= test["q95"]
    for method in METHODS:
        hit = covered(test, method)
        width = (test[f"hi_{method}"] - test[f"lo_{method}"]) / test["q95"]
        marg = hit.groupby([test["basin"], test["h"], test["fold"]]).mean()
        tails = hit[tail].groupby([test.loc[tail, "basin"], test.loc[tail, "h"],
                                   test.loc[tail, "fold"]]).mean()
        wide = width.groupby([test["basin"], test["h"], test["fold"]]).mean()
        wide_m3 = (test[f"hi_{method}"] - test[f"lo_{method}"]) \
            .groupby([test["basin"], test["h"], test["fold"]]).mean()
        for basin in BASIN_SLUGS:
            for h in HORIZONS:
                t = tails.loc[basin, h]
                rows.append({"basin": basin, "h": h, "method": method,
                             "marginal_pct": 100 * marg.loc[basin, h].mean(),
                             "tail_pct": 100 * t.mean(), "tail_folds": len(t),
                             "folds_without_tail": len(marg.loc[basin, h]) - len(t),
                             "width_over_q95": wide.loc[basin, h].mean(),
                             "width_m3s": wide_m3.loc[basin, h].mean()})
    out = pd.DataFrame(rows)
    share = aci.groupby(["basin", "h"], sort=False)["clamped_share"].mean().rename("aci_clamped")
    return out.merge(share.reset_index(), on=["basin", "h"], how="left")


def table7_link(test: pd.DataFrame) -> pd.DataFrame:
    """Tail coverage against the test-window Q95, the definition of Table 7."""
    rows = []
    for (basin, h, fold), f in test.groupby(["basin", "h", "fold"], sort=False):
        threshold = np.quantile(f["y"], 0.95)
        t = f[f["y"] >= threshold]
        rows.append({"basin": basin, "h": h, "fold": fold,
                     **{m: 100 * float(covered(t, m).mean()) for m in METHODS}})
    return (pd.DataFrame(rows).groupby(["basin", "h"], sort=False)[list(METHODS)]
            .mean().reset_index())


def reading(var: pd.DataFrame) -> pd.DataFrame:
    wide = var.pivot_table(index=["basin", "h"], columns="method", values="tail_pct", sort=False)
    wide = wide.reset_index()
    wide["rise_log_pp"] = wide["log"] - wide["original"]
    wide["rise_aci_pp"] = wide["aci"] - wide["original"]

    def label(r) -> str:
        a, b = r["rise_log_pp"] >= RISE_PP, r["rise_aci_pp"] >= RISE_PP
        if a and b:
            return "Both causes contribute"
        if a:
            return "Homoscedastic score dominates"
        if b:
            return "Calibration-to-test drift dominates"
        return "Tail out of reach with this backbone"

    wide["reading"] = wide.apply(label, axis=1)
    return wide


def rows_by(frame: pd.DataFrame, keys: list[str], fields: list[str]) -> dict:
    def label(key: str, value) -> str:
        if key == "basin":
            return BASIN_SLUGS[value].replace("-", "_")
        return f"h{value}" if key == "h" else str(value)

    out: dict = {}
    for _, r in frame.iterrows():
        node = out
        for k in keys[:-1]:
            node = node.setdefault(label(k, r[k]), {})
        node[label(keys[-1], r[keys[-1]])] = {f: r[f] for f in fields}
    return out


@click.command()
@click.option("--out-dir", default="outputs/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
@click.option("--results-dir", default="results/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
def main(out_dir: Path, results_dir: Path):
    tables = results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    test, cal, aci = load_long(out_dir / "preds", out_dir / "r2")
    click.echo(f"[32_rj2] {test['fold'].nunique()} folds, {len(test)} test and {len(cal)} cal rows")
    test.to_parquet(out_dir / "rj2_test.parquet")
    cal.to_parquet(out_dir / "rj2_cal.parquet")

    counts = tail_counts(test, cal)
    deciles, slopes, tails, ks_pool, ks_st = error_scale(test, cal)
    bands = coverage_bands(test)
    clusters, top, events = clustering(test)
    var = variants(test, aci)
    link = table7_link(test)
    read = reading(var)

    outputs = {"T_rj2_tail_counts": counts, "T_rj2_error_deciles": deciles,
               "T_rj2_error_slopes": slopes, "T_rj2_tail_residuals": tails,
               "T_rj2_ks_pooled": ks_pool, "T_rj2_ks_stations": ks_st,
               "T_rj2_coverage_bands": bands, "T_rj2_clustering": clusters,
               "T_rj2_events_top": top, "T_rj2_events": events, "T_rj2_variants": var,
               "T_rj2_table7_link": link, "T_rj2_reading": read}
    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)

    last = str(test["date"].max().date())
    ks_summary = (ks_st.groupby(["basin", "h"], sort=False)
                  .agg(stations=("ks_d", "size"), median_d=("ks_d", "median"),
                       share_p05=("ks_p", lambda p: float((p < 0.05).mean()))).reset_index())
    update_numbers(results_dir / "numbers_revision.json", "rj2", TYPE_DIAGNOSTIC, SOURCE, clean({
        "ultima_fecha_objetivo": last,
        "cola_calibracion_test": rows_by(
            counts, ["basin", "window", "h"],
            ["days", "tail_days", "tail_share_pct", "folds_without_tail"]),
        "pendientes": rows_by(slopes, ["basin", "window", "h"], ["median_slope", "folds"]),
        "residuos_cola": rows_by(tails, ["basin", "window", "h"],
                                 ["tail_days", "abs_r_q50", "abs_r_q90", "signed_r_median"]),
        "ks_cuenca": rows_by(ks_pool, ["basin", "h"], ["n_cal", "n_test", "ks_d", "ks_p"]),
        "ks_estaciones": rows_by(ks_summary, ["basin", "h"], ["stations", "median_d", "share_p05"]),
        "cobertura_bandas": rows_by(
            bands[bands["basis"] == "clipped"], ["basin", "method", "band", "h"],
            ["coverage_pct", "folds", "days"]),
        "cobertura_bandas_crudo": rows_by(
            bands[bands["basis"] == "raw"], ["basin", "method", "band", "h"],
            ["coverage_pct", "folds", "days"]),
        "agrupamiento": rows_by(
            clusters, ["basin", "method", "h"],
            ["tail_days", "miss_pct", "adjacent_pairs", "lag1_autocorrelation",
             "p_miss_after_miss", "p_miss_after_cover", "run_mean", "run_max",
             "runs_3_plus_pct", "events", "peak_covered_pct", "any_covered_pct"]),
        "variantes": rows_by(var, ["basin", "method", "h"],
                             ["marginal_pct", "tail_pct", "tail_folds", "folds_without_tail",
                              "width_over_q95", "width_m3s", "aci_clamped"]),
        "enlace_tabla7": rows_by(link, ["basin", "h"], list(METHODS)),
        "lectura": rows_by(read, ["basin", "h"], ["original", "log", "aci", "rise_log_pp",
                                                  "rise_aci_pp", "reading"]),
    }))

    pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 300,
                  "display.max_colwidth", 42)
    pick = lambda f: f[f["h"].isin(REPORTED_H)]  # noqa: E731
    click.echo(f"last target date: {last}")
    click.echo(pick(counts).round(1).to_string(index=False))
    click.echo(pick(slopes).round(2).to_string(index=False))
    click.echo(pick(tails).round(2).to_string(index=False))
    click.echo(pick(ks_pool).round(3).to_string(index=False))
    click.echo(pick(ks_summary).round(2).to_string(index=False))
    click.echo(pick(var).round(2).to_string(index=False))
    click.echo(pick(link).round(1).to_string(index=False))
    click.echo(pick(read).round(1).to_string(index=False))
    click.echo(pick(clusters[clusters["method"] == "original"]).round(2).to_string(index=False))
    b = bands[(bands["h"] == 1) & (bands["basis"] == "clipped")]
    click.echo(b.pivot_table(index=["basin", "band"], columns="method", values="coverage_pct",
                             sort=False).round(1).to_string())
    click.echo(top[top["h"] == 1][["basin", "fold", "peak_date", "tail_days", "peak_m3s",
                                   "raw_peak_m3s", "peak_over_q95", "peak_covered_original",
                                   "peak_covered_log", "peak_covered_aci"]]
               .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
