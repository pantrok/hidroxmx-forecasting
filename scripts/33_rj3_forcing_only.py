#!/usr/bin/env python
"""RJ-3: forcing-only forecaster shown at held-out station GANSL.

F0-PUB without its eight discharge channels, leave-one-out over the seven Bajo Pánuco
stations with complete forcing in the v2026.06 snapshot. At the target it receives
forcings and the station's training-window statistics, and no recent discharge: a site
with a past record and no current gauge. See docs/revision_log.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch import nn

from hidroxmx.alert import CATEGORIES, build_alert_fis
from hidroxmx.data.features import (
    TARGET_COL,
    TARGET_LOG_COL,
    build_features,
    standardise,
    target_stats_m3s,
)
from hidroxmx.data.splits import TemporalSplit
from hidroxmx.data.streams import LocalRoots
from hidroxmx.data.windows import WindowSpec, collect_windows
from hidroxmx.eval import kge, nse
from hidroxmx.io import seed_everything
from hidroxmx.models.forecaster import LSTMEncDecConfig, LSTMEncoderDecoder
from hidroxmx.revision import (
    ALPHA,
    HORIZONS,
    LOOKBACK,
    TYPE_DIAGNOSTIC,
    alert_levels,
    check_snapshot,
    clean,
    load_climate,
    load_target,
    neighbours_of,
    read_manifest,
    update_numbers,
)
from hidroxmx.uq import order_statistic_rank

SOURCE = "scripts/33_rj3_forcing_only.py"
FOLDS = ("ATCHD", "BLMSL", "GANSL", "OLVVC", "PSNVC", "PUJSL", "STRSL")
TARGET = "GANSL"
SEED = 20260101
FORCING = ["precip_mm", "precip_mm_ma7", "tmax_c", "tmax_c_ma7", "tmin_c", "tmin_c_ma7"]
REPORT_H = (1, 3, 7)
HIDDEN, BATCH, EPOCHS, LR, PATIENCE = 64, 128, 40, 5e-4, 6
EVENT = ("2025-10-01", "2025-10-31")
DRY = ("2024-03-01", "2024-04-30")
RECORD_END = "2025-12-31"


def prepare(roots: LocalRoots, row: pd.Series, split: TemporalSplit) -> dict:
    target = load_target(roots, str(row["clave"]))
    calendar = pd.date_range(target["fecha"].min(), RECORD_END, freq="D")
    target = target.set_index("fecha").reindex(calendar).rename_axis("fecha").reset_index()
    clima = load_climate(roots, neighbours_of(row))
    series, cols, clip = build_features(target, clima, use_clima=True,
                                        train_mask=split.mask(target["fecha"], "train"),
                                        clip_upper=None)
    if not set(FORCING) <= set(cols):
        raise SystemExit(f"{row['clave']}: forcing columns missing from {cols}")
    mask_train = split.mask(series["fecha"], "train")
    std, stats = standardise(series, mask_train, cols, TARGET_LOG_COL)
    spec = WindowSpec(lookback=LOOKBACK, horizons=HORIZONS, feature_cols=FORCING,
                      target_col=TARGET_LOG_COL)
    feats = std[FORCING].to_numpy(dtype=np.float32)
    dates = std["fecha"].to_numpy()
    origins = [t for t in range(LOOKBACK - 1, len(std))
               if not np.isnan(feats[t - LOOKBACK + 1:t + 1]).any()]
    x_all = np.stack([feats[t - LOOKBACK + 1:t + 1] for t in origins])
    train_q = series.loc[mask_train, [TARGET_COL, "fecha"]].dropna()
    climatology = train_q.groupby(train_q["fecha"].dt.dayofyear)[TARGET_COL].mean()
    mu, sigma = stats[TARGET_LOG_COL]
    return {
        "clave": str(row["clave"]), "mu": mu, "sigma": sigma,
        "stats": target_stats_m3s(series, mask_train, clip),
        "train": collect_windows(std.loc[mask_train], spec),
        "val": collect_windows(std.loc[split.mask(series["fecha"], "val")], spec),
        "test": collect_windows(std.loc[split.mask(series["fecha"], "test")], spec),
        "x_all": x_all, "origin_all": dates[origins].astype("datetime64[D]"),
        "climatology": climatology, "precip": series.set_index("fecha")["precip_mm"],
    }


def new_model() -> LSTMEncoderDecoder:
    return LSTMEncoderDecoder(LSTMEncDecConfig(input_dim=len(FORCING), hidden_dim=HIDDEN,
                                               num_layers=1, horizons=len(HORIZONS), dropout=0.0))


def train_fold(target: str, data: dict[str, dict], out: Path) -> tuple[LSTMEncoderDecoder, dict]:
    seed_everything(SEED)
    donors = [data[c] for c in FOLDS if c != target]
    x_tr = np.concatenate([d["train"]["x"] for d in donors if len(d["train"]["x"])])
    y_tr = np.concatenate([d["train"]["y"] for d in donors if len(d["train"]["y"])])
    x_va = torch.from_numpy(np.concatenate([d["val"]["x"] for d in donors if len(d["val"]["x"])]))
    y_va = torch.from_numpy(np.concatenate([d["val"]["y"] for d in donors if len(d["val"]["y"])]))
    model = new_model()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best, best_epoch, best_state, stale = float("inf"), -1, None, 0
    started = time.time()
    for epoch in range(EPOCHS):
        model.train()
        order = np.arange(len(x_tr))
        np.random.default_rng(SEED + epoch).shuffle(order)
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(torch.from_numpy(x_tr[idx])), torch.from_numpy(y_tr[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_va), y_va))
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, out / f"{target}.pt")
    return model, {"fold": target, "train_windows": len(x_tr), "val_windows": len(x_va),
                   "best_epoch": best_epoch, "best_val_loss": best,
                   "minutes": (time.time() - started) / 60}


def predict(model: LSTMEncoderDecoder, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def to_m3s(z: np.ndarray, fold: dict) -> np.ndarray:
    return np.expm1(z * fold["sigma"] + fold["mu"])


def pooled_quantile(target: str, val_scores: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    """Order statistic of the z-space scores of the retained stations, per horizon."""
    scores = np.concatenate([s for c, s in val_scores.items() if c != target and len(s)])
    rank, _ = order_statistic_rank(scores.shape[0], ALPHA)
    return np.sort(scores, axis=0)[rank - 1], scores.shape[0]


def skill_rows(fold: dict, pred_z: np.ndarray, q_z: np.ndarray, mirror: Path) -> list[dict]:
    y = to_m3s(fold["test"]["y"], fold)
    yhat = to_m3s(pred_z, fold)
    lower = np.maximum(to_m3s(pred_z - q_z, fold), 0.0)
    upper = to_m3s(pred_z + q_z, fold)
    q95 = fold["stats"]["train_p95_m3s"]
    f0 = read_manifest(mirror, "F0pub-bajo-panuco-sweep-01", fold["clave"])["metrics"]
    rows = []
    for h in REPORT_H:
        j = HORIZONS.index(h)
        at = pd.to_datetime(fold["test"]["targets_at"][:, j])
        doy = fold["climatology"].reindex(at.dayofyear).interpolate(limit_direction="both")
        const = np.full(len(y), fold["stats"]["train_mean_m3s"])
        tail = y[:, j] >= q95
        inside = (y[:, j] >= lower[:, j]) & (y[:, j] <= upper[:, j])
        rows.append({
            "fold": fold["clave"], "h": h, "n_test": len(y),
            "nse": nse(y[:, j], yhat[:, j]), "kge": kge(y[:, j], yhat[:, j]),
            "nse_clim_mean": nse(y[:, j], const), "kge_clim_mean": kge(y[:, j], const),
            "nse_clim_doy": nse(y[:, j], doy.to_numpy()),
            "kge_clim_doy": kge(y[:, j], doy.to_numpy()),
            "f0pub_nse": f0[f"nse_h{h}"], "persist_nse": f0[f"persist_nse_h{h}"],
            "observed_max_m3s": float(y[:, j].max()), "forecast_max_m3s": float(yhat[:, j].max()),
            "mean_ratio": float(yhat[:, j].mean() / y[:, j].mean()),
            "coverage_pct": 100 * float(inside.mean()),
            "tail_days": int(tail.sum()),
            "tail_coverage_pct": 100 * float(inside[tail].mean()) if tail.any() else np.nan,
        })
    return rows


def period_table(fold: dict, pred_z: np.ndarray, q_z: np.ndarray, name: str,
                 span: tuple[str, str]) -> tuple[pd.DataFrame, dict]:
    """Forecasts whose target date falls in ``span``, with both alert functions."""
    q95 = fold["stats"]["train_p95_m3s"]
    start, end = (np.datetime64(s) for s in span)
    expected = int((end - start).astype(int)) + 1
    frames, complete = [], {}
    submitted, corrected = build_alert_fis(), build_alert_fis(corrected=True)
    for h in REPORT_H:
        j = HORIZONS.index(h)
        target_dates = fold["origin_all"] + np.timedelta64(h, "D")
        keep = (target_dates >= start) & (target_dates <= end)
        z = pred_z[keep]
        yhat = to_m3s(z[:, j], fold)
        lower = np.maximum(to_m3s(z[:, j] - q_z[j], fold), 0.0)
        upper = to_m3s(z[:, j] + q_z[j], fold)
        width = (upper - lower)[:, None]
        frames.append(pd.DataFrame({
            "period": name, "h": h, "target_date": target_dates[keep],
            "forecast_m3s": yhat, "lower_m3s": lower, "upper_m3s": upper,
            "flow_ratio": yhat / q95, "width_ratio": (upper - lower) / q95,
            "level_corrected": [CATEGORIES[i] for i in
                                alert_levels(corrected, yhat[:, None], width, q95)[:, 0]],
            "level_submitted": [CATEGORIES[i] for i in
                                alert_levels(submitted, yhat[:, None], width, q95)[:, 0]],
        }))
        complete[f"h{h}"] = {"days": int(keep.sum()), "expected": expected}
    return pd.concat(frames, ignore_index=True), complete


def fallback_span(fold: dict, pred_z: np.ndarray) -> tuple[str, str]:
    """Largest 5-day neighbour-mean rainfall of 2023-2025 with complete inputs."""
    rain = fold["precip"].loc["2023-01-01":RECORD_END].rolling(5).sum()
    origins = pd.to_datetime(fold["origin_all"])
    usable = rain[rain.index.isin(origins)].dropna()
    peak = usable.idxmax()
    return str((peak - pd.Timedelta(days=15)).date()), str((peak + pd.Timedelta(days=15)).date())


@click.command()
@click.option("--snapshot-root", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-dir", default="outputs/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
@click.option("--results-dir", default="results/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
@click.option("--reuse-models", is_flag=True,
              help="Load the trained folds of a previous run instead of training again.")
def main(snapshot_root: Path, out_dir: Path, results_dir: Path, reuse_models: bool):
    roots = LocalRoots(snapshot_root)
    click.echo(f"[33_rj3] snapshot verified: {check_snapshot(roots)}")
    mirror = out_dir / "r2"
    work = out_dir / "rj3"
    work.mkdir(parents=True, exist_ok=True)
    tables = results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    cat = pd.read_csv(roots.selected_hidro_csv, dtype={"clave": str})
    split = TemporalSplit.default()
    data = {c: prepare(roots, cat[cat["clave"] == c].iloc[0], split) for c in FOLDS}
    for c, d in data.items():
        click.echo(f"[33_rj3] {c}: windows train/val/test "
                   f"{len(d['train']['x'])}/{len(d['val']['x'])}/{len(d['test']['x'])}, "
                   f"origins {d['origin_all'][0]}..{d['origin_all'][-1]}")

    models, runs = {}, []
    if reuse_models:
        runs = pd.read_csv(tables / "T_rj3_runs.csv").to_dict("records")
    for c in FOLDS:
        if reuse_models:
            models[c] = new_model()
            models[c].load_state_dict(torch.load(work / f"{c}.pt", map_location="cpu"))
            models[c].eval()
            continue
        models[c], info = train_fold(c, data, work)
        runs.append(info)
        click.echo(f"[33_rj3] trained {info}")

    val_scores = {}
    for c in FOLDS:
        d = data[c]
        val_scores[c] = np.abs(d["val"]["y"] - predict(models[c], d["val"]["x"])) \
            if len(d["val"]["x"]) else np.empty((0, len(HORIZONS)))

    skills = []
    for c in FOLDS:
        q_z, _ = pooled_quantile(c, val_scores)
        skills += skill_rows(data[c], predict(models[c], data[c]["test"]["x"]), q_z, mirror)
    skill = pd.DataFrame(skills)

    target = data[TARGET]
    q_z, n_scores = pooled_quantile(TARGET, val_scores)
    pred_all = predict(models[TARGET], target["x_all"])
    event, event_days = period_table(target, pred_all, q_z, "event", EVENT)
    used_event = EVENT
    if any(v["days"] < v["expected"] for v in event_days.values()):
        used_event = fallback_span(target, pred_all)
        click.echo(f"[33_rj3] October 2025 incomplete {event_days}; using {used_event}")
        event, event_days = period_table(target, pred_all, q_z, "event", used_event)
    dry, dry_days = period_table(target, pred_all, q_z, "dry", DRY)
    forecast = pd.concat([event, dry], ignore_index=True)

    skill.to_csv(tables / "T_rj3_skill.csv", index=False)
    forecast.to_csv(tables / "T_rj3_forecast.csv", index=False)
    pd.DataFrame(runs).to_csv(tables / "T_rj3_runs.csv", index=False)

    medians = skill.groupby("h", sort=False)[
        ["nse", "kge", "nse_clim_mean", "kge_clim_mean", "nse_clim_doy", "kge_clim_doy",
         "f0pub_nse", "persist_nse", "coverage_pct"]].median()
    verdict = {f"h{h}": ("skill above day-of-year climatology"
                         if medians.loc[h, "nse"] > medians.loc[h, "nse_clim_doy"]
                         else "no demonstrated skill") for h in REPORT_H}
    per_h = {f"h{h}": medians.loc[h].to_dict() for h in REPORT_H}
    summary = {}
    for name, frame in forecast.groupby(["period", "h"], sort=False):
        summary[f"{name[0]}_h{name[1]}"] = {
            "forecast_max_m3s": frame["forecast_m3s"].max(),
            "upper_max_m3s": frame["upper_m3s"].max(),
            "flow_ratio_max": frame["flow_ratio"].max(),
            "levels_corrected": frame["level_corrected"].value_counts().to_dict(),
            "levels_submitted": frame["level_submitted"].value_counts().to_dict(),
        }
    gansl = skill[skill["fold"] == TARGET].set_index("h")
    update_numbers(results_dir / "numbers_revision.json", "rj3", TYPE_DIAGNOSTIC, SOURCE, clean({
        "objetivo": TARGET, "folds": list(FOLDS), "semilla": SEED,
        "periodo_evento": list(used_event), "periodo_seco": list(DRY),
        "dias_evento": event_days, "dias_seco": dry_days,
        "q95_entrenamiento_m3s": target["stats"]["train_p95_m3s"],
        "q999_entrenamiento_m3s": target["stats"]["clip_upper_m3s"],
        "cuantil_z": {f"h{h}": q_z[HORIZONS.index(h)] for h in REPORT_H},
        "n_scores_calibracion": n_scores,
        "habilidad_mediana": per_h, "lectura_habilidad": verdict,
        "habilidad_objetivo": {f"h{h}": gansl.loc[h].drop("fold").to_dict() for h in REPORT_H},
        "periodos": summary,
        "minutos_totales": sum(r["minutes"] for r in runs),
    }))
    (work / "last_run.json").write_text(json.dumps(clean({"runs": runs}), indent=2),
                                        encoding="utf-8")

    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
    click.echo(skill.round(2).to_string(index=False))
    click.echo(medians.round(2).to_string())
    click.echo(f"reading: {verdict}")
    click.echo(json.dumps(clean(summary), indent=1))


if __name__ == "__main__":
    main()
