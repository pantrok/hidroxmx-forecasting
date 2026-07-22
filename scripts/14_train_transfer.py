#!/usr/bin/env python
"""Stage 14 — F0 transfer training with donor-similarity weighting (Path A).

Milestone 4 slice 1: takes the Milestone-3 lumped multi-station
pipeline (stage 12) and adds a **donor-similarity mechanism**. Instead
of treating every donor's samples equally, the loss on each training
sample is weighted by how similar the donor is to the held-out target
in hydrological signature space (S-SIG). This is the first mechanism
of the Path A family enumerated in §4.3 of the experiment spec.

If the similarity-weighted model beats the lumped baseline (Milestone
3, stage 12) on paired PUB folds, the Path A hypothesis clears its
first load-bearing test. If it merely ties, Path A is soft-falsified
and Path B carries the paper.

Key differences vs stage 12
---------------------------
1. Signatures are computed from each donor's *training-window* raw
   streamflow (not the full series — respects the temporal split).
2. Similarity is computed against the target's signature (same
   training window) in a standardised signature space.
3. Weights are turned into a per-sample vector during minibatch
   assembly and passed to a weighted SmoothL1 loss.
4. The output manifest logs the similarity vector so donor weights
   are auditable and reproducible.

The rest of the pipeline (checkpoint resume, R2 mirroring, per-fold
outputs) is identical to stage 12 so results at both stages can be
compared fold-for-fold.
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
    publish_results,
    r2_from_env,
    seed_everything,
)
from hidroxmx.io.checkpoint import collect_rng_state, restore_rng_state
from hidroxmx.transfer import (
    DEFAULT_ATTRIBUTE_KEYS,
    DEFAULT_SIGNATURE_KEYS,
    attribute_vector,
    compute_signatures,
    extract_attributes,
    score_donors,
    signature_vector,
)


DEFAULT_BASIN = "Alto Lerma"
DEFAULT_HOLDOUT = "*"
MIN_TRAIN_ROWS = 500
MIN_TRAIN_WINDOWS = 128


# --------------------------------------------------------------------------- #
# Series loaders (duplicated from stage 12 — kept local for lean imports)
# --------------------------------------------------------------------------- #
def _load_basin_stations(r2, layout, basin):
    manifest = load_selected_stations(r2, layout, kind="hidro")
    if "cuenca" not in manifest.columns:
        raise SystemExit("[14_train] Manifest has no 'cuenca' column.")
    pool = manifest[manifest["cuenca"].astype(str).str.contains(basin, case=False, na=False)]
    if pool.empty:
        raise SystemExit(f"[14_train] No stations match basin={basin!r}.")
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


def _prepare_station(r2, layout, station_row, use_clima, split,
                     spec, feature_cols_ref):
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
    win_train = collect_windows(series_std.loc[mask_train], aligned_spec)
    win_val = collect_windows(series_std.loc[mask_val], aligned_spec)
    win_test = collect_windows(series_std.loc[mask_test], aligned_spec)

    # For S-SIG — signatures from the RAW post-clip streamflow, restricted
    # to the training window (no leakage).
    train_target_m3s = series.loc[mask_train, TARGET_COL].to_numpy(dtype=np.float64)
    signatures = compute_signatures(train_target_m3s)
    # For S-ATTR — static attributes lifted directly from the manifest
    # row (no leakage possible).
    attributes = extract_attributes(station_row)

    return {
        "clave": clave,
        "name": str(station_row.get("nombre", "")),
        "coverage": float(cov),
        "feature_cols": cols,
        "stats": stats,
        "target_stats": tgt_stats,
        "signatures": signatures,
        "attributes": attributes,
        "windows_train": win_train,
        "windows_val": win_val,
        "windows_test": win_test,
    }


def _stack(windows_list):
    xs = [w["x"] for w in windows_list if len(w["x"])]
    ys = [w["y"] for w in windows_list if len(w["y"])]
    if not xs:
        return {"x": np.empty((0, 0, 0), np.float32),
                "y": np.empty((0, 0), np.float32)}
    return {"x": np.concatenate(xs, axis=0), "y": np.concatenate(ys, axis=0)}


def _stack_with_weights(donor_preps, weights):
    """Concatenate donor train windows and emit a matching per-sample weight vector."""
    xs, ys, ws = [], [], []
    for prep, w in zip(donor_preps, weights):
        n = len(prep["windows_train"]["x"])
        if n == 0:
            continue
        xs.append(prep["windows_train"]["x"])
        ys.append(prep["windows_train"]["y"])
        ws.append(np.full(n, float(w), dtype=np.float32))
    if not xs:
        return (np.empty((0, 0, 0), np.float32),
                np.empty((0, 0), np.float32),
                np.empty((0,), np.float32))
    return (np.concatenate(xs, axis=0),
            np.concatenate(ys, axis=0),
            np.concatenate(ws, axis=0))


def _to_batches_weighted(x, y, w, batch_size, shuffle=True, seed=0):
    import torch
    n = len(x)
    order = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield (torch.from_numpy(x[idx]),
               torch.from_numpy(y[idx]),
               torch.from_numpy(w[idx]))


def _weighted_smooth_l1(pred, target, weights, beta=1.0):
    """SmoothL1 loss weighted per-sample. Weights should have mean ≈ 1."""
    import torch
    diff = torch.abs(pred - target)
    piecewise = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    per_sample = piecewise.mean(dim=-1)  # mean across horizons
    return (per_sample * weights).mean()


def _eval_metrics(y_true, y_pred, horizons, prefix=""):
    out = {}
    for i, h in enumerate(horizons):
        out[f"{prefix}nse_h{h}"] = float(nse(y_true[:, i], y_pred[:, i]))
        out[f"{prefix}kge_h{h}"] = float(kge(y_true[:, i], y_pred[:, i]))
        out[f"{prefix}rmse_h{h}_m3s"] = float(rmse(y_true[:, i], y_pred[:, i]))
        out[f"{prefix}pbias_h{h}_pct"] = float(pbias(y_true[:, i], y_pred[:, i]))
    return out


def _persistence_and_climatology(target_prep, horizons):
    win_test = target_prep["windows_test"]
    if len(win_test["x"]) == 0:
        return {}
    mu_t, sigma_t = target_prep["stats"][TARGET_LOG_COL]
    y_true_m3s = np.expm1(win_test["y"] * sigma_t + mu_t)
    persist_x_std = win_test["x"][:, -1, 0].astype(np.float64)
    persist_m3s = np.expm1(persist_x_std * sigma_t + mu_t)
    persist_pred = np.repeat(persist_m3s.reshape(-1, 1), len(horizons), axis=1)
    clim_pred = np.full_like(persist_pred, target_prep["target_stats"]["train_mean_m3s"])
    out = {}
    out.update(_eval_metrics(y_true_m3s, persist_pred, horizons, prefix="persist_"))
    out.update(_eval_metrics(y_true_m3s, clim_pred, horizons, prefix="clim_"))
    return out


# --------------------------------------------------------------------------- #
# One PUB fold with signature-weighted donors
# --------------------------------------------------------------------------- #
def _select_similarity_vector(prep: dict, mechanism: str) -> np.ndarray:
    """Return the vector used for donor-similarity scoring under ``mechanism``.

    ``sig``  → hydrological signatures only.
    ``attr`` → static manifest attributes only.
    ``both`` → concatenation of signature + attribute vectors, standardised
                per-block by :func:`hidroxmx.transfer.similarity.score_donors`.
    """
    if mechanism == "sig":
        return signature_vector(prep["signatures"])
    if mechanism == "attr":
        return attribute_vector(prep["attributes"])
    if mechanism == "both":
        return np.concatenate([
            signature_vector(prep["signatures"]),
            attribute_vector(prep["attributes"]),
        ])
    raise ValueError(f"unknown mechanism {mechanism!r}")


def _run_transfer_fold(target_clave, stations_pool, *,
                       r2, layout, split, lookback, horizons, use_clima,
                       hidden, layers, dropout, batch_size, epochs, lr,
                       patience, seed, run_id, out_root, upload_to_r2,
                       mechanism,
                       similarity_method, similarity_metric,
                       similarity_temperature, similarity_top_k):
    click.echo(f"\n[14_train] === Fold: holdout={target_clave} "
               f"(mechanism=S-{mechanism.upper()}, weighting={similarity_method}) ===")

    target_row = stations_pool[stations_pool["clave"].astype(str) == target_clave]
    if target_row.empty:
        raise SystemExit(f"[14_train] holdout {target_clave!r} not in basin pool")
    target_row = target_row.iloc[0]

    ref_spec = WindowSpec(lookback=lookback, horizons=horizons,
                          feature_cols=[], target_col=TARGET_LOG_COL)
    target_prep = _prepare_station(r2, layout, target_row, use_clima, split,
                                   ref_spec, feature_cols_ref=None)
    if target_prep is None:
        click.echo(f"[14_train] holdout {target_clave!r} has too little data — skipping fold.")
        return {"target_clave": target_clave, "metrics": {}, "n_donors": 0, "skipped": True}
    ref_features = target_prep["feature_cols"]
    if len(target_prep["windows_test"]["x"]) == 0:
        click.echo(f"[14_train] holdout {target_clave} has 0 test windows — skipping fold.")
        return {"target_clave": target_clave, "metrics": {}, "n_donors": 0, "skipped": True}
    click.echo(f"[14_train] Target {target_clave} ({target_prep['name']}): "
               f"coverage={target_prep['coverage']:.2%}  "
               f"windows tr/va/te={len(target_prep['windows_train']['x'])}/"
               f"{len(target_prep['windows_val']['x'])}/"
               f"{len(target_prep['windows_test']['x'])}")

    # Prepare donors.
    donor_rows = stations_pool[stations_pool["clave"].astype(str) != target_clave]
    donor_preps = []
    for _, row in donor_rows.iterrows():
        prep = _prepare_station(r2, layout, row, use_clima, split, ref_spec,
                                feature_cols_ref=ref_features)
        if prep is None:
            click.echo(f"[14_train]   donor {row['clave']} skipped (insufficient train rows)")
            continue
        if len(prep["windows_train"]["x"]) == 0:
            click.echo(f"[14_train]   donor {prep['clave']} skipped (0 train windows)")
            continue
        donor_preps.append(prep)
    if not donor_preps:
        click.echo(f"[14_train] No usable donors for {target_clave} — skipping fold.")
        return {"target_clave": target_clave, "metrics": {}, "n_donors": 0, "skipped": True}

    # ----- SIMILARITY SCORING (the mechanism) ------------------------------
    target_vec = _select_similarity_vector(target_prep, mechanism)
    donor_vecs = np.vstack([
        _select_similarity_vector(prep, mechanism) for prep in donor_preps
    ])
    donor_claves = [prep["clave"] for prep in donor_preps]
    sim = score_donors(
        target_vec, donor_vecs, donor_claves,
        method=similarity_method, metric=similarity_metric,
        temperature=similarity_temperature, top_k=similarity_top_k,
    )
    click.echo(f"[14_train] S-{mechanism.upper()} similarity "
               f"({similarity_method}, {similarity_metric}) weights (mean=1.0):")
    for c, s, w in sorted(zip(donor_claves, sim.raw_scores, sim.weights),
                          key=lambda x: -x[2]):
        click.echo(f"[14_train]   {c:<8s}  sim={s:.3f}  w={w:.3f}")

    # Concatenate donor windows and matching weights.
    train_x, train_y, train_w = _stack_with_weights(donor_preps, sim.weights)
    val_stack = _stack([d["windows_val"] for d in donor_preps])
    click.echo(f"[14_train] Donor windows: train={len(train_x)}  val={len(val_stack['x'])}")
    if len(train_x) < MIN_TRAIN_WINDOWS:
        click.echo("[14_train] Not enough donor training windows — skipping fold.")
        return {"target_clave": target_clave, "metrics": {}, "n_donors": len(donor_preps),
                "skipped": True}

    # Model.
    import torch
    from torch import nn  # noqa: F401
    from hidroxmx.models.forecaster import LSTMEncDecConfig, LSTMEncoderDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_model = LSTMEncDecConfig(
        input_dim=len(ref_features), hidden_dim=hidden, num_layers=layers,
        horizons=len(horizons), dropout=dropout,
    )
    model = LSTMEncoderDecoder(cfg_model).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    fold_id = f"{run_id}/{target_clave}"
    out_dir = out_root / target_clave
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_store = CheckpointStore(
        run_id=fold_id, local_dir=out_dir / "ckpts",
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
            click.echo(f"[14_train] Resumed at epoch {start_epoch}; "
                       f"best_val={best_val:.4f} at epoch {best_epoch}")
        except Exception as exc:  # noqa: BLE001
            click.echo(f"[14_train] Resume failed ({exc}); starting fresh.")

    # Training loop.
    val_x = torch.from_numpy(val_stack["x"]).to(device)
    val_y = torch.from_numpy(val_stack["y"]).to(device)

    history = []
    epochs_since_improved = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for xb, yb, wb in _to_batches_weighted(
                train_x, train_y, train_w,
                batch_size=batch_size, shuffle=True, seed=seed + epoch):
            xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
            optimiser.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = _weighted_smooth_l1(pred, yb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            epoch_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)
        train_loss = epoch_loss / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            # Un-weighted val loss so it's comparable across mechanisms.
            val_loss = float(torch.nn.functional.smooth_l1_loss(val_pred, val_y).item())
        improved = val_loss < best_val
        marker = " *" if improved else ""
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        click.echo(f"[14_train] epoch {epoch:03d} "
                   f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")

        state = {
            "model": model.state_dict(),
            "optimizer": optimiser.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val_loss),
            "best_epoch": best_epoch if not improved else epoch,
            "config": {"hidden": hidden, "layers": layers, "dropout": dropout,
                       "lookback": lookback, "horizons": horizons,
                       "lr": lr, "batch_size": batch_size, "seed": seed,
                       "holdout": target_clave,
                       "donors": donor_claves,
                       "donor_weights": sim.weights.tolist(),
                       "donor_scores": sim.raw_scores.tolist(),
                       "mechanism": mechanism,
                       "similarity_method": similarity_method,
                       "similarity_metric": similarity_metric,
                       "similarity_temperature": similarity_temperature,
                       "similarity_top_k": similarity_top_k,
                       "use_clima": use_clima, "feature_cols": ref_features},
            "target_stats": target_prep["target_stats"],
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
                click.echo(f"[14_train] Early stopping at epoch {epoch:03d} "
                           f"(best {best_epoch:03d} val={best_val:.4f}).")
                break

    best_ckpt = ckpt_store.restore(name="best.ckpt")
    if best_ckpt is not None:
        model.load_state_dict(best_ckpt["model"])
        click.echo(f"[14_train] Restored best.ckpt (epoch {int(best_ckpt.get('epoch', -1)):03d}).")

    test_x = torch.from_numpy(target_prep["windows_test"]["x"]).to(device)
    mu_t, sigma_t = target_prep["stats"][TARGET_LOG_COL]

    def denorm_to_m3s(z):
        return np.expm1(z * sigma_t + mu_t)

    model.eval()
    with torch.no_grad():
        test_pred = model(test_x).cpu().numpy()
    y_true = denorm_to_m3s(target_prep["windows_test"]["y"])
    y_pred = denorm_to_m3s(test_pred)

    metrics = _eval_metrics(y_true, y_pred, horizons)
    metrics.update(_persistence_and_climatology(target_prep, horizons))
    metrics["train_windows_donors"] = len(train_x)
    metrics["val_windows_donors"] = len(val_stack["x"])
    metrics["test_windows_target"] = len(y_true)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_val_loss"] = float(best_val)
    metrics["n_donors"] = len(donor_preps)

    click.echo(f"[14_train] === Test metrics on {target_clave} (physical units) ===")
    for h in horizons:
        click.echo(
            f"[14_train]   h={h:>2d}d  "
            f"F0_txfr NSE={metrics[f'nse_h{h}']:+.3f} KGE={metrics[f'kge_h{h}']:+.3f} "
            f"RMSE={metrics[f'rmse_h{h}_m3s']:6.2f} m3/s  |  "
            f"persist NSE={metrics[f'persist_nse_h{h}']:+.3f}"
        )

    fold_manifest = RunManifest(
        run_id=fold_id, stage="14_train_transfer",
        config={
            "basin_pool_size": int(len(stations_pool)),
            "n_donors": len(donor_preps),
            "donors": donor_claves,
            "donor_weights": sim.weights.tolist(),
            "donor_scores": sim.raw_scores.tolist(),
            "mechanism": mechanism,
            "similarity_method": similarity_method,
            "similarity_metric": similarity_metric,
            "similarity_temperature": similarity_temperature,
            "similarity_top_k": similarity_top_k,
            "signature_keys": list(DEFAULT_SIGNATURE_KEYS),
            "attribute_keys": list(DEFAULT_ATTRIBUTE_KEYS),
            "target_signatures": target_prep["signatures"],
            "target_attributes": target_prep["attributes"],
            "holdout": target_clave,
            "holdout_name": target_prep["name"],
            "use_clima": use_clima, "lookback": lookback, "horizons": horizons,
            "hidden": hidden, "layers": layers, "dropout": dropout,
            "batch_size": batch_size, "epochs": epochs, "lr": lr,
            "patience": patience, "seed": seed, "feature_cols": ref_features,
            "target_stats_m3s": target_prep["target_stats"],
        },
    ).finalise({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    dump_manifest(fold_manifest, out_dir / "manifest.json")

    if upload_to_r2:
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{fold_id}"
        for f in (out_dir / "history.json", out_dir / "manifest.json"):
            r2.upload_file(f"{prefix}/{f.name}", f)
            click.echo(f"[14_train]   -> r2://{r2.bucket}/{prefix}/{f.name}")
    published = publish_results(
        [out_dir / "manifest.json", out_dir / "history.json"],
        stage="14_train_transfer", run_id=run_id, subpath=target_clave,
    )
    for p in published:
        click.echo(f"[14_train]   -> git: {p.as_posix()}")

    del model, optimiser
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"target_clave": target_clave, "metrics": metrics,
            "n_donors": len(donor_preps)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--run-id", default="F0txfr-sig-alto-lerma-sweep-01", show_default=True)
@click.option("--basin", default=DEFAULT_BASIN, show_default=True)
@click.option("--holdout", default=DEFAULT_HOLDOUT, show_default=True)
@click.option("--lookback", default=90, show_default=True)
@click.option("--horizons", default="1,2,3,5,7", show_default=True)
@click.option("--hidden", default=64, show_default=True)
@click.option("--layers", default=1, show_default=True)
@click.option("--dropout", default=0.0, show_default=True)
@click.option("--batch-size", default=128, show_default=True)
@click.option("--epochs", default=40, show_default=True)
@click.option("--lr", default=5e-4, show_default=True)
@click.option("--patience", default=6, show_default=True)
@click.option("--use-clima/--no-clima", default=True, show_default=True)
@click.option("--seed", default=20260101, show_default=True)
@click.option("--mechanism", default="sig",
              type=click.Choice(["sig", "attr", "both"]), show_default=True,
              help="Donor-similarity mechanism. 'sig' uses hydrological "
                   "signatures; 'attr' uses static manifest attributes "
                   "(lat/lon/altitud/region); 'both' concatenates both.")
@click.option("--similarity-method", default="soft",
              type=click.Choice(["soft", "top_k"]), show_default=True)
@click.option("--similarity-metric", default="euclidean",
              type=click.Choice(["euclidean", "cosine"]), show_default=True)
@click.option("--similarity-temperature", default=1.0, show_default=True, type=float)
@click.option("--similarity-top-k", default=5, show_default=True, type=int)
@click.option("--out-dir", default="outputs/f0_transfer", show_default=True)
@click.option("--upload-to-r2", is_flag=True)
def main(run_id, basin, holdout, lookback, horizons, hidden, layers, dropout,
         batch_size, epochs, lr, patience, use_clima, seed,
         mechanism, similarity_method, similarity_metric,
         similarity_temperature, similarity_top_k, out_dir, upload_to_r2):
    load_dotenv(override=False)
    seed_everything(seed)

    horizons_list = [int(h.strip()) for h in horizons.split(",") if h.strip()]
    out_root = Path(out_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    r2 = r2_from_env()
    layout = layout_from_env()
    split = TemporalSplit.default()

    stations_pool = _load_basin_stations(r2, layout, basin)
    click.echo(f"[14_train] Basin: {basin}  stations available: "
               f"{len(stations_pool)}  "
               f"mechanism: S-{mechanism.upper()} ({similarity_method})")

    all_claves = stations_pool["clave"].astype(str).tolist()
    if holdout == "*":
        holdouts = all_claves
    else:
        if holdout not in all_claves:
            raise SystemExit(f"[14_train] holdout {holdout!r} not in {all_claves}")
        holdouts = [holdout]

    fold_summaries = []
    for target_clave in holdouts:
        result = _run_transfer_fold(
            target_clave, stations_pool,
            r2=r2, layout=layout, split=split,
            lookback=lookback, horizons=horizons_list, use_clima=use_clima,
            hidden=hidden, layers=layers, dropout=dropout,
            batch_size=batch_size, epochs=epochs, lr=lr, patience=patience,
            seed=seed, run_id=run_id, out_root=out_root,
            upload_to_r2=upload_to_r2,
            mechanism=mechanism,
            similarity_method=similarity_method,
            similarity_metric=similarity_metric,
            similarity_temperature=similarity_temperature,
            similarity_top_k=similarity_top_k,
        )
        fold_summaries.append(result)

    skipped = [fs["target_clave"] for fs in fold_summaries if fs.get("skipped")]
    if skipped:
        click.echo(f"[14_train] Skipped {len(skipped)} fold(s): {skipped}", err=True)
    completed = [fs for fs in fold_summaries if not fs.get("skipped")]
    if len(completed) > 1:
        rows = []
        for fs in completed:
            row = {"holdout": fs["target_clave"], "n_donors": fs["n_donors"]}
            for h in horizons_list:
                row[f"nse_h{h}"] = fs["metrics"][f"nse_h{h}"]
                row[f"persist_nse_h{h}"] = fs["metrics"][f"persist_nse_h{h}"]
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(out_root / "folds_summary.csv", index=False)
        click.echo("\n[14_train] === Aggregate transfer summary ===")
        for h in horizons_list:
            avg_f0 = float(summary_df[f"nse_h{h}"].mean())
            avg_p = float(summary_df[f"persist_nse_h{h}"].mean())
            wins = int((summary_df[f"nse_h{h}"] > summary_df[f"persist_nse_h{h}"]).sum())
            click.echo(f"[14_train]   h={h:>2d}d  "
                       f"F0_txfr avg NSE={avg_f0:+.3f}  persist avg NSE={avg_p:+.3f}  "
                       f"folds F0_txfr>persist: {wins}/{len(summary_df)}")
        if upload_to_r2:
            prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}"
            r2.upload_file(f"{prefix}/folds_summary.csv", out_root / "folds_summary.csv")
            click.echo(f"[14_train]   -> r2://{r2.bucket}/{prefix}/folds_summary.csv")
        published = publish_results(
            [out_root / "folds_summary.csv"],
            stage="14_train_transfer", run_id=run_id,
        )
        for p in published:
            click.echo(f"[14_train]   -> git: {p.as_posix()}")

    click.echo("[14_train] Done.")


if __name__ == "__main__":
    main()
