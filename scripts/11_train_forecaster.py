#!/usr/bin/env python
"""Stage 11 — Train the F0 (or F1) forecaster on one Cutzamala station.

Milestone 2 slice: first end-to-end train / val / test run that exercises
every piece of the pipeline (R2 streaming → windowing → LSTM encoder-decoder
→ checkpoint-resumable training → metrics registry → run manifest).

Defaults are intentionally small (60 hidden units, 20 epochs) so a Colab
CPU session can complete the sanity run in minutes. The model, splits and
metrics are the same the full experiments use later; the only knobs that
change for the load-bearing test and beyond are the station roster and the
compute budget.

The station is auto-selected as the Cutzamala hydrometric station with the
highest observed coverage of ``gasto_medio_m3s`` inside the reference
window (2010-01-01 through 2025-12-31). Override with ``--clave``.
"""
from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict
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
    load_selected_stations,
    load_station_daily,
    local_roots_from_env,
)
from hidroxmx.data.windows import WindowSpec, collect_windows
from hidroxmx.eval import kge, nse, rmse
from hidroxmx.io import (
    CheckpointStore,
    RunManifest,
    dump_manifest,
    r2_from_env,
    seed_everything,
)
from hidroxmx.io.checkpoint import collect_rng_state, restore_rng_state


TARGET_COL = "gasto_medio_m3s"
FEATURE_COLS = (
    "gasto_medio_m3s",
    "gasto_medio_m3s_lag1",
    "gasto_medio_m3s_lag3",
    "gasto_medio_m3s_lag7",
    "gasto_medio_m3s_lag14",
    "gasto_medio_m3s_lag30",
    "gasto_medio_m3s_ma7",
    "gasto_medio_m3s_ma30",
)
CUTZAMALA_TAG = "Cutzamala"


def _select_station(r2, layout: DatasetLayout, basin: str) -> str:
    """Return the Cutzamala station key with the highest observed coverage."""
    manifest = load_selected_stations(r2, layout, kind="hidro")
    if "cuenca" in manifest.columns:
        pool = manifest[manifest["cuenca"].astype(str).str.contains(basin,
                                                                   case=False,
                                                                   na=False)]
        if len(pool) == 0:
            pool = manifest
    else:
        pool = manifest
    if "cobertura" in pool.columns:
        best = pool.sort_values("cobertura", ascending=False).iloc[0]
    else:
        best = pool.iloc[0]
    return str(best["clave"])


def _load_series(r2, layout: DatasetLayout, clave: str) -> pd.DataFrame:
    local = local_roots_from_env()
    df = load_station_daily(
        r2,
        layout.feature_table_key,
        clave=clave,
        columns=["clave_estacion", "fecha", *FEATURE_COLS, "calidad"],
        years=range(2010, 2026),
        local_path=(local.feature_table if local else None),
    )
    # Reindex to a continuous daily calendar so windows have a well-defined lookback.
    if df.empty:
        return df
    df = df.set_index("fecha").sort_index()
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    df.index.name = "fecha"
    df["clave_estacion"] = clave
    return df.reset_index()


def _standardise(series: pd.DataFrame, train_mask: pd.Series,
                 feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    stats = {}
    out = series.copy()
    for col in feature_cols:
        mu = float(out.loc[train_mask, col].mean())
        sigma = float(out.loc[train_mask, col].std(ddof=0)) or 1.0
        out[col] = (out[col] - mu) / sigma
        stats[col] = (mu, sigma)
    return out, stats


def _to_batches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True,
                seed: int = 0):
    """Yield minibatches from the stacked (x, y) tensors."""
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
        out[f"rmse_h{h}"] = float(rmse(y_true[:, i], y_pred[:, i]))
    return out


@click.command()
@click.option("--run-id", default="F0-cutzamala-topcov", show_default=True)
@click.option("--basin", default=CUTZAMALA_TAG, show_default=True,
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
@click.option("--epochs", default=20, show_default=True)
@click.option("--lr", default=5e-4, show_default=True)
@click.option("--seed", default=20260101, show_default=True)
@click.option("--out-dir", default="outputs/f0", show_default=True)
@click.option("--upload-to-r2", is_flag=True,
              help="Mirror checkpoints and manifest under {R2_PAPER2_PREFIX}/runs/{run_id}/.")
def main(
    run_id, basin, clave, lookback, horizons, hidden, layers, dropout,
    batch_size, epochs, lr, seed, out_dir, upload_to_r2,
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
        clave = _select_station(r2, layout, basin=basin)
    click.echo(f"[11_train] Station: {clave} (basin match: {basin!r})")

    # 2. Series ------------------------------------------------------------
    series = _load_series(r2, layout, clave)
    if series.empty:
        raise SystemExit(f"[11_train] Empty series for station {clave}; abort.")
    cov = coverage_of_station(series, TARGET_COL)
    click.echo(f"[11_train] Rows: {len(series):,}  coverage(target)={cov*100:.1f}%")

    # 3. Temporal split + standardisation on train only --------------------
    split = TemporalSplit.default()
    mask_train = split.mask(series["fecha"], "train")
    mask_val = split.mask(series["fecha"], "val")
    mask_test = split.mask(series["fecha"], "test")
    if mask_train.sum() < 500:
        raise SystemExit(
            f"[11_train] Too few train rows ({int(mask_train.sum())}); "
            f"pick another station or lower the coverage bar.")
    series_std, stats = _standardise(series, mask_train, list(FEATURE_COLS))

    spec = WindowSpec(
        lookback=lookback,
        horizons=horizons_list,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
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
    cfg_model = LSTMEncDecConfig(
        input_dim=len(FEATURE_COLS),
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
    if resumed is not None:
        try:
            model.load_state_dict(resumed["model"])
            optimiser.load_state_dict(resumed["optimizer"])
            start_epoch = int(resumed.get("epoch", 0)) + 1
            best_val = float(resumed.get("best_val", best_val))
            restore_rng_state(resumed.get("rng", {}))
            click.echo(f"[11_train] Resumed from epoch {start_epoch}; best_val={best_val:.4f}")
        except Exception as exc:  # noqa: BLE001
            click.echo(f"[11_train] Failed to resume ({exc}); starting fresh.")

    # 6. Training loop ------------------------------------------------------
    val_x = torch.from_numpy(windows_val["x"]).to(device)
    val_y = torch.from_numpy(windows_val["y"]).to(device)
    test_x = torch.from_numpy(windows_test["x"]).to(device)
    test_y = torch.from_numpy(windows_test["y"]).to(device)

    history = []
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
        click.echo(f"[11_train] epoch {epoch:03d} "
                   f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        state = {
            "model": model.state_dict(),
            "optimizer": optimiser.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val_loss),
            "config": {"hidden": hidden, "layers": layers, "dropout": dropout,
                       "lookback": lookback, "horizons": horizons_list,
                       "lr": lr, "batch_size": batch_size, "seed": seed,
                       "clave": clave},
            "stats": stats,
            "rng": collect_rng_state(),
        }
        ckpt_store.save(state, name="last.ckpt")
        if val_loss < best_val:
            best_val = val_loss
            ckpt_store.save(state, name="best.ckpt")

    # 7. Test-set metrics (de-standardised) ---------------------------------
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x).cpu().numpy()
    mu, sigma = stats[TARGET_COL]
    test_pred_denorm = test_pred * sigma + mu
    test_y_denorm = windows_test["y"] * sigma + mu
    metrics = _eval_metrics(test_y_denorm, test_pred_denorm, horizons_list)
    metrics["train_windows"] = len(windows_train["x"])
    metrics["val_windows"] = len(windows_val["x"])
    metrics["test_windows"] = len(windows_test["x"])
    click.echo(f"[11_train] Test metrics: {metrics}")

    # 8. Persist manifest and history ---------------------------------------
    manifest = RunManifest(
        run_id=run_id,
        stage="11_train_forecaster",
        config={
            "clave": clave, "basin": basin, "lookback": lookback,
            "horizons": horizons_list, "hidden": hidden, "layers": layers,
            "dropout": dropout, "batch_size": batch_size, "epochs": epochs,
            "lr": lr, "seed": seed,
            "coverage_of_station": cov,
            "feature_cols": list(FEATURE_COLS),
        },
    ).finalise({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    dump_manifest(manifest, out_dir / "manifest.json")

    if upload_to_r2:
        prefix = os.environ.get("R2_PAPER2_PREFIX", "paper2") + f"/runs/{run_id}"
        for local in (out_dir / "history.json", out_dir / "manifest.json"):
            r2.upload_file(f"{prefix}/{local.name}", local)
            click.echo(f"[11_train]   → r2://{r2.bucket}/{prefix}/{local.name}")

    del model, optimiser
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    click.echo("[11_train] Done.")


if __name__ == "__main__":
    main()
