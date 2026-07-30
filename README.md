# hidroxmx-forecasting

[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)

Modelling code for the **HidroXAI-MX** project: probabilistic streamflow
and water-level forecasting across four Mexican pilot basins, with a
scoped predictive/assimilation-ready digital-twin loop.

## Data

The code does not ship data. Inputs are consumed from the sibling
repository [`hidroxai-mx`](https://github.com/pantrok/hidroxai-mx)
(snapshot `v2026.06`, DOI
[10.5281/zenodo.21231601](https://doi.org/10.5281/zenodo.21231601)):

- `series_hidrometricas.parquet` — daily hydrometric series
- `series_climatologicas.parquet` — daily climatological series
- `feature_table.parquet` — pre-computed daily features
- `estaciones_seleccionadas_hidrometricas.csv` — station catalogue

Every script streams these on demand from Cloudflare R2 (S3-compatible)
or from a local mirror pointed at by `HIDROXAI_MX_ROOT`.

## What the code does

Under `src/hidroxmx/`:

- `data/` — R2 streaming, feature building, temporal / PUB splits
- `models/` — encoder–decoder LSTM forecaster (F0)
- `transfer/` — hydrological signatures, static attributes, donor
  similarity (Path A mechanisms)
- `uq/` — split-conformal prediction intervals + coverage diagnostics
- `alert/` — Mamdani fuzzy inference for uncertainty-aware alerts
- `twin/` — innovation-persistence assimilation + what-if scenarios
- `eval/` — metrics registry (NSE, KGE, RMSE, POD/FAR, Value) and
  paired bootstrap confidence intervals
- `viz/` — publication-quality figure helpers (dpi, size, palette)
- `coverage/` — Flood Hub / GloFAS coverage overlays

Under `scripts/` (each is a standalone Click CLI):

- `11_train_forecaster.py` — single-station F0 baseline
- `12_train_multistation.py` — F0-PUB (leave-one-out across a basin)
- `14_train_transfer.py` — donor-similarity mechanism (Path A)
- `15_conformal_uq.py` — post-hoc split-conformal intervals
- `16_coverage_map.py` — Flood Hub / GloFAS coverage map
- `17_evaluate_alerts.py` — fuzzy alert vs simple-threshold baseline
- `18_bootstrap_analysis.py` — paired bootstrap CIs → 3 CSV tables
- `19_paper_master_figure.py` — 3-panel forest plot of the CIs
- `20_dt_demo.py` — digital-twin assimilation + what-if scenarios
- `20_figure_pub_summary.py` — per-basin PUB summary figure
- `21_dt_paper_figure.py` — digital-twin paper figure
- `21_figure_cross_basin.py` — cross-basin PUB summary figure
- `22_basin_inclusion.py` — basin-sample inclusion criteria + figure
- `99_sync_results.py` — pull per-run manifests from R2 into `results/`

## Install and run

```bash
git clone https://github.com/pantrok/hidroxmx-forecasting.git
cd hidroxmx-forecasting
python -m venv .venv && source .venv/bin/activate  # Windows: .venv/Scripts/activate
pip install -e ".[dev,geo,uq]"                     # add ",torch" for GPU
cp .env.example .env                               # fill in the R2 vars
pytest -q                                          # optional
```

Every script exposes `--help`. Typical run:

```bash
python scripts/12_train_multistation.py --run-id F0pub-alto-lerma-sweep-01 \
    --basin "Alto Lerma" --upload-to-r2
```

For a GPU session, `notebooks/00_colab_entrypoint.ipynb` clones the
repo in Colab, wires R2 credentials from Colab secrets and runs one
stage per invocation.

## Credit

Sole author: **Daniel Sánchez-Ruiz** — Instituto Politécnico Nacional
(IPN), UPIIT — project IND-2026-0335 (PICDT 2026, Secretaría de
Investigación y Posgrado, IPN). Citation metadata in `CITATION.cff`.
Code released under the MIT License (see `LICENSE`).
