# hidroxmx-forecasting

[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)

Modelling code for the **HidroXAI-MX** project (IPN · PICDT2026):
probabilistic streamflow / water-level forecasting across four Mexican
pilot basins and a scoped, predictive/assimilation-ready digital-twin
loop.

> **Dataset dependency.** This repository does **not** ship the data. Inputs
> are read on demand from the sibling repository
> [`hidroxai-mx`](https://github.com/pantrok/hidroxai-mx) —
> snapshot `v2026.06`, DOI
> [10.5281/zenodo.21231601](https://doi.org/10.5281/zenodo.21231601) —
> either through the DVC remote on Cloudflare R2 or by streaming directly
> from R2 with `boto3` / `s3fs`. No parquet / raster / vector file is
> committed to git.

## Scope

- **RQ1 (Path A):** does mechanism-governed / invariance-gated donor
  selection transfer better to ungauged sub-basins than attribute or
  aggregate-performance selection?
- **RQ2 (Path B):** does calibrated-uncertainty-driven fuzzy alerting beat
  simpler alerts on the decision frontier and on tail reliability?
- **RQ3 (scoped digital twin):** does retrospective data assimilation plus
  what-if scenarios add value consistent with a *predictive* twin
  (Metcalfe et al. 2023 maturity level), not operational closed-loop?

Kill conditions per hypothesis are enumerated in `docs/experiment-spec.md`
and are reported regardless of outcome.

## Layout

```
hidroxmx-forecasting/
├── src/hidroxmx/
│   ├── data/        # R2 streaming, lazy Parquet loaders, windowing datasets
│   ├── models/      # forecaster F0 (LSTM enc–dec), F1 physics-augmented variant
│   ├── transfer/    # signatures, attribute vectors, donor similarity scoring
│   ├── uq/          # split-conformal prediction intervals, coverage diagnostics
│   ├── alert/       # Mamdani fuzzy inference system, rule export
│   ├── twin/        # innovation-persistence assimilation, what-if scenarios
│   ├── eval/        # metrics registry, ROC / cost-loss, paired bootstrap CIs
│   ├── viz/         # J. Hydrology figure spec (dpi, size, palette, save_figure)
│   ├── coverage/    # HydroRIVERS / Flood Hub / GloFAS coverage overlays
│   └── io/          # R2 client, checkpoint save/restore, run manifest, seeding
├── scripts/
│   ├── 11_train_forecaster.py    single-station F0 baseline (LSTM enc–dec)
│   ├── 12_train_multistation.py  F0-PUB (multi-station leave-one-out)
│   ├── 14_train_transfer.py      donor-similarity mechanism (Path A)
│   ├── 15_conformal_uq.py        post-hoc split-conformal intervals
│   ├── 16_coverage_map.py        Flood Hub / GloFAS coverage vs. pilot basins
│   ├── 17_evaluate_alerts.py     fuzzy alert vs simple-threshold baseline
│   ├── 18_bootstrap_analysis.py  paired bootstrap CIs for the core comparisons
│   ├── 19_paper_master_figure.py three-panel forest plot (bootstrap CIs)
│   ├── 20_dt_demo.py             assimilation + what-if scenarios (digital twin)
│   ├── 20_figure_pub_summary.py  per-basin PUB summary figure
│   ├── 21_dt_paper_figure.py     digital-twin paper figure (assim + fan chart)
│   ├── 21_figure_cross_basin.py  cross-basin PUB summary figure
│   ├── 22_basin_inclusion.py     basin-sample inclusion criteria + figure
│   └── 99_sync_results.py        pull manifests from R2 into results/ for commit
├── notebooks/00_colab_entrypoint.ipynb   mounts R2, restores last checkpoint, runs a stage
├── conf/experiments/                     split defs, seeds, cost ratios, thresholds, R2 paths
├── tests/                                unit tests (metrics, IO, splits, viz, twin, transfer)
├── docs/experiment-spec.md               research questions, kill conditions, protocol
├── results/                              text-only artefacts (manifests, tables, figures)
├── pyproject.toml
├── AUTHORS · CITATION.cff · LICENSE
└── .env.example
```

## Quickstart (local, without GPU)

```bash
git clone https://github.com/pantrok/hidroxmx-forecasting.git
cd hidroxmx-forecasting
python -m venv .venv && .venv/Scripts/activate  # Windows
pip install -e ".[dev,geo,uq]"
cp .env.example .env       # fill in the R2 variables
pytest -q
```

## Quickstart (Google Colab, with GPU)

Open `notebooks/00_colab_entrypoint.ipynb` in Colab. The notebook:

1. Clones this repository, installs `.[torch,geo,uq]`.
2. Reads R2 credentials from the Colab environment / secrets.
3. Restores the last checkpoint of the requested run-id from R2 (if any);
   otherwise starts a fresh run and writes `first.ckpt`.
4. Executes exactly one stage (e.g., `11_train_forecaster.py`) with the
   requested config.
5. Flushes checkpoints and metrics to R2 and prints the exact resume command.

Any GPU timeout / disconnect costs at most `checkpoint_every_n_steps`.

## Reproducing the figures

All figures are re-generated from the CSV tables and per-run manifests on
R2. The four CPU-only figure scripts finish in a couple of minutes each:

```bash
python -u scripts/22_basin_inclusion.py                              # Fig. 2
python -u scripts/20_figure_pub_summary.py --run-id F0pub-<basin>-sweep-01 \
    --basin-label "<Basin>" --out results/figures/fig_3_pub_summary_<basin>
python -u scripts/21_figure_cross_basin.py                           # Fig. 4
python -u scripts/19_paper_master_figure.py --from-r2                # Fig. 5
python -u scripts/21_dt_paper_figure.py --from-r2                    # Fig. 6
```

Every script prints the R2 keys it fetches and mirrors the rendered
figure back to `paper2/figures/` on R2 when `--upload-to-r2` is passed.

## Figure export policy

Every figure in this repository is produced through
`hidroxmx.viz.save_figure`, which writes each artefact at the exact
resolution the *Journal of Hydrology* (Elsevier) submission guide
requires:

| kind          | dpi  | typical use                                        |
|---------------|-----:|----------------------------------------------------|
| `halftone`    |  300 | choropleths, heatmaps, satellite composites        |
| `combination` |  500 | plots with lines + fill, most maps and hydrographs |
| `line`        | 1000 | pure line art (schematics, bar charts, boxplots)   |

Column widths follow the Elsevier layout: 90 mm (single), 140 mm (1.5),
190 mm (double); use `hidroxmx.viz.figure_size(column=...)`. Every call
writes one TIFF at the required dpi, one vector PDF, and one PNG
preview in the same directory. The default categorical palette is the
Wong (2011) 8-colour colour-blind-accessible set; sequential defaults
are `viridis` and `cividis`.

## Compute policy

- **Data at rest** in Cloudflare R2 (S3-compatible), streamed on demand.
- **No data / no checkpoints / no artefacts** in git (see `.gitignore`).
- **Every long job is checkpoint-resumable** (model, optimizer,
  scheduler, epoch/step, best metric, RNG states, AMP scaler, config hash).
- **Memory-frugal** (mixed precision, gradient accumulation, chunked lazy
  Parquet reads, iterable datasets, explicit `del` / `gc.collect()` /
  `torch.cuda.empty_cache()` between basins/folds).

## License and credit

- **Code:** MIT (see `LICENSE`).
- **Sole credit:** Daniel Sánchez-Ruiz — Instituto Politécnico Nacional
  (IPN), UPIIT — project IND-2026-0335 (PICDT 2026, Secretaría de
  Investigación y Posgrado). Citation: see `CITATION.cff`.
