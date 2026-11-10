# Revision log

## JOH minor revision (2026-11)

Manuscript HYDROL84901, *Journal of Hydrology*. Criteria written on 2026-11-04,
before any of the analyses below was computed; results go in later commits.
A later change to these criteria is a dated amendment labelled post hoc.

These analyses answer the reviewers. They do not change any pre-registered verdict
or any figure of the submitted version. The only training is the forcing-only
variant of RJ-3.

### Amendments (post hoc)

- **2026-11-04, reproduction gate.** The NSE tolerance becomes
  |ΔNSE| ≤ 1e-5 · max(1, |NSE|). The gate stopped at fold ARBMX: its 48 test windows
  are almost flat (0.22–0.23 m³/s), so NSE runs from −28 to −184 and a floating-point
  difference of relative size 4.5e-6 exceeds the absolute limit. Its conformal
  quantile, coverage and cost–loss values match the manifest, and the fold has no
  events, so it does not enter RJ-1.

### Common rules

- **Snapshot.** Hidro-MX v2026.06 only, taken from the archived snapshot: hydrometric
  series 197 files and 142,891,675 bytes, climatological series 150 files and
  836,473,785 bytes. Scripts check both counts before running.
- **Code base.** Branch `revision-joh-public` from `32690cb`. `src/`, `scripts/`, `conf/` and
  `tests/` match the code that produced the reported runs; the run manifests cite
  commits of `archive/pre-clean`, whose tip has the same tree for those folders.
- **Bootstrap.** Seed 20260721, B = 10,000, 95 % intervals.
- **Reproduction gate.** No saved predictions exist for the 52 conformal folds, so
  they are regenerated from each fold's `best.ckpt`. Every fold and horizon must
  match its manifest: |ΔNSE| ≤ 1e-5, conformal quantile within 1e-4 relative,
  marginal and tail coverage within one window, cost–loss value within 1e-6. Any
  fold outside tolerance stops the analysis.
- **Outputs.** Every number comes from a script and goes to
  `results/revision_joh/numbers_revision.json`, with `"fuente"` and `"tipo"` per
  block: `"revisión (preregistrado, intervalo alternativo)"` for RJ-1 and
  `"revisión (diagnóstico)"` for the rest. Tables go to
  `results/revision_joh/tables/`.
- **Figures.** `results/revision_joh/figures/`, PDF and 600 dpi TIFF, 90 or 190 mm
  wide, fonts of at least 7 pt, same style as Figs. 3–6.

### RJ-1 · BCa intervals for Path B

**Data.** The `alerts-*-sweep-01` manifests, folds with non-zero event rate, value at
C/L = 0.2, fuzzy value with the per-fold cutoff of `scripts/18_bootstrap_analysis.py`.

**Statistic.** The primary statistic is the one in Table 8,
T = median(fuzzy) − median(threshold). The median of the per-fold differences is
reported as secondary. If the two give different classes in a cell, that is flagged.

**BCa.** For all 20 cells of Table 8:

- same resampling indices as `paired_bootstrap`, so the percentile interval
  reproduces Table 8;
- bias correction z0 = Φ⁻¹(p), with p the mid-rank share of replicates below T;
- acceleration from the delete-one paired jackknife of T;
- limits read with the same quantile rule as the percentile interval.

A cell is "not defined" when n < 3, when all jackknife values are equal, or when
p is 0 or 1. Those cells rest on the per-fold values and the sign test.

**Classes** (lower = BCa lower limit):

| Class | Rule |
|---|---|
| Cleared, BCa-robust | lower ≥ +0.05 |
| Encouraging but under-powered signal | lower < +0.05 and T > +0.05 |
| Not cleared | T ≤ +0.05 |
| Not defined | see above |

A class change in Bajo Pánuco (h = 3, 5, 7) or Alto Lerma (h = 5) is flagged, since
those cells support the summary, the highlights and the conclusions.

**Also, per cell.**

- Per-fold values in `T_rj1_folds.csv`.
- Exact one-sided sign test of ΔV > 0 and of ΔV > +0.05, with p = 0.5. Folds exactly
  at the reference value are dropped and n is reported.
- Leave-one-fold-out range of T (minimum and maximum).

**Table 5 (descriptive, supplement).** BCa for the reported cells of F0-PUB −
persistence, same procedure, from the `F0pub-*-sweep-01` manifests.

**Table 8 sensitivities (post hoc, descriptive).** Neither replaces Table 8.

- **S1, corrected membership function.** Trapezoids with vertical shoulders take full
  membership at the edge of the universe, so inputs clipped to that edge still fire
  their rules. The function as submitted stays in place for reproduction.
- **S2, fixed cutoffs.** Each of YELLOW, ORANGE and RED applied to every fold, with no
  per-fold choice.

Both are computed alone and combined, with BCa and the classes above. Any class change
against the main result is flagged.

### RJ-2 · Tail coverage diagnostic

**Scope.** The 52 conformal folds, all horizons; h = 1 and h = 7 are the reported ones.
Calibration is 2021–2022 and test 2023–2025, on the windows the manuscript evaluates.
The last discharge in the record is 2025-09-03, so October 2025 cannot enter this
analysis, and the event list covers the record up to that date.

**Definitions.**

- Bands use each station's training-window quantiles: < q50, q50–q90, q90–q95, ≥ q95,
  and ≥ q99 only where a basin and horizon have at least 20 days.
- Tail means Q ≥ q95 of training. Coverage against the test-window Q95 of Table 7 is
  also reported, to link both.
- Primary results use the clipped discharge of the manuscript; raw discharge is
  reported as a sensitivity.
- Basin values are means over folds, as in Table 7. Folds with no tail day are left
  out and counted.

**Calculations.**

1. Number and share of tail days in calibration and test, by basin.
2. |residual| by decile of Q in each window. Per-station OLS slope of log|r| on
   log(1 + Q) in each window, leaving out r = 0; basin median. For tail days, the 50th
   and 90th percentiles of |r| and the two-sample KS statistic between windows, per
   station with at least 10 tail days in both, and pooled by basin after dividing
   residuals by q95.
3. Empirical coverage of the 90 % interval by band, basin and horizon.
4. Clustering of misses on tail days:
   - lag-1 autocorrelation of the miss indicator over adjacent calendar days;
   - lengths of consecutive-miss runs;
   - events, defined as tail days with gaps under 3 days merged: share with the peak
     covered and share with at least one day covered;
   - the three largest events per basin, ranked by peak/q95.
5. Two variants, with no retraining, same α = 0.1 and same calibration window:
   - **(a) Log score.** |ỹ − ŷ| in log(1 + Q), inverted to m³/s, lower limit floored at 0.
   - **(b) ACI** (Gibbs & Candès 2021). Original score and calibration scores held
     fixed; γ = 0.005, α₀ = 0.1, per fold and horizon, α updated by
     α + γ(α₀ − err). A miss at horizon h enters only once its target date has passed.
     When α leaves the range of the order statistic, the nearest score is used and the
     share of such steps is reported.

   For each variant: marginal coverage, tail coverage and mean width, by basin and horizon.

**Reading** (per basin, at h = 1 and h = 7, change in tail coverage against the original
interval on the same folds):

| Result | Reading |
|---|---|
| (a) rises ≥ 20 pp, (b) rises < 20 pp | Homoscedastic score dominates |
| (b) rises ≥ 20 pp, (a) rises < 20 pp | Calibration-to-test drift dominates |
| Both rise ≥ 20 pp | Both causes contribute |
| Neither rises ≥ 20 pp | The tail is out of reach with this backbone; the problem is model error in events, not the score |

**Figure.** (i) Coverage by band for the original, log and ACI intervals in the four
basins at h = 1. (ii) |residual| against Q in calibration and test for Bajo Pánuco, the
basin with the lowest tail coverage in Table 7. (iii) Optional: the Bajo Pánuco event
with the largest peak/q95, with the original and log intervals.

### RJ-3 · Forcing-only demonstration at station GANSL

The `panuco_008` polygon contains gauge GANSL, so the demonstration runs at that station,
held out, without using its discharge as an input.

**Variant.** F0-PUB without the eight discharge channels: precipitation, maximum and
minimum temperature and their 7-day means. Same architecture and hyperparameters, seed
20260101, one seed, CPU.

**Folds.** The seven Bajo Pánuco stations with complete forcing in v2026.06: ATCHD, BLMSL,
GANSL, OLVVC, PSNVC, PUJSL and STRSL.

**Inputs at the target.** Forcings and the station's training-window statistics (mean,
standard deviation, q95). No recent discharge. This corresponds to a site with a past
record and no current gauge; a site that was never gauged would also need regionalized
statistics and is out of scope.

**Skill.** NSE and KGE on the test windows at h = 1, 3 and 7, per fold and as a median,
against the training mean (the climatology of the manuscript) and the training
day-of-year mean. If the median NSE does not exceed the day-of-year climatology, the
demonstration is reported as having no demonstrated skill, and it stays in the report.

**Interval.** Score |z − ẑ| in standardized log(1 + Q), per horizon, pooled over the 2021–2022
windows of the retained stations, each under the model that held it out, excluding GANSL;
order statistic k = ⌈(n + 1)(1 − α)⌉, α = 0.1. Back-transformed with GANSL's statistics,
lower limit floored at 0.

**Alert.** Daily flow and width ratios against GANSL's training q95. The corrected membership
function is the primary one, because the submitted one gives GREEN for forecasts at or above
3 × q95. Levels from the submitted function are reported too.

**Periods.** October 2025 at h = 1, 3 and 7, and the dry reference 2024-03-01 to 2024-04-30.
If the 90-day input windows of October 2025 are incomplete, the event becomes the largest
5-day neighbour-mean precipitation of 2023–2025 with complete inputs, and this is stated.

**Figure.** Hydrograph of the forecast with the shaded interval, a strip with the alert level
below it, and the note "unverified: gauge record ends 2025-09-03" inside the figure.

### RJ-4 · Regulation (descriptive)

Each of the 62 pilot stations gets the reach of HydroRIVERS v1.0 with the largest upstream area
within 0.01° of the station, read directly from HydroRIVERS; the nearest reach is a sensitivity.
`dor_pc_pva` of RiverATLAS v1.0 is read on that reach, and its unit is confirmed in the RiverATLAS
catalogue before stratifying. Strata: regulated (DOR ≥ 2 %) and not regulated. Per
stratum, with n:

- median ΔNSE of F0-PUB − persistence at h = 5 and h = 7;
- median ΔNSE of mechanism − lumped in Alto Lerma.

Stations with no reach are left out and counted. No test is applied, and no result changes.

### RJ-5 · Figures

- Fig. 6 legend with the labels of Table 4 in the form `precip −20%`, legend fonts of at least 7 pt.
- Figures exported one per file as `Fig1.tif` to `FigN.tif` (600 dpi) and PDF. Fig. 3 becomes one
  file with its four basins.

### Running these analyses

Inputs from outside this branch:

- Hidro-MX v2026.06 archive, `HidroXAI-MX-v2026.06.zip` (DOI 10.5281/zenodo.21231601).
- HydroRIVERS v1.0, North America (`HYDRORIVERS_NA_URL` in `src/hidroxmx/data/geo.py`).
- RiverATLAS v1.0, `RiverATLAS_Data_v10_shp.zip` (DOI 10.6084/m9.figshare.9890531).
- The checkpoints and manifests of the reported runs in R2, with the credentials of `.env.example`.
- `pdftoppm` from Poppler, for the figures of RJ-5.

Published tables and figures used as inputs are in `results/revision_joh/published/`.

```bash
python scripts/30_rj_regenerate_predictions.py --snapshot-zip <zip> --snapshot-root <folder>
python scripts/31_rj1_bca.py
python scripts/32_rj2_tail_diagnostics.py
python scripts/33_rj3_forcing_only.py --snapshot-root <folder>
python scripts/34_rj4_regulation.py --snapshot-root <folder> --hydrorivers <shp> --riveratlas <shp>
python scripts/35_rj_figures.py
python scripts/36_rj5_export_figures.py
```
