# Paper 2 — Experimental Design Specification

**HidroXAI-MX · IPN grant IND-2026-0335 · target: *Journal of Hydrology*, SI "AI-driven digital twins for hydrological systems"**

This file mirrors the runnable blueprint agreed for Paper 2. It fixes *what to
build, what to compare, and what would falsify each claim* **before** writing
model code. Every experiment maps to a claim and to a kill condition.
Streamflow / water-level only (drought / SPI–SPEI is out of scope for Paper 2,
that is Paper 3). The digital-twin claim is scoped to
**predictive / assimilation-ready** maturity (Metcalfe et al., 2023), not
operational closed-loop.

## Compute policy

- Google Colab with time-limited GPU.
- Heavy data at rest in Cloudflare R2 (S3-compatible); streamed on demand;
  **never committed to git**.
- Every long job is checkpoint-resumable so a GPU timeout resumes from
  ``last.ckpt`` with model, optimiser, scheduler, epoch/step, best metric,
  RNG states, AMP scaler and config hash.

## 0. Objects and notation

- Basins (4): Cutzamala, Lerma–Santiago (split operationally into Lerma Alto,
  Bajío, Santiago), Pánuco, Alta del Balsas.
  Sub-basins delineated in the dataset: 123.
- Gauged hydrometric stations retained: 108 (≥60% completeness).
  Climatological stations: 415.
- Core input tables (produced by the ``hidroxai-mx`` pipeline, streamed from
  R2, never regenerated here): ``series_hidrometricas.parquet``,
  ``series_climatologicas.parquet``, ``feature_table.parquet``.
- Static attribute vector per sub-basin ``a_b`` = { area, mean/median slope,
  elevation hypsometry, drainage density, channel length, sinuosity, aspect,
  IDW-interpolated mean precipitation, aridity index }.
- Notation: ``y_t`` target (level/flow), ``x_t`` dynamic forcing
  (P, T, antecedent y), ``a_b`` static attribute vector, ``θ_b`` physical
  parameter vector (recession k, storage S, roughness/lag).

## 1. Research questions, hypotheses, kill conditions

### RQ1 — Path A: mechanism-governed transfer

- **H1a** (load-bearing): behavior-space distance (θ + response-kernel
  signature) predicts transfer skill better than attribute-space distance
  at gauged sites.
- **H1b**: process / invariance-gated donor selection beats attribute-gated
  and performance-gated selection on PUR spatial holdout and on extreme
  events.
- **Kill A**: if, after fairly tuning the attribute and performance
  selectors, process/invariance-gating shows no statistically significant
  gain on PUR + extremes (and the negative control fails), drop the Path-A
  claim and degrade to a regionalization + UQ study.

### RQ2 — Path B: calibrated UQ → auditable fuzzy alerting

- **H2a**: conformal / Bayesian intervals reach nominal coverage including
  the tail; raw ensemble spread does not.
- **H2b**: calibrated UQ → fuzzy layer dominates (i) per-site-optimised
  fixed margin and (ii) uncalibrated spread → fuzzy on CRPS / cost-loss
  frontier and on FAR-at-fixed-POD / lead-time.
- **Kill B**: if the calibrated layer does not improve tail reliability
  over raw spread AND does not win on the cost-loss frontier across cost
  ratios vs. both baselines, collapse to the simpler baseline.

### RQ3 — Scoped digital twin

Does retrospective data assimilation plus what-if scenario capability add
value consistent with a *predictive / assimilation-ready* twin (not
operational)?

## 2. Data splits

- Temporal (fixed): train 2010-01→2020-12 · val 2021-01→2022-12 ·
  test 2023-01→2025-12. Rationale: the 2024 Cutzamala drought and the
  Oct-2025 floods fall inside test, giving built-in OOD / extreme stress.
  Never leak test into calibration.
- Spatial:
  - **PUB** — leave-one-gauged-basin-out inside a region.
  - **PUR** — hold out an entire region (e.g. all Pánuco gauges).
  - **Internal ungauged** — the delineated pour points with no gauge
    (123 minus 108). Forecasts delivered **uncertainty-flagged only**.
- **Extreme-event subset**: per-station exceedance events (≥ P95 level,
  named 2024 drought, named Oct-2025 event) for stratified evaluation.
- Report every fold; no fold hand-picking.

## 3. Forecasting backbone

- **F0** — encoder–decoder LSTM (and a Transformer / TFT variant); horizons
  1–7 days.
- **F1** — F0 + soft physics constraints: mass-balance penalty,
  monotonic rainfall → runoff, non-negativity, recession behaviour
  (cf. Xie et al., 2021; Feng et al., 2023).
- **Parameter head H** — MLP mapping ``a_b → θ_b`` (recession, storage,
  lag / roughness); θ feeds a differentiable δ-routing or conditions F1.
- Metrics: NSE, KGE, RMSE, PBIAS, EHF (high-flow error), peak-timing error.
  Reported per-station and pooled.

## 4. Path A

### 4.1 Process signature

Per gauged donor: (a) θ from head H; (b) response-kernel features —
recession constant, lag-to-peak, baseflow index, runoff ratio, flow-duration
curve shape. These describe the *transformation*, not the forcing.

### 4.2 Donor-selection criteria

Identical forecaster; only the selector differs.

1. **S-ATTR** — attribute similarity (classical regionalization ≈
   Arsenault et al., 2019). *Baseline to beat.*
2. **S-PERF** — aggregate-skill gated (pooled NSE).
3. **S-SIG** — process-signature distance (θ + response-kernel).
4. **S-INV** — invariance-gated (ICP / FAIR-style). Confounder control:
   condition on precipitation (conditional Granger / partial transfer
   entropy) so shared forcing is blocked; cf. Abbasizadeh et al., 2025 —
   we *gate donor selection*, they *predict runoff signatures*.

### 4.3 Load-bearing test (H1a)

For every donor → target gauged pair, compute (distance in attribute
space) and (distance in behavior / signature space); regress against
realised transfer skill (ΔNSE after fine-tuning). Pass if signature-space
distance explains transfer skill significantly better than attribute-space
distance (nested model test / partial correlation). Otherwise **Kill A**.

### 4.4 Negative control (confounder guard)

Basin pairs with high precipitation concurrency but contrasting geology.
S-SIG / S-INV must flag them as different; a raw (unconditioned) Granger
comparison should mistakenly call them similar. Report both.

### 4.5 Main transfer experiment (H1b)

Evaluate S-ATTR vs S-PERF vs S-SIG vs S-INV on PUB and PUR folds, overall
and on the extreme subset. Metrics: NSE, KGE, EHF, peak-timing; on
extremes also POD / FAR at a fixed alert threshold. Significance: paired
bootstrap across folds/stations; report effect sizes.

## 5. Path B

### 5.1 UQ methods

- **Conformal** (split / CQR / adaptive-conditional). Conformal is already
  used in hydrology (Auer et al., 2024); the novelty is the **UQ → decision
  coupling**, not conformal per se.
- **Bayesian / ensemble** comparator (deep ensemble or PI3NN) for
  calibration contrast.

### 5.2 Calibration diagnostics (H2a)

Empirical vs nominal coverage at several levels; reliability diagram; PIT
histogram; **tail coverage**; sharpness (mean interval width); **CRPS**.

### 5.3 Fuzzy alert layer

Mamdani FIS. Inputs = (forecast level percentile, interval width or
exceedance probability). Output = alert class (none / watch / warning /
emergency). Membership functions anchored to local action thresholds
(co-design placeholder with CONAGUA / Protección Civil). Auditable IF–THEN
rules are a deliverable.

### 5.4 Baselines and decision test (H2b)

- **B0** — point forecast + per-site-optimised fixed safety margin.
- **B1** — uncalibrated ensemble spread → same fuzzy layer.
- Decision metrics: ROC (POD vs FAR), precision–recall, Brier of
  exceedance, cost-loss Value curve across a range of cost ratios,
  lead-time at fixed POD / FAR.

## 6. Scoped digital-twin loop

- Assimilation-ready demo: retrospective updating — feeding recent
  observations should improve next-step forecast / UQ
  (cf. Auer et al., 2024).
- What-if demo: scenario runs (± 20 % precipitation; Oct-2025 forcing
  replayed on an ungauged tributary).
- **Honest maturity statement**: map to Metcalfe et al. (2023) level
  *predictive twin*; explicitly *not* operational closed-loop.

## 7. Coverage / blind-spot map

Overlay Google Flood Hub / GloFAS-CEMS reach coverage against the 123
sub-basins. Deliverables: (a) map (Figure 1); (b) table naming specific
internal ungauged tributaries below the global-model coverage threshold
(candidates: Alta del Balsas headwaters Tlaxcala–Puebla; Pánuco upland
sub-basins). These points receive forecasts in a separate,
"experimental / unverified" tier.

## 8. Baselines roster

| ID      | Model / selector                                    | Role                          |
|---------|-----------------------------------------------------|-------------------------------|
| B-REG   | Classical regionalization (Arsenault 2019 style)    | PUB baseline for Path A       |
| B-LSTM  | Plain LSTM, local                                   | Naive DL floor                |
| B-PHY   | Physics-guided LSTM / δ-model                       | Strong physical baseline      |
| B-GLOB  | Global AI flood model reference where reach overlaps| External benchmark (Nearing 2024) |
| B-FIXM  | Point + per-site fixed margin                       | Path B baseline               |
| B-RAWU  | Uncalibrated spread → fuzzy                         | Path B baseline               |

Also positioned vs the closest DT comparator **Rápalo et al. 2024**
(open-source, near-real-time, data-scarce DT, Honduras).

## 9. Evaluation, governance, honesty

- Spatial (PUB / PUR) × temporal folds; per-station and pooled;
  paired bootstrap / DeLong for significance; multiple cost ratios.
- Report regardless of outcome — both kill conditions can fire; that is a
  publishable result.
- Ungauged outputs are shown in a distinct visual tier with action-gradient
  labels ("verify in field / prepare, not evacuate"); thresholds are
  co-designed; a distributional-justice note accompanies them.

## 10. Ablations

- **Path A**: − physics constraints; − invariance test (signature only);
  attribute-only vs + signature; conditional vs unconditional causal
  measure.
- **Path B**: conformal vs Bayesian; calibrated vs uncalibrated → fuzzy;
  fuzzy vs crisp threshold; with / without tail-adaptive conformal.
- **DT**: with / without assimilation update.

## 11. Execution order (milestones)

1. `scripts/16_coverage_map.py` → Figure 1 + blind-tributary table
   (de-risks the Introduction hook).
2. Backbone F0 / F1 + parameter head H → backbone metrics.
3. **Path A load-bearing test** — gate. If it fails, pivot to
   regionalization + UQ framing now, not after full experiments.
4. Path A transfer experiment + negative control.
5. Path B calibration diagnostics → decision / frontier test.
6. Scoped DT demos (assimilation + what-if).
7. Ablations → final tables / figures.

## 12. Risks and fallback

- Path A fails the load-bearing test → paper still stands on Path B
  (calibrated UQ → alert, verifiable at gauges) + open dataset + honest
  DT + coverage map. State this two-legged design explicitly so a null
  Path-A result is a *finding*, not a failure.
- Small sample (4 regions / 123 sub-basins) limits causal-discovery power;
  lean on physics prior; report power limits.
- "Causal" wording: use *process-consistency* except where the invariance
  test is demonstrated.

## Verified anchor references (seed the Methods)

- Rápalo, Gomes Jr. & Mendiondo, 2024 — closest DT comparator.
- Abbasizadeh et al., 2025 — causal discovery → robust runoff prediction.
- Auer et al., 2024 — conformal prediction for hydrological UQ.
- De León Pérez et al., 2025 — scoping review, probabilistic UQ in hydrology.
- Tang et al., 2025 — causality-guided DL + transfer entropy.
- Metcalfe et al., 2023 — DT maturity levels (predictive-twin anchor).
- Xie et al., 2021 — physics-guided LSTM.
- Feng et al., 2023 — differentiable, physics-informed models for ungauged regions.
- Nearing et al., 2024 — global AI flood benchmark (external baseline).
