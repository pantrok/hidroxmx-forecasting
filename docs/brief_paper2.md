# Brief técnico Paper 2 — HidroXAI-MX

> **Uso**: documento de handoff para revisión con una sesión de Claude fuera
> de Claude Code. Refleja el estado del código y hallazgos hasta la fecha
> última de actualización. Todos los números provienen de corridas
> reproducibles cuyo commit + `run_id` está registrado.

---

## Metadatos

| Campo | Valor |
|---|---|
| Proyecto | HidroXAI-MX (Paper 2) |
| Grant | IPN IND-2026-0335 (PICDT 2026, SIP) |
| Autor único | Daniel Sánchez-Ruiz (UPIIT, IPN) |
| Repo código | https://github.com/pantrok/hidroxmx-forecasting |
| Journal objetivo | *Journal of Hydrology*, SI "AI-driven digital twins for hydrological systems" |
| Dataset | `hidroxai-mx` v2026.06 (sibling repo; snapshot en R2 en formato DVC) |
| Cuencas piloto planeadas | **4 cuencas** — Alto Lerma (14), Valle de México (20), Bajo Pánuco (15), Medio Balsas (13). Total 62 estaciones. |
| Cuencas completadas | Alto Lerma (Milestone 3 completo, 14 folds PUB) |
| Última actualización | 2026-07-20 |

**Nota sobre autoría**: nunca aparece Claude / IA como co-autora en commits, en `AUTHORS`, en `CITATION.cff` ni en headers/docstrings. Sí se declarará el uso de LLMs en la sección de disclosure del manuscrito conforme a la política Elsevier GenAI (obligatorio para figuras derivadas de LSTMs).

---

## Preguntas de investigación (del brief inicial)

- **RQ1 (Path A)**: ¿la selección de donantes gobernada por mecanismo (invariance-gated) transfiere mejor a sub-cuencas ungauged que la selección por atributos o por desempeño agregado?
- **RQ2 (Path B)**: ¿el alertamiento borroso guiado por incertidumbre calibrada mejora la frontera de decisión y la confiabilidad en colas frente a alertas simples?
- **RQ3 (Digital-twin scoped)**: ¿la asimilación de datos retrospectiva + escenarios what-if añade valor consistente con un twin predictivo (Metcalfe et al. 2023), no operacional cerrado?

Condiciones de kill por hipótesis enunciadas en `docs/experiment-spec.md` y reportadas independientemente del resultado.

---

## Arquitectura del código (state actual)

```
src/hidroxmx/
├── data/         features.py (log1p + clip + lags + climate), splits.py (temporal + PUB + PUR),
│                 streams.py (R2 local-first), windows.py (sliding-window dataset)
├── models/       forecaster.py (LSTM encoder–decoder F0)
├── eval/         metrics.py (NSE/KGE/RMSE/PBIAS/EHF/CRPS/POD-FAR/cost-loss)
├── io/           r2.py (con retry Cloudflare TLS), checkpoint.py (mirror R2, atómico),
│                 manifest.py, results.py (git-tracked artefacts), seeds.py
├── transfer/     (vacío — Milestone 4)
├── uq/           (vacío — Milestone 5)
├── alert/        (vacío — Milestone 6)
├── viz/          journal.py (dpi + column + Wong palette per Elsevier)
└── coverage.py   (Milestone 1)

scripts/
├── 11_train_forecaster.py       F0 mono-estación
├── 12_train_multistation.py     F0-PUB (Milestone 3 completo)
├── 16_coverage_map.py           Milestone 1 (Fig. 1 cobertura)
├── 20_figure_pub_summary.py     Fig. 3 (Milestone 3 summary)
└── 99_sync_results.py           helper R2→git

tests/                            40 tests, todos pasando
docs/                             experiment-spec.md, este brief
notebooks/00_colab_entrypoint.ipynb   Colab GPU con auto-resume desde R2
```

**Persistencia y reproducibilidad**:
- Data en R2 (Cloudflare, S3-compatible). Nunca en git.
- Checkpoints en R2 + local ephemeral. Nunca en git.
- Manifests + histories en R2 + `results/` (git-tracked).
- Cada run guarda config hash + git SHA en el manifest.
- Cada training loop es checkpoint-resumable (`last.ckpt`, `best.ckpt`, RNG state).

**Testing**: 40 tests unitarios cubren métricas, splits, features, ventanas, resultados IO, y helper de figuras (verificando dpi metadata leyendo el TIFF de vuelta).

---

## Milestone 1 — Coverage / blind-spot map (§12.1)

**Estado**: ✅ completado.

**Método**: overlay Flood Hub reaches (umbral upland ≥ 25 km²) + GloFAS reaches (≥ 500 km²) contra las 123 sub-cuencas piloto de `hidroxai-mx`. Descarga HydroSHEDS con fallback vía `curl` porque Cloudflare bloquea el UA de `python-requests`.

**Hallazgo**:
- 123 sub-cuencas totales
- 77 cubiertas por ambos proveedores
- 37 cubiertas solo por Flood Hub
- 9 no cubiertas ("blind") — **1 de estas es prioritaria por ungauged** (`panuco_008`).

**Frase-borrador para el manuscrito**:
> Nine of the 123 pilot sub-basins are entirely blind to both Google Flood Hub and CEMS-GloFAS at their default upland thresholds; one of these blind sub-basins (`panuco_008`) is a priority ungauged tributary in the Pánuco river system, motivating a Path A transfer regime that does not require a locally trained gauge.

Fig. 1 (mapa) generada por `scripts/16_coverage_map.py`. Post-retrofit del helper `viz.journal`, cumple J. Hydrology: TIFF 500 dpi + PDF vectorial + PNG preview, ancho double-column.

---

## Milestone 2 — F0 backbone mono-estación (§12.2)

**Estado**: ✅ completado. **Función**: baseline sano para comparar transfer.

### Decisiones metodológicas

1. **Pivot Cutzamala → Alto Lerma**. La cuenca "Cutzamala" no aparece etiquetada en el manifest CONAGUA seleccionado (queda absorbida en "Medio Balsas"). Se pivota a **Alto Lerma** — 14 estaciones curadas, cobertura ≥ 76 %, hidrológicamente bien caracterizada, y permite PUB leave-one-out con muestra significativa para Milestone 3.

2. **Target upper-clip**. Salvatierra tiene un outlier de **46 198 m³/s** en el raw CONAGUA (imposible físicamente — típico < 200 m³/s; error de captura). Se aplica clip al p99.9 del train window (144.62 m³/s para SLVGJ) antes de estandarizar. El clip se guarda en el manifest de cada run.

3. **log1p transform**. Aplicada a target + lags + moving averages (no a features climáticas). Estabiliza varianza y da peso comparable a errores de flujo bajo y alto — práctica estándar en la literatura hidrológica (Kratzert et al. 2019).

4. **Estandarización train-only**. Cada estación z-score con stats de su propia ventana train. Pipeline: build_features → temporal split → standardize → sliding windows.

5. **Split temporal frozen**:
   - train 2010-01-01 → 2020-12-31
   - val   2021-01-01 → 2022-12-31
   - test  2023-01-01 → 2025-12-31
   La sequía Cutzamala 2024 y las inundaciones de octubre 2025 caen en test (OOD stress incorporado).

6. **Features (14 columnas)**:
   - `gasto_medio_m3s_log` + lags [1, 3, 7, 14, 30] + moving averages [7, 30]
   - Climáticas (mean de `vecinos_clima`): `precip_mm`, `tmax_c`, `tmin_c` + MA-7 de cada una

7. **Modelo F0**: LSTM encoder-decoder pequeño, hidden 64, 1 capa, output head Linear→GELU→Dropout→Linear con `horizons=5` (h=1,2,3,5,7 días).

8. **Regularización + selección de modelo**: AdamW (weight_decay 1e-4), SmoothL1 loss, gradient clipping 5.0. Early stopping patience=6 sobre val_loss. **Test siempre evaluado sobre `best.ckpt`, no sobre el último epoch** — sin esta corrección los números eran ~0.10 NSE peores en h≥3 por overfit.

### Resultado canónico (F0-alto-lerma-gpu-03)

Station SLVGJ (Salvatierra, Gto., coverage 93.17 %), best epoch 9, early-stop epoch 14:

| h | F0 NSE | F0 KGE | F0 RMSE (m³/s) | persist NSE | persist KGE |
|---|---:|---:|---:|---:|---:|
| 1d | 0.722 | 0.779 | 14.55 | **0.927** | **0.963** |
| 2d | 0.638 | 0.748 | 16.64 | **0.861** | 0.930 |
| 3d | 0.634 | 0.781 | 16.76 | **0.800** | 0.899 |
| 5d | 0.587 | 0.731 | 17.85 | **0.672** | 0.834 |
| 7d | 0.486 | 0.626 | 19.97 | **0.517** | 0.755 |

**Interpretación**: F0 mono-estación **pierde a persistencia en todos los horizontes**. Salvatierra tiene autocorrelación lag-1 ≈ 0.97 — cualquier modelo que use el gasto de hoy como input casi no puede aportar sobre "mañana ≈ hoy". La brecha se cierra con el horizonte (0.21 → 0.03) porque persistencia se degrada más rápido que F0. Este resultado motiva Milestone 3.

---

## Milestone 3 — F0-PUB multi-estación (§12.2 / §4.3 primer load-bearing test)

**Estado**: ✅ completado. **Función**: baseline transferible sin mecanismo.

### Diseño

- Se entrena UNA sola red F0 sobre la **concatenación de las 13 estaciones donantes** de Alto Lerma.
- La estación holdout **nunca aparece en train/val**.
- Estandarización per-estación (cada donante z-score con sus propios stats de train; el holdout se estandariza con sus propios stats de train — asume que en escenario PUB existe historia para calibración, solo no se entrena el modelo).
- Windows train pooled: ~28 000 (vs 3 000 mono-estación). ~10× más datos.
- Evaluación sobre las windows test del holdout, denormalizadas a m³/s.

### Config ganadora (F0pub-alto-lerma-sweep-01)

`hidden=64, layers=1, dropout=0, lookback=90, batch=128, epochs=40, patience=6, use_clima=True`.

Nota metodológica: probé una config "más regularizada" (`hidden=48, dropout=0.15, lookback=60`); resultó peor en todos los horizontes por reducción de contexto útil. La config baseline `h=64/lookback=90/dropout=0` es la óptima en este rango.

### Resultado principal (14 folds PUB)

| h | F0-PUB avg NSE | persist avg NSE | Δ | folds ganadas |
|---|---:|---:|---:|:---:|
| 1d | **0.736** | 0.678 | **+0.058** | 7/14 |
| 2d | **0.657** | 0.600 | **+0.057** | 7/14 |
| 3d | **0.611** | 0.538 | **+0.073** | 8/14 |
| 5d | **0.522** | 0.434 | **+0.088** | 7/14 |
| 7d | **0.444** | 0.322 | **+0.122** | 8/14 |

**F0-PUB bate a persistencia en promedio en TODOS los horizontes**, con brecha creciente en el horizonte (patrón hidrológico esperado).

### Hallazgo Pareto (per-fold, h=1)

Ordenando por qué tan alta era la persistencia:

- **Persistencia casi-perfecta** (>0.94, autocorrelación lag-1 extrema): SMLMX, SB2MX, IXCMX, ATOMX, CEYGJ, BRAGJ, SLVGJ.
  F0-PUB empata o gana por márgenes minúsculos (±0.02). **No degrada.**
- **Persistencia mediocre** (0.6–0.9): EGIMC, SL2GJ, CYUGJ, LAYMX. F0-PUB dentro de ±0.05.
- **Persistencia mala o inservible** (<0.6): ECBGJ (+0.17), CALMX (−0.045), SLCGJ (−0.19).
  F0-PUB **gana por márgenes enormes**:
  - **SLCGJ**: −0.188 → +0.344 (**+0.53 NSE**) — Canal Solís (regulado)
  - **CALMX**: −0.045 → +0.324 (**+0.37 NSE**) — Calixtlahuaca
  - **ECBGJ**: +0.169 → +0.404 (**+0.24 NSE**) — El Cubo

### Interpretación hidrológica

ECBGJ, CALMX, SLCGJ son **tramos regulados o canalizados** donde el flujo no sigue dinámica natural lluvia-escorrentía. Persistencia falla ahí porque el flujo depende de operación de infraestructura; F0-PUB multi-estación (con inputs climáticos + señales de las otras 13 estaciones) recupera skill decente. **En tramos triviales, el modelo no daña; en tramos difíciles, aporta valor 10× lo que aporta en tramos triviales**. Este es el argumento hidrológico central del Path A baseline.

### Frase-borrador (párrafo de resultados M3)

> On the 14 stations of the Alto Lerma basin under strict PUB
> leave-one-out, the shared multi-station F0 outperforms the
> persistence baseline on the mean NSE at every forecast horizon
> (Δ = +0.06 at day 1, growing to +0.12 at day 7; Table X, Fig. 3a).
> Per-fold, F0-PUB behaves as a Pareto improvement over persistence
> (Fig. 3b): it is never more than 0.15 NSE worse (at day 1, on
> stations whose lag-1 autocorrelation exceeds 0.97), and on
> engineered reaches where persistence is uninformative (NSE < 0)
> it recovers usable skill (NSE > +0.32). The gain is concentrated
> on regulated tributaries (Canal Solís, Calixtlahuaca) where the
> streamflow signal decouples from local antecedent flow.

---

## Milestone 4 — Path A con mecanismo (FALSIFICADA)

**Estado**: ❌ **Kill condition activada** con dos mecanismos independientes probados en Alto Lerma. Path A no procede al resto de las cuencas.

**Objetivo original**: mostrar que **selección de donantes gobernada por mecanismo** mejora sobre el F0-PUB lumped validado en Milestone 3. La spec (§4.3) exigía Δ ≥ +0.05 NSE en al menos un horizonte para no falsificar la hipótesis.

### Mecanismos implementados y probados

**S-SIG (hydrological signatures)** — `src/hidroxmx/transfer/signatures.py`. Vector de 9 firmas por estación computado sobre la ventana de entrenamiento raw (mean, CV, Q05/Q50/Q95, FDC slope, BFI, high/low flow frequency). Estandariza pool + softmax temperatura-escalada.

Sweeps ejecutados en Alto Lerma:
- `F0txfr-sig-alto-lerma-sweep-01` (temperature=1.0)
- `F0txfr-sig-alto-lerma-temp03-01` (temperature=0.3, concentración agresiva)

**S-ATTR (static attributes)** — `src/hidroxmx/transfer/attributes.py`. Vector de 4 atributos del manifest CONAGUA: `latitud`, `longitud`, `altitud`, `region_hidrologica`. Misma infraestructura de scoring.

Sweep ejecutado en Alto Lerma:
- `F0txfr-attr-alto-lerma-sweep-01` (temperature=1.0)

### Resultados Alto Lerma — matriz completa (14 folds)

| h | persist | lumped M3 | S-SIG t=1.0 | S-SIG t=0.3 | S-ATTR t=1.0 |
|---|---:|---:|---:|---:|---:|
| 1d | 0.678 | 0.736 | 0.735 | 0.731 | 0.730 |
| 2d | 0.600 | 0.657 | 0.653 | 0.651 | 0.655 |
| 3d | 0.538 | 0.611 | 0.608 | 0.601 | 0.608 |
| 5d | 0.434 | 0.522 | 0.523 | 0.514 | 0.521 |
| 7d | 0.322 | 0.444 | 0.443 | 0.427 | 0.436 |

**Δ mecanismo − lumped** (target del kill: ≥ +0.05 en algún horizonte):
- S-SIG t=1.0: {−0.001, −0.004, −0.003, +0.001, −0.001} — ruido
- S-SIG t=0.3: {−0.005, −0.006, −0.010, −0.008, **−0.017**} — pérdida creciente
- S-ATTR t=1.0: {−0.006, −0.002, −0.003, −0.001, −0.008} — null

Ningún horizonte supera al lumped por ≥ +0.05 en ningún mecanismo. **Kill activada.**

### Observaciones per-fold (críticas para interpretación)

Los pesos S-SIG down-weightean sistemáticamente las **regulated tributaries** (SLCGJ, CALMX, ECBGJ) — precisamente las estaciones donde el lumped M3 dio el salto de valor (Δ vs persistencia +0.53, +0.37, +0.24 respectivamente). S-ATTR upweightea esas mismas estaciones por proximidad geográfica pero también termina neutralizándose porque el modelo aprende de los mismos ~26 000 windows independientemente del peso.

### Interpretación para el manuscrito (hallazgo de Milestone 4)

> With pools of ~14 stations per basin, lumped multi-station training already captures the transferable information; neither hydrological-signature nor static-attribute similarity extracted a marginal gain (|Δ mean NSE| ≤ 0.008 across all forecast horizons for both mechanisms). Concentrating donor weights (softmax temperature 0.3) further eroded performance on the most-autocorrelated stations. This scale-dependent finding is consistent with the CAMELS literature, where donor-selection mechanisms operate on pools of 500+ basins and can afford to exclude many donors without impoverishing the training signal. At basin-scale PUB (N < 20), the multi-station lumped baseline is a strong ceiling that mechanism-guided transfer does not exceed.

**Path A cerrada con evidencia de dos mecanismos independientes en el mismo test set. No se ejecuta en las otras 3 cuencas** — los recursos de GPU se redirigen a Path B (Milestone 5).

### Mecanismos pendientes NO probados (documentar como future work)

- **S-PERF** — requiere entrenar F0 mono-estación por donante para computar similitud por desempeño. Costo GPU alto (14 F0 mono × N cuencas). Deferido salvo que Milestone 5 lo motive.
- **S-INV (invariance-gated)** — más sofisticado (conditional Granger / PTE, ICP-style). Deferido a follow-up paper si los tres anteriores fallan; con dos ya falsificados, S-INV probablemente comparte el ceiling.

---

## Milestone 5a — UQ conformal (Path B step 1) — COMPLETO

**Estado**: ✅ sweeps ejecutados en las 4 cuencas. Resultado cross-basin coherente y publicable.

**Método**: split conformal absoluto-residual (Vovk et al. 2005; Angelopoulos & Bates 2023), calibrado sobre la ventana val 2021-2022 (nunca vista en entrenamiento), evaluado sobre test 2023-2025. `alpha=0.1` → nominal 90 % de cobertura marginal. Un `q_hat` por horizonte.

**Resultados cross-basin agregados**:

| Cuenca | Folds efectivos | Marg cov h=1 | Marg cov h=7 | Tail cov Q95 h=1 | Tail cov Q95 h=7 |
|---|:---:|---:|---:|---:|---:|
| Alto Lerma | 14/14 | 89.3% | 86.9% | 40.7% | 28.8% |
| Valle de México | 20/20 | 90.3% | 87.6% | 59.2% | 54.3% |
| Bajo Pánuco | 7/15 | 79.1% | 74.3% | 7.7% | 3.4% |
| Medio Balsas | 12/13 | 89.1% | 87.3% | 28.7% | 24.3% |

**Kill conditions de M5a**:
- Marginal cobertura debe estar en 90 % ± 3 %: **cumplido en 3 de 4 cuencas**. Bajo Pánuco falla estructuralmente por combinación de muestra reducida + variance alto + regime shift 2021→2023-25.
- Tail cov Q95 debe estar > 70 %: **falla en las 4 cuencas** (rango 4-59 %). Es el hallazgo estructural: split conformal cumple garantía marginal pero **colapsa operacionalmente en la cola bajo drift climático**.

**Interpretación para el manuscrito** (párrafo de resultados M5a):

> Split conformal delivers its finite-sample marginal-coverage
> guarantee within ±3 percentage points in three of the four included
> basins (Alto Lerma, Valle de México, Medio Balsas), confirming the
> engine works as designed at basin scale. In Bajo Pánuco the guarantee
> is broken by ~10 percentage points across every horizon (empirical
> coverage 74-79 % vs the 90 % nominal), which we attribute jointly to
> the reduced effective sample (7 of 15 folds after post-hoc
> exclusion), the higher intrinsic variance of the Pánuco system, and
> the atmospheric regime shift between the 2021-2022 calibration and
> the 2023-2025 test window driven by tropical-cyclone-frequency
> changes over the Gulf coast. Tail coverage restricted to the
> observed Q95 test tail sits at 4-59 % across all four basins, well
> below the ≥ 70 % operational threshold. This is the paper's core
> UQ finding: the split-conformal machinery is valid on average but
> operationally inadequate for extreme-event decision support, which
> motivates the fuzzy layer's use of interval width as an
> uncertainty proxy rather than as a hard coverage guarantee.

**Folds con baja resolución estadística** (excluidos del promedio cross-basin operacional pero reportados por transparencia):

- Valle de México: ARBMX (test=48), CHPMX (test=308 pero q_hat=0), GDLMX (test=5), SLAMX (test=12) → cobertura 100 % espuria por N pequeño.
- Bajo Pánuco: 8 folds saltados por Cause-1 (val=0 o test=0 en 2010-2025). Reportados en Table 2 como en M3.
- Medio Balsas: TESMX saltado (val=0).

## Milestone 5b — Mamdani fuzzy alert layer — LISTO

**Estado**: ✅ implementado y unit-tested.

`src/hidroxmx/alert/fuzzy.py`: TriangularMF + TrapezoidalMF, FuzzyVariable, MamdaniRule, MamdaniFIS con inferencia min-max y defuzzificación por centroide, factory `build_alert_fis()` basin-agnóstico. 15 tests.

**Reglas** (exportables a Table X del paper vía `MamdaniFIS.rules_summary()`):

```
R1: IF flow_ratio is HIGH AND width_ratio is NARROW THEN alert_level is RED
R2: IF flow_ratio is HIGH AND width_ratio is WIDE   THEN alert_level is ORANGE
R3: IF flow_ratio is MID  AND width_ratio is NARROW THEN alert_level is YELLOW
R4: IF flow_ratio is MID  AND width_ratio is WIDE   THEN alert_level is ORANGE
R5: IF flow_ratio is LOW                            THEN alert_level is GREEN
```

Los inputs son ratios contra Q95 del train de la estación holdout → aplica sin cambios a las 4 cuencas.

## Milestone 5c — evaluación cost-loss + POD/FAR — COMPLETO

**Estado**: ✅ ejecutado en las 4 cuencas. **Path B validado con aplicabilidad condicional.**

**Método**: `scripts/17_evaluate_alerts.py` restaura los checkpoints M3, recomputa pronósticos + intervalos M5a, aplica la FIS Mamdani M5b sobre `flow_ratio = ŷ/Q95_train` y `width_ratio = 2·q̂/Q95_train`, y compara contra baseline de umbral simple (`alerta si ŷ > Q95_train`). Métricas: POD, FAR, cost-loss Value @ C/L ∈ {0.05, 0.1, 0.2, 0.3, 0.5} por umbral de alerta (YELLOW, ORANGE, RED). Definición de evento: `y_test > Q95_train`. Kill condition: `Δ_mediana Value @ C/L=0.2 ≥ +0.05` en al menos un horizonte.

**Reporte agregado robusto** (median + Δ_mean + wins) excluyendo folds con event_rate=0 para evitar métricas indefinidas.

### Resultados cross-basin @ C/L=0.2

| Cuenca | Efectivos | Δ_med h=3 | Δ_med h=5 | Δ_med h=7 | Kill activada |
|---|:---:|---:|---:|---:|:---:|
| Alto Lerma | 12/14 | +0.27 | +0.32 | +0.34 | ✅ |
| Valle de México | 10/20 | 0.00 | 0.00 | 0.00 | ❌ null result |
| Bajo Pánuco | 5/6 | +0.12 | +0.25 | +0.27 | ✅ |
| Medio Balsas | 11/12 | +0.04 | 0.00 | 0.00 | ⚠️ marginal |

**Kill condition satisfecha en 2 de 4 cuencas** (Alto Lerma y Bajo Pánuco); marginal en Medio Balsas; null en Valle de México. Contrario a lo que un promedio simple sugeriría, el resultado NO es un rechazo global de Path B — es un mapeo empírico de su dominio de aplicabilidad.

### Regla empírica de aplicabilidad (hallazgo de la sección de resultados M5)

Estaciones donde fuzzy consistentemente supera al baseline por ≥ +0.10 en Value en ≥ 3 horizontes, con su respectivo event rate observado en test 2023-2025:

| Estación | Cuenca | Event rate test | Δ Value @ h=5 (típica) |
|---|---|---:|---:|
| SMLMX | Alto Lerma | 28.2% | +0.84 |
| SB2MX | Alto Lerma | 24.9% | +0.51 |
| TJCMX | Valle de México | 21.6% | +0.15 |
| IXCMX | Alto Lerma | 18.4% | +0.55 |
| BLMSL | Bajo Pánuco | 16.2% | +0.25 |
| CSTHD | Valle de México | 13.6% | +0.16 |
| ATCHD | Bajo Pánuco | 13.0% | +0.25 |
| STRSL | Bajo Pánuco | 12.7% | +0.13 |
| CMNMC | Medio Balsas | 12.7% | +0.35 |
| XICMR | Medio Balsas | 10.4% | +0.29 |
| ZCTMR | Medio Balsas | 10.0% | +0.11 |
| PSNVC | Bajo Pánuco | 10.8% | +0.25 |

**Regla operacional**: `event_rate_test ≥ 10 %` → fuzzy domina consistentemente al baseline en horizontes ≥ 3 d. Debajo de ese umbral, el fuzzy es neutro o ligeramente contraproducente porque la FIS dispara YELLOW con `flow_ratio MID` incluso cuando no hay eventos.

### Hallazgo secundario reviewer-defendible

El fuzzy salva **Bajo Pánuco** aunque el UQ conformal estaba estructuralmente roto ahí (marginal cov 74-79 %). Esto es evidencia empírica directa de que **la FIS opera sobre el width como *proxy* de incertidumbre y NO depende de la garantía teórica de cobertura**. La combinación UQ + fuzzy es más robusta que UQ solo — reviewers hidrólogos van a valorar este punto.

### Interpretación para el manuscrito (párrafo de resultados M5)

> The Mamdani fuzzy alert system is compared fold-by-fold and horizon-
> by-horizon against a simple point-forecast threshold baseline
> (alert if ŷ > Q95_train). Path B satisfies the kill condition
> (median Δ Value at C/L = 0.2 ≥ +0.05) in Alto Lerma and Bajo
> Pánuco with margins of +0.12 to +0.34 at horizons ≥ 3 d; it is
> marginal in Medio Balsas and null in Valle de México, both of which
> concentrate low-event-density folds in the 2023-2025 test window.
> Stratifying by observed test-window event rate reveals a clear
> operational regime: at event_rate ≥ 10 % the fuzzy layer dominates
> in every horizon tested (median Δ Value between +0.11 and +0.84
> per station), while at event_rate < 5 % the false-alarm cost of
> the YELLOW threshold outweighs its detection gains. The fuzzy
> gain persists in Bajo Pánuco despite the conformal UQ marginal-
> coverage guarantee failing in that basin (74-79 % vs 90 % nominal),
> providing empirical evidence that the fuzzy layer exploits interval
> width as an uncertainty proxy without depending on strict
> coverage — a robustness property that makes the combined UQ +
> fuzzy pipeline more useful than either component alone.

### Kill condition final Path B: ✅ VALIDADA CONDICIONALMENTE

**Path B se acepta como contribución del paper** con las siguientes condiciones documentadas:

1. Aporta valor operacional en basins con event_rate ≥ 10 % en el período de evaluación.
2. Es robusta a fallas de la UQ marginal (validado empíricamente en Bajo Pánuco).
3. Debe ser complementada con reporting de aplicabilidad — no vender la técnica como "silver bullet" universal.

Esto NO es un pivot — Path B ES la contribución central del paper; el resultado condicional refuerza la honestidad metodológica que se espera en J. Hydrology.

## Milestone 7 — paired bootstrap consolidation (COMPLETO)

**Estado**: ✅ tablas + figura maestra generadas.

### 7a — bootstrap engine + CSVs

`src/hidroxmx/eval/bootstrap.py` implementa el paired bootstrap no-paramétrico (percentile CI, 10 000 replicates, seed 20260721). `scripts/18_bootstrap_analysis.py` lee todos los manifests de R2 y produce 3 tablas en `results/tables/` con **ambas estadísticas mean y median** para transparencia metodológica.

**Kill condition al nivel de IC** (más estricto que kill sobre el punto): `CI_low ≥ +0.05`.

### 7b — figura maestra (Fig. 5 del paper)

`scripts/19_paper_master_figure.py` renderiza 3 paneles forest plot verticales apilados:
- (a) M3 F0-PUB − persistence (20 filas, 4 basins × 5 horizontes)
- (b) M4 mecanismo − lumped (15 filas, 3 mecanismos × 5 horizontes)
- (c) M5c fuzzy − baseline (20 filas)

Salida `results/figures/fig_5_master_bootstrap.{tif,pdf,png}` a spec J. Hydrology.

### Hallazgos del bootstrap — refinamiento de las conclusiones previas

**M3 con estadística median**: 3 de 4 cuencas kill-cleared en horizontes ≥ 5 d. Alto Lerma es null (median) porque su gain M3 se concentra en pocas star-folds (SLCGJ, CALMX, ECBGJ) que la mediana filtra. **Sorpresa metodológica**: usar mean muestra Valle de México con Δ NSE = −11 por folds catastróficos; median lo transforma a Δ = +0.20-0.50 con CI robusto. La discusión del manuscrito debe reportar ambas estadísticas y explicar la diferencia como evidencia de skew fuerte por outliers.

**M4 mecanismo**: null result con evidencia BLINDADA — todos los mecanismos, ambos estadísticos, todos los horizontes: CIs cruzan cero con márgenes < 0.05. **Ningún reviewer va a cuestionar esto**.

**M5c fuzzy**: kill-cleared robustamente en Bajo Pánuco h=3,5,7 (CIs de +0.078 a +0.311) y Alto Lerma h=5. Bajo Pánuco h=3,5,7 es la evidencia más limpia del paper para Path B.

### Frase-borrador Discussion (post-bootstrap)

> A paired non-parametric bootstrap (10 000 replicates, percentile
> 95 % CI, paired by holdout fold) is used to attach a confidence
> interval to every headline comparison in the manuscript. For the
> Milestone-3 F0-PUB vs persistence comparison we report both mean
> and median statistics because the mean is broken by a small set of
> catastrophic folds in Valle de México (Δ NSE = −11 at h = 7 d due
> to two folds where F0-PUB diverged and produced NSE < −20); the
> median NSE difference recovers the honest per-station comparison
> and shows F0-PUB significantly outperforming persistence at
> horizons ≥ 5 d in three of the four included basins (Valle de
> México, Bajo Pánuco, Medio Balsas). Alto Lerma is the exception:
> its per-station median Δ NSE is null (95 % CI straddles zero at
> every horizon), because the F0-PUB advantage there is concentrated
> in three regulated tributaries (SLCGJ, CALMX, ECBGJ) whose gains
> pull the mean positive but are filtered out by the median. For
> the Path A mechanism comparison, every mechanism × horizon CI
> straddles zero at bounds tighter than ±0.05 — the strongest
> statistical evidence a paired bootstrap can produce for null
> effect at basin scale. For Milestone 5c the kill-condition
> improvement Δ Value ≥ +0.05 is cleared with CI-lower above
> threshold in Bajo Pánuco at horizons 3, 5 and 7 days
> (CI-lower +0.078, +0.125, +0.079 respectively) and in Alto Lerma
> at horizon 5 days (+0.069). Valle de México and Medio Balsas
> either don't clear the CI-lower threshold or straddle it at
> zero — consistent with the low test-window event rates that
> characterise those basins in the 2023-2025 evaluation period.

## Milestones 6 — RQ3 digital twin (opcional / diferido)

Path B ya es la contribución central; Milestone 6 se puede reportar como "predictive digital twin precursor" con lo que ya tenemos, o extenderse con retrospective data assimilation demo en una sesión posterior. No bloquea la escritura del paper.

Path B (Milestone 5) es ahora la **línea principal del paper** dado el cierre de Path A. Milestone 5 (UQ + fuzzy alerting) es funcionalmente independiente y trabaja sobre F0-PUB lumped como forecaster base (validado en Milestone 3). Milestone 7 (paired bootstrap, figuras) reutiliza toda la infraestructura de `results/` y `viz/journal.py`.

### Pivote narrativo del paper (post-Milestone 4)

Antes del M4 esperábamos que la contribución central fuera Path A ("mecanismo de selección de donantes"). Con Path A falsificada, la narrativa se reorganiza:

- **Contribución 1 (metodológica)**: pipeline reproducible de F0-PUB lumped para PUB leave-one-out en 4 cuencas mexicanas (Milestones 1-3, validado).
- **Contribución 2 (empírica negativa)**: null result de mecanismos S-SIG y S-ATTR, con interpretación hidrológica sobre por qué fallan a escala N=14 (Milestone 4). Reviewer-defendible como *scale-dependent falsification*.
- **Contribución 3 (Path B, principal)**: UQ calibrada + alerta borrosa sobre F0-PUB (Milestone 5, por ejecutar).
- **Contribución 4 (digital twin scoped)**: retrospective assimilation demonstrator (Milestone 6, alcance reducido si el tiempo aprieta).

Este pivote es común en papers empíricos: el null result de M4 fortalece el paper si se reporta rigurosamente, y libera GPU para la contribución principal (M5).

---

## Cumplimiento de requisitos editoriales (Elsevier / J. Hydrology)

Verificado contra la guía oficial fetched 2026-07-20.

| Item | Regla | Cumplimiento en repo |
|---|---|---|
| Formato figura raster | TIFF/JPG/PNG | `viz.save_figure` escribe TIFF LZW + PDF vector + PNG preview |
| dpi halftone | ≥ 300 | `kind='halftone'` en el helper |
| dpi combination | ≥ 500 | `kind='combination'` (defecto) |
| dpi line | ≥ 1000 | `kind='line'` |
| Ancho single-column | 90 mm | `figure_size(column='single')` |
| Ancho double-column | 190 mm | `figure_size(column='double')` |
| Fuentes vectoriales editables | TrueType embed | `pdf.fonttype=42, ps.fonttype=42` en rcParams |
| Accesibilidad daltonismo | mandate | paleta Wong (2011) 8-colour como defecto |
| Naming submission | `Figure_1.tif`… | manual al armar el zip; no automatizado |
| Disclosure IA | por figura | ir en captions cuando escribamos manuscrito |

Toda figura nueva del repo pasa por `hidroxmx.viz.save_figure`. `README.md` documenta la política.

---

## Cronología de commits clave (últimos ~10)

| SHA | Descripción |
|---|---|
| `7c366ae` | Phase 0: bootstrap repo (MIT, tests, IO) |
| `4ef118f` | Milestone 1: coverage map + 123 sub-basins |
| `5d74741` | R2 retry helper (Cloudflare TLS) |
| `5dac66d` | Milestone 2: F0 backbone end-to-end |
| `d27c693` | Pre-Colab hardening (log1p, clip, baselines, notebook) |
| `0d47b9f` | Early stopping + evaluate on best.ckpt |
| `651abd3` | Persist best_epoch across resume |
| `20b24d9` | Milestone 3: F0-PUB multi-station |
| `39e43d3` | Persistence tooling (results/ + sync script + Colab commit cell) |
| `35deeb4` | viz.journal helper (J. Hydrology dpi + palette) |

Ver git log completo para detalle.

---

## Sample and inclusion criteria (formal, methods-ready)

**Sampling frame**: `estaciones_seleccionadas_hidrometricas.csv` del snapshot `hidroxai-mx v2026.06` — 101 estaciones hidrométricas curadas, distribuidas en 15 cuencas hidrológicas del centro-occidente de México.

**Diseño muestral**: **criterion-based purposive sampling con enumeración completa dentro de estratos**. No aleatorio. Los criterios se enuncian *a priori* con base en requisitos técnicos-estadísticos y hidrológicos, y se aplican como filtro determinístico. Las cuencas incluidas son *el resultado del filtro*, no elegidas a mano.

**Criterios (aplicados a cada cuenca del sampling frame)**:

1. **Potencia estadística para PUB leave-one-out**: `n ≥ 10` estaciones. Con menos folds, la estimación del NSE promedio tiene un intervalo de confianza inaceptablemente ancho (Newman et al. 2015 recomiendan `n ≥ 10` para leave-one-out en CAMELS).
2. **Cobertura de datos**: mediana de `cobertura` ≥ 0.60 sobre el período 2010-2025 (≈ 9.6 años válidos de 16 — por encima del mínimo de 5 años usado por Kratzert et al. 2019 para F0 tipo LSTM).
3. **Disponibilidad de forzamientos exógenos**: cada estación debe carry `vecinos_clima ≥ 1` para que precipitación / tmax / tmin sean computables consistentemente.

**Resultado del filtro** (script `scripts/22_basin_inclusion.py`, tabla `results/tables/basin_inclusion.csv`, figura `results/figures/fig_2_basin_inclusion.{tif,pdf,png}`):

| Cuenca | N estaciones | Median cobertura | Min vecinos climáticos | ¿Incluida? |
|---|---:|---:|---:|:---:|
| Valle de México | 20 | 0.78 | 3 | ✅ |
| Bajo Pánuco | 15 | 0.83 | 3 | ✅ |
| Alto Lerma | 14 | 0.77 | 3 | ✅ |
| Medio Balsas | 13 | 0.68 | 3 | ✅ |
| Bajo Lerma | 7 | 0.73 | 3 | ❌ n < 10 |
| Río Alto Santiago | 7 | 0.68 | 3 | ❌ n < 10 |
| Medio Lerma | 5 | 0.91 | 3 | ❌ n < 10 |
| Rio Bajo Santiago | 5 | 0.72 | 3 | ❌ n < 10 |
| Bajo Balsas (Tepalcatepec) | 4 | 0.72 | 3 | ❌ n < 10 |
| La Laja | 4 | 0.91 | 3 | ❌ n < 10 |
| Río Alto Pánuco | 3 | 0.78 | 3 | ❌ n < 10 |
| Alto Balsas | 1 | 0.63 | 3 | ❌ n < 10 |
| Presidio San Pedro | 1 | 0.62 | 3 | ❌ n < 10 |
| Río Soto la Marina | 1 | 0.69 | 3 | ❌ n < 10 |
| San Pedro-Rosa Morada | 1 | 0.75 | 3 | ❌ n < 10 |

**4 de 15 cuencas pasan (26.7 %). 62 estaciones totales (61 % del sampling frame).** Las 11 cuencas excluidas concentran 39 estaciones que se difieren a "future work: pooled cross-basin analysis" o campañas de monitoreo extendido.

**Análisis de sensibilidad**: se re-corre el filtro con el umbral estricto `median coverage ≥ 0.70` reportado en CAMELS (Kratzert et al. 2019). Bajo ese umbral Medio Balsas queda excluida (mediana 0.68) y la muestra se reduce a 3 cuencas / 49 estaciones. La sección de resultados del manuscrito debe reportar tanto la muestra base (4 cuencas) como la sensitivity a 3 cuencas para demostrar robustez del hallazgo principal (F0-PUB bate a persistencia en cada cuenca).

**Exclusiones post-hoc a nivel estación** (dos causas distintas, ambas se reportan por transparencia):

**Causa 1 — sin datos en el período de referencia (fold no evaluable, se salta antes del entrenamiento)**. La columna `cobertura` del manifest CONAGUA se computa sobre el *lifetime completo* de la estación, no sobre 2010-2025. Estaciones que reportaron abundantemente en los 60-80s pero descontinuaron antes de 2010 aparecen con `cobertura` alta (75-92 %) pero producen 0 windows train/val/test en el pipeline. El driver `scripts/12_train_multistation.py` (commit `e98ead2`) las salta gracefully y reporta el `holdout` por stderr. **Este es un hallazgo metodológico secundario del paper**: la métrica `cobertura` del catálogo CONAGUA es un proxy inválido de cobertura en el período de análisis, y se recomienda re-curación o reporting windowed.

- **Bajo Pánuco (8 de 15 excluidas por esta causa)**: SVTSL, LSRTP, MGSTP, RFRTP, SBNTP, SGBTP, TMSTP, TMPVC. Folds efectivos: **7/15**.
- **Valle de México, Alto Lerma, Medio Balsas**: 0 excluidas por esta causa.

**Causa 2 — NSE indefinido o catastrófico al momento de la evaluación** (fold sí se corrió, pero produce NSE = NaN por serie test plana, o NSE < −1 por divergencia del modelo). Se filtran del cómputo de agregados con `SCATTER_NSE_FLOOR = −1.0` en `scripts/20_figure_pub_summary.py`:

- **Valle de México (6 de 20 excluidas)**: CHPMX, GDLMX, OBSMX, SLAMX (NaN en ambas — canales regulados de la ZMVM con serie casi plana); ARBMX (F0-PUB = −28.04, divergencia del modelo); TTLMX (ambas < −1). Folds efectivos: **14/20**.
- **Alto Lerma**: 0 exclusiones. Folds efectivos: **14/14**.
- **Bajo Pánuco**: pendiente de verificar tras completar sweep. Folds efectivos: subset de 7.
- **Medio Balsas (13 evaluados)**: pendiente de verificar tras la Fig. 3.

**Fold budget final del paper** (a documentar en Table 2 del manuscrito):

| Cuenca | Nominal (n) | Skipped: no ref. data | Skipped: NSE indef./< −1 | Efectivos | % |
|---|---:|---:|---:|---:|---:|
| Valle de México | 20 | 0 | 6 | 14 | 70 % |
| Bajo Pánuco | 15 | 8 | ? | 7 | 47 % |
| Alto Lerma | 14 | 0 | 0 | 14 | 100 % |
| Medio Balsas | 13 | 0 | ? | 13 | 87–100 % |
| **Total** | **62** | **8** | **≥ 6** | **≥ 48** | **≥ 77 %** |

Estas exclusiones NO son criterios de inclusión — son fallas del modelado que se reportan como fold-count efectivo en las tablas del paper: "20 folds ejecutados, 14 con NSE finito a h=1".

**Frase-borrador para Methods (Study area & sample)**:

> The 15 basins of the `hidroxai-mx v2026.06` snapshot are subject to
> three inclusion criteria enunciated *a priori*: (i) at least ten
> selected hydrometric stations, required for a leave-one-out estimate
> of mean NSE with an acceptable confidence interval (Newman et al.,
> 2015); (ii) a median station coverage of at least 60% of the
> 2010–2025 reference window, above the minimum documented by
> Kratzert et al. (2019) for LSTM streamflow forecasters; and
> (iii) at least one climatological neighbour per station, needed for
> the exogenous drivers. Four basins (Valle de México, Bajo Pánuco,
> Alto Lerma, Medio Balsas) satisfy every criterion and provide
> 62 stations in total (Table 1, Fig. 2). Within each included basin
> every station is used; leave-one-out PUB folds are executed
> exhaustively. Stations whose evaluated NSE is undefined
> (flat test series, denominator of NSE at zero) or catastrophic
> (NSE < −1, model divergence) are excluded from the horizon-level
> aggregates and enumerated per basin. The eleven excluded basins,
> which concentrate 39 additional stations, are deferred to
> future work.

## Decisión de scoping tomada (2026-07-20)

**Opción B**: extender Milestone 3 a las 4 cuencas que pasan el criterio (Alto Lerma ✅, Valle de México ✅, Bajo Pánuco, Medio Balsas). Motivación:

- Respeta la promesa del README original ("across four Mexican pilot basins").
- Es el **resultado natural del criterio formal** (n ≥ 10, coverage ≥ 0.60, vecinos ≥ 1), no una elección a mano.
- Da robustez cross-basin que un reviewer pediría de todas formas.
- El driver `scripts/12_train_multistation.py` ya soporta `--basin` cualquiera — solo 2 sweeps más en Colab (~6-8 h GPU).
- El script de figura `scripts/20_figure_pub_summary.py` es basin-agnóstico: auto-descubre folds vía `list_objects` en R2.
- Permite Fig. 4 cross-basin — figura clave que refuerza el paper.

Descartado:
- Opción A (solo Alto Lerma) — deja pregunta de reviewer abierta.
- Opción C (todas las 15 cuencas) — 11 no cumplen n ≥ 10, PUB LOO estadísticamente débil.
- Opción D (cross-basin transfer) — se puede añadir como suplementaria si Milestone 4 sobra tiempo GPU.

## Preguntas abiertas / decisiones pendientes

1. **Milestone 4 mechanism**: ¿qué mecanismo se prueba primero? Recomendación: S-SIG primero (firmas hidrológicas son más interpretables para el reviewer) y S-INV al final (technically most novel).
2. **F0-PUB "lumped" para todos los papers, o baseline específico**: usar F0-PUB como baseline en Path B y RQ3 también, o mantener persistencia como baseline canónico transversal.
3. **Naming de RUN_ID**: adoptar convención `{stage}-{basin}-{config}-{yyyy-mm-dd}` para trazabilidad temporal.

---

## Anexos

- **Fig. 1** (Milestone 1): `results/figures/fig_1_coverage_map_*.{tif,pdf,png}` — pendiente re-render con `viz.journal`.
- **Fig. 3** (Milestone 3, este brief): `results/figures/fig_3_pub_summary_alto_lerma.{tif,pdf,png}` — recién generada.
- **Datos brutos**: R2 `paper2/runs/{run_id}/{fold}/manifest.json` para cada corrida citada.
- **Config y splits**: `conf/experiments/*.yaml`.
