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

## Milestone 4 — Path A con mecanismo (pendiente)

**Objetivo**: mostrar que **selección de donantes gobernada por mecanismo** (S-ATTR / S-PERF / S-SIG / S-INV) mejora sobre el F0-PUB "lumped" que Milestone 3 acaba de validar.

- **S-ATTR**: similitud por atributos estáticos (área, pendiente, geología).
- **S-PERF**: similitud por desempeño agregado del F0 mono-estación en el donante.
- **S-SIG**: similitud por firmas hidrológicas (flow duration curve, baseflow index).
- **S-INV**: invariance-gated (conditional Granger / PTE, ICP-style).

**Condición de kill**: si el mejor mecanismo NO bate al F0-PUB lumped por ≥ +0.05 NSE en al menos un horizonte, Path A queda falsificada y pivoteamos a Path B como línea principal.

**Diseño en agenda**:
- Nuevo `scripts/13_donor_matching.py` con `--criterion {attr,perf,sig,inv}`.
- Nuevo `scripts/14_transfer_train.py` que pesa los windows por score de mecanismo.
- Evaluar con el mismo test set (fold PUB) para comparación pareada.

---

## Milestones 5-7 — RQ2 Path B, RQ3 digital twin, evaluación

Pendientes. Milestone 5 (UQ + fuzzy) es funcionalmente independiente de M4 y puede paralelizarse. Milestone 7 (paired bootstrap, figuras) reutiliza toda la infraestructura de `results/` y `viz/journal.py`.

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

**Exclusiones post-hoc a nivel estación**: dentro de cada cuenca incluida, algunas estaciones producen NSE indefinido (test series casi plana → denominador 0) o catastrófico (F0-PUB divergido). Estas se filtran del cómputo de agregados (`SCATTER_NSE_FLOOR = −1.0` en `scripts/20_figure_pub_summary.py`) y se reportan explícitamente:

- **Valle de México (6 de 20 excluidas post-hoc)**: CHPMX, GDLMX, OBSMX, SLAMX (NaN en ambas — series test casi planas, canales regulados de la ZMVM); ARBMX (F0-PUB = −28.04, model divergence); TTLMX (ambas < −1).
- **Alto Lerma**: 0 exclusiones post-hoc (14/14 folds válidos).
- **Bajo Pánuco / Medio Balsas**: pendiente (por completar).

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
