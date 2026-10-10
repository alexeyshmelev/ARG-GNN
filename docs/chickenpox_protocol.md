# Hungarian Chickenpox fixed-topology applicability protocol

Status: source acquisition, schema checks and CPU preparation only. No benchmark model has been assessed and no validation or test scores have been computed. This protocol must remain frozen before model assessment.

## Source, licence and retained evidence

The [UCI Hungarian Chickenpox Cases page](https://archive.ics.uci.edu/dataset/580/hungarian+chickenpox+cases) explicitly assigns [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/legalcode), permitting sharing and adaptation for any purpose with appropriate credit. The displayed citation is:

> Hungarian Chickenpox Cases [Dataset]. (2021). UCI Machine Learning Repository. https://doi.org/10.24432/C5103B.

The public archive is https://archive.ics.uci.edu/static/public/580/hungarian+chickenpox+cases.zip. The source was checked and acquired on 2026-10-10. Its unchanged bytes are retained at `data/raw/chickenpox/hungarian-chickenpox-cases.zip`. Acquisition and CSV asset metadata are recorded in `data/raw/chickenpox/source_manifest.json`. No archive assets are executed. The loader checks the pinned hash, member names, bounded compressed and expanded sizes, and symbolic-link/path-traversal exclusions before reading CSV bytes in memory.

| Asset | Bytes | SHA256 |
|---|---:|---|
| Archive | 14,765 | `dc306b1845c86616d2a72c202deeefac8fd1fcf0a34b3067cf6721e377d9da44` |
| `hungary_chickenpox.csv` | 35,065 | `bd1b802f1059b9b0f0d2c29d6171bc2f302994f81a92d7c7bb25191f76dc4a23` |
| `hungary_county_edges.csv` | 1,850 | `0061c603b935f3722b7c621e545a30f9129b0c0c619bd4bd30e62a7975c8d609` |

The UCI page reports 521 instances. The pinned case CSV contains **522 distinct weekly observations**, from **2005-01-03 through 2014-12-29**, with every successive gap exactly seven days. Analyses use the actual 522-row archive and retain this discrepancy instead of silently dropping a row.

## County identity and graph

There are 20 county-level series. The 102 directed adjacency records are symmetric and contain 20 supplied self loops plus 82 directed non-self records, corresponding to 41 undirected neighbor links. Preserve this native adjacency. If an encoder adds another identity connection, record how it avoids unintended double counting.

The case CSV begins `BUDAPEST,BARANYA,BACS,...`, but graph IDs follow:

`BACS,BARANYA,BEKES,BORSOD,BUDAPEST,CSONGRAD,FEJER,GYOR,HAJDU,HEVES,JASZ,KOMAROM,NOGRAD,PEST,SOMOGY,SZABOLCS,TOLNA,VAS,VESZPREM,ZALA`.

The preparation code joins case columns to edge identities **by county name**. In the processed tensors, BACS is ID 0 and BUDAPEST is ID 4. Raw CSV column position is never used as graph identity. The graph remains fixed at every week, including its identities and edges.

## Chronological one-week-ahead forecasting

The task is county-level case-count regression. Each forecast uses the preceding 52 observed weeks and predicts one subsequent week. The boundaries are zero-based and exclusive at `train_end=365`, `validation_end=443`, and `observations=522`.

| Split | Target indices | Target dates | Number of weekly forecasts |
|---|---|---|---:|
| Training | 52–364 | 2006-01-02–2011-12-26 | 313 |
| Validation | 365–442 | 2012-01-02–2013-06-24 | 78 |
| Test | 443–521 | 2013-07-01–2014-12-29 | 79 |

For target index t, the input is exactly `counts[t-52:t]`. Training, validation and test targets are disjoint. History windows naturally overlap. This is a single chronological time series rather than independent simulation replicates or randomly split windows.

Fit one mean and population standard deviation per county using only raw observations `[0:365]`. Replace a zero standard deviation by one. Freeze this scaler for validation and test. No future counts, whole-series normalization, future labels, split-dependent encodings or target-derived features enter a history graph. Fitting preprocessing on the complete training portion is standard training preprocessing and does not use validation or test observations.

Use rolling-origin one-week-ahead evaluation with frozen model weights. Earlier validation/test counts may enter later histories because those weeks have already been observed at each forecast origin. This does not mean open-loop forecasting of 79 future weeks from one origin. Do not retrain or update the scaler while evaluating test forecasts. Tune all choices on validation, then freeze configurations and evaluate the test once under the parent experiment protocol.

Primary metric: MAE in original case units, averaged across target weeks and all 20 counties. Secondary metric: RMSE over the same predictions. Also retain per-county MAE for heterogeneity. Training may use standardized squared error, but report metrics after inverse scaling. Do not retrospectively choose clipping or alternative metrics based on test performance. Any nonnegative clipping rule must be fixed for all applicable methods before evaluation. If uncertainty intervals are used, resample contiguous time blocks rather than treating counties and overlapping histories as independent samples.

## Model contract and applicability limits

`forecast_sequence` packs all 52 observed county graphs, one normalized count feature per node, stable county identities, native edges and uniform weekly intervals on `[0,1]`. It uses a single readout anchor at the midpoint of the last observed interval and queries all 20 counties. This anchor supports existing `GraphSequence` validation. It is a forecast-head adaptation and does not imply that the future target week is inside the observed historical domain.

Tree-clade metadata is an all-zero single column because geographic counties have no supplied genealogy. Do not invent tree ancestry. County IDs are correspondence keys rather than learned numeric input features. Existing classification heads that require `num_classes >= 2` must be adapted to a scalar regression head and an appropriate regression loss by the integration owner. Do not discretize counts merely to reuse a classifier.

This benchmark measures applicability to another temporal graph domain. Its fixed topology and persistent county set cannot validate benefits from changing topology, changing internal nodes, lineage correspondence or ARG structure. A favorable result here is not evidence of biological generalization.

## Required controls and interventions

1. Last observed value: `counts[t-1]`.
2. Seasonal value: `counts[t-52]`. The common 52-week history ensures this control uses the same available observations. This is a lag-52 rule rather than an assertion that every calendar year has exactly 52 weeks.
3. Training-only multivariate ridge autoregression on the flattened 52-week history. Fit only training targets, select a positive regularization value using validation, and freeze coefficients for test. `LinearRidgeForecaster` supplies a centered dual ridge solve and raw-unit predictions.
4. An order-blind model using pooled historical graph summaries, with an explicit current-week readout and the same input history. This distinguishes temporal dependence from simple observed-value pooling.
5. A temporal model without geographic message passing, with comparable widths and tuning resources. This distinguishes temporal modeling from graph benefit.
6. An edge-removal ablation using only identity/self connections, with all temporal inputs and training choices held fixed. Native self loops must be handled consistently.

For the chronology intervention, permute the first 51 history graphs and **hold the last observed graph fixed**. Preserve each history's multiset, county identities, targets, scaler, weekly interval locations and final readout anchor. `history_permutation(52, seed, fix_last=True)` supplies this intervention. Record permutation seeds and use the same sample-level permutations across methods. Evaluate the already-trained chronological model under this intervention after the configuration is frozen. A separate train-and-evaluate shuffled control can be useful, but must be separately named and receive its own validation tuning.

Moving weekly content among fixed interval locations scrambles chronology. Reordering storage while leaving each graph attached to its original interval only changes serialization and is a separate invariance check. Shuffle degradation can show model reliance on chronology. It cannot prove chronology is mathematically necessary or establish a changing-topology benefit. Because the final observed week is fixed, the last-value control is unchanged. The seasonal control is a native chronological baseline rather than a substitute benchmark after its lag mapping has been destroyed.

## Preparation and checks

Run from the repository on anogena:

```bash
. scripts/project_env.sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$ARGGNN_PYTHON" scripts/prepare_chickenpox.py
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$ARGGNN_PYTHON" -m pytest tests/test_chickenpox_protocol.py -q --basetemp="$ARGGNN_ROOT/.runtime/tmp/pytest-chickenpox"
```

The CLI writes `data/processed/chickenpox/chickenpox.npz` with raw counts, ISO dates, county order, adjacency, training scaler and target split arrays. `protocol.json` records boundaries, source and output hashes, dates, the licence and `model_assessment_performed=false`. The default protocol above is the frozen benchmark configuration. Alternate CLI history values create different protocols and must not be compared as if they were this fixed one.

Focused CPU tests check training-only normalization, disjoint chronological targets, target/future exclusion from model inputs, fixed-last shuffling, observed-only naive controls, training-only ridge coefficients, bounded repository writes, and the actual pinned archive's county alignment, dates and graph structure. Synthetic future mutations test leakage without fitting or assessing any model on actual validation/test counts.
