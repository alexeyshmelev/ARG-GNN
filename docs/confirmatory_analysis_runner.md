# Joined confirmatory ID analysis runner

## Status and bounded coverage

This implementation covers the frozen ID primary comparison, the prescribed conditional secondary Holm family, and descriptive primary-grid and dense-tract diagnostics. It is a bounded first analysis phase. It does not implement the separate robustness, demographic OOD, query-count/draw inferential family, finite-source/tile sensitivity or prespecified illustrations. The schema, lock and report retain that remaining coverage and record `full_protocol_readiness=false`.

No actual published confirmatory lock, real cohort, target extraction, model prediction, joined final-data analysis, GPU use or fitting occurred in this work. Tests use handmade NPZ/census artifacts and explicitly mocked frozen-lock/cohort admission. Positive terminal fixtures also explicitly mock CUDA execution metadata; a negative fixture uses actual CPU initialization without a model forward pass. These fixtures do not provide evidence of genuine CUDA execution. An actual final lock must bind the workflow implementation, full artifact schema and positive VAL-only temperatures for every selected method and seed before preparation.

The user's stronger competitor requirement includes genuine ARGMix, ARGformer and official AliPearson/AncestralPaths (commit 77db3...). The current ten-family development proposal does not cover genuine ARGformer or AncestralPaths. That scientific-readiness limitation remains open; no actual final lock or test preparation is authorized by this implementation.

The secondary-control files were separately repaired and frozen before this work. This runner reuses unchanged `confirmatory_metrics.py` and `dense_tract_metrics.py`; it does not alter models, preparation, selection or those metrics.

## Public API and execution boundary

`src/arggnn/confirmatory_analysis_runner.py` provides:

- `analysis_schema_values()`: metadata-only prospective schema and current joined-analysis/CLI/metric source pins. Imported source drift rejects this helper. It returns no lock, data or scientific result.
- `admit_analysis(lock_path, expected_lock_sha256, roster_path, expected_roster_sha256, ledger_path, expected_ledger_sha256, *, project_root=ROOT)`: calls the public frozen-lock validator, validates sealed metadata, complete method/seed/parent roster and observed identities/source hashes. It opens neither predictions nor target arrays.
- `analyze_confirmatory(admission, *, output_path=None)`: requires the admitted immutable metadata, refuses any unresolved cell, preflights all successful predictions, then joins external targets and computes the bounded ID report. Optional emission is exclusive.
- `validate_prediction_terminal(capture, terminal_ref, *, frozen_lock, method_id, seed, parent_id, observed_identity, observed_identity_ref, admission_ledger_ref, expected_status=None, expected_prediction=None)`: shared metadata-only terminal checker for joined admission and custodian roster sealing. The caller must already validate the frozen lock. It returns normalized terminal/prediction refs, terminal status and exact receipt values; it never opens target arrays or queries a device.
- `main(argv=None)`: CLI entry point.

The lock validator is `confirmatory_lock.validate_confirmatory_lock(lock_path, expected_sha256, *, project_root=ROOT)`. Its result supplies payload, original content/hash/path and `check(full=True)`. It verifies the real published pre-test lock and captured source/artifact/version provenance. The analysis runner additionally requires the ID-only lock scope, exact current artifact schema, selected seed bundles and actual NumPy/SciPy/Torch analysis versions.

Admission checks retain canonical in-memory lock/schema/registry/temperature/ledger/roster/identity state, parent ordering and the complete cell mapping. Changes to those objects, imported code, or captured file stats reject subsequent work. Inputs are hash-verified at each read. Observed source bytes are streamed through SHA256 before target access. Paths must remain inside the specified project root, with no traversal or escaped symlink resolution. Duplicate JSON keys, nonfinite JSON and object-bearing unsafe NPZ arrays reject loading.

The full frozen prediction roster must cover candidate, the unchanged strongest required baseline and all required locked controls, crossed with seeds 101/202/303 and every admitted ID parent. It cannot replace a checkpoint, select a different seed, choose a test-best baseline or omit a parent.

Every success or failed roster row must bind its exact terminal receipt reference; unresolved rows require a null receipt. Terminal refs cannot alias predictions or targets. Admission validates receipt evidence before any prediction or target load, retains canonical receipt state and pins the predictor module/CLI in the schema's separate `terminal_receipt` section. The `code_artifacts` section remains exactly the four joined-analysis/metric references required by the lock API. The report preserves all normalized `prediction_terminal_receipts`.

An initial complete roster of `unresolved` entries is valid tracking metadata, but joined analysis rejects it before opening any prediction or target artifact. Missing roster entries reject admission. A complete terminal roster contains only success or failed entries. Failed cells retain reasons, failure rates and null seed losses. They block positive confirmation. Successful cells are all CPU-loaded, hash-checked and validated for finite logits, exact source/method/checkpoint/grid/query identity and array shapes before any external label or census artifact opens.

## Sealed admission and observed release

All artifacts use exact `{path,sha256}` references. The admission ledger binds the published lock, frozen parent-seed registry, ID cohort namespace and fixed target/attempt limits. Attempts must be ordered contiguous registry IDs beginning at 0, admitted or rejected by label-blind decisions with preserved reason codes.

The ledger stops at the first 160 admitted parents. If 160 are not reached, all 192 attempts must be recorded. It then analyzes all admitted parents once and labels the shortfall. No score, class proportion, failing method or storage pressure can change the admitted roster. Rejected attempts have no observed/target join refs. Admitted attempts separately reference observed identity, external primary labels and census tract partitions.

Observed identity binds the parent namespace, ID and exact ancestry/mutation/inference registry seeds. It records 64 distinct integer original query IDs, 32 nonoverlapping diploid index pairs covering all 64 queries, both physical midpoint-grid identities and the exact seven physical tile edges `[125000,250000,375000,500000,625000,750000,875000]` bp. Different edge coordinates reject admission.

Both grids use `DenseGrid.identity_record()`, including its SHA256. The domain is [0,1000000] physical bp. Original 128/4096 global midpoint indices must remain in their original order and share observed coverage. Primary admission requires at least 120 of the original 128 points and at least 0.95 physical coverage. No missing interior midpoint is bridged.

Observed identity has exactly one observed-only NPZ source artifact. Analysis verifies its byte hash. The separately frozen observed prediction workflow validates its raw-eight-feature topology, sample/reference roles, physical partition and reference adapter. Analysis does not treat a metadata assertion as an independent reconstruction of observed geometry.

The observed NPZ arrays and header are frozen in the schema block below. It contains raw observed graph features and geometry only. It stores no positions-as-sources, primary/dense labels, census nodes, tract targets, probabilities or target-derived features. Intervals and window bounds are normalized global bp/1e6; source readouts are declared separately through the physical grids.

## Safe prediction and target arrays

One NPZ per method/seed/parent contains both primary and dense raw-logit arrays, computed by separate frozen forward passes. NPZ loading always uses `allow_pickle=False`. Its exact keys are metadata_json, primary_logits, dense_logits, primary_positions_bp, dense_positions_bp and query_ids. metadata_json is one scalar Unicode/byte string containing JSON, never an object array.

Prediction header fields are exactly:

```text
schema_version=1
kind=sealed_confirmatory_raw_prediction
lock_sha256, namespace, parent_id
method_id, training_seed, configuration_id, family_id
protocol={path,sha256}, checkpoint={path,sha256}|null
observed_identity_sha256, primary_grid_sha256, dense_grid_sha256
class_order=[ANA,CHG,WHG,EHG]
no_gradients_or_optimization=true
```

Method IDs are candidate, strongest_required_baseline and the exact selected control IDs. A deterministic reference smoother may retain its actual null checkpoint; no checkpoint is invented. The protocol/checkpoint fields must equal the corresponding frozen method/seed bundle.

Raw logits are finite float32/float64 arrays shaped [retained original midpoint,64 queries,4 classes]. Positions are float64 physical bp and query IDs are integers in the exact observed order. Primary and dense arrays must match the frozen grid identities. Probabilities, capped scores, predictions on a different coordinate set or reordered queries cannot substitute.

External primary label NPZ keys are metadata_json, labels, positions_bp and query_ids. Labels are integer class slots [retained primary midpoint,64 queries]. Header fields are exactly schema_version=1, kind=sealed_external_primary_labels, lock_sha256, namespace, parent_id, observed_identity_sha256, primary_grid_sha256 and class_order.

External census JSON fields are exactly schema_version=1, kind=sealed_external_census_tract_partitions, lock_sha256, namespace, parent_id, observed_identity_sha256, class_order, query_ids, domain_bp and tracts. Each query has its own exact full-domain maximal partition of [left_bp,right_bp,class_slot] triples in the original query order. It is admitted through the unchanged `TractPartition` helper.

Sparse primary labels must independently agree with these census partitions at every exact retained primary midpoint. Labels and census targets are only outcome joins. They never enter a model, observed source manifest or prediction forward pass.

## Qualified terminal execution evidence

A successful receipt must be terminal, complete and have no reason, assert separate primary/dense forward passes, no gradients or optimization, no target artifacts opened, `cpu_validation_only=false` and `cuda_execution_performed=true`. CPU validation outputs reject before predictions or targets can join.

Its frozen method/seed/parent header, protocol/checkpoint, both grids, classes, observed identity, observed NPZ and admission-ledger references must match the lock and observed release exactly. Software fields bind Python, executable, Torch, Torch CUDA and build details. Completed CUDA metadata requires hardware name/compute capability, positive total memory, logical device 0, typed physical GPU 2–5, the exact process title, typed deterministic settings and nonnegative actual allocator peaks. The canonical GPU UUID must match the visible and launcher UUIDs; the archived lease must have the corresponding project `.runtime/locks/gpu_N.lock` path, typed descriptor, GPU and job identity.

Every declared predictor-source reference is byte-hash verified, with the predictor module/CLI and all frozen `src/arggnn/*` source-binding refs mandatory. Extra valid import-context source pins are accepted; the checker does not require equality to the current process's dynamically imported pin map. Captured provenance must include the exact ledger, identity and observed NPZ. Failed terminal receipts may preserve partial runtime, but require exact cell/lock/ledger/source binding, a nonempty reason/error type and no scored prediction. A failed row's reason must equal its receipt reason and always blocks positive confirmation.

This is a metadata/provenance admission gate. OS custody and genuine GPU execution must be independently established by the launch and custody workflow. Synthetic mocked metadata proves checker behavior only.

## Primary and conditional inference

Primary NLL is recomputed from each individual frozen model's original raw logits using float64 stable log-softmax without a probability floor. Historical 1e-12-floor NLL is retained only as a compatibility diagnostic. No probabilities or logits are averaged into an ensemble.

Each parent contributes the average of its three candidate-minus-baseline losses. The primary estimate equally weights independent parents. The report preserves method/parent/seed NLL vectors, every paired parent/seed difference, the three seed-specific means and their range/SD. Coordinates, queries, tiles and parent-seed cells are never independent sampling units.

The unchanged pure helper computes the paired-parent two-sided 95% t interval/test, N-1 degrees of freedom and a 10000-resample percentile parent-bootstrap sensitivity interval with fixed analysis seed 20261115. The same seed and ordered parents give the same sampled parent-index vectors for complete comparisons. A complete-case subset caused by terminal failure is explicitly descriptive, with its omitted parents listed.

Threshold interpretation preserves interval favorability, the -0.010-nat practical threshold, the 0.005-nat intended t-interval halfwidth, sign disagreement between t/bootstrap and the absence of an equivalence claim. A robust positive confirmation requires the complete successful frozen roster, no admitted-parent target/precision shortfall, and favorable t and bootstrap upper bounds. Failure or unresolved predictions cannot produce that decision. Interval-derived descriptive flags are not a replacement for this eligibility rule.

Controls are derived from the frozen lock: gain-matched diagonal and static local always, plus matched scalar-age full for selected age geometry or noncontextual for selected plain depth 2. Plain depth 1 has no third control. The secondary family preserves its full fixed size. Incomplete contrasts contribute conservative p=1 to Holm adjustment and cannot be eligible positive tests. With any terminal prediction failure, conditional results remain descriptive. Estimates and adjusted values remain visible regardless of the primary outcome. Secondary outcomes never reselect a primary method.

## Descriptive diagnostics and frozen temperatures

The actual pre-test lock must capture `temperature_manifest` with positive temperatures for every selected method/seed and exact selected checkpoint/development-prediction refs. Its fit evidence binds all 16 VAL parents, completed source provenance, fit code/records, unchanged model weights and no final-test access. Public lock validation checks that evidence. Analysis independently checks positive values, the complete method/seed matrix and VAL-only fit scope. It performs no calibration fitting or test-based temperature selection.

For each successful parent/seed/method, primary diagnostics retain raw and VAL-temperature-divided-logit NLL, historical floor NLL, accuracy, Brier, fixed-zero all-four-class macro F1, class support/precision/recall/F1, confusion, 15 fixed equal-width top-label bins/ECE, classwise reliability and explicit diploid dosage MAE. Missing class denominators remain null with zero support.

The unchanged dense helper reports approximate physical span quadrature, reconstructed argmax tracts, one-to-one boundary matching at 1000 bp, switch-count error, censored/visible tract distributions, class-specific geometry/length errors, full-observed short true tracts up to 10000 bp, and diagnostics within 10000 bp of artificial tile edges versus the interior. No true boundary gives null recall/F1 with preserved zero support and any false predictions. Dense reports are descriptive within one parent/checkpoint and do not create extra independent parents.

## Output, CLI and operational limits

The result is `{values,bytes,sha256}`. Its JSON kind is `joined_confirmatory_id_analysis_report`. It records exact original roster/ledger hashes, the lock, all attempted/admitted IDs and reasons, failures and rates, all seed loss vectors, primary/conditional reports, raw/calibrated/dense diagnostics, source/artifact provenance and remaining coverage.

Default analysis emits no file. A supplied output path must have an existing in-project parent and must not already exist. Bytes are staged in an owned temporary file whose original descriptor remains open through publication and cleanup. After flush/fsync, no-follow descriptor reads validate regular-file identity, stable stat fields, exact size and SHA before an exclusive hard link and again at the published path. Same-inode byte drift and foreign inode replacement reject. Cleanup removes only the owned pending/published inode and preserves existing or raced paths. No successful receipt or output is emitted for missing/unresolved roster entries, bad hashes, bad target joins or source/policy drift.

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES="" "$ARGGNN_PYTHON" scripts/analyze_confirmatory.py \
  --lock ACTUAL_PUBLISHED_LOCK --expected-lock-sha256 LOCK_SHA256 \
  --roster ACTUAL_TERMINAL_ROSTER --expected-roster-sha256 ROSTER_SHA256 \
  --admission-ledger ACTUAL_LEDGER --expected-admission-ledger-sha256 LEDGER_SHA256 \
  --output FRESH_OWNED_REPORT_PATH
```

Those names are explanatory placeholders. No actual final input bundle is supplied. The runner does no generation, training, prediction or GPU probing. Large future diagnostic reports require the project resource envelope and measurement of real output sizes before bulk execution. In-memory arrays are loaded one artifact at a time; prediction preflight discards logits, and joins then reload the same hashed artifacts. Complete per-parent/seed diagnostic records are retained rather than silently pooled or dropped.

Exclusive output and deterministic bytes support auditability. They do not enforce operating-system custody or authenticate another process's assertions. Root must control release and the one joined analysis after all terminal predictions seal. Formatting repairs may reuse identical hashes/predictions, without selection, retuning or cohort changes.

## Validation

Final CPU suite after the terminal-roster gate and observed-release schema freeze:

```text
tests/test_confirmatory_analysis_runner.py
tests/test_confirmatory_metrics.py tests/test_dense_tract_metrics.py
121 passed in 113.17s
```

This includes 61 joined-analysis cases and 60 unchanged pure-metric cases. The earlier 100-case pre-repair source/test snapshot and receipt remain preserved under `.runtime/tmp/analysis_pre_terminal_repair_v1/`.

Both module and CLI parsed successfully, and CLI help passed. Tests cover actual public lock-reader refusal of the handmade lock, mocked positive lock admission, metadata-only admission, complete roster/checkpoint/source/coordinate/query guards, source/in-memory/imported-code drift, mandatory pre-test temperatures, prediction object-array/shape/finite/header rejection before target reads, external-label/census consistency, terminal failures and null seed vectors, unresolved-roster target-access blocking, uncapped raw NLL, equal-parent three-seed aggregation, calibration/dense absent-class/boundary conventions and staged publication faults/races. Added cases cover explicit mocked positive CUDA metadata, actual CPU-runtime refusal, 17 terminal runtime/source/identity mutations, terminal alias and row-reason guards, exact tile-edge rejection, same-inode staged-byte drift, and preservation of foreign staged replacements.

Handmade source arrays are never passed through genuine observed topology/custodian/cohort admission. The synthetic prediction logits are mathematical arrays, not outputs of the actual frozen models. Positive-path tests replace the public frozen-lock admission and CUDA terminal metadata with explicit mocks; the production API is also exercised negatively to reject that fixture. These tests establish implementation behavior and do not establish a real freeze, inference release or scientific result.

## Prospective artifact schema

This block is a snapshot of `analysis_schema_values()`. Actual lock binding must capture exact schema bytes and all source pins before any real test preparation.

```json
{
  "admission_ledger": {
    "attempt_fields": [
      "parent_id",
      "status",
      "reason",
      "label_blind",
      "observed_identity",
      "primary_labels",
      "census_tracts"
    ],
    "fields": [
      "schema_version",
      "kind",
      "status",
      "lock_sha256",
      "parent_seed_registry_sha256",
      "cohort",
      "namespace",
      "target_admitted_parents",
      "maximum_attempted_parents",
      "attempts"
    ],
    "kind": "sealed_confirmatory_admission_ledger",
    "policy": "contiguous registry attempts, first160 admitted, all192 attempted before a shortfall",
    "status": "sealed_before_model_scoring"
  },
  "analysis_phase": "id_primary_secondary_and_descriptive",
  "census_json": {
    "fields": [
      "schema_version",
      "kind",
      "lock_sha256",
      "namespace",
      "parent_id",
      "observed_identity_sha256",
      "class_order",
      "query_ids",
      "domain_bp",
      "tracts"
    ],
    "format": "external exact full-domain TractPartition triples in original query order"
  },
  "code_artifacts": {
    "analysis_cli": {
      "path": "scripts/analyze_confirmatory.py",
      "sha256": "5554bd2e91f6b5e1902c15acac8b19309315e5a1cdff5d3936691a1add608e57"
    },
    "dense_tract_metrics": {
      "path": "src/arggnn/dense_tract_metrics.py",
      "sha256": "f878a13615b4e276286eb28c3df7a26ef8715a3c5b0226d7c6c1e451557f2c7b"
    },
    "joined_analysis": {
      "path": "src/arggnn/confirmatory_analysis_runner.py",
      "sha256": "dfde4cf9616c0e5deea3340c38efac369191a83182411df5f68a5c5047e5dd11"
    },
    "metrics": {
      "path": "src/arggnn/confirmatory_metrics.py",
      "sha256": "4e4f906cd6a302ffc422302c6f10694594856faab02d0da7565c508baf4c0a80"
    }
  },
  "full_protocol_readiness": false,
  "kind": "prespecified_joined_analysis_artifact_schema",
  "observed_identity": {
    "artificial_tile_edges_bp": [
      125000.0,
      250000.0,
      375000.0,
      500000.0,
      625000.0,
      750000.0,
      875000.0
    ],
    "fields": [
      "schema_version",
      "kind",
      "lock_sha256",
      "namespace",
      "parent_id",
      "parent_seeds",
      "source_artifacts",
      "query_ids",
      "diploid_pairs",
      "primary_grid",
      "dense_grid",
      "artificial_tile_edges_bp",
      "truth_free_observed_inputs"
    ],
    "grids": "DenseGrid.identity_record, original128/4096 midpoint grids in physical bp, coverage clipped",
    "kind": "sealed_confirmatory_observed_parent_identity"
  },
  "observed_npz": {
    "arrays": [
      "metadata_json",
      "x",
      "edge_index",
      "node_graph",
      "node_ids",
      "clades",
      "intervals",
      "query_node_ids",
      "window_bounds"
    ],
    "coordinate_units": "intervals/window_bounds float64 global bp/1e6",
    "dtypes": "x/clades float32 or float64; index and ID arrays int64",
    "feature_names": [
      "reference_ANA",
      "reference_CHG",
      "reference_WHG",
      "reference_EHG",
      "is_query_leaf",
      "is_leaf",
      "log_time",
      "log_parent_branch"
    ],
    "header_fields": [
      "schema_version",
      "kind",
      "lock_sha256",
      "namespace",
      "parent_id",
      "feature_names",
      "time_scale",
      "truth_free_observed_inputs"
    ],
    "identity_sources": "exactly one observed-only NPZ; analysis verifies hashes, frozen prediction workflow validates topology/reference adapter",
    "kind": "sealed_confirmatory_observed_graph_sequence",
    "time_scale": 10000.0
  },
  "policy": {
    "alpha": 0.05,
    "analysis_seed": 20261115,
    "bootstrap_resamples": 10000,
    "calibration_bins": 15,
    "mandatory_pre_test_val_temperatures": true,
    "positive_confirmation_requires_complete_predictions_and_target_precision": true,
    "practical_improvement_nats": 0.01,
    "target_join_requires_terminal_roster": true,
    "target_reads_after_complete_sealed_admission": true,
    "target_t_halfwidth_nats": 0.005,
    "training_seeds": [
      101,
      202,
      303
    ],
    "uncapped_float64_raw_primary_nll": true
  },
  "prediction_npz": {
    "arrays": [
      "metadata_json",
      "primary_logits",
      "dense_logits",
      "primary_positions_bp",
      "dense_positions_bp",
      "query_ids"
    ],
    "format": "np.load allow_pickle=False, finite float32/float64 original raw logits, no probabilities"
  },
  "primary_labels_npz": {
    "arrays": [
      "metadata_json",
      "labels",
      "positions_bp",
      "query_ids"
    ],
    "format": "external integer class-slot labels, exact ordered identity"
  },
  "remaining_coverage": [
    "robustness paired regimes and demographic OOD",
    "query-count draws and two-workflow Holm family",
    "finite-source and tile sensitivity",
    "prespecified illustrations"
  ],
  "roster": {
    "cell_statuses": [
      "success",
      "failed",
      "unresolved"
    ],
    "entry_fields": [
      "method_id",
      "seed",
      "parent_id",
      "status",
      "prediction",
      "checkpoint",
      "reason",
      "terminal_receipt"
    ],
    "fields": [
      "schema_version",
      "kind",
      "status",
      "lock_sha256",
      "parent_seed_registry_sha256",
      "admission_ledger_sha256",
      "cohort",
      "namespace",
      "entries"
    ],
    "kind": "sealed_confirmatory_prediction_roster",
    "matrix": "candidate,strongest_required_baseline,all locked controls x101,202,303 xall admitted ID parents",
    "status": "sealed_before_target_join",
    "terminal_cell_statuses": [
      "success",
      "failed"
    ]
  },
  "schema_version": 1,
  "status": "frozen_before_test_preparation",
  "terminal_receipt": {
    "failed": "actual terminal failure receipt, exact cell/ledger/lock/source, null scored prediction, preserved error; never positive confirmation",
    "kind": "confirmatory_prediction_cell_receipt",
    "runtime": "complete CUDA initialization; exact frozen software/hardware/strictpolicy; allowed physicalGPU/title; canonical runtime=visible=launcherUUID",
    "source_artifacts": {
      "prediction": {
        "path": "src/arggnn/confirmatory_prediction_runner.py",
        "sha256": "a63bd3dc24954dc9f35e8b30401d07f82ec437b64dbec32ce9449232e6f2dfce"
      },
      "prediction_cli": {
        "path": "scripts/predict_confirmatory.py",
        "sha256": "4f9bb482dcbfba76dd5a65bb138788b64b99a454eb99240233fc6810987d2a79"
      }
    },
    "success_requires": {
      "cpu_validation_only": false,
      "cuda_execution_performed": true,
      "no_gradients_or_optimization": true,
      "target_artifacts_opened": false,
      "terminal": true
    },
    "terminal_statuses": [
      "success",
      "failed"
    ],
    "unresolved_receipt": null
  }
}
```
