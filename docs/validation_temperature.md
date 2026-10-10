# Selected-checkpoint VAL-only temperatures

## Status and scope

This producer implements the prespecified scalar calibration step after completed main and secondary development selection. It performs CPU optimization of one positive scalar temperature for each selected method and seed; it does not fit model weights or alter checkpoints. There has been no actual DEV cohort fitting, genuine temperature manifest, real test preparation, held-out generation/access or GPU execution in this work.

Synthetic tests use handmade complete ten-family evidence, actual toy secondary controls and numerical VAL-like arrays. They explicitly mock factual cohort/producer and selected primary-factory admission. The generic public main and secondary selectors replay that evidence, and synthetic emitted receipts pass the existing frozen lock-compatible validator. That establishes implementation behavior, not factual completion of the real development cohort.

The user's stronger competitor requirement includes genuine ARGMix, ARGformer and official AliPearson/AncestralPaths. The current ten-family development proposal does not cover genuine ARGformer or AncestralPaths. This unresolved scientific scope prevents a claim of full scientific readiness and does not authorize an actual final lock or test preparation. Model and preparation pins were not changed.

## Public APIs and factual admission

The implementation is in `src/arggnn/validation_temperature.py`; the CLI is `scripts/fit_validation_temperatures.py`.

- `admit_temperature_fitting(main_plan_ref, main_selection_ref, secondary_config_ref, secondary_selection_ref, scope_amendment_ref, *, project_root=ROOT)` returns an immutable-state-checked `TemperatureAdmission`.
- `fit_scalar_temperature(logits, labels, *, live_check=None)` is a pure CPU numerical utility. Production admission separately enforces all sixteen actual VAL parents.
- `fit_validation_temperatures(admission, *, output_directory)` emits per-cell optimization records, lock-compatible fit receipts and a complete manifest.
- `main_cli(argv=None)` defaults to validation only; actual scalar fitting requires `--fit` and a fresh output directory.

All supplied references have exactly `{path,sha256}`. Development path guards reject final-data namespaces before capture reads. Production admission requires the complete combined ten-family main plan and scope amendment, factual producer re-admission through `bind_development_selection`, canonical equality to the supplied plan, public replay of all 240 main cells, actual secondary configuration admission, complete public secondary replay, matching source bindings and selected candidate/baseline factory validation.

The selected method matrix contains candidate, the unchanged strongest required baseline, gain-matched diagonal and static local controls, plus the protocol-selected third control when required. Every method binds all three seeds 101/202/303 and exact selected checkpoint and development-prediction references. A genuine deterministic reference smoother retains its null checkpoint. No checkpoint or temperature is invented to complete a matrix.

The calibrator's numerical loading begins only after these factual gates. Shared admission validators necessarily inspect DEV artifacts internally to establish genuine evidence; admission is not described as artifact-blind. The numerical loader requires all sixteen ordered VAL parents, exact coordinate/query fingerprints and logit shapes, finite CPU float32/float64 four-class raw logits, and exact seed/protocol/parent identity. It loads predictions with `torch.load(...,map_location="cpu",weights_only=True)`.

External labels must be integer class slots with matching shapes. The recomputed equal-parent raw NLL must agree with the selected recorded mean within 1e-12 relative/absolute tolerance. Copies of numerical inputs are read-only; actual source label, coordinate and query tensors remain fingerprint-checked during optimization. Changed source stats, in-memory admission, policy or imported implementation reject work. Full factual/source checks occur at admission and before complete manifest publication; lighter stable-stat checks protect each objective evaluation.

## Frozen numerical policy

`FIT_POLICY` and its imported canonical bytes are pinned in source. There are no command-line policy overrides.

Each selected method/seed receives one scalar log-temperature shared across its sixteen VAL parents. Temperature is bounded to 0.05–20. The objective equally averages the sixteen parent NLLs, using uniform coordinates and query haplotypes within each parent. Parents with more coordinates do not receive extra weight.

NLL uses raw float64 logits divided by positive temperature, stable log-softmax and no probability clipping or floor. Nonfinite scaled logits or log probabilities reject. The bounded SciPy `minimize_scalar` solver operates in log-temperature with maximum 500 iterations and xatol 1e-12. The final choice compares the solver candidate, both endpoints and raw T=1; the exact minimum wins, with exact ties ordered by absolute log-temperature then log-temperature. Flat logits therefore retain T=1. Raw inputs and model weights remain unchanged.

Each optimization record preserves the frozen policy, every objective evaluation and sixteen-parent loss vector, raw/calibrated vectors and means, solver convergence/message/iterations/evaluations, all final candidates, bound selection, software versions, external label fingerprints, source/input/code provenance, selected refs and selection receipt hashes.

## Exact lock-compatible artifacts

The existing `confirmatory_lock.validate_temperature_manifest` is the compatibility authority. A complete manifest is only published after that validator accepts every emitted fit receipt.

The manifest fields are exactly:

```text
schema_version=1
kind=prespecified_validation_temperature_manifest
status=frozen_before_test_preparation
main_selection_receipt_sha256
secondary_selection_receipt_sha256
training_seeds=[101,202,303]
records
final_test_access=false
```

Each record has exactly:

```text
method_id, seed, temperature
checkpoint={path,sha256}|null
development_prediction={path,sha256}
validation_parent_ids  # all16, exact selected order
fit_scope=fresh_development_validation_only
fit_evidence={path,sha256}
```

Temperature is finite and positive. Fit evidence fields are exactly:

```text
schema_version=1
kind=validation_temperature_fit_receipt
status=complete
method_id, seed, temperature
checkpoint, development_prediction
main_selection_receipt_sha256, secondary_selection_receipt_sha256
source_binding  # all seven main source pins
validation_parents  # full sixteen coordinate/query/logit contracts
fit_scope=fresh_development_validation_only
unchanged_model_weights=true
final_test_access=false
fit_code_artifacts  # nonempty Python source refs
fit_record_artifacts  # nonempty optimization-record refs
```

The code refs include this module, its CLI and the unchanged pure confirmatory metrics source. Every completed cell emits `METHOD_sSEED_optimization.json` and `METHOD_sSEED_fit.json`. A complete selected matrix has twelve or fifteen scalar fits. Only then may `manifest.json` exist.

## Exclusive publication and failure evidence

The output must be a fresh development-only directory with an existing in-project parent. All selected numerical inputs are verified before it is created. Subsequent work checks the owned directory identity.

Each file is staged with the original descriptor kept open through checks, publication and cleanup. A no-follow read descriptor verifies a regular owned inode, stable stat fields, exact size and SHA before exclusive hard-link publication and again at the published path. Same-inode byte drift and foreign staged/published replacements reject. Cleanup removes only the owned inode, preserves foreign replacements and existing/raced destinations, and cannot mistake a reused inode while the original descriptor remains open.

A failure preserves earlier valid fit records and attempts to write `failure.json` with the active cell, error type/reason, selection hashes and manifest-emitted status. Optimizer failures preserve the evaluations collected by the optimizer. It emits no complete manifest when a cell fails. Checkpoint and raw-prediction hashes remain unchanged.

## CLI

These are explanatory placeholders, not an actual completed input bundle:

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES="" "$ARGGNN_PYTHON" scripts/fit_validation_temperatures.py \
  --main-plan ACTUAL_MAIN_PLAN --expected-main-plan-sha256 MAIN_PLAN_SHA \
  --main-selection ACTUAL_MAIN_SELECTION --expected-main-selection-sha256 MAIN_SELECTION_SHA \
  --secondary-config ACTUAL_SECONDARY_CONFIG --expected-secondary-config-sha256 SECONDARY_CONFIG_SHA \
  --secondary-selection ACTUAL_SECONDARY_SELECTION --expected-secondary-selection-sha256 SECONDARY_SELECTION_SHA \
  --scope-amendment ACTUAL_AMENDMENT --expected-scope-amendment-sha256 AMENDMENT_SHA
```

That command admits evidence only. Adding `--fit --output-directory FRESH_DEV_DIRECTORY` performs the authorized scalar fitting step; it never trains or accesses final data. A production launch remains owned by the integration workflow.

## Validation

The first synthetic CPU suite passed 25 tests in 148.18 seconds; its receipt is preserved at `.runtime/tmp/pytest_validation_temperature_v1/temperature_test_receipt.json`. After the publication repair, the full synthetic CPU suite passed 29 tests in 154.74 seconds. Seven focused publication regressions also passed in 3.92 seconds. Final source parsing and CLI help passed. The remote pytest exit status is saved independently of SSH transport at `.runtime/tmp/validation_temperature_final_v3/transport_status.json`; the test log and final receipt are in the same directory. The earlier failed v2 attempt (a local publication helper-name mismatch subsequently fixed) remains preserved.

Coverage includes admission ordering, missing/incomplete factual evidence, forbidden final paths before reads, equal-parent weighting, raw T=1 tie behavior, uncapped NLL, bad raw arrays, lock-compatible complete synthetic receipts, unchanged checkpoint/prediction bytes, source/policy/label drift, optimizer-failure trace retention, exclusive outputs, interrupted writes/source checks/raced destinations, and same/foreign inode drift before and after hard-link publication. CLI help and Python source parsing are also checked.

No synthetic receipt is presented as a genuine pre-test temperature manifest or evidence of scientific readiness.
