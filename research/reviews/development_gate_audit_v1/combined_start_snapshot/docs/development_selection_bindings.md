# Completed fresh-development selection-plan binder

This binder creates a factual combined search plan after successful fresh
64 TRAIN / 16 VAL / TEST0 preparation. It performs no training, data generation,
score inspection, winner selection, final-test access or confirmatory promotion.
The four new files are the binder module, its CLI, the synthetic tests and this
document. Shared helpers, active pilot/preparation sources and existing plans
were not edited by this task.

## Admission and immutable outputs

`bind_development_selection` requires explicit nonzero SHA256 values for the
completion receipt and the actual v2 scope amendment, plus exactly ten config
references with the exact fields `family_id`, `path`, `sha256`. It defaults to
validation only. It calls the mandatory common `verify_completion` with a full
source snapshot. That verifies the real preparation freeze, seed/parent registry,
recipes, oracle/inferred artifacts, observed boundary banks, age caches and
external supervision. Placeholder or incomplete preparation cannot produce a
plan. The default real completion remains absent.

The plan preserves the existing strict selection schema. It binds all seven
common completed-source/preparation hashes and the five actual completion,
inferred source, boundary, age and preparation-freeze references. Every family
must match its producer's exact input scope, cross-tree aggregation, complete
reviewed dependency map, raw float64 uncalibrated-logit NLL policy, fixed seeds
101/202/303, 48-epoch cap and patience 8. Each has the fixed eight-configuration
grid. There is one candidate and nine mandatory baselines. No missing closer
prior-art or all-record family can be silently removed.

The dependency union preserves original protocol keys, including bare `.py`
keys and full repository-relative keys. Bare keys resolve under `src/arggnn`.
Different keys resolving to one file must have the same hash. Captured config,
amendment, source and code bytes remain sealed through validation and emission.
Lexical and resolved held-out paths are rejected before reads. A fresh output
directory and exclusive writes prevent overwriting existing plans. Failed
writes remove this binder's partial files.

The sixteen validation-parent contracts are ordered by observed parent replicate
0 through 15. They contain only parent identity, verified observed-position and
query-ID fingerprints and the expected position/query/four-class logit shape.
The binder never reads a case's labels, baseline probabilities or validation
scores to derive these contracts.

Before final emission, captured plan bytes are written exclusively in a temporary
project-local directory and validated by the existing
`load_development_selection_plan`. Temporary bytes are removed afterward. Full
source hashes and file identities are checked again before and after final
writes. No output directory is created when admission or strict validation fails.

When explicitly enabled, the two emitted files are `plan.json` and
`binding_receipt.json`. The latter is a plan-binding receipt, not a successful
selection receipt. It records the amendment SHA, normalized proposal aliases,
strict numerical policy, producer policy-field mapping, source provenance and
maximum 216 main optimizer starts plus 72 reserved control starts, total 288.
The plan has 240 configuration/seed cells, including 24 deterministic smoother
cells and zero smoother optimizer starts. The receipt explicitly states that
no scores were inspected and no selection or confirmation occurred.

## Required actual families and metadata naming

| Actual config/run family ID | Plan role | Producer policy field |
| --- | --- | --- |
| candidate | candidate | deterministic_execution_policy |
| gru | same_source_gru | deterministic_execution_policy |
| transformer | same_source_transformer | deterministic_execution_policy |
| reference_smoother | observed_reference_smoother | deterministic_execution_policy |
| canonical_gru | canonical_tree_gru | deterministic_policy |
| argmix_full_tree | argmix_adaptation | deterministic_execution_policy |
| allrecord_gru | allrecord_boundary_gru | deterministic_execution_policy |
| allrecord_transformer | allrecord_boundary_transformer | deterministic_execution_policy |
| gst_ca_span | gst_ca_span_adaptation | deterministic_policy |
| graphssm_zoh | graphssm_zoh_adaptation | deterministic_policy |

Both documented numerical-policy spellings must carry the same exact typed
`DETERMINISTIC_POLICY` value. The binder requires the field used by each frozen
producer and rejects missing, ambiguous or wrong-valued fields. It preserves
actual config bytes and hashes instead of relabeling their fields.

Scope amendment proposal IDs `same_anchor_gru`, `same_anchor_transformer`,
`allrecord_boundary_gru` and `allrecord_boundary_transformer` map to `gru`,
`transformer`, `allrecord_gru` and `allrecord_transformer`. Input config-reference
labels may use those documented proposal aliases, but config/run family IDs
remain the actual IDs above. The binding receipt records the mapping.

## API and CLI

The API is:

```python
bind_development_selection(
    completion_path, expected_completion_sha256,
    scope_amendment_path="research/protocols/development_scope_amendment_v2.json",
    expected_scope_amendment_sha256=reviewed_amendment_sha,
    family_configs=ten_factual_refs,
    output_directory=None,
)
```

The return value is a `DevelopmentSelectionBinding` with `plan`, original plan
`content`, `sha256`, `receipt`, `receipt_content`, `receipt_sha256`, `path` and
`receipt_path`. Paths are `None` in validation-only mode. `project_root` defaults
to the real repository and exists for small explicit synthetic test fixtures.
It does not bypass mandatory preparation or family-contract admission.

On anogena in `/workspace-SR003.nfs2/ARG-GNN`, source
`scripts/project_env.sh`. The CLI accepts `--completion`,
`--expected-completion-sha256`, `--scope-amendment`,
`--expected-scope-amendment-sha256`, and ten repeated
`--family-config FAMILY PATH SHA256` triples. Each path/hash must come from the
actual completed family binders and reviewed bytes. It validates only unless
`--write-plan` is supplied. The default output is the fresh directory
`configs/development_selection_bound_v2`. `--output-directory` can choose a
different fresh project directory with an existing parent.

No placeholder plan or receipt is supplied here. Root must first finish genuine
preparation and generate/review all ten factual family configs. Only then should
root pass their actual hashes and optionally enable `--write-plan`.

## Verification and limits

The final CPU suite passed 97 tests in 54.05s, comprising 64 new binder cases and 33
existing selection cases. The new fixtures use small files and explicitly mock
cohort admission and producer review contracts. Their fake cases forbid label
and baseline-probability access. Production has no admission bypass switch.
Adversarial cases cover absent/wrong/incomplete completion, missing families or
configs, wrong hashes, source/dependency alias conflicts, scope/grid/budget drift,
missing or weakened policies, proposal alias normalization, held-out paths before
reads, malformed observed parent contracts, same-size source edits restoring
mtime, and partial-write cleanup. Original source/config bytes are preserved.

A separate read-only live check confirmed all ten real producer contracts and
coherent source union: 53 keys resolve to 42 code artifacts. Root's frozen common
boundary helper, raw helper and deterministic-policy hashes remained unchanged.
CLI help passed. The real default CLI refused absent completion and created no
output. No genuine completed-cache admission, factual real plan, model fitting,
selection receipt, final-data access, GPU launch, native inference, simulation,
package or Git action occurred in this task.

```json
{
  "kind": "development_selection_binding_source_verification",
  "tests_passed": 97,
  "test_seconds": 54.05,
  "new_binding_test_cases": 64,
  "test_command": "CUDA_VISIBLE_DEVICES='' \"$ARGGNN_PYTHON\" -m pytest -q tests/test_development_selection_bindings.py tests/test_development_selection.py --basetemp .runtime/tmp/pytest_development_selection_binding_final_v3",
  "live_family_contracts": 10,
  "live_dependency_pin_keys": 53,
  "live_resolved_code_artifacts": 42,
  "source_files": {
    "src/arggnn/development_selection_bindings.py": "88a4e8861504545201ca41338374a58e5ace53c5708812274e4c11fb58d92af8",
    "scripts/bind_development_selection.py": "64ef56f7995f6a33220606f53f8f04c716a7858207e4dd1c9f836b4854172400",
    "tests/test_development_selection_bindings.py": "5cbec7ac29f12ab228d59833e24ff27ab34e86f1b39fedc9130007df4f35c4a1"
  },
  "frozen_shared_inputs": {
    "src/arggnn/development_training_bindings.py": "4ecabb42086d9ef34d07f791fa5c5765b8b22b5276822501f0f92a4f76a2f0f0",
    "src/arggnn/argmix_development_bindings.py": "dabfddec8d2b47985d34928f097fbbe05cb2975b58a1be6d35e0b048713ef0e6",
    "src/arggnn/development_execution.py": "4a2f87320615eb655a7700b6bbff4a7d293b1b16749468afa46f0de2d12e276f",
    "research/protocols/development_scope_amendment_v2.json": "89ccb8a0f58d2df4d6cd530568b8a0be48c8eddf2dc2f13d17888899ba2f96f1"
  },
  "genuine_completion_present": false,
  "real_plan_emitted": false,
  "selection_performed": false,
  "model_fitting_or_generation_performed": false
}
```
