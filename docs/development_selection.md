# Fresh-development family selection

`arggnn.development_selection` synthesizes a hash-bound, reviewable receipt from a completed fresh development study. It does not train models, simulate data, load ancestry targets, open final-test inputs, or authorize a final-test freeze. No real selection plan, executable placeholder config, or selected-model receipt is included with this implementation. A future binder must create real plans from completed preparation and reviewed executable configs.

## API and execution boundary

```python
from arggnn.development_selection import (
    load_development_selection_plan,
    synthesize_development_selection,
    tensor_fingerprint,
)

plan = load_development_selection_plan(
    plan_path, expected_plan_sha256, project_root=project_root,
)
receipt = synthesize_development_selection(
    plan_path, expected_plan_sha256,
    run_receipts=run_references,
    failure_ledgers=failure_references,
    output_path=new_receipt_path,
    project_root=project_root,
)
```

The plan and every JSON/config artifact are read and hashed once per synthesis. The reader rejects duplicate JSON keys, conflicting hashes, changes during capture, and identity/stat changes before sealing. The selector also checks its own source bytes against the implementation loaded at import. All input/output paths must resolve inside the supplied project root; final-test and confirmatory namespaces are blocked before opening files. Existing receipt files are never replaced: the optional output uses exclusive `xb` creation and requires an existing parent directory. `receipt.sha256` hashes the exact serialized bytes. If `output_path` is omitted, the returned payload and digest are reviewable but no receipt file is written.

Selected checkpoints and predictions must be real hash-matched files. They are bounded to 512 MiB each and loaded on CPU from captured bytes with `torch.load(..., weights_only=True)`. Source/dependency artifacts are hashed as bytes and are never imported by this module. A prediction/checkpoint cannot serve multiple successful cells. Dependency pins may use safe basenames or safe relative repository paths; aliases naming the same source file capture that file only once and must agree on its digest.

## Frozen plan schema

The top-level object has exactly these fields:

| Field | Contract |
| --- | --- |
| `schema_version`, `kind` | `1`, `fresh_development_family_selection_plan` |
| `development_namespace` | `development-v1` |
| `training_seeds` | Ordered `[101, 202, 303]` |
| `splits` | `{"train":64,"val":16,"test":0}` |
| `nll_policy` | `float64 stable log-softmax of uncalibrated raw logits` |
| `tie_tolerance` | `0.0001` |
| `source_binding` | All seven common completed-source hashes below |
| `source_artifacts` | Five exact `{path, sha256}` references below |
| `validation_parents` | Sixteen ordered parent contracts |
| `families` | Named family specifications with exactly one `candidate` role |
| `required_baseline_families` | Explicit unique list containing every planned noncandidate family ID |
| `implementation_artifacts` | Exact union of family dependency-pin keys, each mapped to a real `{path, sha256}` source artifact |

All digests must be real nonzero lowercase SHA256 values. `source_binding` requires `completion_receipt_sha256`, `source_dataset_manifest_sha256`, `boundary_dataset_manifest_sha256`, `boundary_preparation_sha256`, `age_dataset_manifest_sha256`, `age_preparation_sha256`, and `preparation_execution_freeze_sha256`. `source_artifacts` requires `completion`, `source_manifest`, `boundary_manifest`, `age_manifest`, and `preparation_execution_freeze`.

The selector checks the completed 64 TRAIN / 16 VAL / TEST0 receipt, 640 inferred boundary/age tiles, completion stage-manifest hashes, complete cache status, inferred/boundary/age preparation chain, source parent identities, fresh ancestry master seed 20261101, and fresh inference master seed 20261102. It requires a development-only preparation execution freeze. It does not replay preparation or hash every dataset tensor: that admission responsibility remains with the completed preparation receipt and each runner's completed-source verifier.

Each ordered `validation_parents` item has exactly `parent_id`, `positions_sha256`, `query_ids_sha256`, and `logit_shape`. Parent IDs must be the sixteen distinct `source_dataset_manifest_sha256 + ":val:" + replicate` identities for integer replicates 0 through 15. `logit_shape` is positive `[P,Q,4]`. The coordinate/query digests use `tensor_fingerprint`, which hashes a stable dtype/shape header and contiguous CPU tensor bytes. This freezes both query IDs and their order.

Each family specification has exactly `role`, `grid_kind`, `protocol`, `expected_implementation_sha256`, `input_scope`, and `cross_tree_aggregation`. Family IDs and descriptive role IDs are safe ASCII identifiers. The scope/aggregation fields must be nonempty factual strings matching the captured protocol. They are reported verbatim. Their equality to a common source binding does not assert equal information access or native baseline reproduction.

The protocol must bind its family ID, namespace, splits, seeds, raw-NLL policy, all seven common source hashes, exact implementation pins, and the same 48-epoch / patience8 / weight decay0.0001 budget. Its `selection` must be one of the supported attestation strings:

- `minimum mean parent-validation NLL`
- `minimum mean parent-validation uncalibrated logit NLL`
- `minimum equal-parent mean validation float64 uncalibrated raw-logit NLL; earliest epoch on exact tie`

The raw-NLL field and validated history make the meaning explicit even for the first legacy spelling. The ordinary protocol schema uses `models` (eight distinct IDs), `model_specs` (exact ID-keyed dictionary), and `model_learning_rates`. The independent ARGMix adapter may instead use an exact eight-entry `configurations` list with `{id, hidden_dim, dropout, learning_rate}`. The selector normalizes that list only in memory; it never rewrites config bytes or assigns a replacement protocol hash. Supplying both schemas is rejected.

The currently supported exact grids are:

| `grid_kind` | Eight configurations |
| --- | --- |
| `candidate` | plain/age geometry × integer kernel depth1/2 × LR0.0003/0.001, always full coupling and log rate parameterization |
| `width_dropout_lr` | integer width32/64 × dropout0/0.1 × LR0.0003/0.001 |
| `bandwidth_pseudocount` | integer bandwidth10000/30000/100000/300000 bp × pseudocount0.001/0.01 |

The candidate role requires the candidate grid; the observed-reference-smoother role requires its bandwidth grid; other descriptive baseline roles use the width grid. Roles are not restricted to a fixed set of five baselines. Adding another approved family requires its real protocol, source pins and factual scope in the plan, plus an explicit entry in `required_baseline_families`. Config IDs are globally distinct across all families. Shared dependency-pin keys must have identical hashes; captured references to the same source path must also agree.

## Successful cell evidence

A `run_receipts` reference has exactly `{family_id, path, sha256}`. A run may contain all eight configurations or a strict subset of them, but a successful configuration×seed cell may occur only once. The run requires `family`, `development_namespace`, `evaluation_split="fresh_development_validation_only"`, `training_seeds`, `training_seed`, `nll_policy`, `protocol_sha256`, `completion_receipt_sha256`, `boundary_dataset_manifest_sha256`, `initial_implementation_sha256`, and a nonempty `models` dictionary.

Each model record attests a fixed nonnegative integer `parameters` count, `validation_nll`, sixteen ordered `per_parent` scores, `best_epoch`, complete `history`, checkpoint path/hash, prediction path/hash, `ending_config_sha256`, and `ending_implementation_sha256`. Every parent record contains its source-qualified ID, finite nonnegative `nll`, raw-logit NLL policy, and weighting `uniform fixed coordinates and query haplotypes within one parent`. Its mean must equal the reported validation score to 1e-12 tolerance. Neural histories contain consecutive integer epochs from zero through at most47, each with `val_nll`; `best_epoch` must be the earliest exact minimum. No averages over only successful seeds are accepted.

The selected neural checkpoint contains a nonempty finite tensor `model` state, integer `epoch`, `seed`, `validation_nll`, `protocol_sha256`, `completion_receipt_sha256`, boundary manifest hash, and exact `implementation_sha256`. Its metadata must match the selected cell. The predictor artifact contains `seed`, protocol/completion hashes, exact raw-NLL policy, ordered `parent_ids`, and sixteen CPU finite float64 `logits` tensors plus matching `positions` and long unique `query_node_ids`. Every shape and coordinate/query fingerprint must match the frozen parent contract. Targets/truth fields are rejected. The selector validates artifacts and score attestations; it does not recompute NLL from labels or reconstruct model parameter counts from state tensors.

Only the deterministic observed-reference smoother may have zero parameters, no checkpoint/epoch, empty history, and `smoother_fitting="none; seed-independent deterministic predictor"`. It still supplies three separate hash-bound prediction artifacts/seed receipts. Its parent scores and logit fingerprints must be identical across all three seeds.

Supported field aliases are `per_replicate` integer parent records, `validation_predictions` / `validation_predictions_sha256`, predictor `parent_replicates`, and checkpoint `boundary_dataset_manifest_sha256` in place of `boundary_manifest_sha256`. Parent aliases derive IDs only from the already captured source manifest hash. They never invent a missing seed, completion hash, config hash, or ending dependency attestation. Older receipts lacking those facts remain inadmissible.

## Failures, completeness and selection

A failure-ledger reference is `{path, sha256}`. Its object has exactly `schema_version=1`, `kind="development_failure_ledger"`, `plan_sha256`, and `entries`. Each entry has exactly `family_id`, `configuration_id`, `seed`, safe `attempt_id`, Boolean `terminal`, `failure_class`, nonempty `reason`, and `test_access=false`. Classes are `infrastructure`, `numerical`, `data_admission`, `model_unavailable`, or `other`. Only infrastructure failures may be nonterminal retry records. Duplicate attempts and terminal-failure/success conflicts reject the synthesis.

The receipt preserves every prescribed configuration×seed cell, including missing cells and ledgered failures, and all retry records. A missing cell makes its family unresolved. A terminal failure resolves its cell but disqualifies that configuration. A configuration is eligible only after all three fixed seed cells succeed with invariant parameter count. A family choice requires all24 prescribed cells to be resolved and at least one eligible configuration. An entirely failed required family cannot be silently omitted; it prevents a strongest-baseline choice.

For each family, configuration choice uses the mean of its three seed scores. There is one tie window within0.0001 of the global best in the current comparison, followed by parameter count and frozen lexical configuration ID. The strongest required baseline is available only when every explicitly required family is resolved and has an eligible configuration. It applies one global tie window across **all eligible configurations** in those families, not another window around already tie-adjusted family winners. This prevents chained tolerance drift.

The immutable receipt reports the complete matrix, counts, per-seed scores and status, eligibility, per-family choices and scope, required-family list, strongest required baseline when available, all artifact digests, frozen source bindings, and selection implementation hash. It explicitly states `final_test_access=false`, `promotes_final_test_phase=false`, and that developmental choices do not establish independent confirmation, novelty, mechanism, or predictive superiority. Any final-test design and claim scope remain separate decisions.

## Synthetic CPU verification

`tests/test_development_selection.py` constructs small real weights/logit files and fake completed-source contracts inside an isolated temporary project. It covers the exact matrix, missing seeds, terminal and retry ledgers, complete-but-ineligible families, generic extra families, factual scope reporting, list normalization, source-pin aliases, raw policy and earliest epoch, live capture drift, metadata/hash corruption, coordinate/query order, deterministic smoother identity, path/symlink confinement, and immutable output. It does not read the real study or require a GPU.
