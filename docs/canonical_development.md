# Completion-bound fresh canonical GNN+GRU development

This dedicated implementation completes the pilot comparator's fresh-development execution path. It reuses the unchanged canonical model/preprocessor and the shared ARGMix completion/source-admission APIs. It does not modify the pilot runner/config, canonical model, canonicalizer, active preparation or shared core sources. The search plan is reviewed and source-pinned but **non-executable** until root supplies an actual successful64TRAIN/16VAL/TEST0 preparation receipt and binds its factual hashes. No actual cohort fitting, scoring, generation or GPU job occurred during implementation.

## Model and fair development budget

The input is the full observed parent with its verified original F8 prefix. `prepare_canonical_parent` suppresses anonymous unary vertices, merges exactly identical adjacent records, recomputes retained parent branches and freshly applies the reference adapter to the canonical parent, yielding F48. Fixed physical tiling happens afterward. Every retained canonical record inside each scored tile provides a query embedding to the chronological two-layer bidirectional GRU, including records hit by no readout. The source encoder is the existing two-layer full-node mean MPNN. Original readout positions and query order are preserved through explicit parent-index scatter.

| Choice | Frozen values |
| --- | --- |
| Parent splits | 64TRAIN /16VAL /TEST0 |
| Width | 32,64 |
| Dropout | 0,0.1 |
| Learning rate | 0.0003,0.001 |
| Source /GRU depth | 2 /2 |
| Epoch cap /patience | 48 /8 |
| Training seeds | 101,202,303 |
| AdamW decay /gradient clip | 0.0001 /1 |
| Maximum starts | 8configurations ×3seeds =24 |
| Parameters at width32 /64 | 37,988 /145,604 |

Each configuration/seed initializes independently and trains cross entropy on external targets with shuffled training tiles. Minimum equal-parent mean uncalibrated raw-logit VAL NLL selects the checkpoint. Metrics use float64 stable log-softmax without a probability floor. Strict improvement keeps the earliest epoch on exact ties. Family choice requires all24 starts and averages all three selected-checkpoint seed scores. Within0.0001nats, fewer parameters then lexicographic frozen ID breaks ties. Partial grids remain ineligible, with no successful-seed averaging or replacement starts.

This is ordinary preprocessing/GNN/GRU, with no novelty or predictive claim. Full canonical-node F48 and all-record recurrence differ from the candidate's persistent-leaf F47 source and from fixed-anchor temporal comparators. Depth2 does not imply that every distant internal raw feature has a direct message-passing path to a query; the fresh reference summaries also supply graph-wide information. Tiling limits temporal exchange to each scored physical tile. The comparator does not isolate a strict boundary mechanism or establish independent confirmation.

## Binder and actual completion admission

The root-facing API is:

```python
from arggnn.canonical_development_bindings import bind_completed_canonical_development
settings, original_config_bytes, config_sha256 = bind_completed_canonical_development(
    completion_path, factual_completion_sha256,
    plan_path, reviewed_plan_sha256,
    output_path=None,  # verify only; a fresh path writes one exclusive bound config
)
```

The CLI is `scripts/bind_canonical_development.py --expected-completion-sha256 SHA --expected-plan-sha256 SHA [--write-config configs/NEW.json]`. Defaults point to the future completed preparation receipt and `configs/canonical_development_search_plan_v1.json`. Hashes must be factual nonzero SHA256 values. Omission of `--write-config` performs verification only. An output is written exclusively after admission and canonical preprocessing succeed, with no overwrite or placeholder hashes. The API does not simulate, infer trees or fit a model.

`CanonicalIntegritySnapshot` composes the existing shared `IntegritySnapshot`. The shared verifier remains the single catalog for raw/prepared/metadata artifacts and control bytes. Its existing dependency catalog is extended with five canonical-specific source files, while deduplicating the shared execution-policy helper, rather than copying another live-source file list. The final combined catalog has21 source files. It verifies actual completed64/16/TEST0 and640tile facts, all four sealed stage manifests, the oracle→inferred→boundary→age source chain, the actual preparation execution freeze, original preparation/oracle config bytes, inferred protocol bytes, all80 parent simulation/inference identities, and inferred raw/prepared/metadata hashes.

Held-out, external and traversal paths are rejected before opening control or input bytes. The canonical wrapper additionally checks held-out names before resolving aliases, so a held-out-named symlink to an otherwise permitted file is refused. Only path fields actually consumed by admission are preflighted; informational citations and provenance references are not treated as training inputs. No oracle target bundle or truth genealogy is opened.

After completion/source admission, the unchanged source loader verifies F8 prefix, adapter configuration and observed sequence-field identity. Shared parent-contract checks verify F48, all64queries/24known references, absence of query ancestry in input/reference fields, external four-class labels, paired target/position digests, retained original128midpoint indices and sample/haplotype query order. The canonical preprocessor receives only observed `GraphSequence` and fixed observed provenance. Labels join afterward. Every retained original readout appears exactly once, and validation parent IDs are ordered source-manifest-SHA:val:0 through:15.

## Runner, immutable evidence and selection

Use `python -m arggnn.train_canonical_development --config configs/BOUND.json --expected-config-sha256 SHA --output experiments/results/NEW --seed 101 --device cpu`. A distinct subset of prescribed configuration IDs can be supplied with `--configurations ID ...`. `--validate-config` verifies actual completion/sources and canonical preprocessing before reporting CPU model counts, without fitting or creating outputs. CPU runs require `CUDA_VISIBLE_DEVICES=''`. Later CUDA use belongs to root through the project lease/launcher. Completion, freeze and source verification precede decoded input/model/CUDA admission. Canonical preparation additionally precedes model/CUDA admission.

The protocol also freezes the common `development_execution.DETERMINISTIC_POLICY`. After successful completion/source admission and canonical preprocessing, the runner calls `configure_deterministic_development()` before creating a model or calling device admission. It refuses an existing CUDA context, sets cuBLAS workspace`:4096:8`, enables strict deterministic algorithms, disables cuDNN benchmarking and TF32, enables cuDNN determinism and selects highest float32 matmul precision. The applied policy is recorded in run/checkpoint/prediction bindings and checked through fitting. This guarantees a declared policy within fixed runtime/hardware, not cross-platform bitwise equivalence. Unsupported operations become explicit failures; root should perform a reserved-device admission/replay smoke before24starts. No such GPU smoke was executed by this implementation.

The config must exactly equal settings derived from captured reviewed plan bytes, successful completion facts and current verified source catalog. A caller cannot supply an arbitrary config merely by providing its current hash. Control/code hashes and raw source stat identities are checked through preparation, configurations and before/after epochs. Full raw/prepared/metadata SHA256 checks run at major boundaries. Drift aborts before a newly selected checkpoint is saved, with explicit failure records for fitting failures.

Output copies use original captured config, manifest, inferred-protocol and completion bytes. Canonical observed input/correspondence caches, weights-only checkpoints and raw float64 validation logits live under a fresh ignored `checkpoints/canonical_development_v1` namespace. Cache serialization excludes targets. Prediction artifacts retain ordered parent IDs, original positions/query-node IDs and haplotype keys, seed/configuration ID, raw-NLL policy and factual protocol/completion/source/dependency bindings. Results preserve `original_selected_validation_nll`, `reloaded_validation_nll` and `checkpoint_replay_nll_delta`. Exact replay is required under the frozen policy, and mismatch failures preserve both scores/delta in their diagnostic evidence. Selected checkpoints expose flat generic-selector metadata and an additional nested binding used to verify reload identity. Outputs preserve complete histories, parent raw metrics/calibration, preparation/training time, environment, GPU assignment and peak CUDA allocation.

The protocol and run/model/checkpoint/prediction schemas are compatible with `development_selection`: the configurations-list is normalized only in memory, original byte/SHA identity is retained, all seven common preparation bindings are factual, model results use `models`, histories use integer `epoch` and `val_nll`, and each record has exact raw per-parent weighting and initial/ending locks. `input_scope` and `cross_tree_aggregation` explicitly report canonical F48/full-record recurrence. The actual config/run family ID is `canonical_gru`; root can include it under descriptive role `canonical_tree_gru` in the explicit required-baseline list.

The convenience API `select_family(records, config_bytes, expected_config_sha256)` also requires original captured bytes. It derives frozen settings from those bytes, validates the actual original SHA, all24 typed seed/ID cells, exact record specifications, actual model capacity and source/dependency bindings. It never computes an invented digest from a reserialized settings dictionary. The generic selector remains the root-facing synthesis path for full saved checkpoint/logit/source evidence.

## Synthetic CPU verification

Final integration passed **131 tests in66.82 seconds**, including all38 new canonical-development tests, the corrected pilot49test suite, canonicalizer and raw-logit metrics. Both CLI help checks passed. All21 reviewed source pins were checked again against root's frozen shared helper/policy versions after testing; corrected pilot/model/canonicalizer bytes remain unchanged. Tests use hand-built observed trees and explicitly synthetic completed-cohort fixtures. They cover fixed budget/capacity, shared catalog composition, full canonical preprocessing and original parent order, bad/incomplete completion before input/model/CUDA admission, F8/F48 and external-label counterfactuals, poisoned truth fields, explicit unsorted scatter, finite gradients, small fitting/checkpoint reload, unequal-parent equal weighting, original-byte/config/spec/seed selection guards, source/control/canonical-code drift, pre-read held-out aliases, execution-policy drift before model admission and preservation of original/reloaded NLL evidence on an injected replay mismatch. A real tiny canonical fit on synthetic inputs is also admitted by the public generic selector through saved weights/raw logits, while its missing23 starts remain explicitly ineligible. None of these checks is an ancestry performance or confirmation result.

Reviewed non-executable canonical search-plan SHA256: `9b20cbe541a0083ac510245e989b917246cd46321d74751abba0f4dc560e15b7`. This is a reviewed plan hash for the completion binder, not an invented fresh training-config or cohort hash.
