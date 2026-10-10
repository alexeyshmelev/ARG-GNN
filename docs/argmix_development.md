# Fresh-development whole-tree ARGMix equation adaptation

This dedicated runner preserves the existing `make_study_model("argmix_full_tree",48,4,...)` model and gives it the same eight-configuration, three-seed development budget as the other neural families. It is an **independent paper-equation common-input adaptation**, not an exact native ARGMix reproduction. The architectural and equation differences remain documented in `docs/argformer_control.md`, including the printed column-key attention term, integer-floor elapsed-TMRCA bins and full-tree query-node readout.

All raw observed marginal-tree nodes/topology and the verified F48 reference-adapter features enter the model. The original F8 prefix is preserved. There is no unary suppression, duplicate merging, canonicalization, external pretraining or cross-tree aggregation. The existing input linear/GELU projection and two relative-attention layers remain unchanged. No separate source GNN is added. Four attention heads, FFN multiplier four and elapsed-TMRCA cap1000 retain the current factory settings. Width32 has123,268 parameters and width64 has295,684. Dropout does not change capacity. Full-tree scope differs from both the native nearest-reference scope and the candidate's persistent-leaf F47 source, so equal hidden width does not imply equal information or total capacity.

## Search plan and execution boundary

`configs/argmix_development_search_plan_v1.json` is a **non-executable search plan**. It fixes:

| Choice | Values |
| --- | --- |
| Fresh parent splits | 64TRAIN /16VAL /TEST0 |
| Input /classes /queries /windows | F48 /4 /64 /8 |
| Hidden width | 32,64 |
| Dropout | 0,0.1 |
| Learning rate | 0.0003,0.001 |
| Attention depth /heads | 2 /4 |
| Training seeds | 101,202,303 |
| Epoch cap /patience | 48 /8 |
| AdamW decay /gradient clip | 0.0001 /1 |
| Maximum starts | 8configurations ×3seeds =24 |

Each configuration/seed initializes independently, trains cross entropy on external targets with shuffled tiles, and selects the minimum equally weighted mean parent VAL NLL. NLL uses uncalibrated original logits in float64 stable log-softmax without a probability floor. Exact checkpoint ties keep the earliest epoch. The family helper requires all24 starts and exact config/dataset/protocol/completion/code bindings before averaging each configuration's three selected-checkpoint NLLs. A tie within0.0001nats prefers fewer parameters, then lexicographic frozen ID. A per-seed or partial-grid run records `family_selection_complete=False`.

No completed fresh preparation receipt was available during implementation. No executable training config was fabricated, no actual-data fitting/scoring or simulations ran, and no GPU job was launched. Root owns future binding, scheduling, integration, commits and pushes. Canonical and active shared source files were not edited. Native nearest-reference runtime optimization is outside this task.

## Root-facing completion binder

The API is:

```python
from arggnn.argmix_development_bindings import bind_completed_development
settings, exact_config_bytes, config_sha256 = bind_completed_development(
    completion_path, factual_completion_sha256,
    plan_path, reviewed_plan_sha256,
    output_path=None,  # verification only; a fresh path writes one exclusive config
)
```

The CLI is `scripts/bind_argmix_development.py --expected-completion-sha256 SHA --expected-plan-sha256 SHA [--write-config configs/NEW.json]`. Defaults point to the reviewed search plan and the preparation pipeline's future `experiments/results/fresh_development_preparation_v1/completion.json`. Both hashes must be factual nonzero SHA256 values. Without `--write-config`, it verifies and returns the deterministic prospective config hash without writing a config. With an output path, it writes exact bytes only after all checks pass and refuses overwrite. The tool neither generates data nor fits a model.

The binder checks the successful64/16/TEST0 completion and640tile facts, all four sealed stage manifests, the oracle→inferred→boundary→age source chain, preparation execution freeze, original preparation/oracle config bytes, and all80 parent simulation/inference identities against the frozen registry. The inferred protocol file must match its exact hash and embedded manifest protocol. Every source path is confined before opening it; traversal, external paths and held-out-test namespaces are forbidden. Only inferred raw/prepared/metadata artifacts become training inputs. Truth genealogies and oracle target bundles are never opened by this binder/runner.

It then calls the unchanged source loader to verify raw/prepared/metadata hashes, original F8 prefix, full-parent reference-adapter configuration and all observed sequence fields. Additional checks enforce F48, original64query and known24reference roles, query-source-label absence, external target shape/dtype/classes, paired target and retained-position digests, original128midpoint indices, and paired sample/haplotype query order. Observed tiling receives only `GraphSequence`; labels remain external. Every retained original scoring index must be assigned exactly once.

## Dedicated runner and integrity

The runner is `python -m arggnn.train_argmix_development --config configs/BOUND.json --expected-config-sha256 SHA --output experiments/results/NEW --seed 101 --device cpu`. Optional `--configurations ID ...` selects a distinct subset of the frozen grid. `--validate-config` verifies the bound controls and source artifacts and reports CPU model counts without fitting or creating an output. CPU mode requires `CUDA_VISIBLE_DEVICES=''`; later CUDA execution requires root's project launcher lease and one reserved allowed visible GPU.

A config cannot be accepted merely by supplying its current hash: its contents must exactly equal the settings derived from the reviewed plan and completed factual receipt. The plan binds all15 runner/binder/factory/loader/metric/resource source files, including the binder CLI. The runner also rejects source changes after its imports. It captures exact initial bytes for config, plan, completion, manifests, protocols and preparation controls. Output copies use those initial bytes rather than rereading changed paths after preprocessing.

Control/config/manifest/code hashes and raw/prepared/metadata stat identities are checked through preparation, at configuration boundaries and before/after each epoch, including before a newly selected checkpoint is saved. Full source SHA256 checks run before and after parent loading and each configuration, with stat rechecks after hashing. Changing or removing a frozen artifact aborts. The runner records initial bindings and immutable failure details rather than replacing failed parents, silently adding successful starts or changing the selected protocol.

Checkpoints and float64 original-order validation logits stay in a fresh ignored `checkpoints/argmix_development_v1` namespace. Prediction files retain parent IDs, positions, query-node IDs, sample/haplotype keys, config/source/completion/code hashes, configuration ID and seed. The config and receipts also carry the common seven completed preparation bindings consumed by the generic development selector. Model records use `models`, raw per-parent NLL/weighting, earliest-minimum `val_nll` histories and explicit initial/ending code/config locks. Source-qualified parent IDs come from the captured inferred-manifest SHA and verified split/integer replicate, rather than from a reserialized manifest digest. Checkpoints expose both their flat factual bindings and the nested reload identity.

Output JSON includes complete learning curves, per-parent raw calibration/class metrics, preparation/training time, environment, physical GPU assignment and peak CUDA allocation. No temperature is fitted, no held-out cohort is generated or opened, and no confirmation/superiority claim follows from a completed development search.

## Synthetic CPU verification

Final integration passed **79 tests in39.06 seconds**, including all47 dedicated ARGMix development/binder tests, the existing ARGMix equation-control suite and raw-logit metrics. Both runner and binder CLI help checks passed. The reviewed plan and all15 source bindings were checked again after testing, and the factual completion receipt was still absent. Tests use hand-built observed trees and synthetic completed preparation fixtures, not new simulations or actual development labels. They cover the fixed budget/factory counts, finite full-feature/parameter gradients, tiny CPU fitting/checkpoint reload, label-independent observed inputs/predictions, explicit unsorted parent-index scatter, raw unbounded NLL, unequal-parent-size equal weighting, complete-seed/spec/capacity selection, prefix tampering, safe paths/IDs, incomplete or forged preparation facts, exact config bytes, source/control mutation failures and full-source hash rejection with a simulated stale stat signature. Production code performs no runtime monkeypatch; pytest fixtures isolate synthetic roots only within tests.

Reviewed non-executable search-plan SHA256: `215ead89558d995da8045d9b40b6fd3da4029809274f59953daedae1219b67e9`. This is the plan hash for the completion binder, not a nonexistent fresh training-config hash.

## Current numerical execution proposal

The current raw binder defaults to the v2 source proposal, which adds the common strict deterministic execution policy. The historical v1 plan remains preserved. Both checkpoint-selected and reloaded NLL values are recorded with an explicit delta. See docs/development_execution.md for policy, source hashes and CPU validation. No fresh fitting or GPU admission has yet occurred.
