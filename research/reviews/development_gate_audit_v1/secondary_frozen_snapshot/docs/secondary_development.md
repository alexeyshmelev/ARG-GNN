# Conditional matched secondary development controls

## Status and scientific scope

The implementation is ready for independent gate replay. The frozen proposal is conditional and non-executable. This work produced no genuine main-study selection, executable secondary configuration, real-cohort admission, fresh secondary result, final-data access, or confirmatory promotion. The CPU runs described below are explicit synthetic tests.

These controls are attribution comparisons. They retain observed graph-local exchange. Parameter-schema equality does not establish equal effective capacity. Secondary outcomes cannot replace the selected main candidate or the strongest required main baseline.

All project operations use the existing environment in `/workspace-SR003.nfs2/ARG-GNN` on anogena. No package installation, GPU launch, simulation, native inference, budget extension, or Git operation is part of this implementation.

## Conditional control rule and fixed budget

Both `gain_matched_diagonal` and `static_local` are always required. The former uses the unchanged `AnchorKernelModel`; the latter uses the unchanged `StaticLocalAnchorModel`.

| Selected main candidate | Additional required control |
| --- | --- |
| Age geometry, depth 1 or 2 | `matched_scalar_age_full`, using unchanged `AnchorKernelModel` |
| Plain geometry, depth 2 | `noncontextual`, using unchanged `NoncontextualAnchorKernelModel` |
| Plain geometry, depth 1 | None |

Each required control has exactly two learning rates, 0.0003 and 0.001, with training seeds 101, 202 and 303. This requires 12 optimizer starts for a selected plain depth-1 candidate or 18 otherwise. These starts consume the existing 72-start secondary reserve within the existing 288-start total cap.

The frozen dimensions are F47 inputs, width 32, source GNN depth 2, latent dimension 8, three scales, 32 anchors and four classes. Initial coupling rates are 5, 20 and 80, with damping 1 and log-rate parameterization. Ordinary parameter weight decay is 0.0001. Rate and mixture parameter decay is zero. The ceiling is 48 epochs with patience 8. There are 64 training parents, 16 validation parents and no test parents.

Checkpoint selection uses the earliest exact minimum of equal-parent validation NLL computed from uncalibrated raw logits with float64 stable log-softmax. Each control chooses its learning rate only after both rate rows and all three seeds are resolved. An eligible rate requires three successful seed cells. The global best-plus-0.0001 set is resolved by parameter count and then frozen lexical configuration ID.

## Admission and exact provenance

`configs/secondary_development_search_plan_v1.json` has kind `conditional_secondary_development_search_proposal`. It pins 29 implementation files and the binder script. It is never a runner configuration.

Production binding first admits the actual preparation completion through `development_training_bindings.verify_completion(..., with_snapshot=True)`. This returns the mandatory observed-source snapshot, dataset and age interface. Binding then calls the public `development_selection_bindings.validate_producer_settings` API for every one of the ten main families, using that actual completion and source snapshot.

Main admission requires the complete prescribed ten-family study and all 240 configuration/seed cells resolved, with a selected candidate and selected strongest mandatory baseline. The binder reconstructs actual run and failure-ledger references from the main receipt provenance and reruns the public main selector. Apart from its creation time, the main receipt must equal that reconstructed selection. A winner-only or forged receipt is insufficient.

The binder captures exact completion, main plan, main receipt, protocols, reviewed implementation bytes, checkpoints, prediction artifacts, run receipts and terminal-failure evidence. It also compares all 16 plan validation-parent identities, coordinate/query fingerprints and raw-logit shapes against the verified observed validation parents. This contract is constructed from observed positions and query IDs, not labels.

The selected main candidate specification must contain exactly geometry, kernel depth, full coupling and log-rate parameterization. Its three selected checkpoint cells must succeed. Fixed source dimensions and optimization limits must match the prescribed values. The secondary configuration records these selected checkpoints, the exact main receipt and the original strongest primary baseline.

Admission reconstructs the complete expected secondary configuration and requires its canonical bytes and SHA256 to match. The admission object checks in-memory settings drift, dependency drift and source/control-byte drift before and during execution. Production source/config/case admission precedes deterministic numerical setup and device/CUDA queries.

## Public interfaces

All path/hash references have the exact form:

```json
{"path": "project/relative/or/absolute/path", "sha256": "64 hexadecimal characters"}
```

Public functions in `src/arggnn/secondary_control_bindings.py`:

- `proposal_values()`: reconstruct the current non-executable conditional proposal.
- `controls_for(selected_candidate_spec)`: derive the strict conditional control set.
- `bind_secondary_development(proposal_ref, plan_ref, receipt_ref, *, output_path=None, project_root=ROOT)`: validate by default; exclusively emit a new executable configuration only when an output path is supplied and genuine admission succeeds.
- `admit_secondary(config_path, expected_sha256, *, project_root=ROOT)`: admit only the exact reconstructed completion/selection-bound configuration.
- `synthesize_secondary_selection(config_path, expected_sha256, *, run_receipts=(), failure_ledgers=(), output_path=None, project_root=ROOT)`: independently resolve the two-rate, three-seed secondary matrix.

The last function returns `{"values": receipt, "bytes": canonical_bytes, "sha256": digest}`. This is a separate two-rate selector. It does not pass a shortened grid to the eight-configuration main selector.

`src/arggnn/train_secondary_development.py` exposes `make_control`, `validate`, `fit_configuration` and `main`. The production runner admits fixed budgets before calling the fitting utility. Explicit toy tests use smaller epochs only through that utility. The runner uses the unchanged candidate evaluator and scatter path, optimizer grouping and control implementations.

Each checkpoint preserves selected state, epoch, source/spec/main/config/code/policy bindings and raw-logit fingerprints. Reload must reproduce both NLL and selected raw logits exactly. Run records contain original and reloaded validation NLL, zero NLL replay delta and zero maximum raw-logit replay delta.

## CLI usage

These examples describe future authorized development execution. They were not used to run real fitting in this implementation.

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES="" "$ARGGNN_PYTHON" scripts/bind_secondary_development.py \
  --proposal configs/secondary_development_search_plan_v1.json \
  --expected-proposal-sha256 PROPOSAL_SHA256 \
  --main-plan MAIN_PLAN_PATH --expected-main-plan-sha256 MAIN_PLAN_SHA256 \
  --main-selection MAIN_RECEIPT_PATH \
  --expected-main-selection-sha256 MAIN_RECEIPT_SHA256
```

Validation writes nothing. Add `--write-config --output-config FRESH_PROJECT_CONFIG_PATH` to emit an admitted configuration. The output must have an existing parent and must not already exist. With no genuine completed main receipt, the default CLI refuses and writes no configuration.

After separate authorization for actual fitting, the runner interface is:

```bash
CUDA_VISIBLE_DEVICES="" "$ARGGNN_PYTHON" -m arggnn.train_secondary_development \
  --config BOUND_CONFIG_PATH --expected-config-sha256 BOUND_CONFIG_SHA256 \
  --output FRESH_PROJECT_RUN_DIRECTORY --seed 101 --device cpu
```

`--models` can restrict one process to distinct prescribed configurations. Seeds remain restricted to 101, 202 and 303. Partial successful evidence can be retained in a run receipt; missing cells prevent the corresponding control choice. No automatic retry occurs. GPU scheduling belongs to the primary agent and must follow the project launcher and eligible-device rules.

Independent synthesis uses:

```bash
CUDA_VISIBLE_DEVICES="" "$ARGGNN_PYTHON" scripts/bind_secondary_development.py \
  --synthesize --config BOUND_CONFIG_PATH \
  --expected-config-sha256 BOUND_CONFIG_SHA256 \
  --run-receipt ACTUAL_RUN_RECEIPT_PATH ACTUAL_RUN_RECEIPT_SHA256 \
  --failure-ledger ACTUAL_FAILURE_LEDGER_PATH ACTUAL_FAILURE_LEDGER_SHA256 \
  --output-selection FRESH_PROJECT_SELECTION_PATH
```

Repeat the run-receipt and failure-ledger arguments as needed. Omit `--output-selection` for read-only synthesis.

## Run, failure and selection evidence

Successful run receipts have kind `secondary_development_run` and bind the exact configuration, main selection, source, implementation and deterministic execution policy. Synthesis verifies the expected control/spec/learning-rate matrix, checkpoint state keys/shapes/dtypes, learned-state configuration manifest, parameter counts and zero checkpoint replay deltas. It verifies saved prediction bindings and recomputes raw NLL using the admitted external validation-parent labels. This is outcome verification, not model input construction.

The runner's `failure.json` is a preserved per-run failure record. It is not itself a synthesis ledger. Terminal failed optimizer starts must be represented explicitly in a ledger with this schema:

```json
{
  "kind": "secondary_development_failure_ledger",
  "config_sha256": "BOUND_CONFIG_SHA256",
  "main_selection_receipt_sha256": "MAIN_RECEIPT_SHA256",
  "entries": [
    {
      "configuration_id": "gain_matched_diagonal_lr0300",
      "seed": 101,
      "attempt_id": "safe_recorded_attempt_id",
      "terminal": true,
      "reason": "Actual preserved failure reason",
      "test_access": false
    }
  ]
}
```

Every failure must identify one prescribed cell. Duplicate successes, duplicate failures, success/failure conflicts and extra cells are rejected. A failed rate is ineligible; a fully resolved control can still choose its other rate if all three seeds for that rate succeed. No failure is silently retried.

The output kind is `secondary_development_attribution_selection_receipt`. It includes per-control `matrix_resolved`, configuration summaries and choice; the complete cell/failure matrix; recorded and maximum optimizer starts; exact main plan/receipt, original candidate and original strongest primary baseline; source and numerical-policy bindings; and captured artifact provenance.

The receipt always records `primary_selection_changed=false`, `final_test_access=false` and `authorizes_confirmatory_generation=false`. If any required cell is missing, its control choice is absent. Neither the secondary receipt nor its results authorize main re-selection or final-test promotion.

## Validation and limitations

Final rerun after the strict combined-producer API freeze:

```text
tests/test_secondary_development.py tests/test_anchor_controls.py
70 passed in 110.20s
```

The suite contains 30 new secondary cases and 40 unchanged anchor-control cases. Both CLI help paths passed. Default binding without a genuine main selection refused and wrote nothing.

The secondary tests cover all four conditional geometry/depth cases; complete replay and forged/incomplete main evidence; configuration, implementation and source drift; actual observed coordinate/query contract drift; held-out path preflight; admission before device queries; every main family consuming the strict producer guard; toy fitting with label counterfactuals and exact checkpoint replay; effective numerical-policy drift; full/partial two-rate selection; wrong spec/source/policy/NLL/replay records; and preserved terminal failures with immutable primary choices.

The ten-family synthetic main matrix is constructed from an existing main-selector test fixture. Tests explicitly mock cohort admission, reviewed family dependency contracts and complete producer-settings admission. They use a real `SourceSnapshot` on a small fake observed file. The toy CPU fit uses real unchanged matched-control classes on F47, three-scale, 32-anchor mathematical fixtures. Synthetic three-seed selection evidence is built from toy states and is not actual fresh-development evidence. These tests establish implementation behavior, not real-cohort admission or scientific performance.

## Frozen implementation manifest

The following block is copied from the proposal after the strict combined gate repair. The proposal SHA256 is `b26f5fe07918537b0b1d036daf41c308894f1ac79965015e5aa657b743b357eb`. The public combined gate source pin is `52cfea95bb9613b25fc0a8773f9dd5fcc0430607329b2ed0638d79955be7370c`.

```json
{
  "binder_script": {
    "path": "scripts/bind_secondary_development.py",
    "sha256": "be925fe22d71d500a1fdab2cded11f3e108c1c76b62108dd2ffaa87049d9ee0a"
  },
  "reviewed_implementation_sha256": {
    "age_dataset.py": "b8585bf12c14201f46bda6b333c9dd23e8824c9af44cbe7173232f1c820e8a84",
    "age_interface.py": "15f9615a84a230386d8f71f5a317a46051c395f75408c79a6213f1b07f101f3d",
    "anchor_controls.py": "f4939b4515bafb7ba77373980d2be44f7c9be1a5b6881f2166ca0a2d6265063e",
    "anchor_kernel.py": "2b8cc1c1726f746f961c0d63a4870e806a1a9d3d6e1fa6724dc0bda39967b083",
    "anchor_temporal_baselines.py": "3e88095eb466ea25db5f69553dfa31e6e002fa55ce4d31c4ddb17cb73115518b",
    "boundary_dataset.py": "06f5dfebd40c4a3c39a21c8be7b655e40c422edd1629bdde0b8e75d3c25022a2",
    "boundary_memory.py": "a056cd10d1c848d7dabdc1daf646548876fd90e8f489667c46a374ac28cd11dc",
    "confirmatory_metrics.py": "4e4f906cd6a302ffc422302c6f10694594856faab02d0da7565c508baf4c0a80",
    "data.py": "fb4e4b3ad8744731e6d2c5966cc2679f9f2d59dfd079facc280d69fbc61ef654",
    "development_execution.py": "4a2f87320615eb655a7700b6bbff4a7d293b1b16749468afa46f0de2d12e276f",
    "development_selection.py": "c3771b3cc1c14eb972a3edf565f46ebaa4ff82d595cc6511ab2c35229bb53ed2",
    "development_selection_bindings.py": "52cfea95bb9613b25fc0a8773f9dd5fcc0430607329b2ed0638d79955be7370c",
    "development_training_bindings.py": "89a843a1a92695803682cd1b07e4f790546229ee8e8bfc63e4ff5d765550c73c",
    "genealogy_baselines.py": "388e84c183413a79db8e0e75c86f1ce202b4bb9999473b729be045d7459dc9f8",
    "kron_interface.py": "09ec6286c185e7aa2b13d76dcbe8fea24b3a1c785a6619ac9bc5464e0de66d69",
    "metrics.py": "46af1adbd922e1652d9a1cee14dc38e9d4e952a365d3f026a20476f597c89367",
    "models.py": "8429bd02a40950a3dc4d76046ab4b31b366871499d95e295a16ade730f567df5",
    "reference_encoder.py": "039f3b25383ea92f243090c2eaf592df590768fea7abf41d15fd7c34a7715138",
    "resources.py": "ffc7ac9ce1ebb19b6d7030b9db643d8de9a07179318a9bb1e593779fd3f6d111",
    "secondary_control_bindings.py": "42c0a169adf0e7ea3d98c12b2ebd813dd59cbc665dc60e20f3e4f95568a303c8",
    "simulate.py": "f3e2dcbaa47c0b185d33e3544313fce68697aec9e3223fda36fc54539fcc371a",
    "train.py": "4feadc4b0d4b5dfbfa42f017879f523171f0dd7bd3252e1a1422c4c578c80c42",
    "train_anchor.py": "41c916217ece8dc541e7f5131dc3cb3dcdb839e58493c3e53ea4d3fe86c04cc3",
    "train_anchor_development.py": "8488911bd7ba65ad75ab8b34f1a388d664c29a26791929db24eaafa52e10ef2d",
    "train_anchor_temporal.py": "9f1ea9bfb99e9e40f644867dd18b772fe008b855b41f16ec0abc718a8278a37a",
    "train_anchor_temporal_development.py": "f41ac5355f545df0d6804185cf356d7b4496550166f1134ab48ec7edfa2270f1",
    "train_boundary.py": "ce1705494db9a6777d3d0b21fa79e5a1408c13d6f2e63b4bf67303ab4a575188",
    "train_secondary_development.py": "679ce1d738d53f63ccc14cc9100e0ff5b64e07560b0674521dc42bba075a3db1",
    "train_study.py": "c12c1322f9137576a4fe4c1846655dbb86aad2ea3169db7a0446cc9b32861f56"
  }
}
```
