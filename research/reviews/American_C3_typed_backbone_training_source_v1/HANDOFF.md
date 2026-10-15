# American C3 typed backbone: source-only training handoff

Status: inactive scientific callables. This work opened source and protocol metadata only and used Python compile(source, path, "exec") without imports. No truth/observation/checkpoint payload was read, no model was constructed, and no forward, optimizer update, GPU request, install, admission, execution driver, accounting change or Git operation was performed.

The latest task requests checkpointed block/GRU training. That supersedes the earlier readiness draft's single-block TRAIN integration instruction. Existing planner, model, single-block helper, checkpoint loader and two-pass inference sources remain unchanged. The new wrapper keeps the complete-parent objective and all cross-block gradient paths.

## Scientific condition

Use only the authoritative20 independent2Mb TRAIN parents american_c3_short_train_000 through019, frozen ancestry seeds22001..22020, mutation1022001..1022020 and path92201..92220. All1600 observed haplotypes, every active inferred marginal node/tree, all sites and exactly10000 original common query focals remain. Original query sample columns are600..1599. Class order is EUR/EAS/AFR. No purity filtering or support revision occurs.

This is a fresh typed C3 gated-upward/recipient-excluded-cavity baseline followed by exact two-direction whole-parent GRU recurrence, implemented with Torch/PyG. C4 checkpoints/classes are rejected. C3BackboneV1 describes the extension interface; H043_extension is None and any requested H043 extension is rejected until separately audited. No H043 module or result is included.

The fixed main budget is30 epochs x20 complete-parent updates =600 updates and6000000 focal exposures. Seed70001 initializes the model; one-based epoch e uses random.Random(70002+e).shuffle(authoritative_registry_order). FP32, hidden32, donor cohorts2, site chunk256/checkpointed sites, AdamW(lr.001,weight_decay.0001,betas.9/.999,eps1e-8), clip1 and constant LR are frozen. No early stopping or separate two-update campaign exists. Full training's actual first update checks finite loss/all learned gradients/updated parameters, nonzero global gradient and a changed parameter checksum. It records process peak RSS and the actual admitted device's per-parent CUDA peaks.

## TRAIN observation/target barrier

run_train_v1 requires the complete20-entry SHA-bound actual plan map. Plan constructors read metadata. It then seals every observation block for every parent and releases each parent Dataset state before proceeding. It verifies exact whole-tree coverage, original references/query columns and one-to-one common focal coverage. Every encoded bank/block tensor receives a feature fingerprint. Only after all20 seals are complete may the separate TRAIN-only resolver be invoked. All20 target joins must validate before model creation/optimization.

TrainTargetJoinV1 rejects VAL/TEST, the excluded original30Mb TRAIN and unknown IDs before calling its private resolver or inspecting a private path. The exact same-parent private NPZ is SHA-bound, its five ZIP members and uncompressed bound checked, then its original contiguous half-open ancestry intervals are joined with searchsorted(right) and original query-column lookup. Labels remain a separate CPU torch.long vector, never a PyG Data field, plan field, model input or inference feature. The external TRAIN target manifest binds private refs but stores no labels.

The CPU LRU caches observation-only blocks up to the configured prospective cap, cloning before each device transfer. Dataset source state is released between parent updates. No DataLoader subprocess is created by the training callable. Each recomputation checks the sealed feature fingerprint and refuses changed observations.

## Complete-parent differentiation and memory

checkpointed_complete_parent_step_v1 separately checkpoints each complete-block encoder plus directional GRU scan with use_reentrant=False. The directional hidden tensor and focal outputs are copied to CPU while preserving autograd edges. Forward state crosses every chronological block; reverse state crosses every reverse block. States reset only at true parent endpoints. All10000 focal rows are reordered to the frozen common registry. Checkpointed focal-classifier chunks contribute sums to a single unweighted complete-parent mean CE, one backward and one optimizer step.

There is no detach, truncated BPTT, finite halo, per-block optimizer/loss step, pruning or shortened-parent fallback. CPU boundary-state graph storage grows with block count; this is not a constant-RAM claim. The full inferred tree sequence remains CPU-side for one current parent and genotype arrays are memory-mapped. GPU replay keeps whole-tree block activations rather than all-parent all-node history. Actual peak memory, throughput, replay behavior and successful full-budget execution are unverified. FP32 reductions and max-pooling tie subgradients may vary with grouping; bitwise dense-gradient equality is not claimed. No source-only fit/timing/accuracy claim is supported.

The512MiB CPU cache and1024-focal classifier chunk in the inactive config are prospective bounds. Root must fill a positive max_truth_uncompressed_bytes using actual metadata and admit actual planner/checkpoint/CPU/GPU storage/resources. Existing plan caps are root-chosen rejection limits, never a permission to prune a tree.

## Checkpoints and continuation

Scheduled epochs5/10/15/20/25/30 use the existing fresh C3 checkpoint family, augmented with exact scientific/source/plan signature, observation and target manifest hashes, actual trained IDs/lengths, completed-parent trace, next epoch/parent index/order, optimizer state, constant schedule state and Python/NumPy/Torch CPU/one admitted-device RNG state. Checkpoints contain only weights_only-safe builtins/tensors. CPU snapshots are immutable and copied only after a successful full-parent optimizer update. No snapshot blesses a partial update.

A failed full run attempts to preserve failure_last_complete_parent.pt plus failed_training.json and re-raises the original exception. A failure before any completed update has no resume checkpoint. Serialization failure is recorded as an exception note when possible; root still owns original child status/direct wait/drain. No failure artifacts are deleted.

Resume checks exact source/config/plan/data signature, class/fresh-C3 contract, model, full AdamW moments/step counts/settings, history/cursor/order/schedule and RNG. It does not reset the optimizer or treat weights-only loading as continuation. Previously scheduled checkpoint refs are preserved. A failure exactly after an epoch update but before its scheduled file is written recovers that frozen boundary before another update. Scheduled candidate files exclude their own ref to avoid a hash cycle. Resume-boundary files may include all checkpoints already published.

## Prediction, original VAL selection and deferred TEST

seal_prediction_v1 accepts no truth argument. It uses the existing strict C3 load and exact complete-tree two-pass focal prediction API, with owned worker draining. It accepts scheduled checkpoints only from a completed600-update training completion. It validates the original30Mb held-out plan and writes the original evaluator's exact six identity arrays plus raw float32[N,3] probabilities. It removes the heavy trainer state from inference JSON receipts and retains the SHA-bound checkpoint/training refs.

Create all6 x5 original VAL prediction seals, then seal_validation_matrix_v1 creates the30-cell SHA-bound pretruth barrier. The separate score_validation_matrix_v1 receives an original-VAL resolver only after verifying that barrier. It calls the unchanged external evaluator; no new truth decoder or metric equation is implemented. This score phase is separate from GNN training/prediction. Original metrics and focal support remain frozen.

select_original_val_checkpoint_v1 validates all30 external completion receipts against their sealed predictions, uses greatest mean of the five per-parent original macro-F1 values, and breaks exact ties toward the earlier epoch. It records all parent scores and candidate refs. TEST prediction is refused until that separate selection is sealed; the original10 TEST remain deferred now. Original30Mb american_c3_train_000 remains TRAIN and is excluded from this condition.

## Root integration

These are callable/config contracts, without a main/CLI or root driver. Root imports Torch/PyG only after its existing process-title/lease/runtime guards, provides one explicit admitted device, actual plan refs and an existing admitted writer implementing save(name,value,kind="json"|"torch"|"npz"). The scientific callable never creates admissions, leases, accounting journals, output scope authority or launchers.

runtime_guard(event) must be a fast lease/process/RSS/device resource check. Do not wire the slow global storage journal replay into every block/recompute callback. Output writes occur through the supplied writer at phase/checkpoint boundaries. Root remains responsible for canonical consumer integration, runtime/source closure binding, output bounds, actual launch/process-title/scope supervision/direct wait and final drain. Use only existing current credit-v3/GPU45 authority when eventually admitted.

training_inactive.json has execution_enabled=false and an unset private-read cap. callable_bindings_inactive.json leaves all actual plan/runtime/writer/device/admission refs null. The first two inferred parents are not a20-parent training authority. Do not execute this source as a two-parent fit probe. Required real-update checks are embedded in the full600-update run.

GNN is supervised independent-parent transfer; native ARGformer is fresh transductive per-ARG adaptation with known reference labels. Compare full pipelines on the same observations and common focals. This does not isolate architecture or reproduce author Table1 exactly. Treat independent parents, not10000 correlated focals, as uncertainty units.2Mb TRAIN to original30Mb evaluation is length transfer and one model seed does not establish seed uncertainty.
