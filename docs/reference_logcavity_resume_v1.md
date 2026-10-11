# H030 narrow durable recovery

This is operational continuation of the registered H030 condition after an output-guard failure, not a new model, arm, seed or independent confirmation. Original H030 model, config, driver, V9 adapter and failed output remain immutable. H032 sources/output are also preserved. A separate driver/output uses root's exact retry/backoff adapter without changing budget core or science.

## Failure and durable anchors

The immutable predecessor is `experiments/results/reference_logcavity_gnn_v1_seed101_GPU3/failed.json`, SHA256 `f914bf9082746614cf43da8b7db657d714b7614c0b61adc0ddd584ff05fd99ed`. It records `Process inventory changed during writer inspection` after the immediate-retry guard exhausted five observations. Original input is `research/reviews/novelty_discovery_execution_v1_root/H030/input.json`, SHA256 `20642248f51f54d270118c84fa2a038299ab63d1abc7bf56c85e71c6c2972414`.

| Original arm | Last complete checkpoint/metrics pair | Restored logical steps | Next epoch | New steps |
|---|---:|---:|---:|---:|
| Log-cavity only | 4 | 64 | 5 | 96 |
| Static | 4 | 64 | 5 | 96 |
| Carrier GRU | 3 | 48 | 4 | 112 |
| Carrier matrix | 3 | 48 | 4 | 112 |
| Internal-history sham | 3 | 48 | 4 | 112 |

The three recurrent arms' original epoch-4 probability files have no complete saved checkpoint/metrics pair. They are orphan outputs, not final selection candidates. Their 48 original epoch-4 optimizer updates are discarded and replayed from durable epoch3. Kernel/static epoch4 is complete and is not replayed. Recovery performs 528 new updates; each final trajectory has exactly160 logical updates. Original physical work is320 updates plus528 recovery updates:848 total, versus800 final logical trajectory updates. The48 discarded/replayed updates are openly disclosed rather than relabeled as new evidence.

`resume_state_manifest.json` pins every complete prior metrics row and the latest/selected checkpoint and probability references. Earlier metrics remain history even if their nonselected artifacts were legitimately pruned by the predecessor. Prior earliest minimum equal-parent Present64 NLL selection is restored; a strict improvement alone replaces it, preserving earliest ties. All five durable selected epochs currently equal their anchors.

## Implementation boundary

The new driver copies H030's narrow preparation/resource/training wrapper, replacing only its adapter imports and adding per-arm durable restoration/scheduling/lineage. It loads exact `model_state` and `optimizer_state` into the same model and original AdamW parameter-group order, checks arm/epoch/seed/input/teacher identity and optimizer steps, and verifies checkpoint/seal hashes at actual execution. It keeps original scientific fields, observations, emitter/targets/support, teacher, loss, scorer, learning rate, one FP32 seed101, epoch tile permutation `Random(101+epoch)`, selection and160-step endpoint unchanged.

The unchanged model uses no stochastic training dropout or augmentation, learning-rate scheduler or persistent carrier state between tile calls. Checkpoints need no new carry state for this protocol. This does not promise cross-runtime or cross-device bitwise equality. Root must retain the fixed deterministic runtime policy and disclose recovery lineage.

Rebuild complete target-free geometry once using the copied H030 preparation and require equality with original sealed geometry before opening TRAIN targets. First actual recovery update for each arm records finite loss/gradient and synchronized timing; these are steps49 for the three epoch3 anchors and65 for kernel/static, counted within160. No broad parity sweep or extra fit is scheduled.

Prior references are read-only. Pruning checks membership in the new writer's `owned` set and can remove only new-scope superseded files. Original output is never changed. Reuse original H025 baseline comparison metadata and frozen H026 strong comparator; no baseline refit. Retain new current/best probability-only seals and last/best checkpoints. Even if an original candidate remains selected, final recovery receipt refers to its immutable checkpoint/seals rather than copying or altering them.

## Guard and execution authority

Import `validate_storage_v9`, `budget_check_v9`, `PilotOutputV9` from `src/arggnn/project_budget_v9_consumer_retry_v2.py`, SHA256 `75fd586fc8e0733027451a6d9485b7e49f10c3d8d7cf69c09cdec4b9f1b7ba1e`. Only retry/backoff changed; root owns its review. Exact original V9 consumer remains untouched. New output requires its own later root runtime, active input, held GPU lease and2GB reservation. `execution_guard_ready=true` is source readiness only.

Native-tree temporary writes retain before/after/cleanup guards for exact hashed bytes+8192 overhead; tmp directory allowance8192; CUDA-helper qualification allowance32MB. Copied resource guards use the new adapter everywhere on new storage. No original closed authority is reused for writes. Root reservation consumer_bindings must include BOTH the unchanged original `src/arggnn/project_budget_v9_consumer.py` provenance pin and the actual retry_v2 adapter pin, as required by unchanged V9 successor_paths. Actual new storage imports/calls still use retry_v2 only; no budget-core rewrite is needed.

## Completion and H032 comparison

A fresh recovery output may contain `completed.json` with recovery-specific schema/status, original failed/input/config refs, per-arm anchors, complete previous+new history, new update counts, selected candidates and final160-step counts. Never write an original H030 `completed.json`, rewrite its failure, pretend it completed uninterrupted, or treat orphan epoch4 predictions as selected evidence.

The prospective H032 first-moment comparison remains incomplete until this recovery actually completes. Root's separate final metadata summary must bind BOTH original failed H030 lineage and recovery completed/input/config/source refs, then use the recovered final selected STATIC arm. Label it recovered H030; do not substitute a fake original completion. H032's own standalone fit completion is unaffected.

## Source-only validation and limits

Metadata fixtures verify exact model/AdamW restoration calls without optimizer updates, earliest-NLL ties, external selected references, rejection of malformed latest-pair/step metadata, and the asymmetric schedule528-new/160-final counts. No actual checkpoint tensors, parent payloads, target arrays, GPU or optimizer update was opened/run by the author. Existing numerical/model evidence is reused; no new scientific accuracy, mechanism, priority or generalization claim follows.
