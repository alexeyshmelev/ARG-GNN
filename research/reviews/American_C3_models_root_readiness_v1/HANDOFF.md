# American C3 model root readiness

Source review only. No payloads, runtime imports, simulator/inference/model execution, admissions, guard state/configuration reads, GPU access, installs or Git operations. Existing scientific sources and author code are unchanged.

## Ready and missing

Reuse current observe_v3, paths_v1, the GNN whole-tree planner/model/train-step/checkpoint/inference APIs and common external evaluator. Use the authoritative 20 × 2 Mb TRAIN registry (ancestry 22001–22020, path 92201–92220). Existing 100 kb GNN configs and old native observe_v1 handoff fields are stale. The old C4 native-v7 queues use 280 haplotypes/fixed 100 bp support and are not C3 launchers.

A credit-v3 GPU45 launcher already exists: scripts/launch_reserved_gpu_authorized_GPU45_completed_v1.py (d4cc990e…). Reuse it and the current root evaluation admission pattern. Do not duplicate it or use the retired capacity-v2 GPU45 launcher for new admissions.

Missing integration is a canonical guarded C3 native consumer, a CPU embedding concatenator, separate TRAIN-only truth-to-focal join, GNN trainer/resume orchestration and guarded prediction writer. The launcher accepts arggnn modules and uses the shared interpreter, so raw native author argv is not a valid root launch. Its consumer must invoke the pinned native interpreter under actual title/runtime/lease authority and direct-wait/drain every native/DataLoader descendant.

## Native findings and primary comparator

Pinned author tag v1.0.0, commit f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596. Actual source SHA refs are in source_refs.json.

- embeddings.py:321 randomly chooses among train/val loaders and:335 appends each batch. C3 evaluator:64 assumes train then val sidecar rows. Use four unchanged-author single-split embedding invocations (two views × two splits), then an admitted CPU train+val concatenation preserving every vector, tensor and full raw-path sidecar. No truncated-token, embedding or leaf-only identity matching.
- sequence_contrastive.py:832 passes cfg.model to build_contrastive_model. Its weighting branch:600 accesses cfg.train_loader at:604 and catches failure at:621, leaving uniform weights. Keep this behavior for the primary comparator and disclose it. No primary training config-plumbing fix or claim that weighting worked.
- Preserve MLM 5 ep / batch 512 / bf16 / seed 17 and contrastive 1 ep / full batch 4096 / fp32 / seed 36, 12 layers / 12 heads / 128 tokens, release temperature 0.30 (paper default 0.05). Full batch contrastive loss cannot be replaced by smaller independent microbatches without changing the objective.
- MaxMetricTracker logs a maximum at sequence_contrastive.py:132 but does not save a selected checkpoint. Verify an actual completed final checkpoint and Composer timestamp, rather than assume latest-rank0.pt or the reported maximum exists at the required step.

native_serialization_adaptation_inactive.json provides the exact row-preserving contract. No existing author file was edited.

## Budgets and next sequence

training_budget_inactive.json records unchanged release native budgets and proposes a distinct GNN discovery budget:20 independent 2 Mb parents, 30 complete passes, 600 optimizer steps, 6 million focal-target exposures, fresh seed 70001, FP32, AdamW lr 0.001 / weight decay 0.0001 with fixed LR and clip 1. This is unadmitted and not a convergence or timing claim. Qualify actual complete-parent forward/backward/optimizer memory first. No detach, hidden reset, pruning or silent shorter fallback.

Save GNN epochs 5, 10, 15, 20, 25, 30. Select by mean per-parent macro-F1 on the original 5 independent 30 Mb VAL parents once prepared, ties earlier epoch. They and 10 original TEST parents remain deferred. The existing 30 Mb american_c3_train_000 remains TRAIN and is outside this 20-parent short-training condition. Do not relabel it as held-out data.

root_integration_inactive.json gives dependencies, API gaps, target-join requirements and exact credit-v3 compatibility. Root first admits path materialization and observation-only plans from actual completed observation refs, then pins missing consumers and qualifies counts/storage/RAM/GPU fit. The CPU common CLI already accepts the current adapter through configure_v2 and the exact policy environment. Actual model inputs/runtime/reservations remain null because no scientific payloads or current resource state were inspected.

Native is fresh transductive adaptation per target ARG using unlabelled paths and known reference labels. GNN is supervised transfer from independent simulated query ancestry. Compare complete pipelines on the same inferred observations and frozen 10,000 focals. This does not isolate architecture or reproduce Table1 exactly. Report uncertainty over independent parents, not correlated focal records. 2 Mb training undersamples approximately 8.3 Mb tract context, making 30 Mb evaluation a length-transfer condition.
