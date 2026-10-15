# C3 root handoff: exact native focal registry and PyG plan

**Source ready; root admission and actual preparation remain pending.** The sole new producer is `focal_plan_bridge_v1.py`, SHA256 `b2e7fcfa1d301b01cda62a99408986e89cd75fe14509dc4188bd04d0cf84317e`. Existing producers are unchanged. `source_manifest.json` pins 18 source/authority refs, including the 11-file preparation closure.

The bridge prepares each independent 2 Mb TRAIN parent's exact native focal registry, then calls unchanged `american_c3_gnn_dataset_v1.build_plan` and `american_c3_gnn_stream_train_v1.validate_plan_v1`. It supplies no native MLM corpus or optimizer completion and has no ARGformer-training dependency.

## Root commands

On anogena, from `/workspace-SR003.nfs2/ARG-GNN`:

```sh
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B research/reviews/C3_next_plan_root_handoff_20261015_v1/focal_plan_bridge_v1.py --admit american_c3_short_train_000
```

Use the actual admission path and SHA printed by that command:

```sh
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B research/reviews/C3_next_plan_root_handoff_20261015_v1/focal_plan_bridge_v1.py --execute --admission ACTUAL_ADMISSION_PATH --expected-admission-sha256 ACTUAL_ADMISSION_SHA256
```

Execution is one foreground CPU process. Root retains and records its genuine direct command wait/drain using existing supervision; producer completion does not manufacture that evidence. Each parent requires a fresh admission/output. No replay, overwrite, partial continuation or automatic scope close is supplied.

After all 20 parents genuinely complete:

```sh
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B research/reviews/C3_next_plan_root_handoff_20261015_v1/focal_plan_bridge_v1.py --collect-plans
```

This metadata-only collector emits the exact all-20 `parent_plan_refs` mapping. It does not authorize training.

## Exact native equivalence

The frozen author `get_path_key` places the sampled node first, making different samples' full path keys disjoint. The bridge preserves the original sample-outer/tree-inner first-occurrence order, using membership-only sets that are never iterated and offsets from prior samples' unique counts. Therefore global record IDs are identical.

It selects from the same explicit ordered int64 query-ID array using the same freshly seeded `np.random.default_rng(seed).choice(array, size=10000, replace=False)`, then sorts. A second complete enumeration recovers each selected path's identical first interval and original `(left+right)/2` midpoint. Vocabulary construction does not mutate original sequence order; native splitting shuffles a copy. No distribution-only substitute, truncation or sampling fallback is used.

One completed synthetic fixture compared the original global-dictionary algorithm with the bridge on 1,600 samples, duplicates and 10,000 selected query records: record IDs, sample axes, first intervals and positions matched exactly. `source_check_receipt_v1.json` records the check. The later metadata collector did not change enumeration; AST/default CLI checks passed. No further tests are required for this handoff.

## Products and guards

Per-parent outputs reside under `experiments/results/C3_short_TRAIN_focal_plans_v1/american_c3_short_train_###/`: `focals.npz`, `native_full_path_order_counts.json`, `focal_completed.json`, `block_index.json`, and `completed.json`.

The honest focal receipt declares `completed_C3_exact_native_focal_registry_only_v1` and `full_native_MLM_corpus_prepared=false`. It binds actual observation/focal/preparation sources. The existing planner consumes those fields without unused MLM products. All 1,600 samples, 10,000 unweighted focals, full trees/sites and original parent carry remain. Existing blocks are 32 trees / 250,000 nodes / 4,096 sites; EUR/EAS/AFR class order is unchanged.

Only root's explicit admission/execution invokes the active prospective 980 GB guard. Admission uses original unmemoized v4 mutations and the current 64-reference check; execution uses original memo2 and existing `common.Writer` with the current adapter. Each parent reserves 32 MiB, with a 7,200 s CPU deadline, 512 GB current-process RSS, 100,000-tree and 1,024-raw-path-node caps. Bound failures remain failures.

## Readiness and remaining integration

The saved metadata snapshot verifies actual inference completion and successful direct waits for parents 000–013. Parents 014–019 become eligible individually after their actual inference completions and waits; simulation completion alone is insufficient. Actual bindings use wave_v2. Root's requested 000 admission is not claimed successful here.

The prior `American_C3_PyG_execution_source_20261015_v1/` packet remains the training handoff: root integrates its target-factory patch as appropriate and supplies all-20 simulation bindings, TRAIN truth resolver, writer/device/current guards and GPU admission. Training remains 20 × 2 Mb, 30 epochs, 600 complete-parent FP32 updates; five original 30 Mb VAL parents and the all-30-seal scoring barrier remain downstream.

This worker performed no actual payload/truth/checkpoint/model/GPU read, guard/accounting/admission action, installation, Git operation or existing-producer edit. Packet source is stable; root owns execution and lifecycle evidence.
