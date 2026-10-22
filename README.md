# ARG-GNN

<!-- readable-recovery-queue:start -->
### Current recovery queue
Updated 2026-10-22T00:43:35.712285+00:00. Status: running.
76 pending, 1 running, 231 succeeded, 7 failed, 32 blocked, 0 unknown.
Local active: AM_argformer_american_validation_002_argformer_mlm.
Live training stages: bounded stage metadata pending.
Standalone work: no live standalone generations.
The earlier v9 queue block is a historical snapshot. This block describes the active successor.
Ancestry accuracy is pending the frozen prediction and scoring barriers. Training loss is not accuracy.
One optional external AP001 archive is deferred with its full retained storage charge and zero credit.
[Current queue metadata](research/benchmark_readable_current.json).
<!-- readable-recovery-queue:end -->

**TreeLoom** learns local ancestry inference (LAI) from ordered collections of complete genealogical trees. Its GNN processes tree structure and follows persistent haplotypes across genomic intervals.

## Current task: execute the benchmark campaign

**The active LAI benchmark continuation uses a queue of 347 stages, including storage recovery.** It covers shared American and European simulations, full training, native competitors, paired scoring and Danish data application. Current execution and matched results are published below. All computation and Git commands run on anogena, using physical GPU 5 only.

<!-- BEGIN CURRENT BENCHMARK DIAGNOSIS -->
**22 October: ARGformer is training on physical GPU 5, but the comparison is incomplete.** The current queue has **231 completed, 1 running, 76 pending, 7 failed and 32 blocked stages**. The running job is the third American simulation's full five-epoch ARGformer pretraining, with **34,513 / 223,655 updates** observed. The GPU showed **100% utilization** at the status check. **Fresh matched ancestry accuracy and F1 remain unavailable.**

| Workstream | Verified state | Remaining work |
| --- | --- | --- |
| American TreeLoom | First full-parent backward exceeded the 16 GiB CUDA allocator cap, with physical memory still available. This attempt completed **0/600 optimizer updates** | Repair CUDA memory allocation, complete full training, then predict |
| ARGformer | Native training and ancestry predictions complete for **2/15** simulations. Full pretraining is active on the third | Finish the remaining native pipelines |
| FLARE | **15/15** American predictions complete | Paired scoring after the comparison barrier |
| AncestralPaths | **8/8** European genealogical inference stages complete | Complete dependent ancestry models and comparisons |
| European TreeLoom | Both fits were rejected before launch because their old qualification records do not match the current qualification schema/source contract | Run fresh qualifications for the same inputs and current implementation |
| Released ARGMix | Four archive stages failed because the stock Rust helper could not find `python` on PATH | Add the admitted native runtime bin to the child PATH, then continue the unchanged author pipeline |

Versioned repairs are prepared and have passed focused source and metadata checks. They have not yet been integrated or dispatched. The active ARGformer run and all prior successful work are preserved. The persistent capacity-mutex warning in pending rows is stale reporting after brief controller lock contention, not evidence that training has stopped.

Only physical GPU 5 is authorized. Process control is restricted to verified ARG-GNN jobs. No unrelated process was modified or signalled during this check.

[Current diagnosis](research/repairs/benchmark_followup_20261022/status.json) · [Live queue](research/benchmark_readable_current.json) · [Matched results](research/resource_matched_benchmark_results.json)
<!-- END CURRENT BENCHMARK DIAGNOSIS -->

<!-- BEGIN COMPLETED WORK ACCOUNTING -->
## What the recent GPU jobs accomplished

Audited at 20 October 2026, 01:52 Moscow. These are verified training and prediction outputs, separate from queue preparation counts.

| Completed work | Output | Elapsed job time |
| --- | --- | ---: |
| Fresh ARGformer, first validation simulation | Five pretraining epochs, 226,885 optimizer updates | 7.65 h |
| Fresh ARGformer, second validation simulation | Five pretraining epochs, 224,170 optimizer updates | 7.50 h |
| Fresh ARGformer, first simulation, subsequent stages | 2,125 contrastive updates, embeddings and 10,000 ancestry predictions | 2.58 h |
| Earlier American GNN, separate from this fresh fit | 600 completed updates over 30 epochs, verified on 16 October | 9.12 h to update 600 |

The five completed fresh ARGformer stages took **17.74 hours of elapsed job time**. This includes loading and data movement, and does not measure GPU kernel time. An earlier separate ARGformer chain also completed 2,135 contrastive updates and retrieval. These receipts confirm real training, but do not account for every hour of the previous days, including failed attempts and repairs.

**The fresh matched comparison still has no ancestry scores.** FLARE has completed predictions for all 15 American validation/test simulations on CPU. ARGformer has completed predictions for one. The American scoring job depends on all 45 method/dataset prediction outputs, of which 16 are complete. The failed fresh TreeLoom fit had 20/20 inputs prepared and 0/600 completed updates. Its first recurrent pass ended with a cuDNN error. The repaired replacement has started actual training-data preparation after a successful full-length GPU recurrent test.

### Scores that already exist, from earlier development experiments

These European accuracy scores are means and sample SDs across eight reused validation simulations for the present-day cohort. They are earlier development evidence, with method-specific input and training protocols, not results from the fresh benchmark.

| Method | Accuracy (%) |
| --- | ---: |
| TreeLoom | **97.08 ± 0.57** |
| Released ARGMix | 94.45 ± 1.15 |
| AncestralPaths, probabilities mapped to four classes | 93.78 ± 0.84 |

On a separate earlier set of five American validation simulations, FLARE achieved **98.10 ± 0.34% accuracy** and **97.92 ± 0.41% macro F1**. One simulation was previously inspected during development. These FLARE figures cannot be compared with the European table above.

[Completed-work audit and receipt references](research/reviews/benchmark_status_20261020/completed_work_audit.json) · [Full historical scores](research/current_scoreboard.json)
<!-- END COMPLETED WORK ACCOUNTING -->

<!-- BEGIN CURRENT RESOURCE BENCHMARK WORKFLOW -->
## Active benchmark continuation

Observed 2026-10-20T07:17:30.314165+00:00. 164 succeeded; 1 failed; 0 running; 155 pending; 16 blocked; 0 unknown.
Run status: `drained_for_operational_update`. Physical GPU: 0.

[Current queue details](research/benchmark_resource_current.json) · [Authenticated matched scores](research/resource_matched_benchmark_results.json)

| Stage | Succeeded | Failed | Running | Pending | Blocked | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ap_features | 2 | 0 | 0 | 6 | 0 | 0 |
| ap_fit | 0 | 0 | 0 | 1 | 0 | 0 |
| ap_generate | 9 | 0 | 0 | 0 | 0 | 0 |
| ap_infer | 3 | 0 | 0 | 5 | 0 | 0 |
| ap_predict | 0 | 0 | 0 | 5 | 0 | 0 |
| ap_registry | 2 | 0 | 0 | 6 | 0 | 0 |
| ap_train_targets | 0 | 0 | 0 | 3 | 0 | 0 |
| contrastive | 1 | 0 | 0 | 14 | 0 | 0 |
| embeddings | 1 | 0 | 0 | 14 | 0 | 0 |
| external | 9 | 1 | 0 | 0 | 0 | 0 |
| factory | 3 | 0 | 0 | 21 | 1 | 0 |
| fit_factory | 0 | 0 | 0 | 2 | 0 | 0 |
| focal_factory | 35 | 0 | 0 | 0 | 0 | 0 |
| generation_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| inference_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| mlm | 1 | 0 | 0 | 13 | 0 | 0 |
| native | 15 | 0 | 0 | 0 | 0 | 0 |
| prediction_factory | 0 | 0 | 0 | 2 | 15 | 0 |
| prepare | 16 | 0 | 0 | 13 | 0 | 0 |
| qualification_factory | 3 | 0 | 0 | 0 | 0 | 0 |
| release_archive | 0 | 0 | 0 | 4 | 0 | 0 |
| release_c4_registry | 2 | 0 | 0 | 6 | 0 | 0 |
| release_c4_score | 0 | 0 | 0 | 1 | 0 | 0 |
| release_c4_targets | 0 | 0 | 0 | 3 | 0 | 0 |
| release_fit | 0 | 0 | 0 | 1 | 0 | 0 |
| release_frame | 3 | 0 | 0 | 6 | 0 | 0 |
| release_infer | 0 | 0 | 0 | 5 | 0 | 0 |
| release_native6_score | 0 | 0 | 0 | 1 | 0 | 0 |
| release_native_targets | 0 | 0 | 0 | 4 | 0 | 0 |
| release_seal | 0 | 0 | 0 | 5 | 0 | 0 |
| retrieval | 1 | 0 | 0 | 14 | 0 | 0 |

Running workflow jobs: none.
Named external queue jobs running: none. External queue states are listed separately from workflow counts.

No supported authenticated matched score has completed in this workflow.
<!-- END CURRENT RESOURCE BENCHMARK WORKFLOW -->

<details>
<summary>Previous v8 queue and original failure evidence</summary>

<!-- BEGIN CURRENT BENCHMARK WORKFLOW -->
## Active benchmark continuation

Observed 2026-10-19T23:03:44.227173+00:00. 158 succeeded; 2 failed; 0 running; 139 pending; 27 blocked; 0 unknown.
Run status: `drained_for_operational_update`. Physical GPU: 0.

[Current queue details](research/benchmark_active_current.json) · [Authenticated matched scores](research/active_matched_benchmark_results.json)

| Stage | Succeeded | Failed | Running | Pending | Blocked | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ap_features | 2 | 0 | 0 | 5 | 1 | 0 |
| ap_fit | 0 | 0 | 0 | 1 | 0 | 0 |
| ap_generate | 9 | 0 | 0 | 0 | 0 | 0 |
| ap_infer | 3 | 0 | 0 | 5 | 0 | 0 |
| ap_predict | 0 | 0 | 0 | 4 | 1 | 0 |
| ap_registry | 2 | 0 | 0 | 5 | 1 | 0 |
| ap_train_targets | 0 | 0 | 0 | 3 | 0 | 0 |
| contrastive | 1 | 0 | 0 | 14 | 0 | 0 |
| embeddings | 1 | 0 | 0 | 14 | 0 | 0 |
| external | 8 | 1 | 0 | 0 | 0 | 0 |
| factory | 1 | 0 | 0 | 15 | 1 | 0 |
| fit_factory | 0 | 1 | 0 | 2 | 0 | 0 |
| focal_factory | 35 | 0 | 0 | 0 | 0 | 0 |
| generation_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| inference_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| mlm | 1 | 0 | 0 | 13 | 0 | 0 |
| native | 15 | 0 | 0 | 0 | 0 | 0 |
| prediction_factory | 0 | 0 | 0 | 0 | 17 | 0 |
| prepare | 16 | 0 | 0 | 13 | 0 | 0 |
| qualification_factory | 1 | 0 | 0 | 0 | 0 | 0 |
| release_archive | 0 | 0 | 0 | 4 | 0 | 0 |
| release_c4_registry | 2 | 0 | 0 | 5 | 1 | 0 |
| release_c4_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_c4_targets | 0 | 0 | 0 | 3 | 0 | 0 |
| release_fit | 0 | 0 | 0 | 1 | 0 | 0 |
| release_frame | 2 | 0 | 0 | 6 | 1 | 0 |
| release_infer | 0 | 0 | 0 | 4 | 1 | 0 |
| release_native6_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_native_targets | 0 | 0 | 0 | 4 | 0 | 0 |
| release_seal | 0 | 0 | 0 | 4 | 1 | 0 |
| retrieval | 1 | 0 | 0 | 14 | 0 | 0 |

Running workflow jobs: none.
Named external queue jobs running: none. External queue states are listed separately from workflow counts.

No supported authenticated matched score has completed in this workflow.
<!-- END CURRENT BENCHMARK WORKFLOW -->

</details>

<details>
<summary>Previous v7 queue, retained for its original jobs and failure history</summary>

<!-- BEGIN REPAIRED BENCHMARK WORKFLOW -->
## Active benchmark continuation

Observed 2026-10-19T03:08:12.269397+00:00. 150 succeeded; 9 failed; 0 running; 83 pending; 84 blocked; 0 unknown.
Run status: `drained_for_operational_update`. Physical GPU: 0.

[Current queue details](research/benchmark_repaired_current.json) · [Authenticated matched scores](research/repaired_matched_benchmark_results.json)

| Stage | Succeeded | Failed | Running | Pending | Blocked | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ap_features | 1 | 0 | 0 | 0 | 7 | 0 |
| ap_fit | 0 | 0 | 0 | 0 | 1 | 0 |
| ap_generate | 9 | 0 | 0 | 0 | 0 | 0 |
| ap_infer | 1 | 6 | 0 | 0 | 0 | 0 |
| ap_predict | 0 | 0 | 0 | 0 | 5 | 0 |
| ap_registry | 1 | 0 | 0 | 0 | 7 | 0 |
| ap_train_targets | 0 | 0 | 0 | 0 | 3 | 0 |
| contrastive | 1 | 0 | 0 | 14 | 0 | 0 |
| embeddings | 1 | 0 | 0 | 14 | 0 | 0 |
| external | 7 | 3 | 0 | 0 | 0 | 0 |
| factory | 1 | 0 | 0 | 14 | 2 | 0 |
| fit_factory | 0 | 0 | 0 | 0 | 3 | 0 |
| focal_factory | 35 | 0 | 0 | 0 | 0 | 0 |
| generation_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| inference_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| mlm | 0 | 0 | 0 | 14 | 0 | 0 |
| native | 15 | 0 | 0 | 0 | 0 | 0 |
| prediction_factory | 0 | 0 | 0 | 0 | 17 | 0 |
| prepare | 17 | 0 | 0 | 13 | 0 | 0 |
| release_archive | 0 | 0 | 0 | 0 | 4 | 0 |
| release_c4_registry | 1 | 0 | 0 | 0 | 7 | 0 |
| release_c4_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_c4_targets | 0 | 0 | 0 | 0 | 3 | 0 |
| release_fit | 0 | 0 | 0 | 0 | 1 | 0 |
| release_frame | 1 | 0 | 0 | 0 | 8 | 0 |
| release_infer | 0 | 0 | 0 | 0 | 5 | 0 |
| release_native6_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_native_targets | 0 | 0 | 0 | 0 | 4 | 0 |
| release_seal | 0 | 0 | 0 | 0 | 5 | 0 |
| retrieval | 1 | 0 | 0 | 14 | 0 | 0 |

Running workflow jobs: none.
Named external queue jobs running: none. External queue states are listed separately from workflow counts.

No supported authenticated matched score has completed in this workflow.
<!-- END REPAIRED BENCHMARK WORKFLOW -->

</details>


<details>
<summary>Previous continuation, retained while its original jobs finish</summary>

<!-- BEGIN CONTINUATION BENCHMARK WORKFLOW -->
## Active benchmark continuation

The repaired continuation is being prepared. The 21 CPU tests for worker supervision, live predecessor adoption and report isolation passed. No new scientific score is claimed.
<!-- END CONTINUATION BENCHMARK WORKFLOW -->

</details>

<details>
<summary>Previous workflow, retained while its last inference job finishes</summary>

<!-- BEGIN EXECUTABLE BENCHMARK WORKFLOW -->
## Executable benchmark workflow

Observed 2026-10-18T14:08:16.399441+00:00. 139 succeeded; 4 failed; 0 running; 46 pending; 136 blocked; 0 unknown.
Run status: `drained_for_operational_update`. Physical GPU: 0.

[Current queue details](research/benchmark_execution_current.json) · [Authenticated matched scores](research/current_matched_benchmark_results.json)

| Stage | Succeeded | Failed | Running | Pending | Blocked | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ap_features | 0 | 0 | 0 | 6 | 2 | 0 |
| ap_fit | 0 | 0 | 0 | 0 | 1 | 0 |
| ap_generate | 9 | 0 | 0 | 0 | 0 | 0 |
| ap_infer | 0 | 3 | 0 | 6 | 0 | 0 |
| ap_predict | 0 | 0 | 0 | 0 | 5 | 0 |
| ap_registry | 0 | 0 | 0 | 6 | 2 | 0 |
| ap_train_targets | 0 | 0 | 0 | 0 | 3 | 0 |
| contrastive | 0 | 0 | 0 | 0 | 15 | 0 |
| embeddings | 0 | 0 | 0 | 0 | 15 | 0 |
| external | 6 | 0 | 0 | 0 | 0 | 0 |
| factory | 0 | 0 | 0 | 0 | 17 | 0 |
| fit_factory | 0 | 0 | 0 | 1 | 2 | 0 |
| focal_factory | 35 | 0 | 0 | 0 | 0 | 0 |
| generation_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| inference_factory | 29 | 0 | 0 | 0 | 0 | 0 |
| mlm | 0 | 1 | 0 | 0 | 14 | 0 |
| native | 15 | 0 | 0 | 0 | 0 | 0 |
| prediction_factory | 0 | 0 | 0 | 15 | 2 | 0 |
| prepare | 16 | 0 | 0 | 0 | 14 | 0 |
| release_archive | 0 | 0 | 0 | 0 | 4 | 0 |
| release_c4_registry | 0 | 0 | 0 | 6 | 2 | 0 |
| release_c4_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_c4_targets | 0 | 0 | 0 | 0 | 3 | 0 |
| release_fit | 0 | 0 | 0 | 0 | 1 | 0 |
| release_frame | 0 | 0 | 0 | 6 | 3 | 0 |
| release_infer | 0 | 0 | 0 | 0 | 5 | 0 |
| release_native6_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_native_targets | 0 | 0 | 0 | 0 | 4 | 0 |
| release_seal | 0 | 0 | 0 | 0 | 5 | 0 |
| retrieval | 0 | 0 | 0 | 0 | 15 | 0 |

Running workflow jobs: none.
Named external queue jobs running: none. External queue states are listed separately from workflow counts.

No supported authenticated matched score has completed in this workflow.
<!-- END EXECUTABLE BENCHMARK WORKFLOW -->

</details>

[Queue and current workstreams](docs/benchmark_queue.md) · [Executable dependency graph](configs/benchmarks/executable_workflow_20261018_v7.json)

The maintained interface is [src/arggnn/benchmarking](src/arggnn/benchmarking). It separates method adapters, biological protocols, observed-only data descriptors, single-GPU scheduling and common scoring. Original author code and immutable experiment records stay in place.

- [Workflow and release procedure](docs/benchmark_workflow.md)
- [Method implementations and adaptations](docs/benchmark_methods.md)
- [Paper recipes and remaining scientific gaps](docs/paper_benchmark_recipes.md)
- [Model lineage and lessons](docs/model_history.md)
- [Historical reconstruction protocols](configs/benchmarks/protocols)
- [Current exact-job execution policy](configs/benchmarks/execution_policy.json)
- [Complete prior research history](docs/research_history_before_benchmark_refactor_20261018.md)

## Inspect the plan

On anogena, inside the dedicated repository:

```bash
source scripts/project_env.sh
export PYTHONPATH=.:src:scripts
.runtime/env/bin/python -B -s -m arggnn.benchmarking methods
.runtime/env/bin/python -B -s -m arggnn.benchmarking paper-recipes
.runtime/env/bin/python -B -s -m arggnn.benchmarking paper-plan ancestralpaths_european
.runtime/env/bin/python -B -s -m arggnn.benchmarking status
```

The new paper plans generate each independent simulation once and share its observations, reference/query panel and frozen evaluation targets across methods. They preserve the native three-source, four-source and six-path tasks separately. Fresh seeds are permitted. The AP generator now supports the paper's 200 Mb length, complete age roster and 170-generation BAA event. Other material publication gaps remain listed in the plan and [recipe notes](docs/paper_benchmark_recipes.md).

These commands inspect small metadata. They do not train models or load scientific datasets. The plan reports missing data products, method constraints and unresolved publication details. The current continuation runs fresh native comparators while TreeLoom preprocessing is repaired. The complete American, AncestralPaths and ARGMix workflows are integrated, including shared scoring and current empirical application. Historical admissions do not authorize new launches.

## Model advantages and evidence

TreeLoom preserves complete local trees, genomic order and persistent sample identity. The selected model uses bounded GPU windows for learned history. Its replacement fit uses the repaired full-size CPU preparation and preserves the frozen training schedule.

On eight reused European development simulations, TreeLoom achieved 97.0778 ± 0.5674% modern span accuracy, compared with 94.4461 ± 1.1491% for the released ARGMix pipeline and 93.7830 ± 0.8385% for the derived four-source AncestralPaths endpoint. These are common-data development results with disclosed differences from the publications. TreeLoom-NoMem has higher boundary F1. No overall top-1 claim is established.

Native FLARE previously completed five American validation parents. The new independent American comparison is reported separately in the current workflow results. American three-source ancestry and European four-source ancestry are separate protocols. GST is an adapted architecture control rather than a native LAI competitor.

<!-- FROZEN_BENCHMARK_PROGRESS_START -->
## Launch snapshot and earlier results

Updated UTC: 2026-10-18T12:53:52.089475+00:00. Physical GPU 0 only.

| Stage | Current verified status |
|---|---|
| Execution | Released by the user. Physical GPU0 only. Exact runnable jobs and dependent stages are in the maintained queue. |
| Native ARGformer | Fresh native ARGformer passed the repaired DataSpec check and reached trainer.fit, then our supervisor stopped it at the configured256GiB summed RSS cap. Optimizer update count is unverified. Memory accounting and group cleanup need repair. |
| Native follow-on | Earlier exposed-validation chain completed all four embedding stages and native retrieval with actual success. It was not scored and is separate from the fresh comparison. |
| Fresh American data | All35 independent30Mb simulations, shared inferred trees and fixed focal preparations completed. |
| Frozen TreeLoom | All three fresh fit tasks explicitly held for preprocessing repair. Stopped American first-parent preparation after more than four hours with0 optimizer updates. Preserve full20parents/30passes/600updates. |
| European comparisons | Three European inference stages are active on the completed simulations. The metadata-cap correction passed startup. No fresh European training or score yet. |
| Native FLARE | Native FLARE completed all15 fresh American validation/test predictions. Paired scoring waits for all methods. Earlier exposed validation scores remain historical. |

**Native FLARE, five American validation simulations:** accuracy 98.0960 ± 0.3442% and macro F1 97.9196 ± 0.4059%.

| Population | Precision % | Recall % | F1 % |
|---|---:|---:|---:|
| EUR | 97.3671 ± 0.5621 | 97.7505 ± 0.4517 | 97.5579 ± 0.4452 |
| EAS | 98.5230 ± 0.2364 | 98.6975 ± 0.2639 | 98.6101 ± 0.2463 |
| AFR | 98.2167 ± 0.5343 | 96.9760 ± 1.0077 | 97.5909 ± 0.6883 |

Equal weight per simulation, 10,000 fixed focal records each. SD is across five simulation parents, with ddof 1. VAL000 was previously exposed. This is a common-data development comparison, with no exact publication-table reproduction claimed.
[Complete aggregate](experiments/results/FLARE_original5VAL_aggregate_root02_memo_oct17/FLARE_original5VAL_descriptive_aggregate.json).

The substantial TreeLoom schedule is 20 full 30 Mb TRAIN parents and 30 passes, or 600 complete-parent updates from fresh weights. A disposable one-update check is not representative training. No overall top-1 claim is established.
<!-- FROZEN_BENCHMARK_PROGRESS_END -->

<!-- CURRENT_SCOREBOARD_START -->
## Completed comparison

These previously completed development results are preserved separately from the current matched benchmark results in the executable workflow report.

European development panel: eight reused simulation parents. Values are equal-parent means ± sample SD. These common-data results do not establish exact publication reproduction or independent test performance.

| Method | Present-day span accuracy % ↑ | NLL ↓ | Boundary F1 at 1 kb % ↑ |
|---|---:|---:|---:|
| TreeLoom | **97.0778 ± 0.5674** | **0.088115 ± 0.021993** | 18.3807 ± 1.3141 |
| TreeLoom-NoMem | 97.0000 ± 0.4978 | 0.088541 ± 0.019128 | **19.0488 ± 1.4275** |
| Released ARGMix | 94.4461 ± 1.1491 | 0.219695 ± 0.023975 | 11.7372 ± 1.7328 |
| AncestralPaths, derived C4 | 93.7830 ± 0.8385 | ∞ | 7.0281 ± 0.7779 |

TreeLoom-NoMem removes the persistent sample memory and retains the matched continued-training control. AncestralPaths C4 is an explicit collapse of six native path classes. Native hard calls and raw zero posterior probabilities are retained.

Native FLARE's earlier American three-source results appear above. The executable workflow report tracks new independent comparisons separately. GST expanded fitting is outside the current LAI queue.

[All recorded setup and cohort metrics](research/current_scoreboard.json) · [Prior research and results](docs/research_history_before_benchmark_refactor_20261018.md)
<!-- CURRENT_SCOREBOARD_END -->

## Paper and complete results

The manuscript is a working AAAI draft. Benchmarking is incomplete.

[Working PDF](paper/current_full_panel_working.pdf) · [LaTeX](paper/anchor_kernel_working.tex) · [Overleaf bundle](paper/current_full_panel_overleaf_working.zip)

[Detailed current scores](research/current_scoreboard.json) · [Experiment ledger](research/experiment_ledger.md) · [All recorded metric receipts](research/all_recorded_metric_receipts.json) · [Research history](docs/research_history_before_benchmark_refactor_20261018.md)

Metrics retain their original support, population definitions and provenance. Sample SD across simulations and variation across optimizer seeds are different quantities. The benchmark interface rejects missing evaluation targets and keeps native hard calls and zero probabilities intact.



