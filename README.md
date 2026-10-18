# ARG-GNN

**TreeLoom** learns local ancestry inference (LAI) from ordered collections of complete genealogical trees. Its GNN processes tree structure and follows persistent haplotypes across genomic intervals.

## Current task: execute the benchmark campaign

**The LAI benchmark uses an automated queue of 325 stages.** It covers shared American and European simulations, full training, native competitors, paired scoring and Danish data application. Current execution and matched results are published below. All computation and Git commands run on anogena, using physical GPU 0 only.

<!-- BEGIN EXECUTABLE BENCHMARK WORKFLOW -->
## Executable benchmark workflow

Observed 2026-10-18T07:17:40.655715+00:00. 58 succeeded; 9 failed; 8 running; 94 pending; 156 blocked; 0 unknown.
Run status: `running`. Physical GPU: 0.

[Current queue details](research/benchmark_execution_current.json) · [Authenticated matched scores](research/current_matched_benchmark_results.json)

| Stage | Succeeded | Failed | Running | Pending | Blocked | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ap_features | 0 | 0 | 0 | 1 | 7 | 0 |
| ap_fit | 0 | 0 | 0 | 0 | 1 | 0 |
| ap_generate | 8 | 0 | 1 | 0 | 0 | 0 |
| ap_infer | 0 | 8 | 0 | 1 | 0 | 0 |
| ap_predict | 0 | 0 | 0 | 0 | 5 | 0 |
| ap_registry | 0 | 0 | 0 | 1 | 7 | 0 |
| ap_train_targets | 0 | 0 | 0 | 0 | 3 | 0 |
| contrastive | 0 | 0 | 0 | 0 | 15 | 0 |
| embeddings | 0 | 0 | 0 | 0 | 15 | 0 |
| external | 6 | 0 | 0 | 0 | 0 | 0 |
| factory | 0 | 0 | 0 | 0 | 17 | 0 |
| fit_factory | 0 | 0 | 0 | 1 | 2 | 0 |
| focal_factory | 13 | 0 | 7 | 15 | 0 | 0 |
| generation_factory | 14 | 0 | 0 | 15 | 0 | 0 |
| inference_factory | 14 | 0 | 0 | 15 | 0 | 0 |
| mlm | 0 | 1 | 0 | 0 | 14 | 0 |
| native | 1 | 0 | 0 | 14 | 0 | 0 |
| prediction_factory | 0 | 0 | 0 | 15 | 2 | 0 |
| prepare | 2 | 0 | 0 | 14 | 14 | 0 |
| release_archive | 0 | 0 | 0 | 0 | 4 | 0 |
| release_c4_registry | 0 | 0 | 0 | 1 | 7 | 0 |
| release_c4_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_c4_targets | 0 | 0 | 0 | 0 | 3 | 0 |
| release_fit | 0 | 0 | 0 | 0 | 1 | 0 |
| release_frame | 0 | 0 | 0 | 1 | 8 | 0 |
| release_infer | 0 | 0 | 0 | 0 | 5 | 0 |
| release_native6_score | 0 | 0 | 0 | 0 | 1 | 0 |
| release_native_targets | 0 | 0 | 0 | 0 | 4 | 0 |
| release_seal | 0 | 0 | 0 | 0 | 5 | 0 |
| retrieval | 0 | 0 | 0 | 0 | 15 | 0 |

Running workflow jobs: `AP__ancestralpaths_european_test_004__ap_generate`, `AM_argformer_american_train_012_focals`, `AM_argformer_american_train_013_focals`, `AM_argformer_american_train_014_focals`, `AM_argformer_american_train_015_focals`, `AM_argformer_american_train_016_focals`, `AM_argformer_american_train_017_focals`, `AM_argformer_american_train_018_focals`.
Named external queue jobs running: none. External queue states are listed separately from workflow counts.

No supported authenticated matched score has completed in this workflow.
<!-- END EXECUTABLE BENCHMARK WORKFLOW -->

[Queue and current workstreams](docs/benchmark_queue.md) · [Executable dependency graph](configs/benchmarks/executable_workflow_20261018_v2.json)

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

These commands inspect small metadata. They do not train models or load scientific datasets. The plan reports missing data products, method constraints and unresolved publication details. The reviewed current worker runs the approved native contrastive job. The complete American, AncestralPaths and ARGMix workflows are integrated, including shared scoring and current empirical application. Historical admissions do not authorize new launches.

## Model advantages and evidence

TreeLoom preserves complete local trees, genomic order and persistent sample identity. The corrected execution path moves learned history into bounded GPU windows and avoids redundant preparation where measured reuse fits. Small FP32 calculation checks passed. Full-size GPU efficiency remains to be measured during the approved campaign.

On eight reused European development simulations, TreeLoom achieved 97.0778 ± 0.5674% modern span accuracy, compared with 94.4461 ± 1.1491% for the released ARGMix pipeline and 93.7830 ± 0.8385% for the derived four-source AncestralPaths endpoint. These are common-data development results with disclosed differences from the publications. TreeLoom-NoMem has higher boundary F1. No overall top-1 claim is established.

Native FLARE previously completed five American validation parents. The new independent American comparison is reported separately in the current workflow results. American three-source ancestry and European four-source ancestry are separate protocols. GST is an adapted architecture control rather than a native LAI competitor.

<!-- FROZEN_BENCHMARK_PROGRESS_START -->
## Launch snapshot and earlier results

Updated UTC: 2026-10-18T02:08:58.968415+00:00. Physical GPU 0 only.

| Stage | Current verified status |
|---|---|
| Execution | Released by the user. Physical GPU0 only. Exact runnable jobs and dependent stages are in the maintained queue. |
| Native ARGformer | Native ARGformer completed all2,135updates with actual success. Native embeddings and retrieval now continue automatically. |
| Native follow-on | Four complete embedding stages and native retrieval follow genuine contrastive completion through the finite chain. No query truth or ancestry scoring in this chain. |
| Fresh American data | All six initial 30 Mb TRAIN simulations and shared genealogy inferences completed with actual successful waits. The new workflow adopts them and prepares the remaining 29 parents. |
| Frozen TreeLoom | Complete fresh600-update fit, finalcheckpoint prediction, all45heldout prediction barrier and common scoring implemented. CPU focused tests pass. Fullfit awaits20sharedinputs andGPUqueue. |
| European comparisons | Corrected release v2 actively simulates eight independent200Mb European parents. Full native6/C4 methods and joint TEST barriers are queued. |
| Native FLARE | Previously completed five 30 Mb validation predictions and scores retained below. |

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
