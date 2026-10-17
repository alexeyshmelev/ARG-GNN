# ARG-GNN

**TreeLoom** learns local ancestry inference (LAI) from ordered collections of complete genealogical trees. Its GNN processes tree structure and follows persistent haplotypes across genomic intervals.

## Current task: prepare the benchmark codebase

**Benchmark execution is paused pending explicit user approval.** Only small CPU smoke tests are permitted. No full training, simulation, prediction or scoring starts automatically. After approval, all GPU work uses **physical GPU 0**, with the required process title and existing resource guards. Storage remains below the authorized 1 TB limit.

The maintained interface is [src/arggnn/benchmarking](src/arggnn/benchmarking). It separates method adapters, biological protocols, observed-only data descriptors, single-GPU scheduling and common scoring. Original author code and immutable experiment records stay in place.

- [Workflow and release procedure](docs/benchmark_workflow.md)
- [Method implementations and adaptations](docs/benchmark_methods.md)
- [Paper recipes and remaining scientific gaps](docs/paper_benchmark_recipes.md)
- [Model lineage and lessons](docs/model_history.md)
- [Historical reconstruction protocols](configs/benchmarks/protocols)
- [Execution hold](configs/benchmarks/execution_policy.json)
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

These commands inspect small metadata. They do not train models or load scientific datasets. The plan reports missing data products, method constraints and unresolved publication details. Production enqueue also stays closed until the real per-family admission adapters are integrated and reviewed. This release prepares and smoke-tests the interfaces and does not claim launch readiness. A historical admission never overrides the current user hold.

## Model advantages and evidence

TreeLoom preserves complete local trees, genomic order and persistent sample identity. The corrected execution path moves learned history into bounded GPU windows and avoids redundant preparation where measured reuse fits. Small FP32 calculation checks passed. Full-size GPU efficiency remains to be measured after approval.

On eight reused European development simulations, TreeLoom achieved 97.0778 ± 0.5674% modern span accuracy, compared with 94.4461 ± 1.1491% for the released ARGMix pipeline and 93.7830 ± 0.8385% for the derived four-source AncestralPaths endpoint. These are common-data development results with disclosed differences from the publications. TreeLoom-NoMem has higher boundary F1. No overall top-1 claim is established.

Native FLARE has completed the five-parent American comparison. TreeLoom and native ARGformer scores on that same support remain pending. American three-source ancestry and European four-source ancestry are separate protocols. GST is an adapted architecture control rather than a native LAI competitor.

<!-- FROZEN_BENCHMARK_PROGRESS_START -->
## Frozen benchmark execution

Updated UTC: 2026-10-17T21:12:29.696970+00:00. Physical GPU 0 only.

| Stage | Current verified status |
|---|---|
| Benchmark codebase | Maintained method/protocol/data/scheduler/metric interfaces prepared. Synthetic CPU smoke tests only. Scientific admission-family integration and all execution remain blocked pending review and human release. |
| Frozen TreeLoom architecture | Unchanged H050 stable_leaf_memory model. Bounded execution preserves original equations. No architecture search. |
| Full TreeLoom training | Held. Planned 20 independent 30 Mb TRAIN parents, 30 passes and 600 complete-parent updates from fresh weights. The reviewed execution correction still needs full-size GPU fit and throughput qualification. |
| Full-length American data | TRAIN000 ready. TRAIN001 simulation/inference and TRAIN002 simulation completed earlier. Paused simulation and inference controllers have genuine direct wait 1 and their known roots are absent. Files and failures remain preserved. No new generation is authorized. |
| Native ARGformer | The saved MLM checkpoint remains accepted as an initializer. The contrastive-stage source was reviewed and its earlier metadata admission finished. No contrastive worker was launched. Training, embeddings, retrieval and scoring are held. |
| Native FLARE | Completed five 30 Mb validation predictions and their previously authorized scoring, with 50,000 fixed focal targets total. Metrics below. |
| Old short-to-long transfer | H060 global-only v2 failed before optimization at JSON observation-record limit. Failure preserved. No automatic retry. |
| Remaining CPU preparation | Stopped. Simulation and inference controllers returned direct wait 1 after the human pause. The compact controller failed its finite coordinator wait before preparation. No automatic replay or success claim. |
| TreeLoom GPU qualification | Held. The earlier qualification stopped without a verified full-parent optimizer update. Corrected fast preparation and bounded GPU history passed source review and a tiny FP32 calculation check. Full-size GPU work waits for approval. |

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

Benchmark execution is held pending user approval. No scientific jobs are scheduled by this refactor.

European development panel: eight reused simulation parents. Values are equal-parent means ± sample SD. These common-data results do not establish exact publication reproduction or independent test performance.

| Method | Present-day span accuracy % ↑ | NLL ↓ | Boundary F1 at 1 kb % ↑ |
|---|---:|---:|---:|
| TreeLoom | **97.0778 ± 0.5674** | **0.088115 ± 0.021993** | 18.3807 ± 1.3141 |
| TreeLoom-NoMem | 97.0000 ± 0.4978 | 0.088541 ± 0.019128 | **19.0488 ± 1.4275** |
| Released ARGMix | 94.4461 ± 1.1491 | 0.219695 ± 0.023975 | 11.7372 ± 1.7328 |
| AncestralPaths, derived C4 | 93.7830 ± 0.8385 | ∞ | 7.0281 ± 0.7779 |

TreeLoom-NoMem removes the persistent sample memory and retains the matched continued-training control. AncestralPaths C4 is an explicit collapse of six native path classes. Native hard calls and raw zero posterior probabilities are retained.

Native FLARE's completed American three-source results appear above. TreeLoom and native ARGformer on the same American support remain pending. GST expanded fitting is not implemented. There are no new scores from this code preparation.

[All recorded setup and cohort metrics](research/current_scoreboard.json) · [Prior research and results](docs/research_history_before_benchmark_refactor_20261018.md)
<!-- CURRENT_SCOREBOARD_END -->

## Paper and complete results

The manuscript is a working AAAI draft. Benchmarking is incomplete.

[Working PDF](paper/current_full_panel_working.pdf) · [LaTeX](paper/anchor_kernel_working.tex) · [Overleaf bundle](paper/current_full_panel_overleaf_working.zip)

[Detailed current scores](research/current_scoreboard.json) · [Experiment ledger](research/experiment_ledger.md) · [All recorded metric receipts](research/all_recorded_metric_receipts.json) · [Research history](docs/research_history_before_benchmark_refactor_20261018.md)

Metrics retain their original support, population definitions and provenance. Sample SD across simulations and variation across optimizer seeds are different quantities. The benchmark interface rejects missing evaluation targets and keeps native hard calls and zero probabilities intact.
