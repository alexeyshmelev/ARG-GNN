# Current benchmark status
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

[All recorded setup and cohort metrics](current_scoreboard.json) · [Prior research and results](../docs/research_history_before_benchmark_refactor_20261018.md)


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
[Complete aggregate](../experiments/results/FLARE_original5VAL_aggregate_root02_memo_oct17/FLARE_original5VAL_descriptive_aggregate.json).

The substantial TreeLoom schedule is 20 full 30 Mb TRAIN parents and 30 passes, or 600 complete-parent updates from fresh weights. A disposable one-update check is not representative training. No overall top-1 claim is established.
<!-- FROZEN_BENCHMARK_PROGRESS_END -->
