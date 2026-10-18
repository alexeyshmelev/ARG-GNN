# Current benchmark status
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

[All recorded setup and cohort metrics](current_scoreboard.json) · [Prior research and results](../docs/research_history_before_benchmark_refactor_20261018.md)


<!-- FROZEN_BENCHMARK_PROGRESS_START -->
## Launch snapshot and earlier results

Updated UTC: 2026-10-18T02:06:28.979837+00:00. Physical GPU 0 only.

| Stage | Current verified status |
|---|---|
| Execution | Released by the user. Physical GPU0 only. Exact runnable jobs and dependent stages are in the maintained queue. |
| Native ARGformer | Native ARGformer saved all2,135optimizer updates. Existing embedding/retrieval continuation remains authorized. |
| Native follow-on | Four complete embedding stages and native retrieval follow genuine contrastive completion through the finite chain. No query truth or ancestry scoring in this chain. |
| Fresh American data | All six initial 30 Mb TRAIN simulations and shared genealogy inferences completed with actual successful waits. The new workflow adopts them and prepares the remaining 29 parents. |
| Frozen TreeLoom | Complete fresh600-update fit, finalcheckpoint prediction, all45heldout prediction barrier and common scoring implemented. CPU focused tests pass. Fullfit awaits20sharedinputs andGPUqueue. |
| European comparisons | Complete AP200Mb native six-path comparison and available-release ARGMix four-source comparison integrated, with separate four-head TreeLoom training and all-prediction TEST barriers. |
| Native FLARE | Previously completed five 30 Mb validation predictions and scores retained below. |

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
