# Current progress

Updated UTC: 2026-10-16T19:37:41.687869+00:00.

The ten-page AAAI manuscript and Figure 1 have been rewritten, compiled and visually checked. The paper presents TreeLoom, completed competitor pipelines and one matched memory ablation. Matching trees come from the same simulated ARG. Input vectors, two complete GNN passes, memory before the second pass, genomic BiGRU and LAI outputs are shown. Two fresh image-only readers recovered the broad pipeline. Exact memory mathematics remains in the caption and Methods. Numerical results have not changed.

## Experimental movement

- All five original American 30 Mb validation genealogies are inferred. Native ancestral paths are complete for VAL000–003. VAL003 has genuine worker wait0, owned drain and consumed outer0. Its ordinary close succeeded. VAL004 paths and VAL003/004 GNN plans remain unfinished.
- Our saved American GNN completed 600 updates over 30 epochs on twenty independent 2 Mb training parents. All six saved candidate states are accepted. Independent full30Mb prediction stages for VAL000, VAL001 and VAL002 have been admitted and launched. No complete prediction or ancestry score is claimed. All five parents and all thirty candidate products must be sealed before scoring.
- Native ARGformer retains a verified 22,000/227,990-update checkpoint. Training stopped on a controller timeout. An independent source review found a deterministic guard initialization blocker in the proposed continuation before execution. A narrow repair is in progress.
- The 1,611-diploid Relate run stopped at its runtime cap after 21,695 seconds. Its inner native wait-15, aggregate worker-9, consumed outer1 and owned drain are recorded. There is no complete inferred bank or ancestry score. Completed-stage continuation is being assessed from the actual records.

## Measured European development comparison

Accuracy is mean percent ± sample SD across eight reused 2 Mb validation simulations, with one optimizer seed. Present182, Neo302, Yam26 and All686 identify fixed query cohorts.

| Method | Present182 | Neolithic302 | Yamnaya26 | All686 |
|---|---:|---:|---:|---:|
| TreeLoom | **97.0778 ± 0.5674** | **99.7335 ± 0.0682** | **99.9366 ± 0.1229** | **98.8935 ± 0.2345** |
| TreeLoom-NoMem | 97.0000 ± 0.4978 | 99.6468 ± 0.0595 | 99.9072 ± 0.1399 | 98.8258 ± 0.2215 |
| Current author ARGMix | 94.4461 ± 1.1491 | 99.2955 ± 0.2623 | 99.2218 ± 0.8966 | 97.6130 ± 0.4975 |
| Released Dense AncestralPaths, derived C4 | 93.7830 ± 0.8385 | 97.5693 ± 0.2748 | 95.4303 ± 1.4096 | 95.8071 ± 0.4473 |

TreeLoom exceeds the evaluated ARGMix condition by 2.6317 percentage points. This is a comparison of fitted pipelines with disclosed input, training and release differences, rather than original published-table reproduction or an independent test.

Added sample memory gains only 0.0778 accuracy points over NoMem. Boundary F1 decreases from **19.0488 ± 1.4275%** to **18.3807 ± 1.3141%**. Both runs completed 960 additional updates from the same initializer, with 3,840 cumulative updates each. Independent component utility and new operator novelty remain unproved.

American native FLARE has **98.0100% focal accuracy** and **97.7151% macro F1** on one validation parent, without a multi-parent SD. Native ARGformer and our American GNN have no ancestry score yet. European span accuracy and American focal accuracy cannot be ranked together.

[All cohorts, metrics and training budgets](../../current_scoreboard.md) · [Comparison differences](../../../docs/biological_comparison_scope.md).

## Remaining scientific requirements

Complete the common American comparison, larger European comparison and independent held-out evaluation. Establish a useful architectural improvement beyond standard graph propagation and recurrence. Strengthen biological support using evidence independent of model predictions. Existing real-data PCA and ancestry assignments are descriptive. The Danish genotype contrasts did not support the registered positive hypotheses, and the completed cross-domain trade diagnostic did not support a higher-order advantage. No overall top-1 or submission-ready claim is supported.

[Rewritten PDF](../../../paper/current_full_panel_working.pdf) · [Architecture figure](../../../paper/figures/TreeLoom_architecture.png) · [Overleaf bundle](../../../paper/current_full_panel_overleaf_working.zip) · [Actual paper and fresh-reader verification](../paper_scientific_rewrite_20261016_v3/final_integration.root_v1.json).
