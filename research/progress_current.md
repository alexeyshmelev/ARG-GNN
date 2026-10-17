# Research progress

Updated UTC: 2026-10-17T08:26:54.239487+00:00.

TreeLoom performs local ancestry inference (LAI) from complete inferred genealogies. Figure 1 now shows the observed input vectors, two shared GNN passes, sample memory with explicit group-to-sample and sample-to-group routing, aligned haplotype embeddings, a final BiGRU and ancestry output. Fresh readers recovered the high-level computation from the image alone. The final diagram has explicit broadcast/reduction glyphs, signed subtraction and clear whole-array entry brackets. The nine-page AAAI draft and Overleaf bundle have matching rendered pages. The main paper presents the final model, completed comparators and one meaningful memory ablation.

| European development pipeline | Modern span accuracy (%) | Boundary F1 (%) |
|---|---:|---:|
| TreeLoom | **97.0778 ± 0.5674** | 18.3807 ± 1.3141 |
| TreeLoom-NoMem | 97.0000 ± 0.4978 | **19.0488 ± 1.4275** |
| Current ARGMix release | 94.4461 ± 1.1491 | See detailed scoreboard |
| Released dense AncestralPaths, derived four source endpoint | 93.7830 ± 0.8385 | See detailed scoreboard |

These are eight reused 2 Mb validation simulations and one optimizer seed. SD is across simulations in percentage points. Inputs, training and selection differ between pipelines. Exact publication reproduction, independent generalization, architecture attribution and overall top-1 remain unestablished.

| Current work | Verified status | Remaining result |
|---|---|---|
| Native American ARGformer | All five MLM epochs and 227,990 saved AdamW updates across 77 states verified. Producer completed. Actual worker0/drained, supervisor1 from postfit identity race, independent artifact audit complete | Saved-MLM downstream v2 passed independent source review. Current route and capacity integration, original contrastive fit, all embeddings, retrieval and ancestry scores remain |
| Earlier American backbone | 600 saved updates accepted. All five original 30 Mb observation plans complete. 24/30 prediction products authenticated | Complete all 30 before VAL truth |
| Full American TreeLoom/NoMem, H060 | Corrected full run is preparing observations. Both obsolete duplicates are stopped and drained. No new optimizer update verified | Complete 1,200 additional updates and all 60 predictions before VAL truth |
| Native FLARE | VAL000: 98.0100% focal accuracy and 97.7151% macro F1 over 10,000 records. Full VAL001 conversion completed and closed. Native inference admitted and executing | Complete remaining original cells and the common five-parent summary |
| Larger ARGMix panel | Full 1,611-diploid observations accepted. Relate inference reached its operational limit | Complete inference and common training/evaluation |
| Independent European test, H059 | First reserved parent has generation and observed Relate completion. Original worker is absent, but its failed supervisor did not preserve worker wait or descendant drain | Independently assess saved observed data, finish all eight parents and seal every prediction before truth |
| European marker experiment, H058 | GPU 0 v5 stopped on an RSS monitor process-exit race. Outer 241, worker −15 and owned drain verified. No new optimizer update verified | Review a narrow operational repair, then complete the original 32-parent, 1,920-update experiment |
| Ordered marker extension | Original scientific CPU execution remains unfinished, with no accuracy result | Full-parent learning evidence before any utility claim |
| Biological corroboration | Danish predictions, PCA and genotype contrasts completed. Descriptive, without known ancestry truth. Registered genotype directions unsupported | Stronger independent biological evidence |

The completed Chickenpox trial used six 40-epoch fits and 4,800 updates. On the fixed TEST of 79 weeks and 20 counties, Full MAE is 13.907918 ± 0.242561 cases and NoMem MAE is 13.648097 ± 0.050932. Full RMSE is **24.268971 ± 0.094496**, and NoMem RMSE is 24.408159 ± 0.106380 cases. SD spans three initialization seeds. The registered memory utility criterion failed. This is a forecasting trial, not an ancestry result.

A distinct general graph operator and useful broad transfer remain unestablished. The native same-data comparisons, independent tests and stronger biological evidence remain necessary before submission.

[Detailed scores](current_scoreboard.md) · [Verified final native training](reviews/research_execution_ROOT_20261017_v17/verified_training_milestone.root_v1.json) · [Paper](../paper/current_full_panel_working.pdf) · [Figure](../paper/figures/TreeLoom_architecture.png) · [Overleaf bundle](../paper/current_full_panel_overleaf_working.zip).
