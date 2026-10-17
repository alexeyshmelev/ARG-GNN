# Research progress

Updated UTC: 2026-10-17T07:33:28.503664+00:00.

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
| Native American ARGformer | All five MLM epochs and 227,990 saved AdamW updates across 77 states verified. Producer completed. Actual worker0/drained, supervisor1 from postfit identity race, independent artifact audit complete | Acceptance bridge passed independent source review. Original contrastive fit, all embeddings, native retrieval and ancestry scores remain |
| Earlier American backbone | 600 saved updates accepted. All five original 30 Mb observation plans complete. 15/30 prediction products authenticated | Complete all 30 before VAL truth |
| Full American TreeLoom/NoMem, H060 | Full observed preparation running. Zero new updates verified. A legacy serial branch-support computation remains in provider startup | Reviewed exact legacy correction passed actual admission and its full supervisor is launched. No new update verified. Original 1,200 additional updates and all 60 predictions remain required |
| Native FLARE | VAL000: 98.0100% focal accuracy, 97.7151% macro F1 over 10,000 records. Full VAL001 conversion completed with worker0, owned drain and consumed supervisor0 | Complete remaining original cells and five-parent summary |
| Larger ARGMix panel | Full 1,611-diploid observations accepted. Relate inference reached its operational limit | Complete inference and common training/evaluation |
| Independent European test, H059 | First supervisor failed at its cumulative process roster bound. Four native helper waits0 are recorded, with the original worker last observed live and final completion unverified | Resolve original writer state, separately assess saved observed outputs, then all8parents and all48predictions before truth |
| European marker experiment, H058 | All 32 observed TRAIN parents and full packed cache prepared. First aligned and null scheduled updates completed in 2,754 and 2,467 seconds. No full pass verified | CPU run stopped/drained with final totals UNKNOWN. GPU0v3 passed admission but failed a stale startup check before training, with worker1/outer1 and owned drain. The new GPU0 worker returned1 with owned drain and EOF. Its outer terminal and final traceback remain pending, with no new optimizer update verified |
| Ordered marker extension | Original scientific CPU execution remains unfinished, with no accuracy result | Full-parent learning evidence before any utility claim |
| Biological corroboration | Danish predictions, PCA and genotype contrasts completed. Descriptive, without known ancestry truth. Registered genotype directions unsupported | Stronger independent biological evidence |

The completed Chickenpox trial used six 40-epoch fits and 4,800 updates. On the fixed TEST of 79 weeks and 20 counties, Full MAE is 13.907918 ± 0.242561 cases and NoMem MAE is 13.648097 ± 0.050932. Full RMSE is **24.268971 ± 0.094496**, and NoMem RMSE is 24.408159 ± 0.106380 cases. SD spans three initialization seeds. The registered memory utility criterion failed. This is a forecasting trial, not an ancestry result.

A distinct general graph operator and useful broad transfer remain unestablished. The native same-data comparisons, independent tests and stronger biological evidence remain necessary before submission.

[Detailed scores](current_scoreboard.md) · [Verified final native training](reviews/research_execution_ROOT_20261017_v17/verified_training_milestone.root_v1.json) · [Paper](../paper/current_full_panel_working.pdf) · [Figure](../paper/figures/TreeLoom_architecture.png) · [Overleaf bundle](../paper/current_full_panel_overleaf_working.zip).
