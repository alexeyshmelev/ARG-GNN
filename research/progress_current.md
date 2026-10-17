# Research progress

Updated UTC: 2026-10-17T02:57:03.610294+00:00.

TreeLoom processes complete inferred local trees, follows persistent sampled haplotypes and uses a genomic BiGRU for local ancestry inference. The nine-page AAAI draft has a new conceptual abstract and a larger diagram on page 3. Main text and references occupy pages 1–6, with a scientific supplement on pages 7–9.

| European development pipeline | Modern span accuracy (%) | Boundary F1 (%) |
|---|---:|---:|
| TreeLoom | **97.0778 ± 0.5674** | 18.3807 ± 1.3141 |
| TreeLoom-NoMem | 97.0000 ± 0.4978 | **19.0488 ± 1.4275** |
| Current ARGMix release | 94.4461 ± 1.1491 | See detailed scoreboard |
| Released AncestralPaths, derived four source output | 93.7830 ± 0.8385 | See detailed scoreboard |

These are eight reused 2 Mb validation simulations and one optimizer seed. SD is across simulations. Inputs, training and selection differ between pipelines. Exact publication reproduction, independent generalization, architecture attribution and overall top-1 remain unestablished.

| Current work | Actual status | Remaining result |
|---|---|---|
| Native American ARGformer | GPU 0 supervisor remains live. Last stable saved audit verifies 170,000 of 227,990 AdamW updates across all 77 states, observed at 02:01 UTC | Finish native fit and downstream ancestry evaluation |
| Earlier American backbone | 600 saved updates accepted. Five full 30 Mb prediction supervisors launched, with 6/30 products last authenticated | All 30 predictions before VAL truth |
| Full American TreeLoom/NoMem, H060 | Entered the full scientific training routine. No new optimizer counter verified. Original 600 additional complete-parent updates per arm retained | Full 1,200 updates and all 60 predictions before VAL truth |
| Native FLARE | VAL000 has 98.0100% focal accuracy and 97.7151% macro F1 on 10,000 records. Corrected VAL001 conversion has admission 0 and supervisor launched | Complete remaining native cells and five-parent summary |
| Larger ARGMix panel | Full 1,611-diploid observations accepted. Relate inference stopped at its operational limit | Complete inference and common training/evaluation |
| Independent European test, H059 | Five methods and eight reserved simulations frozen, generation pending | All 40 primary and eight secondary predictions before truth |
| Chickenpox, H061 | Six full fits completed, 4,800 updates total. All 15 TEST prediction products completed and authenticated over 79 weeks and 20 counties. Separate scoring admitted and launched | Genuine TEST metrics and registered utility assessment |
| European marker experiment, H058 | Exact metadata-join repair independently accepted. Full 1,920-update run has not restarted | Fresh admission and full registered run |
| Ordered marker extension | Scientific CPU source passed independent review. No execution or efficacy evidence | Full-parent learning evidence before any utility claim |
| Biological evidence | Danish predictions, PCA and genotype contrasts completed. Descriptive, without known ancestry truth. Genotype contrasts did not support registered directions | Independent biological corroboration |

H061 selected VAL MAE is 15.968411 ± 0.061488 cases for TreeLoom and 16.059604 ± 0.075709 for NoMem. SD is across three initialization seeds. These are disease forecasts, not ancestry accuracy, and no TEST metric is yet accepted.

Figure 1 now has explicit observed input vectors, two shared GNN passes, named sample rows, separate sample memory and the final BiGRU. A fresh image-only reader recovered the actual visible computation and verified the corrected input route. All nine PDF pages were inspected. The Overleaf bundle compiles to pixel-identical pages. No experimental table values or equation bodies changed.

Useful operator novelty remains unestablished. Current development leadership does not establish overall top-1. Native benchmarks and independent validation remain necessary.

[Detailed scores](current_scoreboard.md) · [Latest report](../research/reviews/paper_simplification_ROOT_20261017_v9/REPORT.md) · [Paper](../paper/current_full_panel_working.pdf) · [Overleaf bundle](../paper/current_full_panel_overleaf_working.zip).
