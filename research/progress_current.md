# Research progress

Updated UTC: 2026-10-17T00:22:08.715643+00:00.

TreeLoom is the final European development model. It processes whole inferred local trees, carries memory for persistent sampled haplotypes and uses a genomic BiGRU. The main paper now defines local ancestry inference (LAI), explains the biological task and keeps completed competitor comparisons plus one meaningful memory control in the main results.

| European development pipeline | Modern span accuracy (%) | Boundary F1 (%) |
|---|---:|---:|
| TreeLoom | **97.0778 ± 0.5674** | 18.3807 ± 1.3141 |
| TreeLoom-NoMem | 97.0000 ± 0.4978 | **19.0488 ± 1.4275** |
| Current ARGMix release | 94.4461 ± 1.1491 | See detailed scoreboard |
| Released AncestralPaths, derived four source output | 93.7830 ± 0.8385 | See detailed scoreboard |

These are eight reused 2 Mb validation simulations and one optimizer seed. SD is across simulations. Method inputs, training and selection differ. The completed comparison does not establish exact publication reproduction, independent generalization, architecture-only attribution or an overall method ranking.

| Current work | Actual status | Remaining result |
|---|---|---|
| Native American ARGformer | Running on GPU 0. Latest stable saved audit verifies 91,000 of 227,990 AdamW updates across all 77 states, with one completed epoch | Finish native fit and downstream ancestry scoring |
| Earlier American backbone | 600 saved updates accepted. All five original 30 Mb plans completed and closed, with five prediction supervisors launched | All 30 predictions before truth |
| Original American VAL004 | Native paths and final observation plan completed and closed. Prediction supervisor launched | Complete prediction seals |
| Full American TreeLoom versus NoMem, H060 | The scientific repair for all-isolated zero-SNP inferred intervals and its fresh finite CPU wrapper both passed independent source review. All samples, intervals and the original 1,200-update comparison are retained. Fresh admission is pending. No American TreeLoom accuracy is available. | Complete the original 600 updates per arm and all 60 predictions before VAL truth |
| Native FLARE | One VAL parent and 10,000 fixed focals: 98.0100% accuracy and 97.7151% macro F1. Four remaining cells have reviewed completion source | Run remaining native cells and report five-parent mean and sample SD |
| Larger ARGMix comparison | Full 1,611-diploid observed data accepted. Full Relate inference stopped at its 21,600 s limit | Complete the inferred bank before any score |
| Independent European test, H059 | Five methods and eight reserved simulations frozen. Generation pending | All 40 primary and eight secondary products before truth |
| Cross-domain, H061 | All six Chickenpox fits completed successfully with 40 epochs and 800 updates each, for 4,800 updates total. Actual worker and supervisor returned 0 with owned drain, and ROOT accepted TRAIN completion. Selected VAL MAE is 15.968411 ± 0.061488 cases for TreeLoom and 16.059604 ± 0.075709 for NoMem, with sample SD across three initialization seeds. Comparator freeze, TEST predictions and TEST scoring remain pending. These are forecasts, not ancestry accuracy. | Freeze comparators, then separate TEST prediction and scoring |
| Real data | Danish predictions, PCA and genotype contrasts completed. Descriptive, without ancestry truth. Genotype contrasts did not support registered directions | Stronger independent biological corroboration |

The ten-page AAAI draft and Overleaf source compile, with Figure 1 on page 3. The current v6 figure has matching ARG slices and local trees, explicit vectors, shared graph stages, sample memory and a separate BiGRU. A new image-only reader recovered the broad pipeline and found no major overlap or sample-label mismatch. The v6 memory inset now distinguishes current and historical context, with a fresh reader recovering its role before the second graph pass. The caption and equations supply the exact update. Experimental values are unchanged.

Useful operator novelty remains unestablished. Completing native benchmarks and independent validation remains necessary for a submission. No new American GNN or ARGformer ancestry score is available.

[Detailed scores](current_scoreboard.md) · [Latest execution report](reviews/user_progress_20261017_v1/REPORT.md) · [Experiment decisions](experiment_decisions.json) · [Paper](../paper/current_full_panel_working.pdf) · [Overleaf source](../paper/current_full_panel_overleaf_working.zip)

[H061 selected validation scores, per-seed epochs and update counts](reviews/H061_training_milestone_ROOT_20261017_v1/REPORT.md).

The full same-marker experiment is launched with its complete 1,920-update budget. [H061 six-fit milestone](reviews/H061_training_milestone_ROOT_20261017_v2/REPORT.md). [Independent topology-design no-go](reviews/topology_innovation_candidate_20261017_v1/REPORT.md).
