# Current progress

Updated UTC: 2026-10-16T23:20:46.217488+00:00.

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
| Earlier American backbone | 600 saved updates accepted. All five original 30 Mb plans are closed and all five prediction stages are launched | Complete all 30 predictions before truth |
| Original American VAL004 | Native paths and final observation plan completed with genuine worker 0, owned drain and consumed outer 0. Both ordinary closes succeeded. All five original plans are closed | Predict VAL004, then complete all 30 predictions before truth |
| Full American TreeLoom versus NoMem, H060 | V5 reached scientific startup but failed the single-root check before either arm updated. Worker 1, owned drain and outer 1 recorded. All 20 TRAIN genealogies were checked: 40 isolated intervals without SNPs. A narrow loader and structural compatibility repair is being authored | Run the full 600 additional updates per arm, then all 60 predictions before VAL truth |
| Native FLARE | One VAL parent and 10,000 fixed focals: 98.0100% accuracy and 97.7151% macro F1. Four remaining cells have reviewed completion source | Run remaining native cells and report five-parent mean and sample SD |
| Larger ARGMix comparison | Full 1,611-diploid observed data accepted. Full Relate inference stopped at its 21,600 s limit | Complete the inferred bank before any score |
| Independent European test, H059 | Five methods and eight reserved simulations frozen. Generation pending | All 40 primary and eight secondary products before truth |
| Cross-domain, H061 | Six full 40-epoch fits are registered. Scientific and wrapper reviews passed. Actual CPU admission succeeded and the supervisor launched. Scientific startup is verified. Actual optimizer progress and scores remain pending. Earlier TGB mechanism test was negative | Train all six fits, freeze states and comparators before a separate TEST release |
| Real data | Danish predictions, PCA and genotype contrasts completed. Descriptive, without ancestry truth. Genotype contrasts did not support registered directions | Stronger independent biological corroboration |

The ten-page AAAI draft and Overleaf source compile, with Figure 1 on page 3. The current v6 figure has matching ARG slices and local trees, explicit vectors, shared graph stages, sample memory and a separate BiGRU. A new image-only reader recovered the broad pipeline and found no major overlap or sample-label mismatch. The v6 memory inset now distinguishes current and historical context, with a fresh reader recovering its role before the second graph pass. The caption and equations supply the exact update. Experimental values are unchanged.

Useful operator novelty remains unestablished. Completing native benchmarks and independent validation remains necessary for a submission. No new American GNN or ARGformer ancestry score is available.

[Detailed scores](../../current_scoreboard.md) · [Latest execution report](REPORT.md) · [Experiment decisions](../../experiment_decisions.json) · [Paper](../../../paper/current_full_panel_working.pdf) · [Overleaf source](../../../paper/current_full_panel_overleaf_working.zip)
