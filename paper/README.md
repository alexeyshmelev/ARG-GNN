# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf source bundle](current_full_panel_overleaf_working.zip).

The ten-page AAAI working draft focuses on **local ancestry inference (LAI)** and the final **TreeLoom** model. Main text and references finish on page 7. The scientific supplement occupies pages 8–10.

Figure 1 on page 3 uses a genuine small ARG and its matching local trees. It shows observed features, two graph passes with shared weights, memory along sampled haplotypes, the genomic BiGRU, concatenation and ancestry output. It has panel letters and short module labels, with definitions in the caption. Feature values and predictions are schematic. [Figure](figures/TreeLoom_architecture.png) · [Editable SVG and generator](figure_redesign_20261016_v4/).

Two fresh agents assessed the image without manuscript, caption or conversation context. Their findings led to corrections to weight sharing, residual and classifier paths, recombination connectivity and the illustrative output. Exact memory mathematics remains in the Methods. [Assessment and corrections](../research/reviews/paper_scientific_rewrite_20261016_v3/fresh_image_only_reviews.root_summary_v1.json).

The main tables compare TreeLoom with completed author pipelines, report modern and ancient cohorts, and isolate the effect of added memory. The developmental model catalogue is removed. Best comparable values are bold, including NoMem's higher boundary F1. Additional conceptual controls appear in the supplement. Numerical experiment results are unchanged.

The canonical manuscript and independently extracted Overleaf bundle compile successfully. All ten pages were visually inspected, and all ten bundle pages match the canonical render pixel for pixel. [Build and integration](../research/reviews/paper_scientific_rewrite_20261016_v3/final_integration.root_v1.json).

This remains a development manuscript. The eight European validation simulations were reused for model selection. The evaluated ARGMix release and AncestralPaths archive differ materially from their publication recipes. Complete native American comparisons, larger European comparisons, independent tests, useful architecture novelty and biological corroboration remain unfinished. Scientific TODOs identify missing material details.

[Scientific status](CURRENT_FULL_PANEL_README.md) · [External provenance](current_full_panel_provenance.json) · [Current detailed scores](../research/current_scoreboard.md) · [Comparator audit](../research/reviews/competitor_fidelity_user_audit_20261016_v1/REPORT.md).

Earlier anchor_kernel_working.tex, boundary_kernel_working.tex and methods_working.tex are historical drafts. All research and manuscript files remain in the dedicated server repository.
