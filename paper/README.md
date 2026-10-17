# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf source bundle](current_full_panel_overleaf_working.zip).

The nine-page AAAI draft presents **TreeLoom** for **local ancestry inference (LAI)**. Main text and references finish on page 6, and the scientific supplement occupies pages 7–9. The abstract introduces admixture biology, the representation problem, the model and the supported findings. The main methods explain the actual computation, with auxiliary structural formulas in the supplement.

Figure 1 on page 3 is an editable vector schematic. Recorded edges and genomic spans define the toy ARG, whose three extracted trees match exactly. Explicit allele and reference ancestry vectors feed two complete GNN passes. Named sample rows preserve identity through the separate sample memory and final BiGRU, leading to ancestry probabilities and calls. The image has panel letters, short module labels and no titles. A fresh reader interpreted it without manuscript or caption, flagged an input-route ambiguity, and verified its correction. Definitions remain in the caption and Methods. [Figure](figures/TreeLoom_architecture.png) · [SVG and generator](figure_redesign_20261017_v10/) · [Blind review](../research/reviews/figure_final_blind_read_oct17_v9/v10_visual_addendum.md).

Main results compare the selected final model with completed author pipelines and show one meaningful matched memory ablation. The developmental catalogue stays in the experiment record. Best comparable values are bold, including the higher boundary F1 of TreeLoom-NoMem. All scientific table values, equation bodies and visible scientific TODOs are unchanged by the editorial revision.

The canonical source and independently extracted Overleaf bundle compile successfully. All nine pages were visually inspected, and every bundle page is pixel-identical to the canonical render. There are no overfull or unresolved-reference warnings. [Build and integration](../research/reviews/paper_simplification_ROOT_20261017_v9/final_integration.root_v1.json).

This is a development manuscript. The eight European validation simulations were reused for model selection, with one optimizer seed. The evaluated ARGMix release and AncestralPaths archive differ materially from the publication recipes. Native American comparisons, larger European panels, independent tests, useful architecture novelty and biological corroboration remain unfinished. No overall top-1 claim is established.

[Detailed scores](../research/current_scoreboard.md) · [Comparator fidelity](../research/reviews/competitor_fidelity_user_audit_20261016_v1/comparison_matrix.md) · [Provenance](current_full_panel_provenance.json).
