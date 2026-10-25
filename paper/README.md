# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf source bundle](current_full_panel_overleaf_working.zip).

The ten-page AAAI draft presents **TreeLoom** for **local ancestry inference (LAI)**. Main text and references occupy pages 1–6. The scientific supplement occupies pages 7–10.

Figure 1 was redesigned on 26 October. It shows ordered local trees, observed inputs, two shared GNN applications, memory following sampled haplotypes, and the separate recurrent ancestry readout. Ancestry colors have consistent meanings throughout. Every color and the query symbol has a legend entry. A fresh reader checked the image without manuscript or implementation context. [Figure](figures/TreeLoom_architecture.png) · [Editable SVG](figures/TreeLoom_architecture.svg) · [Generator and review](figure_redesign_20261026/README.md).

The main tables compare the selected model with completed competitors and report query cohorts. The matched TreeLoom-NoMem comparison and detailed scientific controls are in the supplement. Comparable winners are bold, including the control's higher boundary F1. Developmental variants remain in the experiment records.

The introduction states the biological problem and research gap. Methods explain the observed features and model mathematically. The supplement contains demographic assumptions, sampling ages, simulation parameters, meaningful training details and comparator differences. Material unresolved simulation or evaluator details remain visible scientific TODOs. Execution paths and audit records stay in the repository.

The manuscript and extracted Overleaf bundle compile to ten pages without warnings or overfull boxes. Figure 1 is on page 3. The final figure and manuscript contact sheet were visually inspected. All result tables retain their previous values. [Build and review record](figure_redesign_20261026/integration.json).

The manuscript remains a development study. The eight European validation simulations were reused for model selection, with one optimizer seed. The evaluated ARGMix release and AncestralPaths archive differ from the publication recipes. Native American comparisons, larger panels, independent tests, useful architecture novelty and stronger biological corroboration remain unfinished. No overall top-1 claim is established. The completed Chickenpox supplement reports lower RMSE but worse MAE after adding sample memory, failing the registered utility criterion.

[Detailed scores](../research/current_scoreboard.md) · [Comparator fidelity](../research/reviews/competitor_fidelity_user_audit_20261016_v1/comparison_matrix.md) · [Provenance](current_full_panel_provenance.json).
