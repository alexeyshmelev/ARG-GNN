# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf source bundle](current_full_panel_overleaf_working.zip).

The eleven-page AAAI draft presents **TreeLoom** for **local ancestry inference (LAI)**. Main text and references occupy pages 1–7. The scientific supplement occupies pages 8–11.

Figures 1 and 2 were redesigned on 26 October. The overview retains the biological inputs and prediction readout, shows all four tree traversals in two rounds, and draws separate haplotype memory tracks across trees. The companion figure expands the exact node updates and structural corrections. Every color has a consistent meaning and a visible legend. [Overview](figures/TreeLoom_architecture.png) · [Node operations](figures/TreeLoom_operators.png) · [Generator and review](figure_redesign_20261026_v2/README.md).

The main tables compare the selected model with completed competitors and report query cohorts. The matched TreeLoom-NoMem comparison and detailed scientific controls are in the supplement. Comparable winners are bold, including the control's higher boundary F1. Developmental variants remain in the experiment records.

The introduction states the biological problem and research gap. Methods explain the observed features and model mathematically. The supplement contains demographic assumptions, sampling ages, simulation parameters, meaningful training details and comparator differences. Material unresolved simulation or evaluator details remain visible scientific TODOs. Execution paths and audit records stay in the repository.

The manuscript and extracted Overleaf bundle compile to eleven pages, with no errors, undefined references, overfull boxes or oversized floats. Figures 1 and 2 occupy pages 3 and 4. Three underfull-box diagnostics remain and were visually inspected. All eight result tables are unchanged. [Build and review record](figure_redesign_20261026_v2/integration.json).

The manuscript remains a development study. The eight European validation simulations were reused for model selection, with one optimizer seed. The evaluated ARGMix release and AncestralPaths archive differ from the publication recipes. Native American comparisons, larger panels, independent tests, useful architecture novelty and stronger biological corroboration remain unfinished. No overall top-1 claim is established. The completed Chickenpox supplement reports lower RMSE but worse MAE after adding sample memory, failing the registered utility criterion.

[Detailed scores](../research/current_scoreboard.md) · [Comparator fidelity](../research/reviews/competitor_fidelity_user_audit_20261016_v1/comparison_matrix.md) · [Provenance](current_full_panel_provenance.json).
