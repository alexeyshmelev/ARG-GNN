# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf source bundle](current_full_panel_overleaf_working.zip).

The AAAI draft presents **TreeLoom** for **local ancestry inference (LAI)**. The manuscript includes the main paper, references and scientific supplement.

Figures 1 and 2 now separate the overall prediction flow from the two context mechanisms. The overview follows one query leaf across three trees to ancestry probabilities. The encoder expansion explains shared tree rounds, clade patterns and leaf memory. Routine operator internals are left to the methods. Colors identify observations, learned blocks, embeddings and probabilities. [Prediction flow](figures/TreeLoom_architecture.png) · [Encoder](figures/TreeLoom_encoder.png) · [Generators, source check and fresh-reader reports](figure_redesign_20261026_v6/README.md).

The main tables compare the selected model with completed competitors and report query cohorts. The matched TreeLoom-NoMem comparison and detailed scientific controls are in the supplement. Comparable winners are bold, including the control's higher boundary F1. Developmental variants remain in the experiment records.

The introduction states the biological problem and research gap. Methods explain the observed features and model mathematically. The supplement contains demographic assumptions, sampling ages, simulation parameters, meaningful training details and comparator differences. Material unresolved simulation or evaluator details remain visible scientific TODOs. Execution paths and audit records stay in the repository.

The manuscript and a freshly extracted Overleaf bundle compile to 10 pages. Both figure pages and the full document were visually inspected. All eight table source blocks are unchanged. [Build and review record](figure_redesign_20261026_v6/integration.json).

The manuscript remains a development study. The eight European validation simulations were reused for model selection, with one optimizer seed. The evaluated ARGMix release and AncestralPaths archive differ from the publication recipes. Native American comparisons, larger panels, independent tests, useful architecture novelty and stronger biological corroboration remain unfinished. No overall top-1 claim is established. The completed Chickenpox supplement reports lower RMSE but worse MAE after adding sample memory, failing the registered utility criterion.

[Detailed scores](../research/current_scoreboard.md) · [Comparator fidelity](../research/reviews/competitor_fidelity_user_audit_20261016_v1/comparison_matrix.md) · [Provenance](current_full_panel_provenance.json).
