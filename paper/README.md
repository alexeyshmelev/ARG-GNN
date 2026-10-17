# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf source bundle](current_full_panel_overleaf_working.zip).

The nine-page AAAI draft presents **TreeLoom** for **local ancestry inference (LAI)**. Main text and references occupy pages 1–5. The scientific supplement occupies pages 6–9.

Figure 1 uses one recorded toy ARG and three exactly matching marginal trees. It shows observed allele, reference ancestry and node vectors, two complete GNN passes, sample memory, a final BiGRU and ancestry probabilities. Panels have letters and no titles. Two readers with fresh context recovered the main computation from images alone. Exact memory averages and the age cut require the caption. [Figure](figures/TreeLoom_architecture.png) · [Editable vector](figures/TreeLoom_architecture.svg) · [Generator](../research/reviews/figure_v14_redesign_20261017_v1/v14r2/build_figure.py) · [Final blind-reader assessment](../research/reviews/paper_feedback_v14_ROOT_20261017_v1/fresh_image_reader2_ROOT_record.v1.json).

The main tables compare the selected model with completed competitors and report query cohorts. The matched TreeLoom-NoMem comparison and detailed scientific controls are in the supplement. Comparable winners are bold, including the control's higher boundary F1. Developmental variants remain in the experiment records.

The introduction states the biological problem and research gap. Methods explain the observed features and model mathematically. The supplement contains demographic assumptions, sampling ages, simulation parameters, meaningful training details and comparator differences. Material unresolved simulation or evaluator details remain visible scientific TODOs. Execution paths and audit records stay in the repository.

Canonical and extracted Overleaf sources compile. All nine rendered pages match and were visually inspected. Figure 1 is on page 3. [Build and integration verification](../research/reviews/paper_feedback_v14_ROOT_20261017_v1/final_integration.root_v1.json).

The manuscript remains a development study. The eight European validation simulations were reused for model selection, with one optimizer seed. The evaluated ARGMix release and AncestralPaths archive differ from the publication recipes. Native American comparisons, larger panels, independent tests, useful architecture novelty and stronger biological corroboration remain unfinished. No overall top-1 claim is established. The completed Chickenpox supplement reports lower RMSE but worse MAE after adding sample memory, failing the registered utility criterion.

[Detailed scores](../research/current_scoreboard.md) · [Comparator fidelity](../research/reviews/competitor_fidelity_user_audit_20261016_v1/comparison_matrix.md) · [Provenance](current_full_panel_provenance.json).
