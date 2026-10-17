# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf source bundle](current_full_panel_overleaf_working.zip).

The ten-page AAAI draft presents **TreeLoom** for **local ancestry inference (LAI)**. Main text and references occupy pages 1–6. The scientific supplement occupies pages 7–10.

Figure 1 uses a recorded toy ARG and three exactly matching marginal trees. Observed allele and ancestry vectors feed two shared GNN passes, a separate sample memory and the final BiGRU. Panel letters replace titles. The current fresh reader recovered the computation from the figure and caption and found no obvious overlap. Small vector cells, pale links and the detailed memory route still require close reading. [Figure](figures/TreeLoom_architecture.png) · [Editable vector and generator](figure_redesign_20261017_v13/) · [Fresh assessment](../research/reviews/paper_simulation_integration_ROOT_20261017_v1/fresh_figure_reader_assessment.root_v1.json).

The main results show the final model, completed comparators and one meaningful matched memory control. Comparable winners are bold, including TreeLoom-NoMem's higher boundary F1. Developmental variants remain in the experiment records.

The latest supplement revision supplies recovered primary-source demographic locations, sampling age ranges, genotype error and coordinate assumptions. It distinguishes the released BAA date from the paper and our study design from the original benchmarks. Material unresolved details remain visible scientific TODOs. [Editorial report](../research/reviews/paper_simulation_integration_ROOT_20261017_v1/REPORT.md).

Both canonical and extracted Overleaf sources compile. All ten rendered pages match. Changed supplement pages 7–10 were visually inspected, while main pages 1–6 are pixel-identical to the previously inspected revision. There are no overfull or unresolved-reference warnings. Four ordinary underfull-spacing warnings remain. [Build and visual verification](../research/reviews/paper_simulation_integration_ROOT_20261017_v1/final_integration.root_v1.json).

The manuscript remains a development study. The eight European validation simulations were reused for model selection, with one optimizer seed. The evaluated ARGMix release and AncestralPaths archive differ from the publication recipes. Native American comparisons, larger panels, independent tests, useful architecture novelty and stronger biological corroboration remain unfinished. No overall top-1 claim is established.

The completed Chickenpox supplement reports lower RMSE but worse MAE after adding sample memory, failing the registered utility criterion.

[Detailed scores](../research/current_scoreboard.md) · [Comparator fidelity](../research/reviews/competitor_fidelity_user_audit_20261016_v1/comparison_matrix.md) · [Provenance](current_full_panel_provenance.json).
