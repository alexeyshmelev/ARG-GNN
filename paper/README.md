# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf source bundle](current_full_panel_overleaf_working.zip).

The nine-page AAAI draft names our model **TreeLoom**, with defined suffixes for its developmental variants and matched controls. Main text and references finish on page 7. The scientific supplement occupies pages 8–9.

Figure 1 has its own page 3. A code-generated, fully recorded illustrative ARG yields three genuine marginal trees. The vector diagram shows observed inputs, symbolic matrix shapes, two shared whole-tree sweeps, structural residuals, sample memory, sampled-state projection, one bidirectional GRU layer and the ancestry readout. The output tracts are schematic. [Figure](figures/TreeLoom_architecture.png) · [Editable SVG](figure_redesign_20261016_v3/TreeLoom_architecture.svg) · [Generator and toy genealogy](figure_redesign_20261016_v3/).

The integrated manuscript and extracted Overleaf bundle compile without warnings and produce identical document text/layout. All nine final pages were rendered and visually inspected. The bundle contains portable LaTeX, bibliography, official AAAI assets, the vector figure and short compilation instructions.

The figure and naming revision changes no numerical experiment results. Best comparable table values are bold, including baseline winners. [Exact model names and original conditions](../research/reviews/paper_fig_naming_revision_20261016_v2/model_names_and_table_bindings.json).

This remains a development manuscript. The eight European validation simulations were reused for model selection. The evaluated ARGMix release uses 15 references and no ancestors, while the paper describes 25 references and three ancestors. The released AncestralPaths dense classifier and derived four-source endpoint differ from its published convolutional/path task. Original-paper configurations, complete native American comparisons, independent confirmation and useful architectural novelty remain unfinished and visible as research TODOs.

[Scientific status](CURRENT_FULL_PANEL_README.md) · [External provenance](current_full_panel_provenance.json) · [Independent comparator audit](../research/reviews/competitor_fidelity_user_audit_20261016_v1/REPORT.md) · [Final build and visual verification](../research/reviews/paper_fig_naming_revision_20261016_v2/final_integration.root_v1.json).

Earlier anchor_kernel_working.tex, boundary_kernel_working.tex and methods_working.tex are historical drafts. The current paper is current_full_panel_working.tex. All research and manuscript files remain in the dedicated server repository.
