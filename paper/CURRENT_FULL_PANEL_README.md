# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf bundle](current_full_panel_overleaf_working.zip) · [Figure 1](figures/TreeLoom_architecture.png).

The ten-page AAAI draft presents TreeLoom for local ancestry inference (LAI). Main text and references occupy pages 1–6. Scientific supplementary material occupies pages 7–10.

The narrative introduces the biological question, related work, model, evaluation and results. Two main result tables compare the selected TreeLoom model with completed competitors and report query cohorts. Detailed memory controls are in the supplement. Their small accuracy gain and lower boundary F1 remain explicit in the main text. All eight tables retain their numerical entries and bold comparable winners.

Figures 1 and 2 show the complete prediction path and an expansion of its two-round tree encoder. The diagram explicitly separates all-node embeddings from selected leaf embeddings, follows one leaf across tree rows, appends the three geometry columns, and distinguishes group memory from the final BiGRU. Learned blocks are gold, embeddings teal, observed features gray and probabilities purple, with legends. [Architecture](figures/TreeLoom_architecture.png) · [Tree encoder](figures/TreeLoom_encoder.png) · [Generator and independent review](figure_redesign_20261026_v4/README.md).

The manuscript and a freshly extracted Overleaf bundle compile to ten pages, with no errors, undefined references, overfull boxes or oversized floats. The two figures occupy pages 3 and 4. Five underfull diagnostics remain and the layouts were visually inspected. All eight result tables are unchanged. [Build and review record](figure_redesign_20261026_v4/integration.json).

This is a development manuscript. Its eight European validation simulations were reused for model selection, with one optimizer seed. The evaluated releases differ materially from publication recipes. Independent tests, complete American comparisons, useful architecture novelty and biological corroboration remain unfinished. No overall top-1 claim is established.

[Current detailed scores](../research/current_scoreboard.md) · [Provenance](current_full_panel_provenance.json).
