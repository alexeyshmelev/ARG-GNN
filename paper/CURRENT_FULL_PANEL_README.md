# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf bundle](current_full_panel_overleaf_working.zip) · [Figure 1](figures/TreeLoom_architecture.png).

The twelve-page AAAI draft presents TreeLoom for local ancestry inference (LAI). Main text and references occupy pages 1–8. Scientific supplementary material occupies pages 9–12.

The narrative introduces the biological question, related work, model, evaluation and results. Two main result tables compare the selected TreeLoom model with completed competitors and report query cohorts. Detailed memory controls are in the supplement. Their small accuracy gain and lower boundary F1 remain explicit in the main text. All eight tables retain their numerical entries and bold comparable winners.

Figures 1–3 were redesigned on 26 October. The overview shows node embeddings through two complete tree rounds and the genomic readout. Separate figures explain haplotype memory and the exact learned node operations. Colored layer blocks distinguish learned and fixed operations, feature cells show illustrative values, and every color has a legend. [Architecture](figures/TreeLoom_architecture.png) · [Memory](figures/TreeLoom_memory.png) · [Node operations](figures/TreeLoom_operators.png) · [Generator and review](figure_redesign_20261026_v3/README.md).

The manuscript and extracted Overleaf bundle compile to twelve pages, with no errors, undefined references, overfull boxes or oversized floats. Figures 1–3 occupy pages 3–5. Three underfull diagnostics remain and were visually inspected. All eight result tables are unchanged. [Build and review record](figure_redesign_20261026_v3/integration.json).

This is a development manuscript. Its eight European validation simulations were reused for model selection, with one optimizer seed. The evaluated releases differ materially from publication recipes. Independent tests, complete American comparisons, useful architecture novelty and biological corroboration remain unfinished. No overall top-1 claim is established.

[Current detailed scores](../research/current_scoreboard.md) · [Provenance](current_full_panel_provenance.json).
