# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf bundle](current_full_panel_overleaf_working.zip) · [Figure 1](figures/TreeLoom_architecture.png).

The ten-page AAAI draft presents TreeLoom for local ancestry inference (LAI). Main text and references occupy pages 1–6. Scientific supplementary material occupies pages 7–10.

The narrative introduces the biological question, related work, model, evaluation and results. Two main result tables compare the selected TreeLoom model with completed competitors and report query cohorts. Detailed memory controls are in the supplement. Their small accuracy gain and lower boundary F1 remain explicit in the main text. All eight tables retain their numerical entries and bold comparable winners.

Figure 1 was redesigned on 26 October. It shows ordered local trees, observed inputs, two shared GNN applications, memory following sampled haplotypes, and the separate recurrent ancestry readout. Ancestry colors have consistent meanings throughout. Every color and the query symbol has a legend entry. A fresh reader checked the image without manuscript or implementation context. [Figure](figures/TreeLoom_architecture.png) · [Editable SVG](figures/TreeLoom_architecture.svg) · [Generator and review](figure_redesign_20261026/README.md).

The manuscript and extracted Overleaf bundle compile to ten pages without warnings or overfull boxes. Figure 1 is on page 3. The final figure and manuscript contact sheet were visually inspected. All result tables retain their previous values. [Build and review record](figure_redesign_20261026/integration.json).

This is a development manuscript. Its eight European validation simulations were reused for model selection, with one optimizer seed. The evaluated releases differ materially from publication recipes. Independent tests, complete American comparisons, useful architecture novelty and biological corroboration remain unfinished. No overall top-1 claim is established.

[Current detailed scores](../research/current_scoreboard.md) · [Provenance](current_full_panel_provenance.json).
