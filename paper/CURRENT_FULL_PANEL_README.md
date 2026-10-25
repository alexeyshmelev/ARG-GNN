# Current scientific working paper

[PDF](current_full_panel_working.pdf) · [LaTeX](current_full_panel_working.tex) · [Overleaf bundle](current_full_panel_overleaf_working.zip) · [Figure 1](figures/TreeLoom_architecture.png).

The eleven-page AAAI draft presents TreeLoom for local ancestry inference (LAI). Main text and references occupy pages 1–7. Scientific supplementary material occupies pages 8–11.

The narrative introduces the biological question, related work, model, evaluation and results. Two main result tables compare the selected TreeLoom model with completed competitors and report query cohorts. Detailed memory controls are in the supplement. Their small accuracy gain and lower boundary F1 remain explicit in the main text. All eight tables retain their numerical entries and bold comparable winners.

Figures 1 and 2 were redesigned on 26 October. The overview retains the biological inputs and prediction readout, shows all four tree traversals in two rounds, and draws separate haplotype memory tracks across trees. The companion figure expands the exact node updates and structural corrections. Every color has a consistent meaning and a visible legend. [Overview](figures/TreeLoom_architecture.png) · [Node operations](figures/TreeLoom_operators.png) · [Generator and review](figure_redesign_20261026_v2/README.md).

The manuscript and extracted Overleaf bundle compile to eleven pages, with no errors, undefined references, overfull boxes or oversized floats. Figures 1 and 2 occupy pages 3 and 4. Three underfull-box diagnostics remain and were visually inspected. All eight result tables are unchanged. [Build and review record](figure_redesign_20261026_v2/integration.json).

This is a development manuscript. Its eight European validation simulations were reused for model selection, with one optimizer seed. The evaluated releases differ materially from publication recipes. Independent tests, complete American comparisons, useful architecture novelty and biological corroboration remain unfinished. No overall top-1 claim is established.

[Current detailed scores](../research/current_scoreboard.md) · [Provenance](current_full_panel_provenance.json).
