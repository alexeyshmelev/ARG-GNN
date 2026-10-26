# TreeLoom figure redesign, 26 October 2026

The revised figures separate the overall architecture, haplotype memory and exact node operators. This replaces the prior two-figure layout. It changes presentation only. Experiments remain stopped, and no model, checkpoint, dataset or result was changed.

[Architecture](../figures/TreeLoom_architecture.png) · [Haplotype memory](../figures/TreeLoom_memory.png) · [Node operators](../figures/TreeLoom_operators.png) · [Paper](../current_full_panel_working.pdf) · [Overleaf bundle](../current_full_panel_overleaf_working.zip)

## Recorded comments addressed

| Owner comment | Implemented change |
|---|---|
| Redundant standalone direction labels | Removed. Arrows on the tree branches show traversal direction. |
| Internal-node embeddings missing | Figure 1b draws feature vectors at internal nodes and leaves after every traversal. Root outside state is explicitly zero. |
| Unclear number or type of GNN layers | Figure 1b shows two complete rounds with four traversals and shared weights. Figure 3a,b expands the actual node operators. These are not four fixed GCN layers. |
| Meaning of the 32-to-8 and 8-to-32 transformations | Figure 2 shows both as learned Linear blocks. Figure 3 shows the separate structural projections. |
| Unexplained age threshold | Figure 2a shows the resulting descendant groups and their representative ancestors. The selected group contains r1 and r2. The complete tree is retained. |
| Unclear r1/r2 state correspondence | Figure 2a copies the same group context to both haplotypes. Figure 2b keeps their histories separate and labels both directional T2 outputs before averaging. |
| Multiplications lack operands | The connected M expansion shows vector and scalar inputs to both Scale operations. Figure 3 distinguishes elementwise products from scalar weighting. |
| Detached dashed memory inset | A callout connects a specific M update to its expanded calculation. It is not an extra stage. |
| Subtraction order unclear | The memory and structural subtraction inputs now have explicit signs. The memory output subtracts the current group context before its learned projection. |
| Memory confused with GRU | Figure 2 shows exponential averaging and its learned distance scale. Figure 1c separately shows the one-layer BiGRU. |
| Two averages confused | Figure 2b shows a mean over directions for each haplotype, followed by a mean over current group members. |
| Source of interval geometry unclear | Figure 1c derives interval length and adjacent center distances from the chromosome axis. Three features enter each tree's local query vector. Missing-neighbor components are visibly zero. |
| Meaningless colors and uniform feature cells | Learned operations use violet blocks, fixed operations use pale neutral blocks, feature values use a purple intensity scale, and ancestry categories retain four distinct colors. Every figure includes the full legend. |
| Unclear structural flags | Figure 3c uses recognizable descendant-pair icons and presence patterns, with the current upward state feeding each explicit learned map. |
| Too much text and figure titles | Figures have no titles. They use subfigure letters, operation labels and dimensions. Scientific definitions remain in captions and the method text. |

These are implemented changes, not a claim of owner approval. The original feedback remains preserved in the v2 record.

## Scientific checks

The source audit checked the frozen European and American TreeLoom models, the tree operators, the structural corrections and the memory implementation. The final changes retain the correct order of site MLP before mean/max pooling, complete upward/downward rounds, shared weights, memory centering and injection at the group ancestor, per-child transformation before averaging, and the 35-to-64 BiGRU with its local embedding skip into the 96-to-32-to-4 classifier.

The figures reuse the preserved three-tree example. Its intervals are 0–3000, 3000–7000 and 7000–10000 bp. The depicted group ancestor is 212.5 generations old, below the 350-generation cut, while its parent is 400 generations old. Genomic geometry is computed from these recorded coordinates, including zero at missing neighbors. Hidden feature cells and ancestry probabilities are illustrative, not measured activations or predictions.

A fresh reviewer saw the images without the source code or method narrative. It identified an ambiguous T2 reverse state, an apparent early child aggregation, ambiguous subtraction inputs and a crowded structural label. These were corrected. The final targeted review reported no remaining misleading major connection or label collision. The review is limited to the inspected figures and does not establish that every reader will find them immediately clear.

## Integration checks

The canonical AAAI manuscript and a freshly extracted Overleaf bundle both compile to 12 pages. Figures 1, 2 and 3 occupy pages 3, 4 and 5. Main text and references occupy pages 1–8, with the supplement on pages 9–12. There are no LaTeX errors, undefined references, overfull boxes or oversized floats. Three underfull diagnostics remain and their page layouts were visually inspected.

All eight table source blocks are exactly unchanged relative to commit 99b72ca6. The full contact sheet and each figure page were inspected. The canonical and extracted bundle PDFs have identical extracted text. See [integration.json](integration.json) for hashes and build checks, and [review.json](review.json) for the independent review record.

## Rebuilding

Run on anogena from the repository root:

```bash
. scripts/project_env.sh
"$ARGGNN_PYTHON" paper/figure_redesign_20261026_v3/build_figures.py
```

The generator loads the preserved toy tree sequence and writes PDF, SVG and PNG versions beside itself. It does not simulate data, load a model or allocate CUDA. Canonical copies are in `paper/figures/`. The three caption snippets are kept beside the generator. The integration script records the one-time manuscript replacement and verifies table preservation before writing.
