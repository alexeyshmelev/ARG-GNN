# Figure 1 redesign, 26 October 2026

The figure now has one view of the marginal trees, a horizontal encoder, a separate sample-memory inset, and a standard recurrent ancestry readout. The redundant 3D drawing and decorative layer colors were removed. Each ancestry has one color shared by reference labels and predicted probabilities. Gray means learned features or memory. Black means structure or computation. Every color and the hollow query symbol has a visible legend entry.

The source reads the preserved illustrative tree sequence from the previous figure. It does not simulate data, load a model, allocate CUDA or launch experiments. Tree topology, node ages, sample identity, age-cut groups and illustrative alleles come from that one source. Outputs are schematic probabilities, not results.

## Scientific check

A separate agent inspected both frozen TreeLoom implementations and the memory/adapters. The figure preserves two complete upward/downward GNN applications with shared weights. The original features receive clade and memory corrections before the second application. Memory projects group states, copies them to persistent sampled haplotypes, filters along the genome in both directions, averages over current groups, subtracts the current projected state and projects the correction back to the representative. It does not claim internal ancestral-node tracking or cross-tree hidden-state averaging in the selected episode adapter. The genomic BiGRU remains distinct from this memory.

## Image-only review

A fresh agent saw the image without code or manuscript. The first review recovered the overall flow but identified missing memory composition, the absent age cut, and ambiguous readout steps. Those were corrected. The second review recovered projection/copy, identity-wise EMA, group averaging, centering, the returned correction, sequential GNN passes, shared weights, softmax and argmax. It found no collisions or broken connectors. Its remaining requests for explicit tree input, a readable interval input and output interval labels were incorporated. Exact centering and clade-adapter definitions are in the caption. The final PDF rendering was visually inspected by the primary agent.

## Build

The canonical manuscript and extracted Overleaf bundle each compile to 10 pages with pdfLaTeX and BibTeX. Neither final log contains warnings or overfull boxes. Figure 1 is on page 3. A contact sheet of all ten pages and the standalone figure were inspected. All result-table source blocks are byte-identical to the previous manuscript. No model, experiment or benchmark result changed. Experiments remain stopped at the owner's request.

- Generator: `build_figure.py`
- Editable vector: `TreeLoom_architecture.svg`
- Print figure: `TreeLoom_architecture.pdf`
- Caption: `caption.tex`
- Build receipt: `integration.json`
