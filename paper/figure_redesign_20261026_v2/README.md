# TreeLoom visual operations, 26 October 2026

This revision retains the biological inputs and prediction readout, and replaces opaque encoder and memory boxes with visible operations. Figure 1 shows four directed tree traversals in two rounds, the original-feature bypass, the shared weights, sample embeddings and explicit haplotype memory tracks. Figure 2 expands the upward and downward node updates and the factored structural corrections. The model is unchanged.

## What is now visible

- Two rounds, each visiting every tree depth upward and downward. These are not four fixed GCN layers. Upward and downward operators differ and their parameters are reused in round 2.
- Child projection, edge gating, arithmetic mean, concatenation, linear transformation, tanh, residual addition and normalization in the upward operator. The downward operator excludes the recipient's normalized message and has no final residual addition.
- Separate forward and reverse memory histories for the same sampled haplotypes through T1, T2 and T3. The current group context is copied to its members. Both pooling branches tap the updated T2 state, not an incoming history.
- A visible memory update, directional averaging, current-group averaging, centering and projection back into the group ancestor.
- Actual gain, loss and persistence patterns from the saved toy trees, followed by the two low-rank corrections. Both use the current upward state, not a neighboring hidden state.

Black means structure or computation. Gray means feature vectors or learned matrices. The four ancestry colors have full population names in both legends. The query is an open circle. The selected group ancestor is a square around its existing internal-node dot.

## Scientific and visual review

An independent implementation audit checked the frozen European and American TreeLoom sources, the memory implementation and both structural corrections. The illustrative T2 group is exactly r1 and r2. Its representative has age 212.5 generations and parent age 400, straddling the 350-generation cut. The toy clade-presence patterns agree with the preserved tree sequence.

A separate reviewer used images without the code or manuscript. It identified an ambiguous ancestor symbol, missing memory transport, incorrectly placed projection dimensions, a crossing projection wire, a reverse-state tap before its update, and an inset label touching its border. These were corrected. The final targeted image review reported no remaining blocking diagram error. This does not establish that every reader will immediately understand every symbol. The caption defines the operators and the paper explains their biological role.

The primary agent inspected both standalone PDF renders, the full manuscript contact sheet and the two figure pages. The manuscript and extracted Overleaf bundle each compile to 11 pages. Figure 1 is on page 3 and Figure 2 on page 4. There are no LaTeX errors, undefined references, overfull boxes or oversized floats. Three underfull-box diagnostics remain and were visually inspected. The eight table source blocks are unchanged.

## Rebuilding

From the repository root on anogena:

```bash
. scripts/project_env.sh
"$ARGGNN_PYTHON" paper/figure_redesign_20261026_v2/build_figures.py
```

The generator reads the previously saved illustrative tree sequence. It does not simulate data, load model weights or allocate CUDA. It writes PDF, SVG and PNG files beside itself. Canonical figure copies are in `paper/figures/`. Captions are preserved in the two `.tex` snippets. Build hashes and verification are in `integration.json`.

All experiments remain stopped. This revision changes presentation only. No model, checkpoint, data or benchmark result was modified.
