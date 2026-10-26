# TreeLoom: two-figure redesign

The main paper now contains two figures. The first traces observations to ancestry probabilities. The second expands the tree encoder, including its two complete message-passing rounds, structural correction and sample memory. It returns explicitly to leaf selection in the first figure.

[Prediction flow](../figures/TreeLoom_architecture.png) · [Tree encoder](../figures/TreeLoom_encoder.png) · [Paper](../current_full_panel_working.pdf) · [Overleaf bundle](../current_full_panel_overleaf_working.zip)

## What changed

| Recorded issue | Revised representation |
|---|---|
| Ancestor embeddings absent | Matrices retain all node rows through both rounds. Leaf selection is a separate operation afterward. |
| Leaf sequence assembly unclear | The same query leaf is selected across trees, its embeddings become ordered rows, and three observed geometry columns are appended. |
| Three unexplained geometry features | Interval length and distances between adjacent centers are drawn, column names match those distances, and missing neighbors visibly give zero. |
| Oversized repeated tree drawings | Compact upward and downward message-passing blocks show the two complete rounds and shared weights. |
| Memory has no readable entry or exit | The path proceeds from group context to leaf sequences, two directional memories, two distinct averages, subtraction and projection, then injection before round 2. |
| Unexplained memory arithmetic | A concise recurrence replaces the expanded multiplication circuit. The learned physical scale is explicit. |
| Detached operator detail | Figure 2 names its parent block, starts with the same base matrix and ends with an explicit return to Figure 1b. |
| Unclear structural flags | A presence-to-flag table includes overlapping gain/loss/persistence and isolated-clade cases. The learned maps use the current upward state. |
| Missing input transformations | Separate node, site and leaf feature paths show pooling within a leaf and interval and addition only to leaf rows. Both tree and variant inputs connect to learned edge gates. |
| Uninformative colors | Gold denotes learned layers, neutral blocks fixed operations, teal embeddings, gray observations and purple probabilities. All colors have legends. |
| More than two large main figures | The manuscript contains exactly two architecture figures. The earlier operator equations remain in the text and supplement. |

## Independent reconstruction

Three fresh agents inspected successive frozen pairs of images without the manuscript, source code, previous reviews or task history. The first two readings exposed specific missing explanations and were not accepted. The third correctly reconstructed the complete conceptual computation, including the aggregation axes and all three mechanisms carrying information across trees. A fourth fresh reader independently explained the delivered pair after two labels were moved away from bypass lines. That explanation also passed the source comparison.

The [review record](review.md) contains failed rounds, repairs and the final source comparison. This is a conceptual communication check, not code reconstruction or a claim that every reader will understand the figures instantly. It does not constitute owner approval.

## Scientific fidelity

The [source audit](source_audit.md) verifies that the figures describe the frozen implementation. The source audit also corrected a textual error: the edge signal compares observed allele partitions with inferred branches. It is not the placement of inferred mutations.

The toy example comes from the preserved tree sequence used previously. No new simulation ran. Group ancestors and interval geometry are derived from that saved example. Matrix entries and probabilities are illustrative rather than measured activations or results. Four output classes depict the European condition.

No model, checkpoint, dataset, training queue or result was changed. All eight table source blocks are byte-for-byte identical to the preceding committed manuscript. Experiments remain stopped.

## Build checks

The canonical AAAI manuscript and the freshly extracted Overleaf bundle compile to ten pages. Figures occupy pages 3 and 4. Main text and references end on page 6, and the supplement occupies pages 7–10. There are no LaTeX errors, undefined references, overfull boxes or oversized floats. Five underfull diagnostics remain. The full contact sheet and both figure pages were visually inspected.

[Integration receipt and hashes](integration.json)

## Rebuild

Run on anogena inside the dedicated repository:

```bash
. scripts/project_env.sh
"$ARGGNN_PYTHON" paper/figure_redesign_20261026_v4/build_figures.py
```

The generator uses CPU drawing and the saved illustrative tree sequence. Canonical PDF, SVG and PNG copies are in `paper/figures/`. Frozen review images and their generators preserve what each reader saw. The one-time integration script records manuscript replacement and table-preservation checks. It is not a general-purpose manuscript updater.
