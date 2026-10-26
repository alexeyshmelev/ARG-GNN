# TreeLoom figure redesign, 26 October 2026

The revised figures separate the overall architecture, haplotype memory and exact node operators. This replaces the prior two-figure layout. It changes presentation only. Experiments remain stopped, and no model, checkpoint, dataset or result was changed.

[Architecture](../figures/TreeLoom_architecture.png) · [Haplotype memory](../figures/TreeLoom_memory.png) · [Node operators](../figures/TreeLoom_operators.png) · [Paper](../current_full_panel_working.pdf) · [Overleaf bundle](../current_full_panel_overleaf_working.zip)

## Recorded comments addressed

| Owner comment | Implemented change |
|---|---|
| Redundant standalone direction labels | Removed. Arrows on the tree branches show traversal direction. |
| Internal-node embeddings missing | Reopened after owner review. Small vector glyphs were added, but Figure 1b still fails to make the all-node output and node-to-vector correspondence clear. See the open oversight below. |
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

## Open owner-reported oversight: all-node embeddings in Figure 1b

Record only, do not redraw yet. The owner cannot clearly see that the encoder produces embeddings for every node, including internal ancestors and sampled leaves. The current small vector glyphs and the text saying "32 features at every node" do not adequately communicate the resulting collection of node embeddings or which vector belongs to which node. The prior visual review did not resolve this communication defect.

In the next requested revision, make the node-to-vector correspondence and the all-node output explicit after each traversal. Visually distinguish the embeddings of internal ancestors and sampled nodes, and show that selection of sampled haplotypes for the final ancestry readout occurs afterward. Preserve the zero root outside state. This issue remains open. The figures and manuscript are unchanged by this record.

## Recorded terminology correction: leaf embeddings

The owner requests "leaf embeddings" instead of "haplotype embeddings" in the diagram and its explanation because the visible objects are leaves. Use "leaf embeddings" for the sampled-node readout and make the selection of leaves from all node embeddings explicit. Define once in the caption that each sampled leaf represents one haplotype. This wording must also make clear that the same leaf identity is followed across trees. Apply this correction with the next requested figure revision. This record does not redraw the figures or change the model.

## Open owner-reported oversight: stacking leaf embeddings

Record only. The owner cannot see how leaf embeddings are assembled before the BiGRU. For one query leaf q, its 32-component embeddings from successive trees form the rows of a T by 32 sequence matrix, ordered by genomic interval. The same leaf identity is followed across all T trees. Each row receives that interval's three geometry features, producing a T by 35 input to the BiGRU. This is stacking along the sequence axis, not flattening all trees into one vector or pooling different leaves.

The next requested revision must visibly connect the same leaf in successive trees to its individual vectors, show those vectors becoming ordered matrix rows, and show the three geometry columns appended to each row before the BiGRU. The present detached vector glyphs and "Concat per tree" block do not make that assembly explicit. This communication issue remains open. No figure, model or experiment is changed by this record.

## Open owner-reported oversight: meaning of the three geometry entries

Record only. Figure 1c does not clearly map the three cells in each interval vector to interval length L_t, previous-center distance d_minus and next-center distance d_plus. The chromosome measurements, log transformation and resulting vector entries are not connected clearly enough for the reader to infer their meaning. The "3 features per interval" label does not resolve this issue.

In the next requested revision, visibly associate each measured distance with its corresponding vector entry, preserve the entry order [L_t, d_minus, d_plus], explain the zero at missing neighbors, and make clear that these are three additional geometry features appended to the 32 learned leaf features. Describe this as the current design choice, not an established optimal number of features. This issue remains open. The figures and manuscript are unchanged by this record.

## Open owner requirement: continuous representation flow from input to output

Record only. The owner must be able to trace a continuous visual path from the observed inputs to the ancestry predictions and see how the representations change at every important operation. Disconnected embedding glyphs and named blocks are insufficient when their inputs, outputs or correspondence are left implicit.

In the next requested revision, show the observed inputs becoming node features, embeddings at every node through both rounds, the memory and structural correction branches returning to the correct node inputs, selection of leaf embeddings, alignment of the same leaf across trees, stacking into an ordered sequence, addition of the three geometry columns, the BiGRU output, concatenation with the original leaf embedding, and the classifier output probabilities. Visually show meaningful changes of shape and dimensions, selection, stacking, concatenation, aggregation and any actual reshape or transpose needed to explain the computation. Do not invent reshapes or include implementation-only rearrangements that have no explanatory role. Every branch must have an identifiable source and destination. Preserve identity and ordering visibly through the pipeline.

This is a requirement for the whole diagram, not a request for additional explanatory text alone. It remains open alongside the specific recorded issues. No redraw, model change or experiment is authorized by this record.

## Recorded layout correction: compact tree message passing blocks

The owner requests replacing the large repeated tree drawings inside the encoder with compact blocks labeled "Upward message passing" and "Downward message passing", optionally using a small tree icon. The current drawings consume too much space. Preserve the two rounds and their shared weights, but use the recovered space to show the continuous representation flow, node-embedding matrices, dimensions and important transformations.

This supersedes the choice to repeat a full tree inside every traversal block. It does not remove the requirement to make embeddings for all nodes explicit. Show that correspondence through the input/output representation and a compact key where needed. Keep the biological tree example at the input. This is recorded for the next requested redesign. No figures or experiments are changed by this record.

## Open owner-reported oversight: overuse of purple

Record only. The owner objects to purple being used throughout the figures for learned blocks and feature vectors. Reusing this hue obscures the distinction between probabilities, embeddings and operations. A legend alone does not resolve the ambiguity.

For the next requested redesign, reserve a purple intensity scale for probabilities, as suggested by the owner. Choose distinct, restrained colors for other meaningful representation or operation categories, including embeddings and learned layers. Use the same color only for the same semantic category and provide a legend for every color. Keep ancestry identity colors consistent and distinguish them from the probability intensity scale. Do not introduce decorative colors or a different color for every individual block. This supersedes the v3 choice of violet learned blocks and purple feature values. The issue remains open. No figures or experiments are changed by this record.

## Open owner-reported oversight: memory figure has no clear reading path

Record only. The owner cannot determine where to start reading Figure 2 or follow its overall flow. The layout mixes the order of computation with the two genomic scan directions. Copied context vectors are detached from their subsequent inputs, repeated leaf tracks compete for attention, and long perimeter routes send the result backward into an earlier panel. The current connected arrows and operation labels have not made the computation understandable. This is a structural layout failure, not a request for more prose around the same layout.

In the next requested redesign, establish one obvious entry and one dominant reading direction through a small number of ordered stages: derive group context, distribute it to the group's leaves, update each leaf's memory across ordered trees, combine the two directions for each leaf, aggregate the current group's leaves, subtract the current context, project the correction, and inject it before round 2. Keep forward and reverse genomic scans inside one clearly bounded stage so their opposing arrows do not determine the reading order of the whole figure. Place the resulting correction and its destination at the end of that reading path. Show leaf identity, current interval and representation dimensions consistently. Use short local connections and directly associate any expanded arithmetic with its parent operation.

Preserve the exact method and distinguish directional averaging from averaging current group members. Apply the recorded compact-block and semantic-color requirements. The issue remains open despite the prior targeted visual review. No figure, model or experiment is changed by this record.

## Recorded clarification and design guidance: EMA, M and formulas

The owner finds both EMA and M unclear and permits formulas where they explain an operation better than a diagram. EMA means exponential moving average. M is only the matching label between a compact update block and its expanded inset, not a separate learned layer or additional operation. The current notation introduces an unnecessary decoding step.

For the next requested redesign, critically assess each proposed addition for clarity before applying it. Prefer a short, defined recurrence for the memory arithmetic: m_new = alpha * m_previous + (1 - alpha) * f_current, with alpha = exp(-L / ell). Here f_current is the current group context copied to the leaf, L is the current genomic interval length and ell is a learned positive length scale. The same rule applies in either scan direction, and the first state initializes to its current context. Define every symbol close to the formula or in a short caption. Remove the redundant M label if the expanded inset is replaced.

Use the visual flow to explain which representations enter and leave the operation, their shapes and their identities. Use a formula for arithmetic when expanding that arithmetic into boxes and wires creates clutter. Do not attempt to resolve the existing layout failure merely by adding more labels, equations or connectors. This records guidance for the future redesign and does not change the figures or experiments.

## Recorded simplification: remove the ordinary-edge legend entry

The owner flags "Tree edge / data flow" as an unnecessary legend entry. Remove this entry in the next requested redesign. Ordinary tree edges and directed connectors do not need a definition. Reserve legend space for meaningful colors and unfamiliar symbols, retaining the requirement to explain every semantic color. This is a record only. The figures remain unchanged.

## Recorded simplification: replace the expanded EMA inset with its formula

The owner specifically identifies the oversized exponential-average inset as suitable for replacement by a formula. In the next requested redesign, replace the two Scale blocks, coefficient box, addition node and their wiring with the compact recurrence m_t = alpha_t * m_previous + (1 - alpha_t) * f_t, where alpha_t = exp(-L_t / ell). Briefly define the previous memory, current group context, interval length and learned positive distance scale. Keep the memory operation's inputs and output identifiable in the overall flow. The same rule applies in both genomic directions. Remove the redundant M callout along with the expanded inset.

This makes the earlier guidance about using formulas concrete for this panel. The update itself has one learned distance scale, while the current context comes from its separate learned projection. Do not make this arithmetic appear to be an additional multilayer network. Recorded only, with no redraw or experiment changes.

## Proposed redesign: replace horizontal leaf tracks with vertical representation flow

The owner rejects the r1/r2 track layout, where incoming and outgoing vectors sit on the same horizontal paths and the forward/reverse routes are difficult to follow. The requested improvement is a different visual organization, not a cosmetic adjustment of arrow heights.

Proposed layout for the next figure revision:

1. Start at the top with one context sequence matrix per leaf. Each matrix has T rows in genomic order and eight feature columns. Each row is the context of that leaf's group in the corresponding tree, not an ancestry output or the final 32-component leaf embedding. Show how these rows come from the group-context distribution step.
2. Split each sequence into two compact parallel blocks for forward and reverse exponential memory. Represent the recurrence with the recorded compact formula. Indicate genomic direction within the blocks, while all diagram-level input/output arrows continue downward.
3. Put both output matrices below the blocks with rows aligned in the same genomic order. Average matching rows across the two directions. This is an elementwise mean, not concatenation or averaging across genomic positions. It yields one T by 8 memory matrix per leaf.
4. For the illustrated T2 group, visibly select the T2 row from r1 and r2. Stack those two eight-component rows, then average over the current group's leaves. Explain that this selection and group aggregation occur separately at every interval with its actual group membership.
5. Continue downward through subtraction of the current group context, the Linear 8-to-32 projection and injection at the group ancestor before round 2. Show the destination at the bottom, without a long return route around the page.

This proposal preserves the exact memory computation and separates the leaf, genomic-interval and scan-direction axes. It should use compact matrices, distinct meaningful colors and short connections. It remains to be drawn and visually verified during the requested redesign. The current figures and experiments are unchanged.

## Open owner-reported oversight: expanded operators lack a clear parent connection

Record only. The owner cannot tell where Figure 3 starts, how it connects to the preceding figures, or whether it expands an existing operation or adds a new stage. Panels a and b are intended to expand the upward and downward node updates in the overview. Panel c expands its structural correction. These parent relationships are not visually explicit enough. The owner rejects the present presentation despite the prior visual review.

In a future requested revision, make each expansion visibly identifiable with its parent block using consistent names, symbols, representations and clear input/output correspondence. Give each panel an obvious entry point. Do not expect the reader to infer the relationship from a figure reference alone. Avoid disconnected detailed circuits, and use formulas where expanding arithmetic into boxes harms clarity.

The owner requires a future fresh agent with no prior architecture context or explanatory hints to reconstruct the entire input-to-output computation from the figures. The review must cover representation changes, dimensions, identities, ordering, branches, aggregation axes and connections between overview and expansions. Compare its account against the implementation and record all errors and ambiguities. A generic aesthetic verdict or a narrow collision check is insufficient. The acceptance target is a complete, correct account of the depicted flow, not a favorable opinion.

A new blind-reader agent was started during this exchange, then interrupted when the owner clarified "Just record for now". No completed result from that review is claimed. This issue and the future review requirement remain open. No figures, model or experiments are changed by this record.

## Superseded by the two-figure redesign

The owner-reported open issues above remain preserved as the record of this version. They were addressed in the [v4 redesign](../figure_redesign_20261026_v4/README.md), which replaces the three figures with a prediction diagram and an expanded encoder. New independent readers had no source, paper text or prior conversation. Two failed readings led to further changes, and a subsequent fresh reader correctly explained the complete conceptual flow. This is not owner approval. The detailed review and remaining implementation-level scope are recorded in v4. Experiments remain stopped and the numerical tables are unchanged.
