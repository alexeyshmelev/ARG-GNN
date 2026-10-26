# Conceptual figure source check

Date: 2026-10-26. This review checks the meaning of the v6 figures against the frozen TreeLoom contract, not whether every implementation operation is expanded. It is separate from the independent reader test.

## Figure 1

Read `build_architecture.py` and visually inspected its current rendered PNG. The conceptual computation agrees with the frozen model:

- Three ordered marginal trees and observed alleles/reference information enter the TreeLoom encoder. The trees come from the preserved illustrative tree sequence. The highlighted q is the same sampled leaf in each tree.
- The encoder is explicitly marked as two rounds and linked to Figure 2. The node grids are schematic states. At this level, combining final upward/outside states and projecting the selected leaf is part of the encoder abstraction. The picture does not assign a contradictory literal state width or claim that these colors are measured activations.
- The selected q rows form a sequence ordered T1, T2, T3. This is a sequence for one leaf, not pooling all leaves or pooling the tree set into a single vector. The same operation applies to other leaves.
- Observed interval geometry is appended to that leaf sequence before the bidirectional GRU. The direct skip originates in the teal embedding region, not the appended gray geometry region. This agrees with the local embedding plus recurrent context input of the ancestry head.
- The output has one probability row per interval and ancestry columns. It does not imply one global class for the tree sequence. The pictured probabilities are illustrative, not benchmark results.

The collapsed encoder and ancestry head are appropriate conceptual abstractions. They introduce no additional message operator, pooling stage, recurrence, supervision source or output aggregation. Definitions of population abbreviations and observation preprocessing can remain in the caption or methods. No extra implementation detail is requested by this source check.

## Figure 2

Read `build_encoder.py` and visually inspected the final simplified PNG. Its conceptual paths agree with the frozen implementation:

- **Original-base residual.** The long skip starts at the original node-state input, not at the first-round output. The clade and memory corrections join it before round 2. This represents B+S+Δ correctly. Both rounds perform upward then downward tree messages, and the sharing bracket connects only these two rounds.
- **Current-node clade correction.** The observed presence of the same descendant leaf set is compared across ordered trees. Its learned correction receives the current T₂ node embedding, with no hidden-state arrow from T₁ or T₃. For the illustrated {r1,r2}, absence/presence/presence agrees with the recorded toy. The resulting correction returns to its current T₂ representative b. This accurately separates structural pattern information from transport of neighboring embeddings.
- **Persistent-leaf memory.** Group context is copied to the group's members. The same r1 is then followed across T₁/T₂/T₃, with its source group changing from d={q,r1} to b={r1,r2}. The two directional arrows, distance decay and direction mean summarize the selected stable-leaf recurrence correctly. The example is explicitly one leaf, while the current-group mean collects all member leaves, here r1 and r2.
- **Group return.** The fixed group average followed by the learned Group correction and return to b matches the source. Group correction encapsulates subtraction of the current group's forcing and its learned output projection. The caption may explain this centered return without expanding every internal operation in the figure.
- **Biological meaning.** The small trees are local descendant-set sketches, not disconnected complete genealogies. Shared leaf membership is not a claim of persistent inferred ancestor identity. The same letter b denotes the selected group's current representative in each depicted tree. The separate T₂ recipient pictures show where the two corrections are applied. The simplified display makes no false assertion about a continuous ancestral genome, topology conversion, new recombination event, or hidden-state alignment of internal nodes.

The matrices and vectors are explicitly schematic. Their displayed cell counts are not tensor-shape claims. The final all-node state picture is consistent with retaining internal nodes before the leaf readout used in Figure 1. No additional operation or biological correspondence is implied by these abstractions.

## Scope

The reference is the frozen source contract already audited for v5, especially `FrozenEuropeanTreeLoomV1.encode_trees` and `forward_holds`, the shared tree message routines, the selected stable-leaf memory and own-state structural adapters. This v6 pass read both figure sources and displayed both PNGs. No model computation, simulation, training, benchmark or GPU operation ran. Only this review document was written. Figure assets, model sources and canonical manuscript files were not changed by the reviewer.

## Clarifications after the first fresh reading

The three revised labels agree with the source. The memory groups use a fixed age cut, represented by `HORIZON` in the frozen cut plan. Memory decays with genomic span in base pairs through `exp(-spans / ell_bp)`. Both calls to the tree sweep reuse the same node modules across trees and rounds. These labels make existing mechanisms explicit without adding any new computation to the figure. The final PNGs were visually inspected after these wording changes.
