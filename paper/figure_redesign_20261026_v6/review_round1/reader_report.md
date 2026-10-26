# Fresh-reader report

Scope: I viewed only the two supplied PNGs, without the paper, captions, code, or earlier reviews.

## What I understand

The main figure has a clear left-to-right sequence. Trees T1–T3 cover successive genomic intervals. The marked leaf q is the same query individual or haplotype across those trees; the image does not specify which of those biological units a leaf represents. Trees, alleles, and reference labels enter the TreeLoom encoder. It returns node embeddings for each tree, from which the q row is selected. Interval information is added to this sequence. A BiGRU supplies context across trees, and the ancestry head receives both its output and the direct q sequence. The result is an ancestry probability distribution over A, C, W, and E for q at each tree/interval. I read those letters as reference ancestry classes, without knowing their expanded names.

The encoder figure reads naturally as the top overview followed by the two mechanism panels. Each tree undergoes upward then downward message passing, two cross-tree mechanisms produce corrections, those corrections are added to the original node states, and a second upward/downward round produces the final states. The drawn addition uses the original states as its baseline; it does not show a direct residual from the Round 1 output.

The clade mechanism follows the identity of a descendant leaf set, rather than relying on an internal-node name remaining stable across trees. The set {r1, r2} is absent as a clade in T1 and present in T2 and T3. Its change/persistence pattern and the current T2 node embedding produce a correction to node b.

The leaf-memory mechanism follows the same leaf through changing groups. For r1, the copied group embedding comes from d with {q, r1} in T1 and from b with {r1, r2} in T2 and T3. Information is accumulated in both directions with distance decay and the directions are averaged. Memories of leaves in the current group are then averaged, and a learned group correction returns to the current internal node. The distinction between stable leaf identity and recurring clade identity is communicated particularly well.

## What remains unclear

- The consequential gap is the meaning of “group” in the copy-to-leaves step. A leaf belongs to several nested clades. The examples suggest its immediate parent group, but the figure does not establish whether that is the rule or which group supplies its state. This affects the conceptual interpretation of the memory, not merely implementation detail.
- “Distance decay” does not name the distance. Genomic distance is a plausible reading given the main figure, but the encoder panel alone also permits tree-index distance. A short label would resolve this.
- The bracket visually links the two rounds, while “shared tree weights” could mean sharing across trees, across rounds, or both. The weight-sharing scope deserves a more explicit label.

The main connections otherwise communicate their roles adequately. The direct and contextual q signals visibly meet at the ancestry head, and the lower encoder panels identify where the two corrections act. I do not need tensor sizes or standard-layer internals to understand this.

## Overall judgment

These feel like an architecture overview plus a mechanism explanation, rather than an implementation wiring manual. The main prediction story is readily recoverable, and the encoder's extra detail earns its space by explaining its two distinct ways of carrying information across trees. The group-selection ambiguity is the main point that prevents a fully confident conceptual reading; the distance and sharing labels are smaller, readily fixable uncertainties.
