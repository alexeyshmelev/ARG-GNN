# Fresh visual reading of the TreeLoom diagrams

This report is based only on the two supplied PNGs. I did not consult source code, a paper, external captions, or previous reviews. I treat the little matrices as schematic embedding vectors, not literal dimensions.

## Main computation

The architecture takes a sequence of trees ordered along genomic position. Alleles and reference labels are additional observed inputs. The repeated marked leaf q is the query whose ancestry is to be predicted; the question mark among the reference labels suggests that its ancestry label is unknown.

The TreeLoom encoder produces node representations for each tree. The highlighted q row is selected from every tree and assembled into one ordered sequence for leaf q. Interval information is supplied to that sequence. A bidirectional GRU contextualizes the sequence across the genome. The ancestry head receives both the sequence directly and the GRU output, so I read the final prediction as using both local query representations and their sequence context.

The output is a table of ancestry probabilities: one row for each illustrated tree interval and one column for each ancestry category A, C, W, or E. The darker purple cells illustrate stronger probabilities, not hard class assignments.

## Representations and identity across trees

The trees and observation box are structural and observed inputs. Teal strips and grids are learned representations. The encoder returns representations for all nodes; the end-to-end prediction path then selects q.

Leaf identity persists across trees. In the architecture this is shown by the repeated q label. In the encoder, r1 is explicitly followed through T1, T2, and T3 even though its group changes. I would not infer that every internal node has a universal identity or that an arbitrary row in each node-state grid names the same internal node.

For the clade mechanism, the identity being compared is the same descendant leaf set {r1, r2}. That clade is absent in T1 and present in T2 and T3. This is a different correspondence from merely following the same leaf through different groups.

## Tree rounds

Each tree begins with node states. Round 1 passes messages upward and then downward through the tree. Its result feeds two context mechanisms, Clade pattern and Leaf memory. Their corrections are combined with the original node states before Round 2, which again performs upward then downward tree messages and yields all node states.

I read the two rounds as two passes of this procedure over each tree, not as processing T1 and then T2. The sharing bracket says the tree-message weights are reused across rounds and across trees. The diagram does not show a third tree pass or context corrections applied after Round 2.

The visible residual connection comes from the original node states. There is no separately drawn direct path from the Round 1 output to the addition symbol, beyond the two context branches. That is a definite implication of the drawing and should match the intended method.

## Context mechanisms

### Clade pattern

For a node in the current tree T2, the model examines whether its descendant leaf set forms a clade in other trees. The illustrated no/yes/yes pattern provides information about clade change and persistence. That pattern and the current node representation enter a learned block, which produces a correction for the corresponding internal node b in T2.

This gives a node context about how consistently its particular grouping of leaves occurs along the tree sequence.

### Leaf memory

Age-cut groups provide representations that are associated with their constituent leaves. The example follows r1: it belongs to a group with q in T1 and to a group with r2 in T2 and T3. Its group-derived representation can therefore change while its leaf identity stays fixed.

Information is accumulated along genomic position in both directions, with contributions decaying with genomic distance; the two directions are averaged. The resulting leaf memories are averaged within the current group {r1, r2}, and a learned Group correction block turns that aggregate into a correction for b in T2.

This gives a current group context assembled from the histories of its leaves, including earlier or later membership in different groups. The example successfully communicates why this is distinct from checking whether one exact clade persists.

## Ambiguities and potentially misleading connections

1. **Interval information is underspecified.** The gray columns and incoming Intervals arrow make it clear that interval-related observations accompany q's embeddings. I cannot tell whether they encode boundaries, interval lengths, position, or more than one of these. This is the most material unresolved feature meaning in the architecture. It does not prevent understanding the overall pipeline.

2. **The age-cut construction is named but not demonstrated.** I can understand that groups are selected by an age criterion and mapped to leaves, but no age threshold or cut line is shown. A reader unfamiliar with age cuts would understand the subsequent memory path while remaining unsure how its groups were selected.

3. **The top-level combination of the two context branches is compressed.** The two branches first join on an unlabeled line and then enter a plus symbol with the original states. I read this as adding the two corrections to the original states. The figure does not explicitly distinguish separate additions from a prior fusion of the two context outputs. If the exact combination is important to the conceptual claim, two independently entering arrows or a short sum label would remove the uncertainty.

4. **Leaf memory compresses a collection into one example.** Only r1's trajectory is displayed before Mean within current group {r1, r2}. The text each leaf: r1 shown makes the intended repetition clear, so I infer an analogous memory for r2 before the mean. Without that text, the connection would wrongly suggest that a single leaf's history alone supplies a group mean. With it, this is a manageable schematic simplification.

5. **Some operations are only qualitatively specified.** Genomic distance decay and mean directions convey the direction and purpose of the memory computation, but not its exact recurrence. The interval-to-sequence attachment and the ancestry head's fusion of direct and GRU inputs are likewise schematic. I do not regard these as failures for a conceptual architecture figure.

6. **Input labels could use one explicit identity cue.** I infer that the unknown label belongs to q and the other labels belong to reference leaves. The observation box is not explicitly aligned or linked to those leaves. This does not alter the main flow, but the intended association currently depends on conventional reading of the question mark and repeated q.

## Overall assessment

Yes, the conceptual flow is understandable from the images alone. The strongest parts are the extraction of one query leaf across trees, the explicit two tree-message rounds, and the distinction between exact-clade persistence and leaf histories that survive changing group membership. The direct path from q's sequence to the ancestry head is also clearly visible.

I did not find a connection that makes the overall computation incoherent or reverses its apparent direction. The interval features and the precise correction merge remain the most consequential ambiguities. These are bounded uncertainties rather than a reason to reject the figure's overall conceptual readability.
