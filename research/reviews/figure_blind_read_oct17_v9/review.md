# Blind architecture image review — 2026-10-17 v9

Inspected only the rendered current image and its 2x rendering. No manuscript, caption, source code, image metadata, or earlier reviews were read.

Image: paper/figure_redesign_20261017_v7/TreeLoom_architecture.png
SHA-256: 83360cff05d45e500b995a65c3fc15793129f1d667a991718cf810e2f856a020

## What the image communicates

The overall sequence a → b → c → d is readable. An ancestral recombination graph is decomposed into three rooted local trees along a 0–10 kb genomic interval, split at 1.46 and 6.65 kb. Solid, diagonal-hatched, and crosshatched interval strips consistently identify the three intervals. Time increases upward. Diamonds denote recombination; the inset shows a descendant lineage entering a recombination vertex and two ancestral lineages leaving it. I cannot determine the intended meaning of each very faint connector between the local graph drawings.

Each local tree contains the same five named tips, reordered by topology: four references r1–r4 with known Ana/CHG/EHG/WHG labels and a query q1 shown by a magenta square and question mark. The query is consistently sister to r3 in all three displayed trees. Tip-level feature glyphs include a small gray array above the ancestry labels and a colored/blank vector below them. I infer an allele-like input and an ancestry-like input, but their dimensions, ordering, and missing-value encodings are not independently recoverable. A single internal-node callout displays (τ, 0, 0), suggesting that internal nodes have a different feature pattern. This is a useful hint, not a complete definition of the input features.

The bracket below the tree inputs leads into an MLP and an N_t × d node-feature array. An upward-then-downward GNN pass leads into a clade-update block. The result is added to a bypass from the input features and to a correction from the lower sample-memory computation; a second GNN follows. The dashed “same weights” connection suggests that the two GNN applications reuse parameters. Thus I can infer two applications of a shared block, although I cannot tell how many actual graph layers or node updates each application contains.

The lower computation stores sample-like rows across T1–T3; the magenta top row is q1 and horizontal two-way arrows suggest cross-tree information exchange. A small group of samples feeds a vector f. The memory contributes a vector z̄, which is shown subtracting f before a Linear transformation returns a correction to the residual addition. Interval widths w1–w3 probably influence this aggregation, but this is not explicitly shown.

The second GNN produces one array per tree, labeled S × d. A magenta row e1q/e2q/e3q connects to the magenta row of a one-layer BiGRU. A genomic side input contains (w_t, Δc_t^−, Δc_t^+) and a log(1 + ·) transform. A bypass from the pre-GRU embeddings and the GRU output meet at a concatenation-like symbol, followed by an MLP. The output is a set of ancestry probabilities p_tq along the genome; argmax produces an ancestry track for q1. The output segment widths appear to preserve the unequal genomic interval lengths. The illustrated hard calls are all WHG.

## Ambiguities that materially impede understanding

1. The feature glyphs are too small and insufficiently keyed to explain the model input. “Alleles,” “ref.,” and “query” explain categories, but not what occupies the feature vector at a leaf or internal node. The gray array and lower one-hot-looking vector could represent different feature blocks, or a visualization of the same information. I cannot establish which.
2. Row identity disappears after panel b. The S × d arrays and hidden-state columns have no r1–r4 row labels. Because the tree leaves change horizontal order, I cannot verify whether rows retain sample identity or are reordered with each tree. Only q1 remains identifiable.
3. “Clade update” contains a forest-like collage without an explicit before/after or a visible state transfer. I cannot tell what is updated, which clades are matched across intervals, or whether topology, features, or memory are being changed.
4. The sample-memory row groups above f are unexplained. They look like clades or sample sets, but the grouping rule is not visually defined. The weighting/reduction producing z̄ is absent; the bracket into z̄ does not say whether this is a mean, weighted mean, sum, or selection. The destination of the linear correction is a global plus, so its node/sample scope is also unclear.
5. The BiGRU has one h box per sample/time position with two-way connecting arrows. This communicates bidirectionality in broad terms but not two direction-specific hidden states or how they are combined. The concatenation symbol has two incoming routes, yet the exact two tensors and their dimensions are not labeled.
6. The positional side vector is visible but its attachment is underspecified. The arrow enters the BiGRU enclosure rather than a concrete per-step input. I cannot determine whether log(1 + ·) transforms every component, whether these values are concatenated to e_tq, or how the Δc terms are computed.
7. GNN “Up → Down” and shared weights are visible, but no updated node-state vectors or message arrows tied to individual edges show the computation. The diagram communicates traversal direction more strongly than the actual feature update.

## Clutter and legibility

I see no obvious literal text/shape overlap or clipped content in either rendering. Panel b is relatively clean. Panel c is the main bottleneck: several long bypass/feedback routes surround small tree and matrix glyphs, and the sample-memory machinery is compressed against the lower edge. At ordinary viewing size, the inset states inside the clade block, the input feature cells, the lower memory row boundaries, and the positional-vector symbols require magnification. The long routing lines are visually more salient than some of the operations they connect. Gray sample rows are low contrast, so they are present but difficult to follow individually.

The figure already has very little explanatory prose. More prose is not the remedy. The remaining problem is missing visual definitions and identity correspondence.

## Mandatory improvements

1. Make one leaf feature and one internal-node feature visibly decomposable into their actual blocks, with a compact symbol key and explicit dimensions/order; connect those same blocks to the MLP input. Make the query’s unknown ancestry encoding explicit. Do not require decoding tiny gray cells.
2. Keep sample identity registered through c and d: use five consistent rows with r1–r4/q1 labels or an equally unambiguous shared row key, irrespective of changing tip order. Maintain the magenta q1 trace through the final probability/argmax track.
3. Replace the clade collage and unexplained memory groups with an explicit example of the actual grouping/update: identifiable input states, the matched group, the operator, and its resulting state. Show the reduction that creates z̄, including where w_t enters; label the recipient of Linear(z̄ − f).
4. Show the GNN traversal as edge-level messages updating node vectors, with the true repetition/layer count and shared parameter identity visible. An abstract duplicate tree plus “Up → Down” is insufficient for the requested explicit architecture.
5. Attach the positional vector to a specific per-tree GRU input; show the true combination operator and transformation scope. Distinguish forward and backward hidden states and explicitly label the two inputs to the final concatenation.
6. Give the lower computation more space and shorten its feedback routes. Enlarge the scientifically meaningful glyphs; simplify redundant small cells/tree miniatures. Preserve the clean output probability-to-argmax relationship and unequal genomic segment widths.

Verdict: the high-level model narrative is recoverable, and geometric overlap is largely resolved. A reader can identify the model stages but cannot yet reconstruct the input schema, cross-tree memory update, or tensor identities from the image alone.
