# Independent image-only reader review

Image: paper/figure_redesign_20261017_v6/TreeLoom_architecture_2x.png
SHA-256: 18b1db63f91cdaceac8780f28b18515f1f83322f34b4048ee511452002a771a4
Dimensions: 3080 × 2222 px. Inspected full-resolution regions and a 1180 px full-width preview. No manuscript, caption, source, or previous review was read.

## Recovered method

An ARG over three genomic intervals becomes local trees T1–T3 containing four reference samples and query q1. Features visibly include alleles and known reference ancestry categories/one-hot indicators; query ancestry is unknown. The internal-node callout shows τ, 0, 0, suggesting a time value and zero-filled feature fields whose exact meanings need a caption.

The drawn pipeline is MLP → upward then downward GNN → clade update → residual sum → another upward then downward GNN. Initial MLP embeddings bypass to the sum. The sample-memory branch taps the first GNN output BEFORE clade update; its Linear(z̄ − f) correction also enters the sum. The two GNN blocks share weights. The drawing shows two up/down applications (four directional traversals); no further repeated loop is drawn.

Memory has per-sample states z linked bidirectionally across trees. The role of across-tree sample memory is visible; its cell type, layer count, pooling definition, and exact f/z definitions are not specified. Widths w_t and z̄ suggest weighted aggregation but do not uniquely define it. This is separate from the explicitly one-layer final BiGRU. The latter processes the tree sequence and receives log(1 + ·)-transformed interval/context features. Its output is concatenated with the tree embeddings and passed through an MLP to ancestry probabilities p_tq; argmax produces a genomic ancestry track.

## Trees and connections

All three ARG slices match their extracted rooted binary trees and leaf order:
- I1/T1: (r1,(r2,(r4,(q1,r3)))).
- I2/T2: (r2,(r1,(r4,(q1,r3)))).
- I3/T3: (r4,(r2,(r1,(q1,r3)))).

q1 and r3 remain sisters. Reference labels/categories follow the reordered sample identities correctly. The inset shows the expected recombination pattern: one descendant and two ancestral paths with time increasing upward. The pale interslice lines need a caption to establish their correspondence role; they are visually distinct from the dark within-slice ancestry branches. No concrete topology mismatch or nonbinary extracted node was found.

The first-GNN→memory T-junction has no dot. At first glance I placed that branch after clade update; enlarged tracing resolves its actual origin before clade update. A junction dot or a slightly separated tap is an optional small clarity improvement. The residual skip and final embedding-to-concatenation bypass have visible dots and clear endpoints.

## Caption and legibility

Required caption explanations: feature order/τ and zero fields; reference/query masks and ancestry abbreviations; Nt, S, d; f, z, z̄ and memory aggregation; whether the two shared-weight GNN applications are the complete count; w_t, Δc_t−, Δc_t+ and their input transformation; concatenation symbol and p_tq. These are definitions, not reasons to add more figure text. An optional sentence can say that magenta traces q1 through computations performed for samples generally.

At 1180 px full width, the main order, trees, Up→Down, memory, one-layer BiGRU, and output are readable. Individual allele cells, the τ/0/0 callout, memory symbols, and Δc subscripts/superscripts require enlargement or caption support. No actual label overlap or connection through text was found. The route beside “Clade update” has a narrow but visible gap. Repeated T labels, q1 highlights, and ancestry names support identity across panels; no clearly harmful redundant prose was found.

No mandatory graphical or tree correction follows from this image-only check. The pre-clade memory tap is the corrected reading of the image, and its subtle junction is the main optional graphical improvement.
