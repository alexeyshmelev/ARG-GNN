# Complete sparse joint ancestry model, 29 October 2026

The model initializes all nodes from all descendant sample observations, performs
synchronous native and cross-tree updates, and reads every ancestor of each query
leaf after the joint layers. This closes the shallow-hop failure in which a
branch exchanged information but could not deliver it to a leaf within a few
native layers. The construction keeps complete native trees; it does not cap
path depth, prune unary nodes or identify ancestors across trees.

The fixed data protocol is five complete tree occurrences per parent/site: one
forced focal occurrence with source mass zero and four uniform physical draws
in the clipped plus/minus 50 kb domain, each with source mass 0.25. Registration
and RNG construction belong to the data protocol. Exact duplicate coordinates
remain occurrences with their own quadrature weights. Complete native intervals
supply all their observed SNPs, including SNPs outside the coordinate radius.

The observation encoder uses observed reference/cohort allele summaries, the
sample metadata and categorical SNP observations. It has no target-query marker.
Each site's complete context and unique focal SNP are encoded once for all its
canonical targets. Labels and example IDs remain outside the observation API.
Stable parent/sample/haplotype identity is validated before cross-tree routing.

## Immutable routing and native computation

Let D(u) be the sampled descendants of native node u. The descendant matrix has
entry 1/|D(u)| for each descendant leaf and initializes node u from their mean
encoded observations plus geometry. A nonlinear node network transforms that
mean. Native child-mean and parent maps then support synchronous learned updates.

After each layer's native and cross updates, the model applies a residual GELU
and per-node LayerNorm. There is no normalization across site windows. After the
last layer, each leaf combines its own hidden state with a weighted mean of all
ancestor states. The same initialization, native maps, all-depth readout, focal
SNP encoder and classifier are used for all control arms.

All graph maps are cached CSR. The `SparseSum` PyG MessagePassing subclass uses
fused `message_and_aggregate` and `torch.sparse.mm`. It never constructs Python
edge-by-hidden-channel message tensors during normal forward passes. CSR maps,
indices and weights are prepared once, not coalesced each training step. This is
an implementation fact, not a claim that every model activation is small: SNP
pooling still materializes SNP-row by sample by channel activations.

## Branch correspondence measure

For a finite branch u spanning [a(u), a(parent(u))), use the fixed log-age
measure d mu(a) = da/(100+a), with ages in generations. Its mass is

    m(u) = log1p((a(parent(u))-a(u))/(100+a(u))).

The direct interval expression avoids cancellation when a short positive branch
occurs at a very large age. For target branch u and source branch v, let I be the
number of shared descendant sample identities and let m(u intersect v) be the
same log-age measure over the overlapping age intervals. The branch weight is

    B(u,v) = I * m(u intersect v) / (|D(u)| * m(u))
             * I / (|D(u)| + |D(v)| - I).

The final factor is descendant Jaccard specificity. It suppresses a source clade
that contains the target clade but is much broader. The relation is directional
because its measure denominator belongs to the target branch. For each tree
pair, the numeric row mass is substochastic. It is not an ancestor-identity
probability; it combines dating availability and descendant specificity. A
single hidden vector is constant over a branch's lifetime, so the method is not
an independently age-resolved hidden-state field.

Exact shared-descendant counts are accumulated by walking aligned sample paths
through overlapping branch intervals. Root nodes have no finite branch and no
branch-cross message. Zero-duration branches also have no branch-cross message.
Unary and zero-duration nodes remain in native computation; full-network
invariance to their insertion/removal is not claimed.

For source occurrence j and target occurrence i, genomic weight is

    q(j) * max(0, 1 - |x(j)-x(i)| / 50000).

It is not divided by a neighbor sum, so absolute distance attenuation persists
with a single available source. Left and right channels are distinct; exact
coincidence splits weight equally between them. Separate exact same-leaf
channels and branch-overlap channels are used. Each cross residual is

    C_r H - s_r H,  where s_r = C_r 1.

The row mass s_r is computed from the exact coalesced numeric CSR, ensuring that
constant hidden states yield zero residual. Learned linear projections and
sigmoid gates operate on each relation. The structure arm uses four relations;
leaf_only uses the two same-leaf relations; no_cross uses neither. Inactive
cross parameters have no gradients and active capacity differs across controls.

## All-ancestor readout

Along a leaf's full native path, each finite branch gets its log-age mass. A
leaf atom of one and a root atom of one are added before row normalization. A
singleton root/leaf receives both atoms, resulting in a normalized weight of
one. The root atom deliberately retains root context without inventing an
infinite root branch. These atoms and the fixed 100-generation age scale are
scientific priors, not tuned quantities in the registered first benchmark.

This readout allows deep foreign branch content to affect a focal prediction
regardless of the number of native-hop layers. Qualification sets native maps
to zero and shows a nonzero gradient from a foreign branch 161 native hops away
under structure, with exact zero under leaf_only. It demonstrates information
access, not a biological accuracy improvement.

## Complexity and limits

Let P be the sum of sample-to-root path lengths, N the node count, J the number
of occurrence SNP rows, H the sample-panel size, d the width, and E the cached
nonzeros. Topological validation is O(N). Descendant/readout construction stores
O(P) entries. For K occurrences, overlap preparation walks leaf paths for each
eligible tree pair; it has an explicit witness-work bound and checks intermediate
and final nonzero counts. P can be quadratic in node count for pathological
unary trees. No universal linear-complexity claim is made.

Forward graph aggregation is O(E*d) per layer/readout, plus node linear layers
O(N*d*d). SNP features include J*H*d pooling activations. The immutable cache
stores CSR maps and raw observation tensors, never learned activations. Execution
bounds reject oversized windows explicitly without truncation. Complete training
cache preparation and representative GPU/full-epoch profiling must precede a
training run; a single small CPU timing is not a scaling qualification.

The no_cross arm can exactly discard nonfocal occurrences because every active
operation is tree-local until final target selection. `no_cross_view.py` performs
that immutable CPU view once and retains every focal ancestor and SNP. It is a
separately tested execution optimization, not a change in control information.

## Qualification and prior art

`qualification_receipt.json` records full outputs and all-parameter VJPs versus
an independent materialized-scatter oracle on toy and real observed packets for
all three modes. It also records descendant-oracle equality, sample/internal-ID
permutations, isolation, cache roundtrip, distance attenuation, Jaccard specificity,
constant residuals, duplicates, huge-age small branches, singleton forests and
tall native paths. Independent shared-query and batch-isolation tests are in
`../review/SHARED_SITE_MODEL_REVIEW.json`. Historical failed harness runs remain
alongside corrected source and logs. The first failure was a bitwise gather
assertion; its actual error was 5.96e-8 and the recorded FP32 tolerance is 3e-6.

Earlier prior-art research is preserved in
`../../sliding_graph_convolution_20261028/`. MS-G3D describes sliding joint graph
windows analogous to 3D convolution, and STSGCN uses adjacent graph slices with
central outputs. We make no generic new-convolution claim. Exact descendant
branch overlap is an integral of known tree-cut incidence, not inferred ancestor
identity. This model has not yet established accuracy gains or biological
calibration; the registered paired benchmark determines those outcomes.
