# Prospective exact observed-clade flag control

This is a design contract only. No source was implemented or executed. Root may commission a separate, registered control after examining the current H040 comparison. H040's frozen witness/model/worker and completed evidence must remain unchanged.

**Question:** Does a complete but simple indication of *where observed rooted topology changed* explain the benefit, without learning any old-node/new-node pairing? This is a tree-specific feature control, not a new general ML method.

## Observation-only construction

Input only the observed parent forest, its structural levels, each native tree's sampled-leaf indices and their stable haplotype-column keys. Require the same full sampled panel in every native tree. Use no reference class, target, age, branch time, mutation evidence, prediction or validation score to construct flags. Retain the existing model's observations separately and unchanged.

For one native tree, create an ephemeral exact bitset for every node: a sampled leaf has its column bit set, and an internal node takes the OR of its children's bitsets. A 952-leaf mask requires 15 64-bit words, or an exact arbitrary-precision integer. Equality must be exact. A hash can accelerate dictionary lookup only if full-bitset equality resolves collisions. These masks never become model input.

Let C_t contain the distinct *proper nontrivial* descendant-leaf sets in tree t: sizes 2 through H-1. Empty/singleton sets and the full-panel root set are excluded. Collapse duplicate sets introduced by unsampled unary chains for this comparison. Retain all original network nodes. For flag placement, choose the deepest node in each duplicate-set chain using structural depth, independent of numeric node IDs or branch ages. Set the other equivalent nodes' flags to zero. Check that equal-set duplicates form such a chain; reject unsupported structure rather than silently conflating it. Sampled-interior cases require an explicitly extended definition and must be rejected by this initial full-leaf contract.

For each adjacent pair, exact set differences identify lost and gained clades. Return two small channels in original native-node order:

    gained_from_left[v in tree t] = 1[C_t(v) not in C_(t-1)]  if t > 0
    lost_to_right[v in tree t]    = 1[C_t(v) not in C_(t+1)]  if t < T-1

Apply these definitions only at the representative of a proper clade; other entries are zero. The unavailable outer-neighbor channel is zero at each sequence end. Channel meaning is fixed in the schema. Reversing genomic coordinates reverses tree order and swaps the channel roles.

Process adjacent trees sequentially. Keep only their temporary bitsets/representatives plus the accumulating N-by-2 flags, then discard bitsets once comparisons are complete. Retained CPU flags can be uint8 (2N bytes), converted to FP32 on the model device if desired (8N bytes). No dense bitsets or cross-tree correspondence matrix is stored on disk, cached across parents, or sent to the GPU. Compact provenance may include bank/source hashes, shape and flag-content hashes. Straightforward preprocessing uses bottom-up bitset unions plus exact set membership, rather than enumerating H cubed triples or all node pairs. Root must measure actual costs if this control is commissioned.

## Matched network use

Use the identical selected H038 warm start, original width-32 whole-tree body, all native holds and sampled haplotypes, original query/readout mappings, two shared whole-tree sweeps, ordered bidirectional GRU and classifier. After the first sweep, use its node state h_v and the two channels to create a *local* injection. One concrete 1,024-weight adapter is

    z_v = gained_from_left[v] * ((U h_v) * (V h_v))
          - lost_to_right[v] * ((U h_v) * (W h_v))
    injection_v = O z_v

U,V,W are bias-free 32-to-8 maps and O is a bias-free 8-to-32 map: exactly 1,024 added weights, matching H040. All maps participate when their corresponding flags occur. Initialize O to zero and initialize the other maps under the same recorded seed/CPU policy. Inject at original nodes, run the second sweep from base + injection, and retain the original sequence geometry/GRU/head. There are no old/new endpoint pairs and no gathered neighboring-tree latent states.

This adapter matches parameter budget and message placement depth, not the tensor arm's hypothesis class. Its purpose is to test simpler observed topology information. The zero-O initialization preserves the inherited model's initial output and initially blocks U/V/W gradients until O moves, as in H040. Match optimizer reset, training exposure, parent order and checkpoint opportunities if commissioned. A no-injection continuation with the same exposure remains necessary to isolate added optimization. Root owns registration, execution admission, actual source closure and all evaluation. This contract authorizes no run or validation/ground-truth access.

## Exact meaning and limits

For rooted leaf-labelled trees after unary suppression, the proper-clade set determines rooted topology. Therefore any change of that topology yields at least one gained/lost flag. In particular, (L,((x,y),z)) versus (L,(x,(y,z))), with every landmark inside L, produces flags on the changed nonlandmark clades even though H040's K4 witness produces no event.

This is complete *topological-change detection*, not complete genealogical information. Equal clade sets can coexist with different node times, branch lengths, mutations, phase quality or evolutionary histories. Flags cannot certify recombination, ancestry transitions, ancestral homology or identity. Identical observed descendant sets across trees are a set-equality relation, not evidence that their internal nodes represent the same historical ancestor. Existing-tree inference error can generate flags. Suppressing unary equivalences in the feature does not make the unchanged GNN invariant to extra unary nodes.

Before root execution, only small synthetic CPU properties are needed: internal-node renumbering and sample-column permutation with stable keys preserve flags, identical topology with altered ages yields zero flags, unary subdivisions create no new changed clades, the landmark-blind counterexample is detected, and a known subtree rearrangement marks exactly the symmetric-difference clades. These are a short specification for future source validation, not tests run by this audit.

If the local-flag control performs as well as H040 under matched evaluation, the evidence favors useful tree-change feature engineering. If H040 also outperforms this control and an effective degree/anchor-preserving rewiring null, that would support useful correspondence within this task. Neither outcome alone would establish a general machine-learning contribution.
