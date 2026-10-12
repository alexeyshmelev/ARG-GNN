# One architecture hypothesis: sample-anchored latent clade exchange

2026-10-12. Separate design subtask requested after the independent novelty review. That review remains immutable. This is a source-level specification, not implementation, admission, experiment registration or a novelty certification. It leaves the biological CRF accuracy proposal separate.

## Intended operator

Add one latent boundary layer between complete-tree encodings. A rooted-triplet reassociation links two anonymous clades through a genuinely persistent observed sample. Their learned states interact before being injected into both whole trees and rerouted to every sampled leaf. It uses neither ancestry transition labels nor final four-class logits.

The possible ML contribution is this **sample-derived incidence lift and latent clade rerouting for ordered rooted trees with anonymous changing internals**. The tensor kernel, higher-order interactions, antisymmetry and interleaving are established ideas. This is a concrete candidate within higher-order temporal GNNs, not a claim to escape the general MPNN family.

## Eligible observation-only witnesses

Fix K=4 distinct observed haplotype landmarks per parent, selected uniformly with a frozen seed over stable sample keys, independent of donor class, outcomes and prediction confidence. Save their actual sample/haplotype keys and parent namespace. These are actual sampled leaves at their observed ages, not internal ancestors or synthetic census points. The learned operator never embeds those keys.

Use every distinct sample triple containing at least two landmarks. With H leaves the count is (H-K) choose(K,2)+choose(K,3). Resolve each triple separately in two adjacent native trees by structural LCA ancestry. A winning pair has a strictly lower LCA than the common LCA of the other two pairs. Ties, unresolved triples or unsupported sampled interiors produce no witness, with coverage recorded. The original H038 sampled-leaf contract still applies. No query or scored bin is excluded because its witness set is empty.

For a triple whose winner changes, its two winning pairs necessarily share one sample u:

    old winner = (u,v)
    new winner = (u,w)

Here u is the shared sampled identity, v is the losing partner and w the gaining partner. The shared sample need not be the non-landmark member. Define anonymous receiver nodes

    ell_old = LCA_old(u,v)
    ell_new = LCA_new(u,w).

Only within-tree structural LCA lookups are used. The two ell indexes are not matched ancestral entities. Equal numeric indexes across trees have no meaning. Unchanged resolutions give zero boundary events even when node ages or latent states change.

## Latent coupling and injection

Run the existing full-tree upward/outside encoder once. Let p_t(n) be its upward node state and h_t(s) its full sampled-leaf context, both width d=32. For each eligible event at boundary i between holds i and i+1, set

    a = (h_i(u) + h_(i+1)(u)) / 2
    b = p_i(ell_old)
    c = p_(i+1)(ell_new)

and use bias-free U,V,W in R^(r x d), O in R^(d x r), with r=8:

    z(a,b,c) = (U a) * [(V b)*(W c) - (V c)*(W b)].

Products are channelwise. z is a rank-r vector, not an ancestry score. It compares the old and new clade latent states jointly, conditioned on their shared observed sample. b and c are actual complete-tree messages, not engineered class summaries.

Let d_old(ell) and d_new(ell) count eligible events incident to each receiver at this boundary. Weight each event by

    eta = 1 / sqrt(d_old(ell_old)*d_new(ell_new)).

Counts are positive for every eligible event. Scatter -eta*z to ell_old in the old tree and +eta*z to ell_new in the new tree. Sum contributions from both neighboring boundaries at each native node, obtaining R_t(n) in R^r. Then rerun the same complete-tree cavity sweeps with

    base_second_t(n) = base_first_t(n) + O R_t(n).

Reuse the same gates and learned sweep weights. Gather the second-pass leaf contexts and retain the original span-aware genomic GRU and classifier identically in both comparison arms. The second pass starts with fresh within-tree upward/outside buffers. No anonymous internal state is carried along a presumed ancestor identity.

O initialized to zero makes the new layer initially reproduce the original H038 tree representation: the second sweep sees the same base and gates. Gradients initially reach O, and then the relation projections after O becomes nonzero. That warm-start behavior must be disclosed. No learned normalization or nonlinear transformation is placed between the signed rank-r message and scattering in this first design.

## Precise current-source hooks and proposed API

Current PyG encode_trees computes base/gates/degree at lines 103–123, then upward, message_sum, child_message and outside at lines 124–153, and returns only leaf_context at lines 154–156. The latter return currently hides exactly the intermediate p states needed here. The existing forward_holds calls encode_trees and applies geometry/GRU/head in whole_tree_gnn lines 410–420.

A future separate module, for example src/arggnn/anonymous_clade_tensor_gnn_v1.py, can inherit the full-panel model and implement shared helpers:

    prepare_tree_base_v1(bank, control)
      -> base[N,d], ug[N,d], dg[N,d], degree[N]

    sweep_tree_state_v1(bank, base, ug, dg, degree)
      -> upward[N,d], outside[N,d]

    couple_boundary_clades_v1(bank, boundary, upward, leaf_context, interaction)
      -> R[N,r]

    forward_holds(bank, *, boundary, control="aligned", interaction="tensor")
      -> logits[T,H,4]

Both tree passes must use the existing self.pyg_upward/self.pyg_downward with identical level arrays and original message exclusion. Compute site inputs/base/gates only once per call. The sweep helper retains child_message and message_sum internally exactly as the current operator does. No claim of an exact probabilistic cavity is introduced.

The future BoundaryCladeWitnessInputsV1 sidecar contains event arrays:

- boundary_index[E], integer i in 0..T-2
- old_node[E], new_node[E], global bank indexes within their respective native trees
- shared_sample[E], stable sampled column in 0..H-1
- weight[E], observed degree normalization eta
- audit triple/old-pair/new-pair indexes and frozen landmark keys in observation metadata.

It must validate that each receiver is the correct within-tree winning-pair LCA, that sample joins follow bank.sample_index, and that all fields came from inferred topology/observed identities. Existing input caches and dataclass need not be rewritten. Training readout selects bank.query_column, and evaluation retains bank.readout_hold on the original grid. Candidate forward_train_holds/forward wrappers accept the separate boundary argument and retain those exact gathers. interaction is a prospectively fixed registered arm, not a choice made after seeing validation outcomes.

## Mathematical distinction and its limits

For independent latent perturbations da,db,dc, the core has

    D_a D_b D_c z[da,db,dc]
      = U da * [V db*W dc - V dc*W db],

which is generically nonzero. Thus the raw core cannot be written as a sum of functions involving at most two of a,b,c. This only distinguishes an explicitly additive pair-message ablation. It does NOT prove superiority or greater expressivity over a generic nonlinear event MLP, which can also have a nonzero third mixed derivative.

Unlike H038 GNN+GRU, this layer brings two within-tree clade states together at the boundary before graph rerouting. Unlike an ordinary interleaving change, its candidate object is the particular two-tree event incidence and coupled latent update. A sufficiently flexible temporal factor-graph MPNN given the same witnesses can implement it. That comparator is essential.

It is not OT: no coupling optimization or marginal-mass matching is performed, and the message is a signed joint interaction of latent values. It is not a diagonal SSM or semigroup recurrence: there is no generator, exponential, interval integration or persistent anonymous-state trajectory. No interval-subdivision invariance is claimed. H035 tested signed compositions of tree/alignment linear operators, whereas this uses sample-anchored old/new clade latent interactions. H032's within-tree reference Gram did not test this two-tree incidence. Those differences specify a new condition, not grounds to rerun unchanged negative experiments.

### Reversal covariance

At the isolated operator level, reversing the boundary swaps b,c and the two receiver roles while leaving a unchanged. z(a,c,b)=-z(a,b,c). eta is symmetric in its endpoint degrees. Consequently the reversed operator sends the same perturbation to each physical tree, with its two output trees exchanged.

This requires the same landmark set, unordered triple enumeration, structural resolution rule, U/V/W/O and degree normalization in both directions. A learned directional gate, asymmetric distance feature, bias after antisymmetrization, independent left/right weights or orientation-dependent witness sampling would break this guarantee. The retained bidirectional GRU is not automatically guaranteed to give identical predictions under chromosome reversal.

Storage-node renumbering changes only structural indexes, not outputs. Sample-column permutation is equivariant when stable keys/landmarks and sidecar indexes are transformed consistently. Empty event sets and b=c give zero injected messages. These are operator properties, not empirical findings.

## Closest prior and honest attribution

**Strongest kernel prior: Trouillon et al., ComplEx, arXiv:1606.06357, §3/equation 11.** Its imaginary-relation term is antisymmetric trilinear factorization. Setting the complex entity channels to Vb+iWb and Vc+iWc, with imaginary relation Ua, yields the same antisymmetric channel product. Multiplication, low rank and antisymmetry are therefore inherited, not new. Here the relation is a dynamic shared-sample state, the result remains a vector, and the incidence injects it into two anonymous clades before whole-tree propagation rather than outputting a scalar link score.

**Strongest graph-layer prior: Bodnar et al., MPSNs, arXiv:2103.03212, §4 and Theorem 14.** Higher-order local message passing and orientation-equivariant constructions already exist. The proposed incidence is formed by a rooted reassociation across two different trees through persistent leaves, rather than the boundary/co-boundary adjacency of one simplicial complex. MPSNs can be augmented to model these events. No 3-WL or general expressivity improvement is inferred from their theorem.

**Closest temporal motif prior: Wang et al., CAW-N, arXiv:2101.05974, §§1–3.** It extracts temporal motifs, preserves relative node identity through anonymization, and encodes sampled walks with RNN/set pooling. Anonymous motifs and persistent-relative identity are not new. This proposal deterministically lifts rooted-triplet exchanges to actual old/new LCA message states and reroutes vector messages into both trees.

Full arXiv HTML for all three was freshly accessible. GST's local/global interleaving and ordinary MPNN machinery remain additional attribution. A targeted search for prior latent LCA-event layers is still required before any priority claim.

## Discriminating comparison

The required main comparison is tensor versus an ordinary event MLP with exactly the same witnesses, a/b/c states, signed scattering, normalization, extra sweep, GRU/head, initialization and optimization:

    z_eventMLP = GELU(Ua + Vb + Wc) - GELU(Ua + Vc + Wb).

Apply the same O afterward. Both arms have exactly 4dr=1,024 added learned weights and no biases. Both obey receiver-swap antisymmetry and see every input. This control can learn three-way interactions. A tensor win would support this structured coupling bias/efficiency, not the mere availability of topology features or interleaving.

A focused secondary mechanism ablation replaces the trilinear core with the pairwise expression

    z_pair = (Ua)*[(V+W)b - (V+W)c] + (Vb)*(Wc) - (Vc)*(Wb).

It uses all the same matrices and routing, but its raw core contains only pair interactions. It must not be described as a generic-MLP expressivity baseline. If only tensor-versus-pair wins while eventMLP ties, the result favors higher-order conditioning, not this particular operator.

Before a general-ML claim, a controlled non-ancestry ordered-tree task should require the joint shared-leaf/old-clade/new-clade relation with matched one-/two-input marginals. A parity-style latent triplet target can diagnose that requirement, but the generic eventMLP can solve it too. This is a proposed discriminating task, not a tested result or permission to launch.

## Efficiency and decision

At K=4,H=952 there are 5,692 candidate triples per boundary. For T=1,000, at most about 5.69 million events are examined before unresolved/unchanged masks. A query-to-landmark plus landmark-to-landmark LCA cache costs about 15.3 MB per parent in int32, subject to verified node bounds. Do not assume inferred tree edits are sparse without observing event counts.

With M total native nodes and E eligible events, projecting node/sample states, computing interactions and scattering costs O(Mdr + Er), plus one additional full-tree sweep O(Md²). Avoid an H×H matrix and all H³ triples. Generate/gather event blocks from the LCA cache, scatter rank-r accumulators, and checkpoint blocks. Peak autograd memory, CPU preprocessing and NFS sidecar I/O remain unmeasured. Old/new gradients remain connected. No detaching or hidden tiling approximation is part of the specification.

This is implementable and testable, with substantial novelty risk. If the ordinary eventMLP matches it, the new incidence representation may still be useful, but a claim for this tensor operator is unsupported. Even a win would require closer prior-art checking and broader evidence. The design offers a concrete architecture hypothesis beyond final-logit CRF features, not a guaranteed publishable generic-ML contribution.
