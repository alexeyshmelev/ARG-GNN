# Candidate: aligned mixed-path difference on adjacent whole trees

**Source-only proposal.** One candidate, no implementation, qualification, registration or result. Its potential contribution is a precisely controlled node-level interaction between changing tree neighborhoods and observed correspondence. Commutators, graph filters and multiplex message passing are established. Historical priority and predictive usefulness are unresolved.

## Evidence and changed condition

H029's static, carrier-matrix and history-sham outputs were effectively tied. That establishes no history-specific benefit beyond their shared static correction; it does not identify why history failed. H030 supplies normalized reference cavities, while H032 adds static donor-support Gram gates. Neither asks whether *propagation across two aligned neighboring trees depends usefully on the order of the spatial and correspondence steps*. The candidate changes that condition without another retained memory or pooled graph token.

I inspected H021 and H024 before selecting it. H021 already contains census-cut block diffusion and clade-turnover retention. H024 already registers bidirectional numerator/mass carry across tile edges; its ledger records one successful FP64 reset-equivalence cell, a schema interruption and deferred remaining qualification, not matched predictive carry results. This proposal is neither another census-clade transport field nor a reset/carry rerun. H031's weak direct SNP field supplies no demonstrated cause that this operator solves.

## Observed alignment

Retain every native vertex/edge of each complete sample-ancestor forest, including context samples and the same analytic350 points/reference policy as H030. Construct a sparse partial node map T[h←g] between adjacent holds:

1. Match sampled vertices by the same qualified actual sample/haplotype identity. Dates and roster joins must agree.
2. Match unlabelled vertices only when their observed signature — exact descendant set of actual sampled identities plus exact observed age — is unique on both sides.
3. Leave ambiguous/unmatched vertices unmapped. No resurrection, soft invented ancestor or arbitrary native-ID embedding.

A signature match means “the same uniquely observed tree role”, not “the same real ancestor”. Equality of inferred ages is deliberately conservative; small dating changes may remove matches. T has at most one1 per row/column. Native-ID renaming and common arbitrary renaming of actual sample IDs preserve the relation. Older proxies stay at their actual sampled vertices; matching never fabricates their350 descendants. Classification still reads actual eligible query paths at350.

This is an alignment assumption, not ancestry truth. It needs recorded match/ambiguity rates and dating sensitivity before biological interpretation. Exact observed-role joins may still connect distinct biological ancestors.

## Exact operator

Let H_g∈R[N_g,D] be a shared complete-tree reference encoder output. Use H030's observed proxy values; optional future changes to those values are outside this proposal. Let A_g be a sparse row-normalized tree-neighborhood operator with self loops and nonnegative scalar edge gates from observed endpoint ages/branch lengths. A uses the native/cut adjacency, not reference truth or query targets.

For one direction g→h compute two D-channel paths:

z_spatial_then_align = T[h←g] A_g H_g,
z_align_then_spatial = A_h T[h←g] H_g,

and define

Δ[h←g] = (z_align_then_spatial − z_spatial_then_align)/2.

Every multiplication is sparse. Both paths have the same source H_g and matching geometry. Δ identifies a difference between the two message routes; it is **not** causal attribution of a recombination event. When the observed trees are identical, all roles match, T is the corresponding permutation and A_hT=TA_g, giving exactly Δ=0. With partial matches, Δ also responds to loss/creation of observed contextual support; that must not be described solely as changed edges.

Compute left→current and right→current Δ in parallel over the ordered tree bank. Project each through one shared learned D→D map and a linear complete-tree decoder whose gates depend only on current-tree reference/geometry fields. Static current context remains a separate stream. There is no GRU/DeltaNet cell, past-state reservoir, graph-level embedding aggregation or feature outer product.

At physical readout x inside hold h=[a_h,b_h), combine decoded streams as

c_h(x)=c_static,h + exp(-(x-a_h)/ell_bp)c_left,h
                         + exp(-(b_h-x)/ell_bp)c_right,h,

omitting absent neighbors. Use ell_bp>0 shared across directions; genealogical edge gates use generations, not this genomic scale. The existing centered zero-init residual head and H025 skip can remain matched. The linear decoder permits distance scaling after gathering without inventing within-hold trees. Parameters are shared across all hold pairs.

Before pairing, coalesce adjacent records representing exactly the same observed labelled metric tree. This makes an artificial duplicate hold create no event; without coalescing, nearest-neighbor pair weighting is not generally segmentation invariant. This does not guarantee unary-subdivision invariance of the finite encoder/decoder. Moving graph contents along fixed genomic coordinates changes pairings and distances and can change predictions. Reversing the chromosome with shared bidirectional parameters can remain reflection-equivariant; reflection equivariance is compatible with order sensitivity.

## Cost and decisive control

For N packed vertices, E native/cut edges, M matched adjacent-node pairs and width D, the additional two paths cost O((E+M)D), their channel projections O(ND²), and the whole-tree decoder O((N+E)D²). Each graph participates in at most two adjacent pairs. Additional forward storage is O(ND+M); autograd introduces fixed-depth activation factors. Descendant-signature preparation with B actual samples costs O(N ceil(B/word_bits)) bitset work/storage plus canonical hashing. No all-node pair attention, dense donor-pair tensor or chromosome-long recurrent state is required. These are operation counts, not measured speed/memory.

The decisive matched control replaces Δ by

Σ[h←g]=(z_align_then_spatial + z_spatial_then_align)/2,

with the same encoder, alignment, two computed paths, gates, channel width, decoder, positions, residual head, initialization and tuning budget. Thus it changes the signed path interaction while retaining both trees' information and all allocated states. A tie with Σ fails to support this particular difference operator; a gain only over a weaker static/emitter comparator is insufficient. Tiny corrections or near-zero Δ also cannot establish biological usefulness. Independent-parent confirmation and class/boundary/short-tract behavior remain necessary; two reused VAL parents cannot support generality.

## Closest collisions and claim boundary

A two-layer relation-specific supra-graph MPNN can reproduce these two ordered path channels and subtract them. This is a direct equivalence witness, not a first-spatiotemporal-GNN claim. [Graph Matching Networks](https://arxiv.org/abs/1904.12787) already reason jointly over graph pairs; [GraphSSM](https://arxiv.org/abs/2406.00943v2) already combines graph filtering and persistent-node alignment; [TGN](https://arxiv.org/abs/2006.10637v3) can accept such paired messages through a general message interface. The difference here is the fixed finite mixed-path construction, not an inability of those broad frameworks to express it.

Dynamic tree message-update algorithms already maintain pre/post messages and propagate topology-induced changes. H029 itself uses complete pre/post cavity differences. A_g, A_h and T make this proposal an explicit difference of two propagation routes, not a newly invented update identity. Earlier IntervalLineageGNN also already has all-node clade attention and collection feedback. H032's tree OU/second-order machinery remains unrelated to a new claim for the simple subtraction above.

A supported future claim would concern a useful observed-role aligned mixed-path bias for whole genealogies, with a defined equivalence/control boundary. It would require measured advantage over Σ and an equally informed supra-graph/GraphSSM comparator, plus a focused priority search. The proposed computation is materially different from the current memory and static Gram routes, but it does not yet establish a novel general operator or an evidenced biological bottleneck.

**Inspected definitions:** src/arggnn/reference_logcavity_gnn_v1.py; src/arggnn/reference_dependence_gnn_v1.py; docs/census_cut_exchange_v1.md; docs/biological_cross_tile_state_carry_proposal.md; research/experiment_ledger.md H021/H024–H026; docs/architecture.md; docs/prior_art.md; docs/novelty_assessment_v1.md; existing fresh originality notes. No model/data arrays or current running outputs were read for this proposal.
