# Next graph-collection operator: one constrained-coupling hypothesis

**Recommendation: hold implementation.** This audit retains one mathematically specified utility hypothesis: jointly weighting edit-compatible ancestor-role correspondences and the task-affinity messages they transmit. Its distinguishing ingredient would be a *globally coupled distribution over observed tree-edit explanations*. Graph matching, probabilistic message passing, iterative correspondence refinement and posterior feedback already exist. No defensible new general operator, priority claim, observed biological failure or execution-ready next experiment has emerged.

H048 remains frozen. The fresh H048 review establishes the exact RFMix conditional reset family and commuting fixed-pi scalar generators. A full class-pair rate matrix would revisit H039/neural CTMCs. Neither another rate parameterization nor a GRU/attention wrapper addresses this audit's question.

## What the project has already tried

| Prior route | Relevant mechanism and recorded result | Consequence |
|---|---|---|
| H001 / IntervalLineageGNN | src/arggnn/models.py:254–353 already sends all-node foreign content with identity/Jaccard biases back through tree layers | Generic ancestor attention or context-before-query readout is already present. |
| H004–H005 | Query anchor states feed later source projections; noncontextual control was essentially tied | Feeding predictions/context back into sources alone is not a new condition. |
| H019/H021 and the incidence audit | Normalized bipartite/partition diffusion, incidence lift/project and clade-overlap kernels | A new name, age basis or normalization does not escape that family. |
| H025 | Qualified persistent ancestral branch traces and sample-only/sham controls | Ancestral persistence is already a tested hypothesis; genuine traces did not beat simpler controls. |
| H029/H030 | Whole-tree reference cavities and learned carrier GRU/matrix history before query pooling | Moving memory inside tree decoding is already covered. History did not beat shuffled controls in accuracy. |
| H033 | Repeated tree inference and genomic TV feedback, interleaved versus blocked | Posterior-like task fields already enter tree inference. Selected accuracy was essentially tied, 85.969702% versus85.969468%. |
| H035 | Structural role map T; (A_new T − T A_old)/2 versus the matched sum | Tree/alignment compositions and noncommutativity are already tested; signed difference lost. |
| H039 | Directed graph-witness-conditioned native-hold CRF | Directed transitions, chain sum-product and graph→decoder conditioning are established here. Directed95.7119%, plain95.5635%, generic95.4964%,320 updates/arm,8reusedVAL/one seed. |
| H040/H042/H043/H045 | Changed-triplet interactions, local flags, nearest changed-clade foreign states and exact-clade episodes | Foreign internal-state exchange is already explicit. H042 flags beat H040; H043 gain0.0153pp and H045 gain0.009568pp did not meet the0.10pp development criterion. |

These outcomes do not prove all graph exchange useless. They make another generic feedback or fixed-correspondence experiment a weak next step. H044's localization gain0.00526pp with a boundary-F1 decrease also supplies no causal evidence that an uncertain correspondence model fixes the remaining errors.

## The sole retained candidate

Call it **an edit-explanation mixture factor**, a descriptive name rather than a new hypothesis ID.

At boundary b, let T and T' be complete observed dated trees with the same actual haplotype roster. Construct A_b, the set of distinct *observationally admissible* zero/one-rooted-subtree-prune-regraft explanations. Each explanation a induces a partial one-to-one map M_a between surviving internal vertex roles. Pruned/suppressed/new roles are unmatched. Persistent samples have their known identity correspondence separately. An explanation is not an inferred true recombination event or proof of ancestor identity.

This is more restrictive than independently scoring every node pair: all pairs in M_a must arise from the **same edit explanation**. The map may connect surviving coalescent roles whose descendant sets changed during the edit. Raw descendant overlap cannot certify these continuities. Different descriptions inducing the same partial map must be quotiented or assigned explicit prior mass; enumerator multiplicity must never become accidental evidence.

A source-only three-leaf example shows why topology alone need not determine continuation. Transform ((a,b),c) into (a,(b,c)). With a root stem available, one may prune a and regraft above the residual (b,c), prune c and attach beside b, or prune b and attach beside c. These explanations preserve different old internal vertices. They are topology-only possibilities. Exact true dates can eliminate alternatives. This example does **not** establish ambiguity in the actual inferred observations or calibrate their dating error.

Biological motivation is continuity of an ancestral *reference-affinity context* across a plausible local recombination edit, with explicit uncertainty about which reconstructed vertex role continues. A K-state internal variable H_g,u describes a latent task-affinity category. It does not assert a real ancestor's population label, admixture time, recombination rate or biological identity. Query Y_x,i variables remain ancestry labels on the unchanged original marker grid. Only observed source-reference labels can be clamped.

An existing complete-tree encoder provides observed-only node unaries and marker evidence. Ordinary within-tree compatibility factors and an unchanged query-coordinate decoder link H to Y. Add one factor per boundary:

    F_b(h,h') = sum_(a in A_b) rho_b(a)
                   product_((u,v) in M_a) psi_b,u,v(h_u,h'_v).

Here rho is a normalized, predeclared observed-only weight over admissible explanations. psi is a positive shared compatibility, for example exp(kappa*1[c=d]). That Potts form is established. No new transition law is claimed. If A_b is empty, F_b=1: no anonymous-ancestor exchange crosses that unresolved boundary. The ordinary persistent-sample route stays present. Empty support must be reported, never repaired by forced nearest-node matching or top-k overlap.

A uniform prior over distinct feasible maps is a structural modeling convention, not an SMC likelihood. Substituting a genuine SMC/SMC-prime transition density would reuse established population-genetic inference. Observed dating uncertainty is currently unspecified. An arbitrary age tolerance must not be selected to make the candidate work.

## Concrete message operator and feedback

Let r_u(c),s_v(d) be normalized incoming cavity messages, excluding the destination boundary factor. Define

    Z_uv = sum_(c,d) r_u(c) psi_uv(c,d) s_v(d).

Because each M_a is a partial matching, its pair factors have disjoint endpoints. The standard sum-product message from F_b to a right node v is

    m_(F→v)(d) proportional to
      sum_(a:v unmatched) rho(a) product_(e in M_a) Z_e
      + sum_(a:(u,v) in M_a) rho(a)
          [sum_c r_u(c) psi_uv(c,d)]
          product_(e in M_a excluding (u,v)) Z_e.

The left-directed formula is symmetric. An unmatched role contributes no fabricated counterpart. Normalization occurs after summing explanations. In exact arithmetic this is the **ordinary exact factor-to-variable BP update** for this factor under its incoming product message. On the whole tree/coordinate/boundary factor graph, loops remain: these messages do not make global inference exact or guarantee convergence.

Use a predeclared two-round schedule if this ever becomes an executable experiment. Initial unaries come only from observations and observed references. Each round includes all original tree nodes/edges, the complete genomic coordinate support and boundary factors. Updated query beliefs feed tree messages, changing the next round's internal cavity messages and the relative support for competing edit explanations. No query target, teacher-forced TRAIN label or VAL truth enters these updates. Full autograd crosses both rounds and all genomic positions. Two rounds are a concrete small hypothesis, not a tuned iteration count or a convergence certificate.

This transmits foreign internal-node belief vectors into local tree inference. It is not merely a graph scalar modifying a leaf-chain transition. The additional effect is **explaining away between correspondence choices**: evidence at another endpoint can reweight the explanation and thus change a node's foreign context.

A generic algebraic witness makes that distinction precise. Take two old roles u1,u2 and two new roles v1,v2, with two equally weighted full matchings, identity and swap. Let r_u1=(1,0), r_u2=(0,1), s_v2=(1/2,1/2), and psi(c,d)=3 for equality and1 otherwise. Changing s_v1 from(1,0) to(0,1), while holding the left vectors and v2 fixed, changes normalized m_(F→v2) from(3/8,5/8) to(5/8,3/8). A fixed overlap matrix or independent per-target row map with those inputs fixed cannot reproduce this boundary-block dependence on v1. This is a hand-derived matching-factor witness, **not** proof that these two full permutations are admissible SPR maps on the project trees. It does not prove a universal expressive separation from sufficiently deep GNNs.

For independent internal-node permutations, the admissible-map family and variables permute together, so the factor/message equations are equivariant. Actual sample IDs are joins, not numeric features. There is no claimed ancestor-identity, arbitrary graph-order, unary-insertion or serialized-hold-subdivision theorem. Extra identity factors can change a model; any duplicate-hold handling must be explicitly fixed and shared.

## Falsifiable matched control

The primary contrast would isolate **adaptation of the joint correspondence weights**, rather than generic feedback or extra computation.

For destination v and round ell write

    W_(a,v)^ell = rho(a) product_(e in M_a not incident to v) Z_e^ell,
    Q_(a,v)^ell(d) = 1                         if v is unmatched,
                     sum_c r_u^ell(c)psi(c,d)  if (u,v) is matched.

The candidate uses sum_a W_(a,v)^ell Q_(a,v)^ell. The **fixed-correspondence-evidence control** uses sum_a W_(a,v)^0 Q_(a,v)^ell. Both retain current foreign messages Q, tree/query posterior feedback, the same globally admissible edit family, node unaries, all trees/SNPs/queries, parameters, initial weights, two rounds, optimizer and training budget. The control recomputes W^0 from the same initial observation-derived messages in each round to retain the declared factor-calculation budget. It does not detach W^0: ordinary gradients still reach its initial encoder.

The first round is identical. Only later dependence of correspondence confidence on updated, off-target cavity beliefs differs. The control is an intentional algorithmic ablation; do not call it exact Bayesian inference for a second joint model. Equal asymptotic work and parameter count are proposed, not measured equal runtime or effective capacity.

A positive result would support adaptive constrained correspondence for this task. It would not establish the usefulness of all ancestor exchange, the novelty of posterior feedback or a new graph-collection algebra. A tie/negative result falsifies the narrow adaptation hypothesis at the frozen budget. A conventional one-way encoder+chain result remains useful context, but an unequal-budget historical score cannot isolate this mechanism.

Before any actual score, root would need a new registration and a fixed endpoint. A reasonable utility rule would be mean Present182 accuracy gain≥0.10pp without mean1kb boundary-F1 loss, with per-parent paired effects and sampleSD across the original8development parents. This is a suggested future rule, **not registered**. Reused VAL and one optimizer seed remain exploratory. H045/global-best comparison must be reported separately. No confirmation/outerTEST or cross-domain claim follows.

## Why it is not a new general operator

- CRF-RNN formulates mean-field CRF inference as a recurrent neural module and trains end to end. Iterative inference and gradients through feedback are established.
- Factor graphs/sum-product and generalized BP establish the inference machinery above. The displayed message is a factorization derivation, not new BP.
- GMN §3.2 jointly reasons over graph pairs with foreign-node attention. Its row-attention route is a close baseline, not the entire graph-matching literature.
- Wang et al., ICCV2019 already learn combinatorial node/edge affinity with a permutation objective; structured matching constraints are established.
- Deep Graph Matching Consensus, ICLR2020 iteratively refines correspondences using neighborhood message passing. “Update correspondence using graph beliefs” alone is already prior art.
- ARGweaver performs SMC-conditioned genealogy threading and extends it to subtrees. SMC-prime is an established ARG approximation. A genealogy-edit prior and joint uncertainty do not create a new population-genetic principle.
- H001/H005/H025/H029/H030/H033/H035/H039–H045 cover the project's nearest fixed-attention, feedback, identity/role transport, cavity, chain and foreign-state routes.

The candidate's restricted novelty question is whether *this particular observed SPR-admissible mixture and task-affinity coupling* has an unreported/useful application. This bounded search neither establishes publication priority nor proves absence of an equivalent implementation. Removing the global explanation constraint recovers ordinary pairwise matching/attention; fixing a map recovers a static correspondence/supra-graph route; freezing role states yields graph-conditioned decoder features; using true qualified identities revisits H025; using overlap weights revisits the incidence kernel.

## Reasons to stop before implementation

1. **No observed failure has been tied to correspondence uncertainty.** Negative foreign-state results make the hypothesis possible, not supported. A new full fit without a distinct diagnostic would be speculative.
2. **Dating/identity semantics are unresolved.** We have observed point-date/topology estimates, not a calibrated uncertainty model. Exact dates may collapse explanations to a fixed map; a permissive tolerance may invent identity. Neither is evidence of a new mechanism.
3. **Admissible-map support is unknown.** Adjacent inferred trees may require multiple SPRs or reflect estimation errors. Empty/unique families make this candidate ineffective or equivalent to fixed transport. Source reads establish no distribution of those cases.
4. **Exact complete-boundary cost is unqualified.** Given A explanations and n internal roles, the factor work is O(A*n*K^2) per boundary/round after pair scores; full-tree work is additional. Enumeration and storing all induced maps can be much worse than the current linear passes. We have not proved a sparse complete factorization for the actual952-haplotype trees. Restricting to top-k explanations would change the operator and require a separate disclosed hypothesis.
5. **Feedback can reinforce a wrong affinity or edit.** Cavity exclusion prevents immediate self-use at a factor; it does not remove loop double counting, establish identifiability or turn discriminative affinities into biological ancestor states.

Do not start a broad validation campaign for this operator. These unresolved questions are reasons to stop implementation now. Existing routine evidence could later motivate reopening a precise condition, but no such evidence is established here. Prefer the single boundary-risk experiment in NEXT_EXPERIMENT.md, which targets a measured weakness using an unchanged completed architecture and a reused baseline. As of this handoff, no defensible distinct operator is ready for implementation.

## Evidence and limits

The fresh H048 report, H001–H048 ledger conclusions and cited project source definitions were inspected. New public primary retrievals cover CRF-RNN's abstract, GMN full HTML/§3.2, structured graph matching and matching-consensus primary abstracts, ARGweaver full text, and bibliographic metadata for sum-product, generalized BP and SMC-prime. CRF-RNN's HTML conversion is empty; it was not treated as full-text evidence. Failed author-code URLs remain in retrieval receipts. Crossref-only entries establish bibliographic identity, not newly verified equations.

No model/data/checkpoint/truth/GPU/guard/Git activity occurred. No H048 file was edited. Only research/reviews/next_graph_collection_operator_design_20261015_v1 was written. This packet supplies one design and a hold recommendation, not a hypothesis registration, implementation, experiment result or priority certificate.
