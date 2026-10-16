# One architecture hypothesis after H050

This is an unregistered design, conditional on H050 failing meaningful utility. It changes no live source or experiment. No model result or architectural priority is claimed.

## What the evidence actually says

H045 foreign-episode accuracy is96.8270±0.5870% on Present182, versus96.8175±0.6142% for its matched own-state adapter. Its paired gain is0.009568percentage points, below the0.10criterion. BoundaryF1 gains0.100147pp. These are eight reused development simulations and one optimizer seed, not confirmation. The saved residual report gives only22.59% true-boundary recall and18.16% F1, despite high span accuracy.

H045 has a precise limitation. It finds consecutive occurrences of an identical proper descendant set, excludes roots/singletons, and averages other first-sweep states by physical span. The rank8 residual receives this mean instead of its own state. It has no order within an episode. Algebraically foreign_i-own_i is the weighted mean of(h_j-h_i) over other members. If the states are constant, both arms are exactly identical at this adapter. Shared descendants and redundant features make small innovations plausible. The two complete-tree sweeps and existing genomic BiGRU can also make a further pooled context redundant. None of these explanations has been measured in saved hidden-state diagnostics. The score alone does not identify why the gain is small. Stable-clade eligibility also misses the very changed-clade events at which a different joint representation could matter.

H050 tests a different, still unresolved question. It starts every arm from the completed H045 own-state pass30 and retains the direct native-hold predictor. Three arms add the same513parameter rank8 memory. Their carry uses triple-support cut history, an adjacent-marginal null or stable sampled-leaf history. The fourth is unchanged H045 continued training. The three memory arms have32325parameters, continuation31812. They share observations, full30pass/960update budget, selection and readout. The triple rule is conditional-mean transport on observed350generation cut partitions, followed by genomic-width exponential forcing. It tests usefulness of that historical pooling bias beyond richer persistent-leaf history and extra training. It does not test identified ancestral individuals, a new transport primitive, or within-hold boundary resolution. No H050 score was available in the inspected ledger snapshot. This report does not infer one.

## Candidate: shared ancestry-switch factors on observed clade changes

The falsifiable question is whether queries that share an observed changed clade benefit from a jointly inferred switch variable, beyond equally informed independent-query transitions. This is a change from embedding averaging to explicit dependence among ancestry transitions.

Keep a whole-tree PyG encoder over every original vertex and all observed haplotypes. Stable sampled-haplotype IDs join observations across genomic positions. At each distinct adjacent-tree change, construct candidate support sets from appearing/disappearing proper descendant clades. Keep both sides and their ambiguity. A changed inferred clade is a witness, not an identified recombination event or ancestor.

For candidate e with query support D_e, introduce a latent variable z_e with13states: no ancestry change, or one of the12directed pairs of four source classes. For query q and consecutive observed marker intervals i,i+1, ancestry variables are Y_qi,Y_q,i+1. Use the factor

    F_e(Y_left,Y_right)
      = sum_z phi_e(z | left/right whole-tree states, reference support)
          product_(q in D_e) psi_e(Y_qi,Y_q,i+1,z | query-local evidence).

phi and psi are small shared neural potentials. They use observed left/right clade and outside states, dates, real source-reference availability, phased query/reference concordance and genetic/physical distances. Query ancestry labels are training targets only. No true ARG, true boundary, ancestral population path or query cohort label enters features. Support labels refer to observed pure-source references, not certified ancestors.

A query-specific participation gate makes each witness optional. A query outside D_e receives neutral factor1. Missing reference evidence leaves the corresponding factor neutral rather than fabricating donors. Even z_e=no change must remain possible at every topology change. Candidate events with conflicting or uninformative support can be ignored. The model must not force every member of an inferred clade to share an ancestry transition.

Retain a common marker-level local emission and per-query continuity factor in every arm. This allows switches within unchanged native holds. Marker gaps bound location information. A topology edge cannot be treated as an observed ancestry boundary. Known SNP/HMM machinery here is common infrastructure, not the proposed contribution.

The cross-graph layer is three fixed rounds of log-domain factor-to-query and query-to-factor messages, with ordinary chain forward/backward messages between rounds. Overlapping descendant sets create loops, so this is disclosed approximate inference. Feed the resulting factor/query messages into the common readout. All arms use the same encoder, potentials, marker support, iterations and optimization. This is a structured GNN/factor layer over a sequence of complete trees and persistent sampled leaves. It is not a claim of exact biological likelihood or calibrated posterior.

One factor can be updated using products/sums over its memberships and13x4x4state tables. Component work is O(L*C^2*13*sum_e|D_e|), plus encoder/chain work. There is no dense tree-pair matrix. Highly overlapping clades can still make membership count large. No measured memory or speed advantage is claimed. Collapse only exactly duplicate candidate records. Approximate top-k pruning or dropping original trees would change the model.

## Matched falsification

Use three matched architectures if a later literature check justifies this direction.

1. Shared z_e with the genuine observed descendant support.
2. The same potentials/weights but a separate z_eq for every query. This preserves all per-query inputs and removes shared latent dependence. No dummy parameters are added.
3. Shared z_e with support alignment permuted within prospectively fixed observed age/callability strata at each boundary. Preserve candidate sizes and overlaps by permuting leaf columns together, while keeping each query's genotype/local encoder input unchanged. Do not stratify by ancestry truth.

The cloned-variable model could learn the same output marginals from the complete graph features. No expressivity or information dominance is asserted. The test is whether the explicit dependence is a useful inductive bias.

Freeze one boundary-focused primary rule before implementation or new fits: gain at least1pp mean1kb boundaryF1 over EACH matched control with no mean exact-span accuracy decrease. Report false short tracts/Mb, false-switched bp/Mb, recall of true<=10kb tracts, signed boundary displacement, NLL/Brier and all cohort rows. Reduced fragmentation with worse short-tract recall would not support the proposed mechanism. Keep selection by the original accuracy/NLL rule unless a new protocol is explicitly agreed before outcomes exist. Report all parent-paired changes and sampleSD. Independent parents are required before confirmation, because the current eight VAL parents have been repeatedly reused. Full original-paper benchmarks remain a separate requirement.

This does not repeat H043/H045/H046 context means, H047 cohort reweighting, H048 independent reset-chain conditioning, H049/H050 historical partition pooling, or the already owned donor-identity-versus-remix condition. It also differs from the existing reference-only memory/refinement-consistency concept. Its single new question is joint descendant-supported switch dependence.

## Biological identifiability and originality gates

Genomic order and genealogical age remain different axes. Native internal IDs are not persistent ancestors. Older samples are never projected to an age before their existence. Source ancestry at350generations is a simulation target, not the sample cohort. Coupled sampling ages/mixture pulses and rare ancient boundary counts preclude an age-only explanation.

Neither a tree change nor shared descendant membership identifies an ancestry switch. Uninformative SNP gaps, missing source proxies, uncertain phasing and uncertain inferred topology can leave alternatives observationally indistinguishable. Joint factors can be confidently wrong when the inferred support is wrong. Therefore evaluate boundary precision/recall and calibrated uncertainty by observation-defined reference support, and retain a neutral fallback. Real-data agreement or disagreement with H055 is descriptive, not accuracy or historical proof.

The new operator is not established as novel. Generic latent factor graphs, higher-order CRFs, sum-product inference, temporal GNNs, HMM smoothing and source-conditioned graph evidence are known. A shared latent variable inducing dependence is an elementary fact, not a new theorem. Existing project audits already identify TGN, DiffPool, higher-order memory networks, GraphSSM, FLARE, AncestralPaths, ARGMix, ARGformer and GST as close machinery. Combining them does not establish an A* contribution.

The defensible possible contribution is narrower: an explicit, uncertainty-aware joint ancestry-transition factorization whose support is constructed from ambiguous observed genealogical changes, with a tract-level benefit that disappears under independently cloned factors and correspondence shams. This remains a proposal, not a priority finding.

Before implementing, inspect primary literature/code for joint or pedigree-aware LAI, recombination HMM/SMC tree-transition factorization, tree-based ancestry painting, shared-breakpoint multi-sequence segmentation and neural factor-graph layers. Check especially whether marginalizing shared genealogical/recombination variables already gives this same factorization. The four supplied approaches must be checked at their actual joint inference equations, not by architecture labels. These are specific unresolved literature gaps, not invented citations. If an equivalent formulation exists and no distinct useful property survives, reject a new-architecture claim. A biological utility result alone may be valuable but would not automatically meet the intended machine-learning novelty bar.

Only this proposal directory is written. No ledger registration, source modification, truth/probability/checkpoint/native payload read, model/fixture execution, GPU job, guard action or Git operation occurred.
