# Addendum: categorical ancestry at census-crossing lineages

2026-10-16. Targeted source audit and mathematical specification requested by root. No implementation, registration, experiment or novelty claim.

**The exact output constraint was not found in the inspected implementation or tests.** Existing census code predominantly uses the cut to mix latent context. A shared categorical ancestry variable at the census is a different statistical restriction, but its inference machinery is an established latent factor graph. The distinction must survive the limiting cases below to be more than a reparameterized group readout.

## What was actually implemented/tested

| Prior source | Exact relation to the candidate |
|---|---|
| H021, census_cut_exchange_v1.py:99–147,361–379,539–611; census_tree_encoder_v1.py:301–363 | Eligible queries are partitioned by inferred MRCA age≤350. Finite-rate block diffusion acts on per-query hidden states; gated residual updates retain their individual local states. The final decoder is applied separately to each scored query. No single categorical population state is assigned to a cut lineage and no equality of descendant ancestry outputs is enforced. |
| H021 tests/test_census_cut_exchange_v1.py:186–230,247–270 and test_census_tree_encoder_v1.py:199–264 | Tests cover cut/date/proxy policies, block-projector algebra, diffusion, gradients, shared components and genomic dependence. They do not test a categorical ancestry variable shared by all descendants. Existing reported passing checks are engineering evidence, not a completed structured-output ancestry comparison. H021's predictive run was stopped after 22 epochs and is recorded partial, not a final 30-epoch result. |
| H025, persistence_trace_kernel_v1.py:392–442 | Screened continuous latent-state propagation over sample/ancestral-trace edges, then a per-query residual over frozen unaries. It does not share a census categorical label. Genuine traces did not beat simpler controls. |
| H026, reference_census_innovation_v1.py:102–221 | Moves eligible query/younger-reference affinity locations to census350 and computes eight reference-context channels. Older reference fields remain unchanged proxies. A per-query correction head follows. Census conditioning did not improve on the age-free condition. |
| H033, interleaved_tree_tv_gnn_v1.py:236–293; ancestral_response_operator_design.../REPORT.md | Gaussian/quadratic surrogate states and anchored tree elimination. It is not categorical ancestry inference. Schur elimination and categorical sum-product are different operations. Interleaved and blocked H033 accuracy were essentially tied. |
| H050/H051 cut memory | Cut-group contexts, occupied-cell conditional means and persistent-sample history enter the tree representation. The output remains per query. The completed H050 and trade H051 outcomes do not test this new categorical restriction. |
| post_H050_shared_switch_design.../DESIGN.md:15–33,51–59 | Proposed, unregistered shared variable over no-change/directed ancestry-transition states on changed-clade support. This is the same broad latent-star factor family, but a different variable and support. Its rejected inference that changed-clade descendants identify shared recombination support is not required by a fixed-census local-lineage relation. Its uncertainty, correlation and generic-factor-graph cautions still apply. |

Search scope: relevant src/arggnn source text, census tests/docs, named census/kernel/memory/Schur/shared-switch reports and current ledger/decision summaries. This is not proof that no unindexed historical draft ever described the idea.

## Minimal structured probabilistic formulation

Fix the ancestry estimand at census age tau=350 generations. Let O contain the frozen observed variants, inferred trees and dates, sampled identities and observed proxy-reference labels. For each interval t and query q with sampling age a_q<tau, the inferred tree defines a local cut-lineage group g_t(q). These local group names are not persistent ancestral identities. An actual crossing requires lower_time≤tau<upper_time. A root ending at/below tau, missing upper endpoint or invalid inferred date is not an authenticated observed crossing; it needs an explicit unresolved case or separately specified model extension, not an invented ancestor. H026 raises when its native path has no genuine crossing.

Use the following categorical variables and potentials:

- Y_tq in {1,...,4}: source ancestry of query q at the census.
- C_tl in {1,...,4}: the shared census ancestry assigned to inferred local lineage l.
- u_tq(c)=exp(f_tq(c)): positive unary potential from the frozen/current complete-tree predictor, identical in the joint model and control. If f includes the existing BiGRU, it already uses genomic context. This is a discriminative potential, not a conditionally independent genotype likelihood.
- a_tl(c)>0: optional lineage potential, uniform in the minimal specification. Any learned alternative must use only O and be identical in the control.
- psi_tq(y,c)=(1-epsilon_tq)1[y=c]+epsilon_tq/4, with epsilon_tq in [0,1]: noisy compatibility with the inferred membership g_t(q). An observation-conditioned learned epsilon is a relaxation parameter, not automatically the posterior probability that the inferred merger is wrong.
- T_tq(y,y')>0: the same ordinary per-haplotype temporal compatibility in every comparison, using physical/genetic distance and permitted observed features. All ancestry switches remain possible. No inferred-tree boundary forces a switch or a no-switch.

Then

    p(Y,C | O) ∝
      product_(t,q) u_tq(Y_tq)
      product_(t,l) a_tl(C_tl)
      product_(t,q) psi_tq(Y_tq,C_t,g_t(q))
      product_(t,q) T_tq(Y_tq,Y_(t+1),q).

The temporal links follow the same sampled haplotype across adjacent partitions. They do not require matching internal ancestors. Integrating out C creates a local higher-order factor

    F_tl(Y_D) = sum_c a_tl(c) product_(q in D_tl) psi_tq(Y_tq,c).

This is ordinary latent-variable marginalization. Exact sum-product applies on an acyclic factor graph. Adjacent groups sharing several sample trajectories generally create loops, so any finite-round BP would be approximate, with no automatic convergence or exact-posterior claim. No inference algorithm is selected here.

The hard limit epsilon=0 ties all descendants' Y to one categorical C, giving shared assignments as well as equal marginals. The limit epsilon=1 removes the constraint. A wrong merger can be escaped through low-compatibility memberships, although this simple escape does not identify two alternative reliable sublineages or reconstruct a genealogy posterior.

## When it reduces to pooling, and when it can differ

With no temporal factors, hard compatibility and one local group D,

    p(C=c | O) ∝ a(c) product_(q in D) u_q(c),
    log p(C=c) = const + log a(c) + sum_(q in D) f_q(c).

For marginal predictions, this is exactly a group-sum-logit readout followed by broadcast. Averaging instead of summing changes temperature/group-size scaling. There is no need to introduce a new inference layer for this special case. A convex mixture of local/group logits or hidden states generally has no specified joint categorical distribution, but may reproduce the same marginals in a limiting case.

A potentially substantive difference arises only when changing local groups and same-sample temporal links make evidence from one query alter another query's possible ancestry paths through shared categorical variables. This is joint conditional inference, not automatically new expressivity or a new GNN primitive.

If shared probabilities are merely broadcast before independent per-query GRUs/classifiers, those later decoders can diverge. The final outputs then no longer obey the claimed constraint. The constraint must remain in the joint final ancestry distribution if that is the scientific claim. Otherwise it returns to a shared-context design close to H021/H050.

## Biological validity and failure conditions

On the correct dated genealogy at a fixed genomic site, two queries younger than tau with MRCA age≤tau share one lineage at tau, and therefore one population label at tau. Different cut lineages may still have the same source ancestry. This is a statement about the exact census estimand, not shared sample cohort or a shared recombination event. The target-producing source biological_paths_simulation.py:258–289 walks query ancestry to a flagged node at exactly350 and errors if none exists.

On inferred trees, topology errors or node-age errors can merge genuinely different source lineages below tau. Hard tying is then wrong and imposes unavoidable bias. A literal hard joint model assigns zero probability to a truth assignment containing different classes within that group. If training instead scores only marginal cross-entropy, it can learn a shared compromise; that does not make the hard joint assignment valid. With conflicting true-class masses within an inferred group, a hard shared probability can only compromise, and one hard call cannot satisfy every descendant. A learned escape gate must be able to relax this, and should not be called calibrated genealogical uncertainty without separate evidence.

Queries at/older than tau are outside this specification: no backward ancestral path takes them to a more recent time. Existing H021 explicitly rejects them. The H050 latent-context API's older-sample singleton convention is not a biological census output state. It must not be borrowed as such. Actual eligible rosters require their existing observed-age validation.

Older proxy references remain at their real observed nodes and supply encoder/reference evidence. They cannot be projected to nonexistent descendants at tau or pinned to a query lineage as known census labels. Even younger observed-source labels may pin a census state only under an explicit justified label/demography assumption. They are otherwise proxy evidence. H021's tests and H026's source already enforce this distinction.

Real short tracts remain possible through Y_tq and nonzero temporal transitions. Hard inferred mergers or overly strong temporal/lineage factors can still delete them. Native-hold intervals cannot resolve an unrepresented internal switch. If finer marker intervals are introduced simultaneously, that changes representation and confounds attribution. Reduced false short tracts accompanied by damaged genuine short-tract recall is failure, not improved tract discrimination.

Unaries already contain shared tree/reference evidence. Their products and multiple sampled-haplotype links must not be interpreted as independent biological observations. Repeating correlated evidence can create overconfidence or excessive switch penalties. A valid conditional model does not by itself establish posterior calibration.

## One decisive fair redundancy control

A future source-level comparison can define an **independently cloned-lineage control**: replace C_tl by C_tlq for every membership, with exactly the same lineage features/potential, unary encoder, compatibility gates, per-sample temporal factors, learned parameters, information, inference schedule and completed optimization/selection budget. No clone gets an independent learned parameter. This removes only shared categorical dependence among query paths. Group summaries may still be supplied identically to every clone, preventing a straw-man information deficit.

This is the decisive falsifier: **if the shared model fails to improve the prespecified ancestry/tract objective over the cloned model, or its gain consists of deleting true short tracts or worsening NLL/calibration, distinct joint-lineage utility is unsupported even if both beat the original predictor.** Gate collapse to the uncoupled limit also defeats the constraint explanation. A win attributable to unequal calibration, additional chain smoothing, data access or training would not answer the question. The source-only formulation does not set a new numerical threshold, add an arm to H058 or authorize a run.

## Three directly checked primary precedents

Only these three cached primary works were looked up for this follow-up:

1. Baran etal. (2012), *Fast and accurate inference of local ancestry in Latino populations*, [DOI10.1093/bioinformatics/bts144](https://doi.org/10.1093/bioinformatics/bts144), §2.3. It explicitly discusses a standard factorial HMM for joint trio paths, but implements trio phasing followed by independent haploid LAI. Known Mendelian family constraints are stronger evidence than inferred cut groups. This establishes relevant joint-ancestry precedent, not this exact factor.
2. Hubisz, Williams and Siepel (2020), *Mapping gene flow between ancient hominins through demography-aware inference of the ancestral recombination graph*, [DOI10.1371/journal.pgen.1008895](https://doi.org/10.1371/journal.pgen.1008895), Results “ARGweaver-D can sample ARGs…” and posterior introgression. Its HMM states include branch, time and population path, and sampled ARGs give ancestry probabilities. Genealogy/population-state uncertainty is established; this differs from a point-tree neural C4 conditional model.
3. Garcia Satorras and Welling (2021 revision), *Neural Enhanced Belief Propagation on Factor Graphs*, [arXiv:2003.01998v5](https://arxiv.org/html/2003.01998v5), §§2.1–2.2,4.1–4.2. Arbitrary-variable factor graphs, sum-product and learned message refinement are explicit. The categorical formulation above supplies no new general inference principle.

Cached passages and metadata were read directly, with hashes recorded. No new network retrieval was necessary. No citation claims exact prior implementation of this census-cut composition or exhaustive priority.

## Conclusion

The candidate is **not already tested by the named latent census-pooling experiments**, and it is **not algebraically the Schur/Gaussian response**. Its soft version nevertheless belongs to the established high-order latent-factor family already identified in the shared-switch audit. The biologically valid local census relation differs from the invalid changed-clade-to-shared-switch inference. Whether a useful joint categorical restriction survives inferred-tree errors, correlated evidence and short-tract requirements remains unresolved. This is a clarified hypothesis, not a justified novelty claim or an implementation decision.
