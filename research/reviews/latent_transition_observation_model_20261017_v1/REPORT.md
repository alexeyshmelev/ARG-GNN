# TRAIN-only latent transition observation model — source assessment

17 October 2026. **No new architecture or experiment recommended.** TRAIN-only simulator supervision is a legitimate and materially different statistical condition from H043/H048/H050. An explicit mutation likelihood and point-identifiable events are not prerequisites for learning a simulation-conditional distribution. However, the concrete efficient implementations here reduce to auxiliary structural prediction or learned matching. Genuine joint uncertainty marginalization is established latent-structure inference and still lacks a specified valid, tractable state space in this proposal. The negative conclusion concerns the requested architectural opportunity, not the legitimacy of simulator supervision.

## Actual retained latent annotations

The European full-panel producer calls sim_ancestry with Hudson ancestry, record_migrations=False, and neither record_full_arg nor additional_nodes (prepare_ancestralpaths_full_panel_discovery_v1.py:334–337). It writes the unsanitized census_ancestry.trees and external ancestry intervals (:347–378). The shared demography adds a census at350 (biological_paths_simulation.py:181), and external_truth_intervals walks each query to a flagged census node (:262 onward).

Thus the source supports deriving true local topology/dates, recorded census-lineage identity, and its source population. It does **not** establish a retained complete recombination-event history, including all prune/recoalescence events or invisible recombinations. The installed msprime ancestry.py documents record_full_arg=False by default and enables recombinant/common-ancestor event nodes only when requested. Raw marginal-tree differences cannot fill the missing event history uniquely.

The American producer records migration rows and traces source destinations at the12-generation pulse (american_c3_simulate_v1.py:48–72,100–101), but likewise requests no full ARG event recording. These are pulse-migration annotations, not a universal recombination-event roster. Do not reuse migration-row identity as a European census-lineage label or claim these producers retain the same latent ontology.

All conclusions above concern source behavior. No true/inferred tree or target payload was opened. A new structural target exporter would still be required.

## A well-defined target without true-to-inferred node matching

On the existing observation-defined100bp centers x_i, let

    L*_iq = the recorded true census350 node ancestral to sampled q at x_i,
    R*_(i,q,r) = 1[L*_iq = L*_(i+1),r].

Use only samples younger than350 for this target. Raw true node IDs are compared only within one true tree sequence, never embedded as vocabulary, compared across simulations, or joined to inferred internal IDs. Equal IDs denote the same recorded census lineage, not an identified historical diploid individual. Observed sample/haplotype identity supplies the true/inferred leaf join.

R* is richer than ancestry-boundary supervision: a census lineage can change while its source class stays the same. The diagonal relation distinguishes recorded-lineage continuity from source-class continuity. It is nevertheless an endpoint relation, not the number, time or descendants of recombination events between x_i and x_(i+1). Multiple changes and return to an earlier lineage need not be recovered.

This coordinate join handles disagreeing true/inferred breakpoints: evaluate the true genealogy at the pre-existing observed coordinate and gather the inferred tree covering that coordinate. Do not snap a true breakpoint to its nearest inferred boundary, invent a true/inferred internal-node match, truth-select candidate repairs, or use truth intervals as input windows. A center label is not a claim that the entire observed hold has that relation. If the downstream model is constant within holds, unresolved within-hold changes remain. Giving only the proposed arm a finer marker readout would confound the comparison.

Within/cross-position lineage equality induces equivalence constraints over the sampled endpoints. Inferred cut groups can mix or fragment true lineages. Their correspondence is therefore not generally a one-to-one assignment, and balanced Sinkhorn constraints are not biologically justified merely by calling it matching. An observation-defined candidate family must include unmatched/unresolved outcomes and disclose truth relations outside its support.

## What learning would and would not establish

Given independent TRAIN simulations generated through the same observation/inference pipeline, a proper conditional loss can fit q_phi(Z|O), where Z is a precisely declared structural target and O contains only observed variants, inferred trees/dates, reference labels, sample ages/identities and geometry. With adequate data/modeling this approximates p_sim(Z|O). Non-identifiability permits a non-degenerate conditional distribution and does not prevent learning it.

This q(Z|O) is a discriminative latent observer, not a fitted generative observation likelihood p(O|Z). It is an amortized approximation under the simulator and inference pipeline. A learned attention matrix is not automatically a calibrated event posterior. Demography, mutation, missingness, phasing and inference error define the training distribution and can shift on real data. Current census annotations can supervise a census-relation model, not retroactively create unavailable full event labels.

Three computations must be separated:

1. Adding -log q_phi(Z*|O) while leaving ancestry inference unchanged is auxiliary/multitask supervision. An ancestry-boundary BCE target alone is derived from labels already available to the ancestry loss.
2. Using expected relation weights in h'_u = h_u + sum_v E_q[Z_uv] m_theta(h_u,h_v,O) is ordinary learned soft correspondence/message passing. It differs from H043's fixed incidence, but does not itself marginalize nonlinear ancestry inference.
3. Computing p(Y|O) = sum_Z q_phi(Z|O) p_theta(Y|O,Z) can differ from p_theta(Y|O,E_q[Z]) because the decoder is nonlinear. This is a non-redundant computation relative to a mean-relation decoder, but it is an established latent-mixture/structured-inference construction. A Bernoulli distribution over unrelated pair edges is a valid random-graph model, not automatically a distribution over genealogically consistent partitions or recombination histories.

Pairwise relation BCE identifies conditional pair marginals, not the joint correlations required by a partition or event posterior. It therefore cannot by itself qualify those correlations. A full joint distribution over lineage partitions could carry information beyond pair marginals. Its normalization, valid support, approximation and handling of unsupported topology errors must be specified. Merely naming that missing distribution does not supply an architecture. Reusing latent population/shared-switch factors after learning their membership weights returns to the previously audited factor family with a changed observation model.

## Prior work and no-repeat boundary

The inspected H019–H058 decisions and actual H040/H043/H045 source/design do not show this census-relation supervision. H043 learns ancestry through a deterministic sparse incidence; H048 learns a persistence parameter from ancestry risk; H050 pools observed cut histories. None is an unchanged test of learned truth-to-observation structural calibration. The census categorical addendum uses fixed inferred memberships with optional compatibility relaxation. The rejected attachment/shared-switch proposals lacked a qualified observation distribution. Simulation supervision is a possible way to address that statistical prerequisite, rather than requiring an explicit likelihood.

Closest verified primary precedents limit architectural claims:

- **CoalNN**, Nait Saada etal.2023, *Inference of Coalescence Times and Variant Ages Using Convolutional Neural Networks*, DOI10.1093/molbev/msad211. Methods jointly train simulated pairwise TMRCAs and breakpoint probabilities using regression plus cross-entropy, including weighted breakpoint BCE. This is adjacent amortized genealogy prediction, **not** a verified full ARG posterior system. It directly anticipates the simple auxiliary-boundary version.
- **SuperGlue**, Sarlin etal.2020, §3.3 Eq10, arXiv1911.11763v2. Ground-truth correspondences and unmatched labels supervise a differentiable graph matcher. Genealogical many-to-many relations differ from its image-point assignment; supervised uncertain matching itself is established.
- **Neural Relational Inference**, Kipf etal., ICML2018, published abstract. An amortized latent interaction graph and GNN reconstruction decoder are explicit. The full methods PDF was not inspected in this follow-up, so no equation-level exact equivalence is claimed.
- **Structured Attention Networks**, Kim etal.2017, §§2–3, arXiv1702.00887v3. Graphical distributions and differentiable inference over latent structures provide expected features within neural networks.
- **ARGweaver-D**, DOI10.1371/journal.pgen.1008895, cached primary methods and previous audited passages. Uncertain genealogical/population histories induce ancestry probabilities. It uses model-based sampling, not the proposed supervised observer.
- **Mo and Siepel2023**, DOI10.1371/journal.pgen.1011032. Domain-adaptive SIA explicitly addresses simulated-data mismatch and ARG inference error. This is a warning about transfer assumptions, not an identical structural observer.

The bounded public search found these adjacent precedents. It establishes neither absence of a closer amortized ARG method nor priority for census-relation supervision. Fresh primary locators, inspected passages and retrieval hashes are recorded separately.

## Fair falsifier, cost and decision

A future concrete model must compare against an **equally supervised observer-plus-soft-matching control**: identical structural labels, observer, input support, ancestry encoder, parameter/training/selection budget and uncertainty features. Compare joint nonlinear marginalization only against consumption of the same observer's relation marginals/uncertainty. An ordinary backbone without the structural loss can describe the value of extra supervision, but cannot isolate marginalization. The model must also demonstrate held-out structural probability accuracy/calibration. An ancestry gain with unchanged structural prediction, or one attributable to finer output resolution, extra fits or loss reweighting, would not support the proposed mechanism.

The falsifier is failure to improve prespecified ancestry boundary utility over that equally supervised control while preserving span accuracy, NLL and genuine short-tract recall. A shared-factor version also inherits the already identified shared-versus-independently-cloned-variable comparison. No new threshold, arm, registration or run is proposed here.

Dense cross-position relations cost O(B*n^2) scores/targets before graph inference, where B is the observation-grid length and n the eligible panel. Full equivalence partitions over2n endpoints have Bell-number support. A K-particle nonlinear decoder costs roughly K times the affected decoder work and only approximates its expectation. Reusing the first observed-tree encoder may help, but the changed nonlinear graph propagation cannot generally be cached as one expected adjacency. An anchor/top-k or local-SPR restriction is a changed estimand with candidate-coverage and unresolved-mass requirements, not free scalability. No fit or runtime is claimed.

TRAIN-only latent labels are legitimate extra supervision and must be disclosed to competitors and ablations. Split parents before target construction, keep true node identities/populations/breakpoints outside inference features, preserve query masking and prohibit held-out truth from candidate construction or calibration. Older proxy samples remain observed evidence without fictitious descendants at350. The TRAIN target and observer hyperparameters must not be selected against the repeatedly used VAL outcomes.

**Decision:** the follow-up fixes one overly narrow premise of a topology-only no-go: conditional uncertainty can be learned from latent simulator supervision. It supplies a definable new supervision question. It does not yet supply a new, tractable non-redundant architecture beyond established matching/auxiliary prediction or latent-factor inference. No candidate is advanced for implementation or training.

Scope: source/literature reasoning only. No project payload/array/model import, numerical probe, experiment, guard/resource/process query, install, registration or Git operation. Writes are confined to this new review directory.
