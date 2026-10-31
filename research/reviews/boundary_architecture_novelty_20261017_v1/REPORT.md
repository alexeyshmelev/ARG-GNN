# Boundary architecture audit: no new operator qualified

17 October 2026. **NO_GO for another architecture proposal from the inspected evidence.** No implementation, hypothesis registration or experiment is recommended. The most concrete remaining information gap is an observation-conditioned distinction between a genuine change of source ancestry and a change of matching donor or uncertain genealogy within the same ancestry. Neither the current aggregate metrics nor the inspected architecture identify that error mechanism. Explicitly adding donor history is not a new direction: the project already excluded donor-copying/remix hybrids, and the primary literature supplies the defining latent states, switching processes and efficient inference.

This is a bounded source/literature assessment, not proof that no useful or novel genealogy architecture can exist. Novelty can reside in a representation, coupling or exact algorithm built from established primitives. Here no materially distinct coupling or algorithm, tied to a demonstrated error, survives the reductions below.

## Evidence that the direction must explain

The current European selection is H050 stable-sample memory: modern span accuracy **97.077844 ± 0.567395%**, versus NoMem continuation **97.000028 ± 0.497841%**. Boundary F1 at 1 kb is **18.380657 ± 1.3141%** versus **19.048847 ± 1.4275%**. These are eight reused development simulations, one optimizer seed101 and 960 added complete-parent updates per arm from the same initializer, with independent checkpoint selection. The differences do not establish that memory causes oversmoothing. Changed selection, missed transitions, false transitions and their genomic masses can produce the same two aggregates.

The current-release ARGMix pipeline scores 94.4461 ± 1.1491% on the European development endpoint. The material input, release and training differences in the manuscript prevent assigning that pipeline gap to one TreeLoom operator or equating it with a publication reproduction.

There is stronger, but historical, boundary evidence in the H038 diagnostic: 1,094 of 4,008 true boundaries lacked an allowed hold edge within 1 kb; 2,195 of 3,289 observed misses still had a nearby candidate. Predictions contained 6,729 changes versus 4,008 true changes. Strict false internal islands ≤10 kb explained 35.46 ± 2.71% of extra boundaries. This supports investigating transition recognition as well as resolution. These are H038 measurements, not H050 residuals or proof of a donor/LD/phase mechanism. The H057 TRAIN-only grid-minus-hold oracle accuracy gap, 0.1044431 ± 0.0098382 points, establishes limited extra representability, not an achievable gain or a boundary-recall ceiling.

H044's fixed local SNP boxes versus pooled evidence gained only 0.00526 accuracy points and reduced F1 from18.6750 to15.5374%. H043/H045/H046 foreign context, H048 graph-conditioned persistence and H050 higher-order cut history failed their registered utility rules. These answered conditions must not be renamed and repeated. They do not establish that every marker representation, donor-copying model or uncertainty model is ineffective.

H061 provides an independent limit on a universal memory claim, not an ancestry explanation: three-seed Chickenpox TEST MAE is13.907918 ±0.242561 cases with memory versus13.648097 ±0.050932 without; RMSE slightly favors memory. Its registered memory criterion failed. H058, separate ordered_marker_placement_v2 and H060 are full-training live **per ROOT's current handoff**. The inspected decision snapshot contains older H058 operational wording. No live process, payload or new result was queried here.

## The precise information gap

Direct source inspection confirms that the original site route maps each query allele to eight source×age reference-frequency/mask channels, panel frequency, entropy and callability; it then mean/max pools learned site features within native holds. H058 retains same-marker source×age×allele/tree-path alignment before pooling. Ordered-marker v2 retains the order of those 24 aligned channels through width8 bidirectional GRUs, reset at native holds. Both retain native-hold outputs.

The gap is therefore **not “the model has no SNPs, LD or marker order.”** Full inferred trees and their mutation support already derive from correlated variants, genomic recurrence can encode correlations, and the live ordered route explicitly addresses order. What is absent as an explicit statistical object is a separate donor-identity trajectory, with a distinction between within-ancestry donor turnover and ancestry turnover, and a qualified treatment of point-tree error/reference mismatch. Absence of that explicit object is a source fact; its necessity and predictive value are unproved.

A query may match a different reference through historical recombination, incomplete lineage sorting, sampling noise or imperfect reference proxies while retaining the same source ancestry. Conversely, a real source change can occur without an inferred topology change. Thus a clade edit, nearest-donor change or an increased number of transitions cannot by itself label an ancestry event. Normalized attention weights also do not become an evolutionary posterior, and genotype-derived trees and genotype emissions are not independent evidence factors.

## One audited construction and its exact known-method reduction

The following is a **rejected construction used to expose the reduction**, not a new proposal.

Allowed inputs would be the original phased query/reference calls, missing masks, reference source labels and ages, observed marker coordinates or a declared map, complete inferred trees and the existing whole-tree representation. No query truth, true genealogy, census-state annotation or true boundary is an input.

For query q at marker s, let the hidden state be Z_sq=(C_sq,R_sq), source c∈{1,…,C} and copied reference r∈{1,…,R}. Allow donor source to differ from c. Let π_sq,c,r be a normalized reference-selection distribution, μ_sq,c a source distribution, a_sq the probability of an ancestry-redraw event and b_sq,c a within-source donor-redraw probability. A standard hierarchical transition is

    K_sq[(c,r),(d,u)] =
       (1-a_sq) 1[c=d] {
          (1-b_sq,c) 1[r=u] + b_sq,c π_sq,c,u
       }
       + a_sq μ_sq,d π_sq,d,u.

Each row sums to1. An ancestry-redraw event can choose the same source and is not necessarily an observed label transition. A positive match/mismatch emission e_sq(c,r) uses the target/reference calls; unavailable comparisons are neutral. A genetic emission would additionally require a declared mutation/error/ascertainment model. Using neural scores instead defines conditional potentials, without automatic biological calibration.

The conditional chain is

    P(Z_q | O) ∝ initial(Z_1q)
                  ∏_s e_sq(Z_sq)
                  ∏_(s>1) K_sq[Z_(s-1)q,Z_sq],
    P(C_sq=c | O) = Σ_r P(Z_sq=(c,r) | O).

For fixed donor-panel probabilities π_c,r=p_c,g(r)/n_g(r), ordinary mismatch emissions, a=1−exp(−TΔg) and b_c=1−exp(−ρ_cΔg), this is the established ancestry×donor-copying HMM family in HAPMIX/FLARE [1–3]. HAPMIX explicitly has state (ancestry, copied population, copied individual), allows miscopying and explains why forbidding miscopying can create spurious ancestry breaks through incomplete lineage sorting or imperfect references. FLARE explicitly distinguishes pre-admixture donor changes from post-admixture ancestry changes.

With one ancestry this becomes the ordinary haplotype-copying specialization. If b=1, π and emissions are constant across donors within each source and sites, marginalizing donors leaves an ordinary source-state chain. A learned tree-dependent π or neural emission gives a covariate-dependent HMM/conditional chain. That changes its inductive bias, not the defining inference operation. Feeding posterior beliefs back into the tree would require an explicit joint model and solver; ordinary cavity exchange is already rejected as sum-product in the separate audit. No new exact factorization has been established here.

The donor dimension need not create all-pairs transitions. Write F_prev(c)=Σ_r F_prev(c,r), F_prev=Σ_c F_prev(c), with normalized forward weights. Then

    F_s(c,r) ∝ e_s(c,r) [
       (1-a_s)(1-b_s,c) F_prev(c,r)
       + (1-a_s)b_s,c π_s,c,r F_prev(c)
       + a_s μ_s,c π_s,c,r F_prev
    ].

This is O(CR) per query-marker, rather than O(C²R²). MOSAIC's primary appendix already derives the sum-plus-self correction that avoids donor-pair enumeration [4]. Torch could batch normalized/log-domain forward/backward over queries and donor states; PyG could supply the unchanged full-tree covariates. Block recomputation could trade memory for time while retaining all gradients. Those are established implementation techniques, not a new efficient exact algorithm. The inherited whole-tree work, computing tree-conditioned donor weights and full-parent training storage remain additional costs. No implementation, numerical check, measured fit or resource bound is supplied or authorized.

The primary-text reduction is decisive against this particular novelty claim. It does not claim that FLARE already implements every possible whole-tree neural coupling, or that a carefully controlled biological application could never be useful.

## What the existing conditions can distinguish

| Existing condition | Actual discriminating information | What it cannot establish |
|---|---|---|
| H058 aligned versus alignment-null, plus exact continuation | Same-marker reference allele/source/age alignment to inferred branches beyond the same capacity and source-frequency counts. The frozen whole-reference diploid permutation preserves reference sequences and LD while disrupting branch/exact-date attachment within allowed groups. | Donor-history utility, marker-order utility, topology alone, or calibrated genealogy. No new donor-indexed state exists. |
| ordered_marker_placement_v2 before versus after the second tree sweep | Whether the same ordered aligned evidence is more useful when it conditions full-tree computation than at sample readout. Equal33966parameters and fixed training are central. | Order itself: both arms are ordered. It also retains constant native-hold output, so it does not validate new within-hold breakpoint locations. An advantage would support this biological coupling, not generic GRU/graph-feedback priority. |
| H048 transition versus confidence control, already completed | The tested graph-conditioned reset-chain persistence versus confidence calibration. | A donor-copying or tree-uncertainty mechanism. Its negative result cannot be generalized to those untested models. |
| H059 frozen independent European confirmation | Generalization of the selected complete packages beyond reused development parents. | Attribution to donor history or a new operator. Preserve its frozen purpose. |
| H060 matched American Full/NoMem | Transfer of the actual full architecture and the memory addition. | Dense tract-boundary fidelity from sparse fixed focals, or a cause of European boundary errors. |

None of the live existing comparisons directly isolates donor-identity continuity. It would be incorrect to relabel H058 or ordered-marker placement as that test. This audit does not create the missing test or revive the already excluded copying/remix direction.

The useful next evidence is the **already planned** matched marker results, with their frozen boundary precision/recall, correctly typed transitions, true-short-tract preservation and false-island diagnostics. They can distinguish useful tree-aligned evidence/coupling from added training/capacity. A score gain that removes genuine short tracts cannot establish improved tract recovery. Historical H038 diagnostics must not substitute for those actual outcomes.

## Decision and provenance

**NO_GO: no additional architecture, registration or run.** Keep the concrete information question: can observed evidence distinguish genuine source transitions from same-source donor turnover and uncertain genealogy beyond the strong whole-tree predictor? The obvious donor state and two switching processes are occupied primary prior art. A future design needs a specific additional relation the established ancestry×donor model plus observed-tree covariates cannot express, a materially distinct representation/coupling or exact algorithm, and an error mechanism supported by observations. Neither renaming an HMM nor requiring every primitive to be unprecedented is a suitable novelty standard.

Primary sources actually inspected:

1. Li & Stephens (2003), *Modeling linkage disequilibrium and identifying recombination hotspots using single-nucleotide polymorphism data*. Genetics165:2213–2233. [DOI10.1093/genetics/165.4.2213](https://doi.org/10.1093/genetics/165.4.2213). Bibliographic/abstract record and primary PMC page verified. Its full scanned body was not used for a new technical claim; HAPMIX explicitly states its Li–Stephens extension.
2. Price etal. (2009), *Sensitive detection of chromosomal segments of distinct ancestry in admixed populations*. PLoS Genetics5:e1000519. [DOI10.1371/journal.pgen.1000519](https://doi.org/10.1371/journal.pgen.1000519). Full primary XML, Materials and Methods / “Modeling genetic variation in admixed populations,” state definition and Eq0.1–0.2.
3. Browning, Waples & Browning (2023), *Fast, accurate local ancestry inference with FLARE*. AJHG110:326–335. [DOI10.1016/j.ajhg.2022.12.010](https://doi.org/10.1016/j.ajhg.2022.12.010). Full primary PMC HTML, Subjects and Methods / “Hidden Markov model” and “Computational techniques.” Explicit ancestry×donor state, two crossover histories, panel mismatch, emissions and posterior readout.
4. Salter-Townshend & Myers (2019), *Fine-Scale Inference of Ancestry Segments Without Prior Knowledge of Admixing Groups*. Genetics212:869–889. [DOI10.1534/genetics.119.302139](https://doi.org/10.1534/genetics.119.302139). Full primary XML, “Model,” “Thinning,” and appendix forward/backward computation. Nested ancestry/copying states, estimated copying matrix and efficient sum/self transition evaluation.

Exact primary excerpts and retrieval records are in PRIMARY_EVIDENCE.v1.json and PUBLIC_RETRIEVALS.v1.json. Failed XML retrievals and unsuccessful compact searches are recorded; no claim relies on their absence of hits. Internal source hashes/read scopes and a compact decision snapshot are in SOURCE_RECEIPTS.v1.json and DECISION_CONTEXT_SNAPSHOT.v1.json. MANIFEST.v1.json seals the delivered files. Public retrievals were bounded and all saved material is inside this assigned directory.

Only instructions, manuscript/source text, concise ledger/decision metadata, named prior reviews and public literature were read. No production arrays, truth/checkpoint/prediction contents, runtime/process/resource state, model import, numerical test, experiment, installation, accounting, existing-source mutation or Git operation occurred. ROOT retains integration, experiments, resources and Git.
