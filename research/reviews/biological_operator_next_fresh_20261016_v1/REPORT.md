# Fresh assessment: retain marker-specific reference support as a utility test

2026-10-16. Independent source-only scientific assessment. **No defensible new general operator was found.** The most credible next biological question is whether the observed branch placement of a reference allele matters at the same marker as the query allele, before the current model pools sites within a native-tree hold. This independently selects the existing marker-specific support proposal. It is not a new registered condition and should not spawn a duplicate experiment.

## Evidence and its limits

The current paper and progress report record H050 stable-leaf memory at Present182 span accuracy **97.0778 ± 0.5674%**, compared with **97.0000 ± 0.4978%** for equal-budget continuation. Its boundary F1 is **18.3807 ± 1.3141%**, below continuation at **19.0488 ± 1.4275%**. These are eight reused VAL simulations and one optimizer seed101. Pair-cell/triple-support memory fails both registered utility rules. TGB conditional-triple TEST NDCG@10 is **0.830455 ± 0.011007**, below first-order **0.832384 ± 0.011250** and persistence **0.846883 ± 0.014389**. That SD is across three dependent years, not optimizer seeds. Neither campaign supports useful higher-order novelty.

The saved H038 diagnostic says 2,195 of 3,289 missed boundaries had an allowed readout edge within 1 kb. Missing output resolution therefore cannot explain all misses. Its false short-island statistics must not be relabelled as H050 measurements. H045 and H050 retain weak boundary recovery despite higher span accuracy. These facts establish a residual problem. They do **not** identify allele-to-branch alignment as its cause.

The actual `_site_chunk` and `encode_sites` code encodes a query or reference allele through the 19 donor-frequency/callability channels, then mean/max pools within each native hold before tree propagation. This explicit route discards same-site alignment across different haplotypes before their graph states interact. The complete model still sees SNP-derived topology, mutation support, every sampled leaf, and a genomic BiGRU. It can encode relevant correlations indirectly, so no global claim that it lacks linkage or allele/topology information is justified.

Current native ARGMix **94.4461 ± 1.1491%** and derived-C4 AncestralPaths **93.7830 ± 0.8385%** are descriptive pipeline comparisons with different inputs, training and selection. They cannot establish architecture-only superiority or reproduce their published tables.

## One biological mechanism

**Hypothesis:** a source-frequency-matched allele carried by references close to a query in the observed genealogy is more informative than the same source-wide allele frequency without that local alignment. This may help distinguish real source changes from false calls at existing candidate edges. References are temporal proxies, and identity by state is not proof of identity by descent. No posterior or independent likelihood interpretation follows.

Retain the complete H045 own-state backbone, every native tree vertex and all 952 observed haplotypes, 686 query outputs, observed SNPs, source labels and dates, both whole-tree sweeps, the full genomic BiGRU, classifier and native-hold output resolution. This tests evidence placement without simultaneously changing the output grid, adding a new memory rule, or adding a chain decoder.

Within each hold, predict a positive attenuation `gamma_e` for each native undirected edge from a small shared gate on existing observed edge channels and first-sweep endpoint context. Use the same attenuation in both message directions. At SNP s, seed only genuinely labelled reference sampled nodes with eight source × existing age-family channels and their called allele category b:

    x_rs[g,b] = 1[group(r)=g, called_rs, allele_rs=b].

Two additive recipient-excluding tree sweeps give

    m_(u->v)[g,b] = gamma_(u,v) *
                   (x_us[g,b] + sum_(w != v) m_(w->u)[g,b]).

At a query q this equals the ordinary weighted path sum

    N_qs[g,b] = sum_(r in g, called_rs) W_qr * 1[allele_rs=b],
    W_qr = product_(e on path(q,r)) gamma_e,
    D_qs[g] = sum_b N_qs[g,b].

For its observed called allele `a_qs`, use the contrast

    z_qs[g] = N_qs[g,a_qs] / D_qs[g] - f_sg(a_qs),

where f is the ordinary called-reference allele frequency in the SAME marker and group. Add called-mass and availability channels. Empty groups and missing query calls remain explicitly masked. Older references stay at their actual sampled nodes and dates. No older reference is assigned a fictitious 350-generation descendant, and unknown query ancestry never supplies a seed.

Perform the query-allele lookup before pooling. Pool these marker-specific contrasts within the original hold, then use a small rank8 adapter with zero-initialized output to inject a residual at actual query sampled nodes before the second complete tree sweep. Zero output preserves the common initial predictor. Exact source and allele definitions should follow the already written proposal rather than creating an almost identical variant.

This is deliberately a source summary, not a donor-indexed posterior. It cannot test donor continuity across SNPs. It may fail because local allele alignment is already encoded sufficiently, because inferred branches are unreliable, or because weighted identity by state adds noise.

## Specific prior art and novelty verdict

| Directly inspected material | Consequence |
|---|---|
| ARGformer, DOI10.1093/bioinformatics/btag438, saved full-text XML, method paragraphs on leaf-to-root node tokens, masked-node pretraining and contrastive retrieval | It exploits shared ancestral-node occurrences and genealogical reference retrieval, without direct genotype matrices. The proposed same-marker allele/source/path contrast differs in input computation, but path-based reference information is already established. |
| ARGMix supplied v1 text, DOI10.64898/2026.04.05.714076, TMRCA positional encoding and selected-reference subgraphs | It already learns reference-conditioned genealogical relationships. Its current author source also has `TmrcaKernelWeighted`, an exponential-TMRCA weighted reference-label classifier, and `GruRefiner`, a bidirectional genomic GRU. Weighted reference votes or graph-plus-sequence processing cannot be claimed as new. The source/allele alignment statistic is more specific than its label-only kernel. |
| AncestralPaths Nature supplement, supplied DOI10.1038/s41586-023-06705-1, Note3 printed pp30–33 | Ancestor-shell reference-group proportions and node ages already encode local genealogical support. Printed p33 explicitly discusses flanking-SNP topology inertia and correlated short false tracts. Boundary noise is not a newly discovered biological problem. |
| GST arXiv2606.05116v1, Section3.1 Eq1–12, cached primary text | Local graph propagation, context pooling and feedback into nodes are already central operations. Adding this residual before a second sweep does not establish feedback/interleaving novelty. |
| Ralph, Thornton and Kelleher2020, DOI10.1534/genetics.120.303253, Framework/Definition3 and site-statistic discussion | Vector-valued sample weights, additive tree accumulation and allele-weight summaries are established. Their ordinary site statistic can be tree-independent, whereas learned path attenuation here is tree-dependent. That distinction does not make additive weighted tree message passing a new primitive. |

The useful potential finding is **incremental value of correctly aligned same-marker source/allele/observed-branch evidence after a strong predictor**. All algebraic ingredients are familiar. The exact-combination priority search is incomplete, and the proposal already exists in this repository. A positive controlled result would support a biological representation contribution. It would not prove a new ordered-graph operator, historical priority, calibrated ancestry probabilities, or broad benchmark superiority.

A general ordered-graph contribution does **not** presently follow: this new route acts within snapshots and relies on the inherited GRU for order. It adds neither a new cross-snapshot relation nor a new recurrence. Any later claim about a general event-driven or incremental operator would require its own algorithm, exactness/complexity result and priority search. It cannot be inferred from ancestry accuracy or attached retrospectively to this experiment.

## One decisive matched experiment, prospectively frozen

Use the existing H045 own-state pass30 checkpoint as the common initializer, preserving the already proposed condition. Do not silently replace it with a validation-selected H050 winner. Run only these mechanistic arms:

1. **Aligned support:** actual reference allele sequences occupy their observed reference nodes in the new route.
2. **Alignment null:** only in the added route, one observation-seeded parent-wide permutation of complete reference diploid blocks within source × existing younger/older age-family strata. Keep it constant across sites. This preserves per-marker group allele/call counts and each reference sequence/LD while disrupting its extra association with branches and exact dates within its age family. Original backbone inputs and gates remain untouched. The interpretation is branch/date alignment, not pure topology with exact dates controlled. Record moved/called mass and immovable strata. Weak movement makes that part of the test inconclusive.
3. **Exact backbone continuation:** same whole-parent optimization without the added route.

**Reuse the completed seed101 H050 continuation as arm3, rather than rerunning it.** It already starts from the same H045 own-state checkpoint and finishes 30 passes/960 updates. New arms must inherit its exact FP32 AdamW recipe, parent ordering, loss, training exposure and checkpoint-selection schedule. Verify this equivalence from manifests/source before calling the comparison matched. The seed101 comparison therefore needs 1,920 new updates, not another baseline replay.

For a decisive prospective study rather than another one-seed observation, predeclare two additional last-stage seeds and run all three arms at each new seed, 30 complete passes/960 updates per arm. Together this is **7,680 new updates**, with the existing 960-update seed101 continuation reused. All arms keep all32 TRAIN parents, full observations, gradients and exact-span supervision. Six checkpoints use the original equal-parent Present182 accuracy/lower-NLL/earlier-pass rule. For each future evaluation, seal all new arm/checkpoint products before its scorer opens targets. Prior use of these VAL simulations remains disclosed. This quantifies continuation-stage seed uncertainty conditional on one inherited backbone, not full-pipeline initialization uncertainty.

Before new outputs, freeze the utility rule: aligned support must gain at least **1.00 percentage point Present182 boundary F1 at1kb** over BOTH mechanistic controls, lose at most **0.05 point span accuracy**, and not worsen mean NLL. Preserve source-transition identity diagnostics as well as the existing positional scorer, true-short-tract recall, false short tracts and false-switched bp per Mb. Report parent-paired effects with sampleSD, all seven cohort rows, precision/recall/F1 and event counts. Removing genuine short tracts cannot count as improved tract discrimination. No boundary-based checkpoint selection or validation-selected smoother.

Lock all choices before one final evaluation on the eight untouched outer European simulations, with every arm included. Reused VAL alone cannot make the result decisive. A seed101 comparison against the frozen H050 stable-leaf score has equal added training exposure and is a useful practical reference, but only aligned-versus-null attributes an effect to the proposed evidence alignment. A model that beats its null but fails continuation should be rejected as a useful addition. A development tie should end this condition, not trigger another unchanged replay.

The added arithmetic is approximately `O(sum_s V_t(s) * 8*A_s)`, using two scalar-channel tree sweeps, with marker-block working memory `O(B*V_max*8*A_max)`. It avoids a dense query-by-reference matrix and a full32-channel nonlinear graph rerun at every SNP. This is neither an O(edge-edits) claim nor measured feasibility. Stable log-domain sums/masks and exact checkpointed blocks need implementation review. The first scheduled full-parent optimizer update, counted in its real training budget, should establish feasibility. Any later execution belongs to root under physicalGPU0 and existing storage/lease policy.

## Sources actually read and actions

Required documents: `AGENTS.md`, `paper/current_full_panel_working.tex`, `research/progress_current.md`, `docs/biological_comparison_scope.md`.

Method/code: `src/arggnn/mutation_routed_whole_tree_gnn_v1.py` especially281–341, `docs/observed_variant_evidence_v1.md`, `docs/observed_variant_residual_v1.md`, `docs/reference_census_innovation_v1.md`, `docs/edge_innovation_gnn_v1_proposal.md`, `docs/interleaved_tree_tv_gnn_v1.md`, `docs/prior_art.md`, `docs/novelty_assessment_H038_v1.md`, `docs/biological_kernel_design_v2.md`, `docs/gst_baseline.md`; author ARGMix `src/joint_support.py:29–61` and `src/joint_interval_model.py:214–318` under `.runtime/author_sources/argmix_c2481248eb0d/`.

Primary texts: `research/reviews/ARGformer_published_benchmark_audit_20261015_v1/paper.fulltext.xml`; `research/reviews/ARGMix_published_benchmark_audit_20261015_v1/paper_text.txt`; `.runtime/author_sources/ancestralpaths_supplement_41586_2023_6705_MOESM1_ESM.txt` and the Note3 page excerpts; `.runtime/cache/gst-baseline/paper.txt:84–128`; `research/reviews/post_H050_shared_switch_primary_novelty_20261016_v1/tree_sequence_statistics_full.txt` Framework/Definition3 passages. FLARE cached primary text was also inspected for its extended Li–Stephens mechanism.

Relevant local reviews read: `next_biological_architecture_after_H050_20261016_v1/REPORT.md`, `next_useful_biological_mechanism_20261016_v1/REPORT.md`, `post_H050_attachment_uncertainty_direction_20261016_v1/REPORT.md`, `post_H050_marker_feedback_reassessment_20261016_v1/REPORT.md`, `post_H050_marker_tree_feedback_design_20261016_v1/REPORT.md`, `interval_tree_operator_novelty_20261016_v1/REPORT.md`, and `TGB_H051_completed_seed17_root_20261016_v1/REPORT.md`, all under `research/reviews/`.

Only this report was written. No scientific arrays, query truth, predictions or checkpoint payloads were opened. No experiment, model forward, scorer, fixture, resource admission, accounting mutation, source edit, installation or Git action occurred. Literature conclusions concern the inspected passages, not an exhaustive proof of novelty absence.
