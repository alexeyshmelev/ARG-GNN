# Next accuracy hypothesis: directly supervised mutation-supported whole-tree classification

2026-10-11. Independent source/literature proposal only. No implementation, fit, outcome, GPU use or experiment registration is reported. This file is the sole project write for this assignment. The user-authorized 1 TB limit supersedes the old 500 GB statement in AGENTS.md.

## Recommendation and falsifiable mechanism

Test one architecture: a directly supervised, width-32 whole-tree cavity GNN whose reference messages are gated by **the concordance of original observed allele splits with their positions in each inferred tree**, followed by sample-aligned genomic recurrence and a direct C4 classifier. Train encoder, gates, recurrence and classifier together. Supply neither ARGMix/H025 predictions nor a teacher loss, query-age centering, or zero-initialized correction head.

The hypothesis is that reference proximity has different ancestry value on branches supported by observed allele sharing than on branches with little or contradictory support. Learned class-competitive routing might correct some confidently wrong genealogical assignments, whereas scalar reference-affinity correction mostly changes confidence. This is plausible, not established. Original variants and inferred trees derive from the same observations. Support is neither independent evidence nor a branch posterior.

The decisive contrast is genuine versus shuffled **branch-support placement**, with identical direct models, original SNP leaf features, complete trees, observed reference labels, genomic ordering, parameters, seed and updates. Both arms access genotype evidence and static genealogy. Only placement of additional support vectors changes. A tie rejects a useful branch-local support mechanism in this pilot. Likelihood improvement alone does not meet the accuracy objective.

This differs from H029/H030/H032 residual calibration, H033 tree/TV ordering and H035 signed mixed paths. Supervised ancestry prediction, gating, cavity sweeps and recurrence are established machinery.

## Evidence and limits

Inspected README/H025–H035 records show smoothing at 85.94% accuracy/NLL 0.3891, H030 static at 85.94%/0.3747, and H032 dependence controls with essentially the same accuracy/likelihood as that first-moment model. Genuine history/dependence has not won matched controls. H033's epoch 5 arms tie at 85.94%/0.3881. H035 has no predictive outcome in the inspected record.

H028's fixed 10 kb categorical-frequency residual did not improve accuracy. H031's fixed SNP classifier reaches 42.34% overall and 26.84%/29.16% on the two parents' baseline-error support, below the ANA source-prior prediction on those spans. There is **no measured strong complementary-SNP signal**. The unresolved possibility is conditional use of mutations to route genealogical reference information, rather than a strong SNP-only classifier.

H030 normalizes class-balanced reference sources, encodes subtree/outside mixtures, centers decoded context over same-age queries, and adds a linear correction to detached H025 logits. It does not learn standalone ancestry representations from original alleles. H028 fixes pseudocount frequencies, density-weighted 10 kb composition and softmax before its 32-scalar correction. Removing these constraints is an information/readout discovery change, not a novelty finding.

## Observation contract and target separation

Use one frozen genealogy producer and the four existing development parents. Prefer qualified tsinfer/tsdate trees already used by ARGformer and H029–H035, joined to original phased VCF by (sample_id,haplotype) and physical position. Disclose the genealogy difference from Relate-based native ARGMix. Do not mix producers within one encoder or call that historical comparison matched-topology.

Every marginal interval retains the complete connected sample-ancestor union, all 280 sampled haplotypes, native internal branches and observed ages. Disconnected administrative rows belonging to other marginal trees are not vertices here. Do not suppress mutation-bearing unary branches. Node numbers are storage indexes, never embeddings.

Only 96 observed pure-source proxy classes enter features, ordered ANA,CHG,WHG,EHG. Other 184 leaves, including 64 mixed-reference context leaves, carry no input ancestry label. All280 leaves participate in encoding and recurrence. Only actual query 120 enter supervised loss or scored prediction. Present-day 64 remain primary. Older references stay at their true observed sampling dates, with no downward projection to 350 or fabricated query target.

The target remains the simulated query's population at the350-generation census, mapped to C4. Prohibit hidden census flags, populations, ancestry paths, migration annotations, raw simulated trees and cohort mixture proportions as features. Mutation support comes from original observed alleles and inferred topology, **never simulated mutation nodes or true ancestral states**. Seal observation structures before opening TRAIN targets. VAL truth opens only after prediction sealing through the existing scorer. Split by biological parent before windows/samples. No TEST access.

## One small architecture

**Site-to-leaf route.** At each original site, construct allele-index-invariant features for each called haplotype: its allele frequency in the eight younger/older observed donor-class groups, callability, total-panel allele frequency and site entropy. Retain H028's pseudocount 0.5 and absent-class/missing-call rules, but keep log frequencies and masks before softmax or centering. A shared small site MLP encodes these tuples. Mean/max pool within the site's genuine marginal-tree interval, append called-site count and interval span, then project to width 32. Empty intervals have zero mutation input and an empty flag. Retain every original multiallelic site. Neither SNP nor allele index receives an embedding. This is learned pooling, not an independent-SNP likelihood. Original variants excluded by genealogy inference can contribute within existing coverage, with provenance disclosed.

**Label-free branch support.** At site s, for nonroot branch e, let D_e be its sampled descendants restricted to called samples, and C_sa the called samples carrying allele a. For each nonempty proper allele-versus-rest split define

m(s,a,e) = min(|D_e symmetric_difference C_sa|, |D_e symmetric_difference complement(C_sa)|) / N_called(s).

Complements use the called roster. No ancestral polarization is required. Compute descendant allele counts by tree sweeps, without dense sample-pair matrices. For each allele split find all minimum-discrepancy branches and divide support equally across ties, including unary branches with identical descendants. Per branch retain exact-split support, closest-split support weighted by 1-minimum_discrepancy, average discrepancy across callable splits, callable split count, and an empty flag. Normalize support counts by the tree's callable split count and retain that count separately. Native endpoint ages, duration and descendant count remain separate geometry channels.

These are concordance summaries, not inferred mutation events. Recurrent/multiallelic sites need not be explained by one mutation. Contradictions remain present rather than filtered away. Inference might make these summaries almost redundant with topology. Measure variation and sham moved-field fraction before fitting. Nearly constant support cannot sustain this hypothesis.

**Whole-tree routing.** Use two directed cavity sweeps with shared width 32 parameters. Sampled nodes start from site-to-leaf input, observed reference/no-label flag and age. Anonymous internals start from age, root/sample status and zero leaf evidence. Learned upward messages are multiplied by sigmoid gates of geometry and mutation support. Use invariant sum plus child-count normalization. The downward message uses parent outside/local state and sibling messages while excluding the receiving child's own upward message. Residual nonlinear updates limit attenuation along deep paths. Every reference can reach every query through complete-tree context, with no fixed ancestor cutoff or sampled-path restriction. Logits derive from learned states rather than fixed normalized reference affinity.

**Ordered readout.** Gather each tree's 280 leaf vectors by verified persistent sampled-haplotype map. Scan **every original interval** in genomic order with one bidirectional width 32 GRU, supplying interval span and positive physical distances. A shared MLP maps local context and forward/reverse states to four logits. Holds without scoring centers still participate. Readouts gather the containing tree and exact requested haplotype, preserving all original clipped bins.

Cross-tree correspondence means the same observed haplotype. There is no inferred-ancestor memory or true-ancestor identity claim. Position x orders trees, age t describes branches, and genomic recurrence is not evolution in genealogical time. This remains an ordered collection of complete trees although cross-tree states live on sampled leaves. A gain would not validate the earlier persistent-ancestor mechanism.

## Decisive matched control and decision

Fit exactly two arms, aligned and branch-support shuffled. Freeze sham before target access: derange entire mutation-support vectors among nonroot branches **within each tree**, stratified by observed duration and descendant-count bins. Fix bin edges from TRAIN observations alone, apply unchanged to VAL, and shuffle only strata with at least 4 members. Report unmoved branches and physical/query-weighted moved-field fraction. If fewer than half of supported vectors move, mark the mechanism test inconclusive. No permutation depends on accuracy or query ancestry.

Both arms retain geometry, donor labels, original SNP leaf route, per-tree support distribution, complete trees, sample correspondence, decoder, active parameter count and optimizer schedule. Sham keeps geometry original and permutes support. The intended broken information is its relationship with a branch. This pair tests placement, not the separate benefit of removing centering, teacher dependence or the residual head, since those change together versus H030.

Use FP32, one seed 101 and at most 160 updates per arm with inherited optimizer/clipping. Fix width 32, two sweeps and one bidirectional GRU. Process full-parent ordered leaf sequences with checkpointed batches of complete trees. Do not silently detach history, drop leaves or truncate tree access. If full-parent gradients are infeasible, revise the explicit protocol before fitting. Match schedule within the pair and disclose differences from historical tile updates. Select the earliest minimum equal-parent Present 64 VAL NLL checkpoint and report its accuracy, without accuracy reselection.

Reuse frozen H025/native ARGMix/H030-static/native ARGformer receipts on identical queries and physical support. No native refit is needed here. Score the new models standalone. A later trained blend with ARGMix is a separate hypothesis. Oracle error unions or hindsight mixtures do not demonstrate selectable complementary signal.

Suggested practical discovery gate: aligned accuracy exceeds both sham and 85.94% smoothing by **0.20 percentage points**, with positive paired direction on each VAL parent. This is a decision margin, not significance/power evidence. Also report NLL, class, short-tract/boundary results and correct-to-wrong/wrong-to-correct changes against smoothing after prediction sealing. Aligned beating sham but losing smoothing is a mechanism lead that misses the objective. Beating smoothing with sham tied is an input/direct-model finding without branch-placement evidence. Failure under the fixed small budget is a negative discovery result, not a theorem against all mutation/tree supervision.

Only a convincing lead motivates fresh-parent confirmation and H034 full-panel comparisons. Reused VAL, historical in-sample TRAIN emitter outputs, genealogy differences and historical training budgets remain explicit. Do not reopen parity matrices, seed sweeps or answered memory/Gram pilots.

## Closest prior methods and novelty

- **ARGMix / Graph transformer for ancient ancestry inference**, Shanks et al., bioRxiv 2026, DOI [10.64898/2026.04.05.714076](https://doi.org/10.64898/2026.04.05.714076). Inspected current-release commit[c2481248eb0de5fda659d68f59c51e0e27becd20](https://github.com/AI-sandbox/argmix/tree/c2481248eb0de5fda659d68f59c51e0e27becd20) uses population tokens/TMRCA-biased attention in tmrca_attn.py/tree_sampler.py and genomic GRU refinement in joint_interval_model.py. Direct reference-conditioned supervised genealogy plus recurrence is already close prior art. Current release is not an exact paper-era reconstruction.
- **ARGformer**, Bonet et al., Bioinformatics 42 Supplement 2, btag438, 2026, DOI [10.1093/bioinformatics/btag438](https://doi.org/10.1093/bioinformatics/btag438). Inspected cached published XML specifies leaf-to-root node-ID tokens, masked-node learning, supervised contrastive reference training and retrieval, expressly without genotype matrices. Complete-tree sibling context and observed allele-support gates differ from that inspected implementation. Shared genealogies/reference learning do not.
- **AncestralPaths**, Irving-Pease et al., The selection landscape and genetic legacy of ancient Eurasians, Nature 625:312–320, 2024, DOI [10.1038/s41586-023-06705-1](https://doi.org/10.1038/s41586-023-06705-1). Cached Supplementary Note 3, pp 30–32 describes disjoint ancestor-shell group proportions and ages at first 5 informative nodes, supervised on Relate trees. The paper CNN and released Dense implementation are distinct. Reference-labelled neighborhoods, ages and simulated supervision are established.
- **Tree-LSTM**, Tai, Socher, Manning, ACL 2015,[arXiv1503.00075v3](https://arxiv.org/abs/1503.00075v3), inspected primary text and child-sum composition. Learned tree aggregation/gating is inherited. Cavity/rerooting also is ordinary machinery.
- **Graph Set Transformer**, Escrig Molina, Chen, Probst, 2026,[arXiv2606.05116v1](https://arxiv.org/abs/2606.05116v1). Existing project source audit covers graph propagation and graph-collection attention/feedback. Whole-tree collection processing is not sufficient novelty.

The prospective contribution is narrowly **observed mutation-concordance placement for ancestry reference routing**, conditional on accuracy and matched-control success. Novelty confidence is low. Edge conditioning, graph denoising, genotype/genealogy fusion and sequence recurrence are broad existing ideas. This cached-source inspection is not exhaustive priority evidence. A positive pilot initially supports a domain mechanism/representation finding, not a new general GNN claim or publication outcome.

## Lean execution and inspected sources

Preprocessing is linear in called-site/tree-node incidences using batched descendant counts. Trainable actions are linear in vertices times width squared plus tree-by-sample genomic scans. No donor Gram, quadratic all-node attention or matrix carrier state. Cache compact branch/leaf summaries, retain sparse full topology, and checkpoint whole-tree batches. Runtime/memory are unmeasured. Basic identity/missingness and finite FP32 loss/gradient checks suffice before discovery, without broad equivalence matrices.

Read AGENTS.md, development_priorities_20261011.md, README.md, experiment_ledger.md, experiment_decisions.json, biological_estimand_and_features_v1.md, observed_variant_residual_v1.md, novelty_assessment_v1.md, cached ordered-collection priority note, and relevant model source. No result was rerun. Selected inspected-source SHA256 hashes:

- reference_logcavity_gnn_v1.py:8970aa2e82a43b45d1ec10b2eaae6e2203dab12356a4b968bea3c65b2e21fc0b
- observed_variant_residual_v1.py:695a78d266d8dc4cf226f6e8ff54669979eaf94f30e5cc8cb3062f14ec02c904
- census_tree_encoder_v1.py:09351221d2d91f72c8b1df12bc3799ed72e538597c31a55737ed1bec111f886e
- Cached AncestralPaths supplementary text:f4d21b9b08ead8f4f9a25ba6510123ff91d1f83005a46d8839b0feb87c839b7d
- Cached ARGformer published XML:53ab26710804f0b9ca6f355095893b365413caf56cf7bb722b39907f6db0619f
- Cached Tree-LSTM text:e94871fc1c2e0875ed8dff49db55b668bd0c0d8454e6d4eec3607b2ea2106a6b

No code, configuration, ledger, accounting or Git history changed. Registration and an explicit executable data/training freeze are future prerequisites if selected.
