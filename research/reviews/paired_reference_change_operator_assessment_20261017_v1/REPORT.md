# Paired reference changes in inter tree ancestry inference

17 October 2026. One source-only operator hypothesis.

**A feasible, distinct utility hypothesis is to use paired changes of labelled references across adjacent complete trees to condition a correction of query representation changes before the second tree sweep.** This uses actual sample correspondence, rather than inferred ancestral identity or topology edits. Its matrix operation is established ridge shrinkage. It therefore does **not** yet supply a defensible new general GNN operator or close TreeLoom's scientific contribution gap. The possible contribution is narrower: a useful reference-anchored relation for distinguishing nuisance genealogy changes from ancestry boundaries under transfer. That relation is unvalidated and can fail biologically.

## The gap supported by the record

Selected European TreeLoom is H050 stable_leaf_memory: Present182 span accuracy 97.077844 ± 0.567395%, versus 97.000028 ± 0.497841% for its continuation control. Its 1 kb boundary F1 is lower, 18.380657% versus 19.048847%. These are eight reused development parents, one optimizer seed and 960 added updates per arm, with separate checkpoint selection. They establish neither oversmoothing as a cause nor independent generalization.

Actual source implements SNP-conditioned complete upward/outside sweeps, gained/lost-clade and own-state residuals, rank-eight stable-sample exponential memory before the second shared tree sweep, then a complete-parent BiGRU and per-sample classifier. H050's memory is explicitly rho*m+(1-rho)*forcing followed by current-cut pooling and centering. Merely changing its decay, gate, name or carrier is insufficient here.

H043, H045 and H046 failed practical foreign-state utility; H044 localization lost boundary F1; H047 risk balancing worsened accuracy/NLL; H048 persistence placement failed; H049 was prepared without fitting; H050 higher-order carry failed; H051 trade conditional memory lost to its first-order control. H052–H054 establish baselines, active observation support and adaptive-carrier reasoning, not a LAI operator gain. H055's empirical affinity direction failed and H056's descriptive model assignments are not validated transfer accuracy. H057's TRAIN hold/grid oracle gap is only 0.1044431 ± 0.0098382 accuracy points. H058, H059 and H060 must retain their existing scientific questions; their decision snapshot is not a new efficacy result. H061's registered TEST memory criterion failed. No unchanged condition is proposed again.

The current boundary, shared-switch, attachment, categorical-census, random-frontier, adaptive-carrier and continuous-heat audits were consulted. The following hypothesis neither identifies recombination from clade edits nor introduces a donor-copying posterior, attachment mixture, new census variable or heat transport on clades.

## One explicit hypothesis

Within a source-labelled reference group, local genealogy and genotype evidence vary even though the supplied source label stays fixed. Some directions of **paired representation change** may therefore describe variation a query should tolerate without an ancestry switch. Use those directions to condition a small signed learned correction. The hypothesis is that correct reference pairing improves boundary precision and transfer beyond generic smoothing with the same local observations.

The assumption is deliberately weaker than calling references known invariant ancestry at census350. Ancient proxy labels are not authenticated census states. Fixed metadata labels do not make their genomic variation noise. Query admixture can also react differently to genealogy error than reference representations do. These are central failure possibilities.

Let t index all original native holds, w_t their physical widths, and i an observed sampled haplotype. Let h_ti in R^32 be the first complete tree sweep's sampled-node context, before any genomic GRU. Compute it on every original vertex and sample, including internally sampled nodes. Keep the existing structural adapters, stable-sample memory, second complete sweep, BiGRU and readout unchanged.

For boundary e=(t,t+1), define

    z_ei = U(h_(t+1),i - h_ti),       U: R^32 -> R^8.

Only genuine same-sample identities supply this join. Let R_g be the labelled references in existing nonempty source × donor-age groups, and G their number. The boundary second moment is

    S_e = (1/G) sum_g (1/|R_g|) sum_(r in R_g) z_er z_er^T,
    Q_e = S_e (S_e + lambda I)^(-1), lambda=0.0001+softplus(theta).

This is an uncentered second moment, intentionally including coherent reference drift. It is **not** an ancestry posterior covariance. No query label enters it. The reference-indexed h values still contain permitted transductive whole-tree evidence, so this is not a reference-only evidence channel or query-panel invariance claim. Empty reference support gives Q_e=0.

Set d_ei=O Q_e z_ei, with O: R^8 -> R^32 initialized to zero. For each boundary add the following simultaneous sampled-node injections to the existing second-pass input:

    I_ti       += [w_(t+1)/(w_t+w_(t+1))] d_ei,
    I_(t+1),i  -= [w_t/(w_t+w_(t+1))] d_ei.

Sum contributions from the two incident boundaries and use the original sample-to-native-node map. Chromosome endpoints have one incident boundary. All original trees remain complete. These equations define one two-slice correction, with no scan, recursive memory or extra supervision. U and O contribute 512 weights and theta one, an analytic 513-parameter addition. Zero O exactly preserves the common initial predictor.

Each boundary's added feature mass is span-weighted zero. Q_e is symmetric PSD with eigenvalues s/(s+lambda) in [0,1). Neither fact proves that the learned U/O residual, simultaneous multi-boundary block or complete nonlinear predictor is contractive or improves calibration. Native-hold outputs remain native-hold outputs: the operator cannot locate a new within-hold boundary. Arbitrary hold-refinement invariance is not claimed.

## What is distinct and what is established

For group-weighted reference matrices, S contains

    U [M_right + M_left - C_(right,left) - C_(left,right)] U^T.

The C terms are paired cross-tree moments. One-tree reference-state multisets do not determine them. Thus the proposed relation is not just source counts, a within-tree donor-support Gram, an EMA, or the existing cut-intersection table. It differs from H032's same-tree donor-path dependence and H050's state carry. An unrestricted temporal GNN could compute such features; no expressivity dominance over existing models is proved.

The algebra itself supplies a candid limit. Put weighted reference jumps in rows of D, so S=D^T D. Then

    Q z = D^T (D D^T + lambda I)^(-1) D z.

This is exactly the fitted component of the ridge problem

    min_a ||z-D^T a||^2 + lambda ||a||^2.

The production formulation uses the 8-by-8 solve, not the reference-by-reference matrix. It is a regularized projection onto observed reference-change directions, followed by ordinary cross-tree messages. Calling it a new inverse, denoiser, attention principle or temporal GNN would be unjustified.

Primary nearest antecedents:

| Primary source actually checked | Consequence |
|---|---|
| Gagnon-Bartsch and Speed, *Using control genes to correct for unwanted variation in microarray data*, Biostatistics 13:539–552 (2012), [DOI 10.1093/biostatistics/kxr034](https://doi.org/10.1093/biostatistics/kxr034), author abstract | RUV-2 already estimates unwanted variation using negative controls. Its controls are defined to lack the biological effect of interest. The proposed source-labelled references do not automatically satisfy that condition, making this a harder biological assumption. Full RUV methods were not accessible in this retrieval; no exact RUV algorithm identity is asserted. |
| Sun, Feng and Saenko, *Return of Frustratingly Easy Domain Adaptation*, [arXiv1511.05547v1](https://arxiv.org/html/1511.05547v1), §3.1–3.2 and Algorithm1; Sun and Saenko, *Deep CORAL*, [arXiv1607.01719v1](https://arxiv.org/html/1607.01719v1), §3.1 | Covariance-conditioned feature adaptation, ridge regularization and differentiable feature-statistic operations are established. CORAL aligns domain covariances; this hypothesis conditions genomic jumps on same-reference cross moments, rather than aligning two marginal domains. That is a specific relation, not a new covariance primitive. |
| Whitehouse, Ray and Schrider, *Tree Sequences as a General-Purpose Tool for Population Genetic Inference*, [DOI10.1093/molbev/msae223](https://doi.org/10.1093/molbev/msae223), “GCN Architecture and Training” | Whole marginal-tree GAT processing, node GRU and tree-sequence GRU already coexist; consistent sample order is explicit. The proposed reference-paired statistic is more specific, but generic ordered-tree GNN priority is unavailable. |
| Hoerl and Kennard, *Ridge Regression: Biased Estimation for Nonorthogonal Problems*, Technometrics12:55–67 (1970), [DOI10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634) | Bibliography verified; the ridge identity above is derived directly, without attributing this biological construction to that paper or claiming to have read its full proof. |

The bounded literature comparison supports withholding generic novelty. It does not claim that a missing search hit establishes priority or that any cited paper implements this exact ancestry block.

## Feasibility and one matched falsifier

For T holds, H sampled haplotypes, R references, width d=32 and rank r=8, incremental work is

    O(T [H d r + R r^2 + r^3 + H r^2]).

Use batched small SPD solves and scatter sampled-node injections with Torch/PyG. Forward working storage per boundary block is O(Hr+r^2), excluding existing first-pass states and an O(THd) injection buffer if materialized. Full gradient storage/recomputation, backend solve overhead and the unchanged whole-tree/BiGRU cost remain additional. There is no H-by-H, donor-pair or tree-pair tensor, and no claim of measured throughput or admitted memory fit.

A single future matched comparison must retain one common full TreeLoom initializer, every observed tree/SNP/sample, the same 513 added weights in all correction arms, independent identical-budget optimization and unchanged selection. No live H058/H060 arm is altered.

The decisive null changes only the reference pairing used to build S: within each source × age group, replace h_left,r by h_left,pi_g(r), with a fixed observation-only nonidentity permutation. Every per-tree reference-state multiset, reference label, group count and original backbone input remains exactly the same. The sham uses the same solves, maps and schedule. Single-member groups are unchanged; report actual covariance perturbation and abandon the pairing interpretation if the null barely changes S. Under chromosome reversal, invert the pairing so the same matched pairs are compared.

Include an equally parameterized isotropic arm S_iso=trace(S)I/r, computing the same full S before discarding orientation, and the unmodified full-TreeLoom continuation. A tie with the pairing null rejects the claimed value of genuine reference correspondence. A tie with isotropic correction rejects a direction-specific nuisance-removal explanation, even if both beat continuation. Gains only over historical scores cannot identify this mechanism.

Freeze new independent dense-label simulation parents and a reference-panel transfer stress before outcomes. Do not amend H059's frozen confirmation or use its outputs to tune this condition. Within each stress cell, all arms must see the same inference result and source/age-labelled panel. A suitable transfer stress is prespecified reference thinning/source imbalance; genuine empirical proxy replacement requires its own qualified labels. Train only on TRAIN query targets; reference pairing and inference-time matrices use observations alone.

A deliberately stringent boundary gate would require at least +1.00 percentage point equal-parent Present182 1 kb F1 over both correction controls and continuation in the standard and shifted-panel conditions, at most 0.05 point accuracy loss, no mean NLL regression, and no loss of uncensored true ≤10 kb tract recall. Report matched/missed/extra boundaries, false short tracts and false-switched bp per Mb. Fewer false calls obtained by deleting true short tracts is failure. Sparse American fixed focals can assess accuracy/NLL transfer, not dense boundary fidelity. This is a mathematical falsifier, not an experiment registration or execution request.

## Decision and evidence scope

**Retain only as one low-confidence, unregistered biological utility hypothesis. Reject a new-general-operator claim at this stage.** Paired reference change is a precise additional relation and a feasible bias with a strong null. Its negative-control assumption may be false, and current residual summaries do not establish that reference-shared drift causes TreeLoom's errors. Useful, independently confirmed boundary/transfer gains would support a specialized method contribution; the equations alone do not.

Direct source ranges: mutation_routed_whole_tree_gnn_v1.py:252–388; ordered_clade_episode_gnn_v1.py:319–403; H050 direct_readout_cut_memory_v1.py:123–147; order2_cut_memory_v1.py:375–434; American full-mechanism v2 classes at67–153 and253–367. Read experiment_decisions.json's H043–H061 scientific summaries and the relevant architecture/transfer, H043/H045–H051/H054, biological-outcome, H032, boundary, census, attachment/shared-switch, random-frontier and continuous-transport audit sections. Operational status is not inferred from those historical summaries.

Public retrievals were RAM-only. Full primary hashes: Whitehouse XML e3f86d6fa6a29a88ab9dd13aa626f3f652e8ae585f277cd11af2623afb77931d; CORAL HTML e6880c92138ef3cd5b411a015c214c0d14f7c925d02c93b563d87e1376aabd7b; Deep CORAL HTML 15855b077c2d0ced53f2f3aa45b149557db683296e9cea39daa3046ec49e6eeb. RUV author abstract from Europe PMC PMID22101192 core response: 8f19c152bc43d978699584a7d34b977951a763e98a6193ad71c9aeffe8a21af9. A first incorrect CORAL identifier resolved to an unrelated quark-matter paper and was excluded. The RUV PMC page returned a browser challenge, so it supplied no methods evidence.

Only this report is written. No model import/implementation, scientific payload/truth/checkpoint/prediction read, training, registration, guard/state/resource/process/GPU operation, installation or Git action occurred.
