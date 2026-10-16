# TreeLoom next-mechanism novelty audit

2026-10-16. Research-direction audit, not a paper peer-review verdict. Source text and existing concise outcomes only.

**No justified new architectural idea emerges from the inspected evidence.** TreeLoom is a useful development-stage whole-genealogy predictor assembled from established operations. The record neither demonstrates a new general operator for ordered graph collections nor isolates a useful distinctive cross-tree mechanism. Another memory, graph/set feedback, reset chain, donor-copying hybrid or spectral flow would repeat an examined direction or combine known components. This is a bounded assessment, not an exhaustive priority theorem.

## Exact model and evidence

The selected European TreeLoom condition is H050 stable_leaf_memory, not the triple-support arm. It scores **97.077844 ± 0.567395%** Present182 span accuracy against matched NoMem continuation **97.000028 ± 0.497841%**, a descriptive **+0.077816 percentage-point** difference. Boundary F1 at 1 kb is **18.380657% versus 19.048847%**, a **−0.668190-point** difference. Both arms receive 960 new complete-parent updates, 30 passes, fresh AdamW, seed 101 and independent checkpoint selection from the same H045 own-state pass30 initializer. Data: 32 full 2 Mb TRAIN simulations, eight repeatedly reused development VAL simulations, 952 input haplotypes and 686 queries, including 182 modern queries. Accuracy SD is across eight simulations in percentage points, not optimizer seeds. No superiority or independent-test conclusion follows. [Ledger H050](../../experiment_ledger.md), [manuscript](../../../paper/current_full_panel_working.tex).

| Actual computation | Established component and limit |
|---|---|
| Per-SNP 19-channel donor-frequency/callability tuple, shared MLP, mean/max pooling within each native tree interval | Local encoding and set pooling. The explicit site route loses marker order before graph interaction. This does not prove the complete model lacks correlated genetic evidence. |
| Full upward and recipient-excluding outside traversals, then another complete traversal with shared weights | Learned tree message passing, gates, normalization and residual refinement. All observed nodes remain. Recipient exclusion prevents immediate self-return, not all statistical dependence. |
| Gained/lost proper-descendant-clade flags and a rank-eight eligible own-state residual | Topology-change features and low-rank conditioning. The selected own-state arm imports no neighboring clade embedding. A changed clade identifies neither an ancestry switch nor a persistent biological ancestor. |
| Width-eight cut 350 context carried for the same sampled haplotype with rho=exp(−span/lambda), two directions, current-group averaging and centering before injection | Persistent-leaf state, exponential relaxation and aggregation. The 513 additional parameters act before the second traversal. No internal-ancestor identity is required. |
| Per-haplotype alignment, full genomic BiGRU, local/recurrent concatenation, four-class classifier | Genomic sequence modeling. Predictions are constant within native holds. There is no explicit donor trajectory or learned recombination/population posterior. |

Direct source: mutation_routed_whole_tree_gnn_v1.py:252–341, ordered_clade_episode_gnn_v1.py:319–403, H050 direct_readout_cut_memory_v1.py:123–147, and H049 order2_cut_memory_v1.py:375–434. These files were read as text, not imported.

Directly checked primary precedent: Whitehouse etal., *Tree Sequences as a General-Purpose Tool for Population Genetic Inference* (2024), [DOI 10.1093/molbev/msae223](https://doi.org/10.1093/molbev/msae223), “GCN Architecture and Training,” already combines full marginal-tree graph convolution, node GRU and tree-sequence GRU. Its tree-level compression and sequence-level tasks differ from TreeLoom's retained per-haplotype outputs, so this is not exact equivalence. ARGformer [DOI 10.1093/bioinformatics/btag438](https://doi.org/10.1093/bioinformatics/btag438), §2.1, explicitly uses repeated ancestral-node tokens across samples/intervals. ARGMix's primary §§2.1/4.1 use TMRCA-conditioned graph attention for LAI. These cached primary XML sections were read directly. No new public retrieval or absence-of-prior-art claim was made.

## Unresolved mechanism

The desired contribution is **useful extra ancestry information obtained from the ordered collection inside graph computation, beyond the same local observations and ordinary sampled-haplotype sequence processing**. Retaining complete trees and sample correspondences is worth assessing. A pipeline advantage over differently trained native methods cannot assign its gain to this choice.

Existing experiments narrow this question:

- H043 neighboring-state, H045 exact-clade episode and H046 foreign-flank context failed registered practical accuracy criteria.
- H048 graph-conditioned persistence failed both rules: transition 96.379965% versus confidence 96.388605%, with lower boundary F1.
- H050 triple 97.064043%, marginal 97.050364% and stable-leaf 97.077844% do not support the registered higher-order advantage. NoMem has the strongest boundary F1.
- H051's biological identity gate was source-blocked: the inspected Relate export uses fresh internal keys across trees. Its separate trade test was negative: TEST NDCG@10 conditional-triple 0.83045 versus first-order 0.83238 and native persistence 0.84688, one seed and three dependent TEST years. This does not prove all ordered genealogy mechanisms useless.

H058 asks a different utility question: query-allele-conditioned source/age evidence is gathered with positive tree-path weights before pooling, then injected before the second sweep. It is a within-tree alignment computation, with no new tree-to-tree state, event posterior or collection transport. Donor identity and SNP order are still marginalized within holds. Current decision: corrected one-hold FP32 arithmetic PASS, zero optimizer updates, full-parent training pending, no ancestry score. [Scope audit](../H058_same_marker_scientific_scope_audit_20261016_v1/REPORT.md), [decision](../../experiment_decisions.json).

## Strongest biological and ML gap

**A validated distinction between genuine source-ancestry transitions and changes/noise in inferred genealogy, while preserving real short tracts, remains missing.** High span accuracy with low boundary F1 makes this consequential but does not identify its cause. H057's TRAIN-only native-hold/grid oracle gap is just 0.1044431 ± 0.0098382 points. This establishes limited extra representability, not an achievable trained gain or an explanation of different VAL errors. Finer output alone is therefore not an evidence-backed cure.

Inputs are point inferred trees. Clade turnover, repeated internal keys and nearest reference alleles cannot be promoted to identified recombination events, ancestry posteriors or independent likelihood factors. A biological advance needs an observed, controlled failure repaired by a specific operation. An ML advance additionally needs a nontrivial distinction from message passing, temporal memory, latent-factor inference or ensembling. Renaming the current stack supplies neither.

## Decision: no new experiment proposed

There is insufficient evidence to recommend one new architecture meeting the requested novelty bar. Prior audits already considered higher-order conditional means, shared-switch factors, marker feedback, donor-continuity/remix HMMs, tree-uncertainty operator/state covariance and Haar semigroup/lazy propagation. Some lack prerequisite observation posteriors or proved algorithms. None becomes useful from the current scores.

The **already registered** H058 decision remains falsifiable: aligned same-SNP path evidence must gain at least 1.00 boundary-F1 point over BOTH its equal-capacity alignment-null and exact H045 continuation, lose at most 0.05 accuracy point against either, show no mean NLL regression and not damage uncensored true-short-tract recall. The null preserves source/age-group allele counts and whole reference sequences but changes branch attachment, exact age association within groups and spatial callability. It does not isolate topology alone. Losing either control ends that condition. Passing would support independent confirmation of alignment utility, not general graph-collection priority.

This audit changes no H058 arm, proposes no additional run and authorizes no execution. **No justified new idea is offered for registration.** A negative result should change the research question rather than trigger another wrapper.

## Scope

Read instructions, manuscript, source text, decision metadata, ledger summaries, named prior reviews and cached primary sections. Evidence manifest records hashes and reading scope. No scientific model/data/truth/checkpoint/probability payload, model import, fixture, resource check, experiment, installation, registration, existing-source edit or Git action occurred. Only this new audit directory was written. Root retains integration, experiment and publication authority.
