# H049/H050: operator novelty and the tract-inference gap

**Generic operator priority remains unsupported. The defensible question is whether a particular pooling bias improves ancestry tracts on inferred genealogies.** H050 is a useful controlled utility condition on a stronger predictor. It reuses H049's memory operator, rather than introducing another transport primitive.

This review adds a source-level representation-invariance finding and a reference/query communication distinction. It carries forward the existing Rosvall/Scholtes/De Bruijn conclusion and does not repeat the H048 reset-decoder audit. No scientific outcomes or live job status were assessed.

## What the actual code computes

For old cut-pair cells (A,B), new cells (B,C), and stable sampled-leaf incidence matrices L_old and L_new, the carry is

P = D_new^-1 L_new^T L_old,  D_new = diag(L_new^T 1).

Thus P[(B,C),(A,B)] = n_ABC/n_BC. Code lifts each old cell state to its leaves and averages into new cells (core lines399–402). It is conditional expectation over the observed sample panel. The adjacent-marginal null averages inside B first (403–407). The leaf control retains each leaf's rank8 state (386–393). These are different historical pooling biases over the same observations. Full leaf history is richer, so no information or expressivity dominance follows.

Between events the update is m_next = exp(-width/ell)m + (1-exp(-width/ell))f. Forcing f is U times first-sweep upward plus outside states at actual cut representatives. Shared forward/reverse histories are averaged, pooled to current cut groups, centered by subtracting current forcing, then mapped through O (core413–434). The cut is at350generations, with real sampled singletons above the horizon and real root-saturated representatives (204–245). Those are disclosed biological modeling choices, not persistent ancestor identities or learned demographic rates.

H049 injects this residual before its second tree sweep, then uses its genomic BiGRU and H048 decoder (454–494). H050 imports this exact CutMemoryV1 and adds its residual to H045's trained own-state adapter before the second sweep (H050134–147). It retains H045's direct readout. H049 versus H050 also changes common architecture/warmstart, so their cross-pipeline difference alone is not a clean decoder-placement ablation.

## Closest prior art and the actual distinction

| Prior art | Relevant established machinery | What remains specific here |
|---|---|---|
| [GCRN, Seo et al.,2016](https://arxiv.org/abs/1612.07659), abstract; [TGN, Rossi et al.,2020](https://arxiv.org/html/2006.10637v3), §3.1 Eq3–4 | Combining graph aggregation and recurrent memory, event-indexed states, message aggregation and memory updates | Carriers are anonymous cut-partition intersections across genomic coordinates. Genomic order is not causal clock time, and bidirectionality is permitted for this inference task. Graph plus recurrence is already established. |
| [TGAT, Xu et al.,2020](https://arxiv.org/abs/2002.07962), abstract; [CAWN, Wang et al.,2021](https://arxiv.org/abs/2101.05974), abstract; [DyGFormer, Yu et al.,2023](https://arxiv.org/abs/2303.13047), abstract | Temporal topology, identity-free walk motifs/hitting counts, and historical neighbor co-occurrence used in learned representations | Exact all-leaf cut contingencies, rather than sampled temporal walks or first-hop event co-occurrences. This does not establish priority for history, anonymous identity handling, or co-occurrence features. These models are related mechanisms, not literal algebraic equals. |
| [DiffPool, Ying et al.,2018](https://arxiv.org/html/1806.08804v4), §3.2 Eq3–6 and Proposition1 | Assignment-matrix pooling/coarsening and permutation arguments | Here assignments are fixed observed cut memberships and means use explicit cell counts. Hard incidence lift/project and its equivariance are not a new graph-to-graph primitive. |
| [Fused Gromov–Wasserstein, Vayer et al.,2018](https://arxiv.org/html/1811.02834v1), §2 | Optimizing a coupling using feature and structural costs | H049 already knows leaf correspondence and fixes the empirical coupling L_new^T L_old/H. It solves no OT objective or unknown-node matching problem. Calling the same count matrix “optimal transport” would not create a new contribution. |
| [Rosvall et al.,2014](https://www.nature.com/articles/ncomms5630); [Scholtes et al.,2014](https://www.nature.com/articles/ncomms6024); [De Bruijn goes Neural,2022](https://arxiv.org/html/2209.08311v1) | Path-indexed memory states, conditional transition counts, higher-order neural message passing and projection to original nodes | The previous audit already established this closest higher-order overlap. The incoming count-normalizer is not literally the forward probability normalizer, but transposing/conditioning the same incidence relation does not establish a new memory-network principle. |

Width-aware dynamics also belong to an established continuous-time modeling landscape, e.g. [Kidger et al.,2020](https://arxiv.org/abs/2005.08926). That reference is background, not a claim that this affine recurrence is a neural CDE or inherits its theorems.

The new retrieval includes eight verified primary abstract pages and full TGN, DiffPool and FGW texts. Bibliographic titles/versions are preserved exactly in verified_primary_bibliography.v1.json. The FGW source is the mathematical-foundations manuscript, not the differently titled ICML paper. This is a focused mechanism audit, not an exhaustive priority search.

## Properties that matter for biology

| Property | Source-supported assessment |
|---|---|
| Leaf-identity equivariance | In exact arithmetic, consistently permuting the sampled leaves and every corresponding genotype, age, role, recipe and readout axis conjugates the incidence maps and permutes query outputs. First-encounter group numbering can change, but it is an index, not a learned identity. FP32 reduction order can change rounding. This is not a claim about re-running the tree-inference algorithm after permuting its input. |
| Transport consistency | At a carry event P is nonnegative and row-stochastic. Cell-size-weighted totals are preserved before subsequent decay/forcing. This follows from complete leaf partitions. It does not conserve an ancestry probability, establish calibration, or identify a real ancestor. |
| Redundant marginal-tree subdivision | **Not a full-predictor guarantee.** Episode collapsing avoids repeated pair-cell projection when the cut is unchanged. Constant-forcing carry has a semigroup. Neither fact makes the per-hold memory readout or the inherited predictor invariant to arbitrary duplicate tree records. |
| Reference/query separation | The reviewed interface keeps query ancestry unknown, and the memory adds no truth reader. But all952 leaves enter support counts, and forcing comes from shared tree states containing allowed query evidence. This is label-free transduction, not a directed reference-only evidence channel or invariance to query-panel composition. That distinction is not an accusation of leakage. |
| Tract resolution | H050 produces native-hold logits and expands them through readout_hold. A real ancestry switch strictly inside one inferred hold cannot be independently localized by that readout. A better average accuracy therefore need not mean better recovery of short tracts or boundaries. |

### A literal subdivision counterexample, without execution

Even isolate the memory, keep the cut/forcing unchanged, and ignore every neural-encoder complication. Consider an interior interval with forcing f=0 and incoming left/right scalar states both1. Let two half-width decay factors equal1/2, so the whole-width factor is1/4. The current code averages the two states after each direction has consumed the full current hold.

One unsplit hold emits (1/4+1/4)/2 =1/4. After splitting it into two identical holds, the first emits (1/2+1/4)/2 =3/8 and the second emits (1/4+1/2)/2 =3/8. Their width-weighted mean is3/8, not1/4. The end-to-end carry across the interval is still correct. The readout changed. This concerns learned nonzero O, not the deliberately zero initial residual.

There is a second independent H050 issue: inherited H045 own_state residuals are masked by episode_size>=2 (ordered_clade_episode_gnn_v1.py339,351–354). A singleton proper-clade hold split into two identical records can enable that trained adapter even when new O=0. The inherited BiGRU also advances once per record and uses per-hold/neighbor spans (anonymous_clade_flag_control_v1.py291–303). Site pooling itself includes hold span (mutation_routed_whole_tree_gnn_v1.py339).

These are source/algebra findings, not measured failure rates. They do not invalidate a fixed-input H050 comparison. They rule out advertising full subdivision invariance from the unchanged-cut optimization alone.

## What H050 can establish

Its three memory arms match the513 added weights and initialization. The unmodified H045 continuation separates memory-package gains from another960updates, with its smaller31812 versus32325 parameter count disclosed. The draft requires triple support to beat the adjacent null, leaf memory and continuation, with boundary safeguards and all192pretruth products.

A successful result would support: **conditional cut-history pooling is a useful inductive bias on this predictor and this inferred-data condition.** It would not establish a new generic operator, ancestry identifiability, a biological interpretation of ell, subdivision robustness, independent-test performance, or superiority to every leaf-aware model. Conditional mutual information in the cut tables is an opportunity statistic, not evidence that those conditionals predict ancestry.

## One prospective direction: reference-anchored memory for robust tract inference

If a new condition is justified after H050, target one biological claim:

**On independent admixed genomes inferred from noisy genotypes, a reference-anchored memory that is invariant to lossless tree-record refinement reduces spurious short ancestry tracts without erasing true short tracts.**

The biological issue is specific. Inferred-tree changes can reflect uncertainty rather than an ancestry switch. Conversely, indiscriminate memory smoothing can merge genuine short tracts. Pure-source reference evidence should not be reweighted just because many unlabeled queries occupy a cut cell.

A prospective design would use an ancestry-evidence carrier whose support counts *and forcing* come only from genuinely labeled reference haplotypes, with queries reading that channel through observed membership. Empty-reference cells need an explicit unavailable state and a local-evidence fallback. Restricting only the denominator is insufficient if shared query-derived states still write the forcing. Keep a separate local query genotype/topology path.

Treat representation consistency as an entry requirement, not a novelty slogan: losslessly coalesce identical tree/date/sample records while preserving the original SNP point measure, or make every downstream state/readout operate on physical-coordinate events. Check the complete prediction at unchanged scoring centers, including the inherited adapter/GRU. A width-aware memory alone is insufficient. This does not require reviving the H048 reset-chain.

The new scientific test would compare the role-separated condition with a same-capacity transductive memory and the existing leaf/null controls, on independent parents with prespecified inference-error levels. Freeze scoring and tract cutoffs before outcomes. Report false predicted short-tract count and false-switched bp per Mb, recall of true short tracts (the existing<=10kb stratum), boundary displacement/1kbF1, exact-span accuracy and calibration. Keep within-hold resolution misses explicit. Lower false-tract burden must not be purchased by destroying true-short-tract recall. Leaf relabeling, redundant subdivision and reference/query communication checks establish the intended mechanism before any utility claim.

This is a new robustness/biological-error hypothesis for later registration, not new generic recurrence/OT/incidence machinery. It does not alter current H050's fixed condition or authorize another experiment.

## Evidence and scope

reviewed_source_refs.v1.json records16 exact source/background refs, function locations and hashes. property_assessment.v1.json separates proved algebra, source limitations and untested utility. prior_art_matrix.v1.json records claim-level primary locators. prospective_direction.v1.json is an inactive concept only.

All new files remain in this dedicated review folder. No model/source execution, fixture, scientific payload/truth/checkpoint read, experiment, source/ledger change, guard/accounting operation, process/GPU probe or Git operation was performed.
