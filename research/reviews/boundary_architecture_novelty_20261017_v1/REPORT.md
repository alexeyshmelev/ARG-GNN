# Ordered marker evidence through complete trees: constructive candidate and novelty boundary

17 October 2026. **One concrete candidate for efficient ordered marker evidence; priority and utility remain unestablished.** Preserve marker position by computing query-specific reference evidence at each ordered SNP, pass its ordered state into the second complete-tree traversal, and use marker states at prediction. A sparse subtree-event calculation can evaluate the existing H058 path kernel exactly. Its potential contribution is the efficient realization together with a biologically useful information path, if a focused priority check and matched boundary results support both. Every component need not be new; adding a SNP GRU alone is insufficient. An age-conditioned tract-duration decoder is a separate precise no-go as a new architecture (DURATION_NO_GO.md).

## What the actual interface discards

The original site route uses a shared SNP function then mean/max/count pooling within a native tree hold (mutation_routed_whole_tree_gnn_v1.py:281–341). H058 performs same-marker query/reference comparison first, but again sum/max/count pools before its residual (same_marker_packed_csr_v2.py:411–459). Its path gate is fixed per hold and uses pooled first-sweep states (462–474).

**Conditional interface theorem.** Fix the point inferred trees, topology/branch summaries, sampled identities, hold geometry and all other non-site bank fields. Jointly permute the aligned observed marker bundles inside a hold, moving genotype, donor-frequency/callability and other site-indexed summaries together while keeping the marker positions fixed. These pooled routes return identical values in exact arithmetic. H058's per-marker comparison commutes with that permutation. Subsequent whole-tree states, sample memory and the genomic BiGRU therefore receive identical inputs. This is not a claim that re-inferring genealogies from permuted raw chromosomes yields the same tree bank, or that all ancestry information is absent. Floating-point reduction roundoff is not meaningful positional information.

A tiny representation example at ten fixed equally spaced sites is

    A A B B A A A A A A
    A A A A A A B B A A.

The marker multiset, endpoint class and tract-duration multiset agree (A lengths2 and6, B length2), but the short interior B tract moved. Pooling cannot identify its position. A downstream stationary duration prior also assigns the same product of duration terms. Symbols illustrate an interface distinction, not a validated demographic fixture. The current hold-constant output is an additional restriction. More downstream recurrence or uncertainty feedback cannot reconstruct this discarded arrangement from identical summaries.

This does not diagnose existing boundary errors. H057's TRAIN-only native-hold versus100bp oracle gap is only0.1044431±0.0098382 accuracy points across32 parents. This limits extra representable span, not boundaryF1 or an achievable gain. Small tracts can create many missed/extra boundaries with little wrong span. H058 remains an unfinished alignment experiment and is unchanged.

## The concrete candidate: exact event-evaluated marker path evidence

For one fixed inferred tree, let 0<gamma_e<=1 be a learned, site-independent edge gate from its complete first sweep. Define the symmetric path kernel

    K(q,r)=product_(e on path(q,r)) gamma_e.

Actual H058 eligibility is known pure references only, never query labels. Group g is `reference_class + 4*1[sample_age>=350]`: younger Ana/CHG/WHG/EHG, then older in the same order. Its null routes complete reference sequences by the fixed within-group permutation, while querying the *original*, unpermuted query genotype. Any sparse encoding must use the same assigned reference sequence at its destination node. All nodes remain.

**No degree divisor belongs in this kernel.** H058 `_block_evidence` adds one `log_gamma[child]` on both upward and downward edges. `degree` there controls sibling log-prefix scans only. The inherited first neural sweep has degree-normalized upward messages, and this influences its gate inputs, but is not an additional directional factor in H058's path field. Gate logs are clamped to[-16,-0.0001].

Let w_g(v)=1 for a reference node in group g and0 otherwise. Precompute

    U_vg = w_g(v) + sum_(child u) gamma_u U_ug,
    O_root,g=0,
    O_ug = gamma_u (O_vg + U_vg - gamma_u U_ug).

U is path-weighted reference mass below v; O is outside mass. Full-tree sweeps cost O(NG), G=8. The actual H058 field gathers **outside at the query**, not U+O. Thus it excludes all reference descendants of an internally sampled query q, even though reference/query identities themselves are disjoint. Define

    C_g(q,v) = sum_(r in D_v but not D_q) K(q,r) w_g(r).

Laminar descendant sets and unique paths give the exact cut identity

    C_g(q,v) = 0,                           v in D_q,
             = O_qg - K(q,v) O_vg,          q strictly in D_v,
             = K(q,v) U_vg,                 D_q and D_v disjoint.

The root baseline is O_qg. For terminal queries with no sampled descendants, this agrees with the simpler all-reference formula; the distinction must not be dropped for ancient internal samples. These are rank-one cut factorizations, not a globally low-rank kernel matrix. R kernels apply channelwise at O(R) cost.

Represent the *observed* allele vector at marker s as a root baseline plus signed descendant-set events:

    e_(allele_s(r)) = e_(root_symbol_s)
                     + sum_(v on root-to-r) Delta_sv.

Choose algebraic symbols for internal nodes, constrain every sampled node to its observed symbol, and let Delta_sv be child symbol minus parent symbol. The identity telescopes. Deterministic observed-data parsimony is one possible encoding; terminal corrections always give an exact fallback. Include multiallelic symbols and a separate missing-call symbol. Internally sampled nodes need their own observed constraints. These events are a lossless representation of calls, **not inferred biological mutations, recombinations or ancestry boundaries**. Never use true simulator mutation locations.

The exact source/age/allele field is then

    B_sqga = e_(root_symbol_s)[a] O_qg
             + sum_(v:Delta_sv!=0) Delta_sv[a] C_g(q,v).

For exact H058 feature semantics, let n_sga be the *unweighted all-reference-group* called allele counts after the same route permutation, c_sg=sum_a n_sga and D_sqg=sum_(a=0..3) B_sqga. If query x_sq>=0 and c_sg>0, choose its original allele x_sq and use

    contrast = B_sqg,x / D_sqg - n_sg,x/c_sg,
    logmass  = min(0, log(D_sqg)-log(c_sg)),
    available = 1.

Missing references contribute to no called allele slot or D; missing queries mask contrast/mass/availability. The availability-channel pooling mask still follows query-called status, exactly as H058. For positive outside support these are its24 features. An internal query can have c_sg>0 but D_sqg=0 if all called members are below q: current code sets safe_logD=0, matching=0, retains valid=1 and computes mass from that safe value. This is a material edge semantic, not a calibrated zero-mass denominator. Exact equivalence must retain it; changing validity to require external support is a separately disclosed scientific change.

No donor or winning branch is sampled. Nested/recurrent algebraic changes telescope; they are not independent evidence factors. For high event counts, an exact dense-sweep fallback can retain all calls. The sparse and dense methods are equal in real arithmetic, conditional on exact event encoding, identical routing and gates.

**FP32 risk is unresolved.** H058 uses positive log-domain sums and subtraction-free sibling exclusion. The sparse formula includes subtraction and signed allele events; cancellation can destroy small masses or gradients. A positive ancestor-segment composition can evaluate the strict-ancestor C term without subtraction, using O(NG logN) binary-lift preparation and O(G logN) per path query. It does not by itself eliminate cancellation in nested signed allele events. An implementation needs a positive region representation or a dense fallback and genuine FP32 output/gradient comparison. Clamping a negative reconstructed mass does not establish exactness. No numerical equivalence or fit is asserted here.

**Information flow.** First complete-tree sweep → fixed hold gates and U/O fields → each SNP's exact query evidence z_sq in physical order → bidirectional marker states m_sq with distance inputs. Inject a zero-initialized projection of the hold's forward/reverse marker endpoint states at the sampled query nodes before the second complete-tree sweep. Keep all reference and internal nodes in that sweep. Predict at markers/bins from the local marker state and the second-sweep sampled-node state. This retains marker order both before graph refinement and at prediction, while sampled identity links changing trees. An identical-evidence one-way decoder is the essential control for the second-sweep feedback. GRU, residual injection and bidirectional sequencing are established components; they are not the contribution.

## Honest complexity and the proposed depth-only bound

With O(1) LCA/range-membership queries and weighted root depths, K(q,v) is evaluated from the weighted path length. An O(N) static-tree LCA construction is available in principle; ordinary binary lifting instead costs O(N logN) preprocessing and O(logN) per query. Avoid unstable division of tiny path products. Let M=sum_s m_s count nonzero algebraic branch events, Q query samples and S markers in this hold. Conditional real-arithmetic field evaluation costs

    O(N G R + Q M G R),

or an extra logarithmic factor for positive ancestor-segment queries, plus O(S Q G) feature selection/normalization, marker recurrence O(S Q d^2), first/second neural tree passes O(N d^2), and all encoding/preprocessing costs. Working field storage is O(N G R); recurrent backpropagation/checkpoint storage is additional. A dense fallback makes per-site field work O(min(N,Q m_s) G R), with exact output equivalence, but selecting/constructing the encoding still costs work.

Crucially, raw calls require Omega(S H) access, and a naive exact parsimony encoding requires O(S N). Sparse m_s is an empirical property, not guaranteed: discordant calls can require O(H) events. Existing inferred mutation tables cannot be assumed lossless without checking their reconstruction against *all* observed calls. A supplied exact sparse representation can shift preprocessing to a preparation stage; its cost remains in total pipeline accounting.

Therefore **O(N+S·depth·rank) is not a valid total bound for dense marker predictions across Q samples**. At least Omega(SQ) output/state materialization remains. A lazy linear state can sometimes update one sparse leaf event along ancestors in O(depth·rank), but querying it at all samples adds cost. Arbitrary nonlinear outside-state updates can change every descendant after one event, giving Omega(N) materialization. The current full-tree GNN cannot be incrementally refreshed with the linear identity while retaining arbitrary nonlinear behavior.

Rebuild the U/O fields once for each hold under this candidate. Adjacent-tree edge diffs do not imply local learned-gate changes: globally conditioned first-sweep states can change throughout a tree, and training updates all weights. No generic cross-tree cache or sublinear nonlinear-GNN claim follows.

## Closest primary prior art and the actual novelty status

- H058 already supplies the positive unique-path kernel, per-marker reference allele comparison and complete-tree sweeps. The new kernel is therefore **not new**. The change is exact sparse evaluation and retaining its marker sequence before graph refinement/readout.
- [Ralph, Thornton & Kelleher2020, Efficiently Summarizing Relationships in Large Samples](https://doi.org/10.1534/genetics.120.303253), “Framework and Statistics” and Algorithms S/B: additive sample-weight accumulation, signed mutation/subtree effects including back/polyallelic mutation, and ancestor-path updates across tree-sequence edge changes are explicit. The paper's generic site statistic is not literally this learned query-distance kernel. Its incremental principle is direct prior.
- [Ho & Ané2014](https://doi.org/10.1093/sysbio/syu005), primary abstract retrieved through EuropePMC: linear-time phylogenetic covariance/inverse calculations for a broad three-point-structure class, including OU models, are established. Full text was not inspected here; no claim of an exact algorithmic identity follows from its abstract. It is a required near-neighbor for a later focused fast-tree-kernel priority check.
- [Naman & Simmhan2025, Ripple](https://arxiv.org/html/2505.12112v1), §§4.1–4.3: exact incremental GNN inference via cached linear aggregates and subtract-old/add-new messages is explicit. It retains nonlinear layer updates over affected neighborhoods, under fixed trained weights; it does not make arbitrary full-tree affected regions small. Training/autograd and changing gates are outside the proposed cache guarantee.
- Public tskit [haplotype_matching.c](https://raw.githubusercontent.com/tskit-dev/tskit/main/c/tskit/haplotype_matching.c), cached15October: mutation-subtree probability changes, detach/graft updates, compression and sequence HMM inference are explicit. Configurable rounding means that source alone does not prove exact real-arithmetic compression. The candidate above uses no donor HMM, but genealogy-compressed sequence computation is occupied territory.
- [Kong, Dyer & Smith2015, Segmental Recurrent Neural Networks](https://arxiv.org/html/1511.06018v1), §§2–4: ordered neural segment evidence, explicit durations, joint segmentation/labeling and exact dynamic programming already exist. An SNP GRU followed by segment decoding is not an original operator.

The cut identity is an elementary consequence of the established path kernel. It provides a useful precise implementation target but no demonstrated priority theorem, dynamic-tree data structure advance, learned-operator distinction or end-to-end speedup. A bounded primary search cannot rule out all more specific algorithmic originality; the current evidence does not meet the requested serious new-architecture bar. **A new-operator claim is not established.** A distinct efficient architecture realization plus retained ordered evidence could still be an integrated contribution; the source/FP32, end-to-end efficiency and matched-utility requirements above must earn that claim.

## One falsifiable utility experiment to leave ROOT

If ROOT chooses later, freeze one information-matched comparison: (A) ordered event-evaluated marker evidence with a direct marker decoder; (B) identical marker evidence and marker recurrence with order-aware residual feedback before the second full-tree pass. Both retain the same complete trees, sampled identities, observation access, marker readout, inherited initialization and full training schedule. Allocate the same residual parameters to A's own sampled readout so B does not win merely from added capacity. Compare the existing symmetric mean/max route separately as a representation reference; its information loss prevents it serving as the sole architectural control.

For attribution, add a separately frozen identity-check diagnostic: dense per-marker sweeps and sparse-event evaluation must agree in values/gradients on synthetic small trees including multifurcations, internal samples, missing/multiallelic and nested events. Time both at the same parameters and exact outputs, including encoding, LCA, transfers, backward and all tree passes. This is an efficiency comparison, not another fitted arm. A fixed within-hold permutation of the new evidence route can diagnose reliance on placement; its damage does not prove novelty or a causal genealogy effect.

The scientific question is whether ordered evidence can improve boundary recovery and whether feedback adds beyond an equally informed one-way model. Use a frozen practical boundary gate, e.g. at least1.00F1 percentage point at the existing1kb tolerance over each matched control, at most0.05accuracy-point loss, no mean NLL regression, and no decrease in uncensored true-short-tract recall. Report typed switches, counts, false short tracts and false-switched bp/Mb, parent-paired effects and SD. Independent parents and multiple optimizer seeds are needed for confirmation; eight reused development parents remain selection evidence. Reject feedback utility if A matches B, even if both beat pooling. Reject an efficiency claim if preprocessing or dense event counts erase its end-to-end benefit.

This is an unregistered prospective experiment, not authority to execute. No part changes H058 or its frozen controls. The best current conclusion is **a specific candidate for source development, with a proved interface distinction and exact real-arithmetic target, but unresolved numerical, efficiency, utility and priority questions**.

Scope: source text, concise existing outcomes and public/cached primary literature only. No scientific observation/truth/checkpoint/probability payloads, model imports, qualification, process/device/resource/guard checks, admissions, training, installs, Git or edits outside this assigned directory. All examples are symbolic and no new ancestry result is claimed.
