# Bounded full-tree learning for TreeLoom

28 October 2026. Architecture assessment and CPU prototype specification. This is a NEW finite-context model, not an exact executor for historical TreeLoom. No accuracy result, GPU work or native-baseline change is implied.

## Recommendation

Establish a finite-window TreeLoom control with legitimate observed cohort metadata and minibatch optimization. Compare it with a model that inserts multi-scale clade and persistent-leaf communication between two complete PyG tree sweeps. A genomic group is a statistical example, not a tile of one chromosome-sized optimizer update. Root owns experiment registration and later execution.

Use the bounded consecutive-window form for the first data/CPU contract. The stronger primary research variant samples a fixed number of genomic coordinates in a radius around the focal site, always including the focal tree, and processes each selected full tree. This gives a fixed memory budget and a useful serialization invariance. It omits unsampled native trees and must be called sampled full-tree context, not full-window coverage. The ordinary consecutive-window control distinguishes this approximation from the mechanism.

## Actual architecture and bottlenecks

The frozen European implementation is `research/reviews/frozen_benchmark_training_ready_20261017_v1/european_frozen_treeloom_v1.py:98`:

1. Observed SNP/donor-frequency vectors are mean/max pooled within tree intervals. Leaf inputs include reference role, four founder labels for known references and sample age. Queries have unknown reference labels. The audited fit omitted legitimate observed query cohort identity.
2. A complete upward sweep mixes descendants. A downward cavity sweep subtracts the receiving child's own upward contribution. These are actual PyG operations (`src/arggnn/anonymous_clade_gnn_v1.py:110`, `src/arggnn/mutation_routed_pyg_gnn_v1.py:24`).
3. Gained/lost exact-clade flags and an eligible own-state episode adapter are injected. The selected `own_state` uses `member + 0*foreign`, so it does not carry a foreign tree's learned clade state (`src/arggnn/ordered_clade_episode_gnn_v1.py:346`).
4. Rank-eight forcing at the 350-generation cut is lifted to persistent leaves, exponentially filtered forward/backward, pooled at current cuts, centered against current forcing, and injected at representatives. Ancient samples older than the cut are singletons. This is ALREADY shared-leaf cross-tree communication (`research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py:347`).
5. A second complete shared-weight sweep precedes a whole-parent bidirectional GRU and classifier.

The field `branch_support_true` denotes observed genotype concordance aligned with an inferred tree, not ancestry truth or calibrated topology confidence. Full-tree encoding is not a guarantee of injective information preservation.

The European fit made 90 optimizer steps over three 200 Mb parents, versus CNN's 8,335 minibatch steps over five passes of the same 50,000 TRAIN rows. The comparison does not establish convergence or that matching update counts is scientifically sufficient. Audit: `research/comparability_audit_20261028/REPORT.md`, `model/REPORT.md`.

The European qualification bank has 95,625 holds and 181,974,375 node occurrences. Earlier American full-bank profiling measured 30.0 GB prepared tensors and 63.5 GB preparation/pack peak RSS; one [N,32] state array is 26.3 GB. Device tiling still leaves whole-parent banks and learned VJPs on the host. Sources: `research/scalability_20261024/profiling/REPORT.md`, `src/arggnn/benchmarking/paper_argmix/paper_argmix_treeloom_executor.py:215`.

## A. Finite-window minibatch repair

Partition ordered full trees into disjoint label cores and add left/right observation halos. Keep all nodes, edges, leaves and SNPs in selected full trees. A provisional starting point is 32 core trees and 16 halo trees on each side, minibatches of four windows. This is a resource starting point, not a validated hyperparameter. Tree count is not physical distance; report span distributions and test a physically bounded rule.

Retain the existing SNP encoder, cavity sweeps, stable-leaf memory and BiGRU, adding common observed cohort inputs. Initialize recurrence at window endpoints, score only core focal records, and use the same context rule in training and prediction. No hidden-state carry across randomized minibatches or optimizer updates. This changes the statistical predictor and must never be called exact whole-parent equivalence.

Own-state eligibility is local: an exact clade belongs to a contiguous episode of length >=2 iff it occurs immediately before or after the current tree. One additional structural neighbor suffices to establish that binary condition. This does not apply to foreign-episode weighted means. Gained/lost flags also need adjacent observations. Preserve those local conditions or label their removal as an ablation.

Let N fixed registry rows be partitioned among cores with n_w rows. Draw cores with p_w=n_w/N and minimize

    L_batch = (1/B) sum_w (1/n_w) sum_(j in core_w) CE(logit_j,y_j).

This is an unbiased estimator of unchanged uniform sampled-row risk. Halo labels have zero loss, and registry duplicates retain multiplicity. Uniform cores plus unweighted within-core means instead overweights sparse cores. Label/cohort stratification requires inverse-probability correction if retaining the same risk. Query truth never enters observations or cut construction.

A supplies the essential control for B: improvements from optimization and observed metadata must not be credited to a new cross-tree mechanism.

## B. Multi-scale shared-leaf communication

Each selected full forest has persistent leaves i=1,...,H. Use M predetermined observed-time cuts. A leaf maps to its highest observed ancestor at or below the cut; samples older than the cut remain singleton representatives. Roots below the cut saturate. Disconnected roots are preserved without inventing ancestors. Internal integer IDs are routing indexes, never ancestral identities across trees.

At scale s, A_ts is the H by C_ts one-hot leaf-to-cut incidence matrix and D_ts=diag(A_ts^T 1). The cut token z_ts is a projection of the complete first-sweep state at the representative. Transport between trees uses

    P_(t<-u,s) = D_ts^-1 A_ts^T A_us.

This preserves constants (P 1=1) and leaf-weighted mass (n_ts^T P=n_us^T). Averaging equally over intersecting clades does neither in general. Symmetric normalized overlap D_ts^-1/2 A_ts^T A_us D_us^-1/2 is appropriate for sqrt(size)-scaled tokens, not ordinary clade means.

Given normalized temporal weights W_tu, compute

    m_ts = D_ts^-1 A_ts^T sum_u W_tu A_us z_us.

Implement gather-to-leaves, temporal aggregation independently for each persistent leaf, then scatter-mean to target cuts. Never form dense internal-clade overlap or H by H leaf matrices. Combine scales with a normalized mixture, and retain a separate same-leaf term without cut pooling.

Practical two-pass form:

    h1 = complete_tree_sweep(base)
    delta_cut = mean_s inject_rep_s[O_s(m_ts - z_ts)]
    delta_leaf = inject_leaf[O_leaf(sum_u W_tu h1_ui - h1_ti)]
    h2 = complete_tree_sweep(base + gate(delta_cut,delta_leaf))
    logits(x,i) = head(h2[tree(x),i], direct_observed_SNP(x,i), observed_metadata_i)

Initialize residual gates small. The same-parameter no-cross control sets W to identity; centered residuals vanish. Leaf-only and single-cut controls determine whether clade pooling or multiple scales earn their complexity. A separately trained single-tree condition is the direct no-neighbor comparator.

The direct focal SNP/cohort head permits distinct predictions inside one tree interval. It retains called/missing state and donor evidence and never reads query ancestry. A tree topology change is not necessarily an ancestry boundary, and an ancestry boundary need not coincide with a tree change. Sparse focal accuracy does not establish dense boundary fidelity.

### Two genomic weighting rules

For consecutive complete intervals I_t, a reference kernel is

    K_tu = integral_(I_t) integral_(I_u) k(q,q') dmu(q') dmu(q)
    W_tu = K_tu / sum_v K_tv.

Use physical or genetic measure explicitly. Physical length remains positive in zero-recombination regions; genetic measure needs a declared flat-map policy. Initial CPU reference uses positive bp spans and exponential distance decay. Equal weights per inferred tree overrepresent regions with many short trees. Label-loss weights remain separate from context-integration weights.

For the stronger sampled-coordinate variant, fix a focal coordinate x and radius R and draw K context coordinates u_j from a known q(u|x), preferably stratified and with deterministic RNG independent of tree serialization. Include x explicitly. Retrieve the entire tree at each u_j. The contribution of a sampled source is proportional to k(x,u_j)/q(u_j|x), NOT interval length times this weight. Repeated draws selecting one tree may share the tree computation but must retain their quadrature multiplicity. The focal deterministic atom has a separate learned/residual route, or a clearly declared atom in the sampling measure.

Inverse-density weighting is unbiased for a linear integral estimate. Dividing by an estimated normalizer produces ratio bias, and the nonlinear network output is not an unbiased estimate of the full-context network. Sampling can miss short informative tree intervals; always include the focal tree and measure omission/short-tract effects. Deduplicating topology alone is unsafe when observed features differ. Use canonical tree plus observed-feature identity.

### Scoped invariances, not a new kernel theorem

The proposed separable kernel

    <1_S/sqrt(|S|),1_U/sqrt(|U|)> * integral_I integral_J k(q,q') dmu(q')dmu(q)

is PSD for a PSD k by the usual tensor-product feature construction. This is established kernel mathematics. Row normalization is not generally symmetric in Euclidean coordinates, and learned asymmetric gates need not preserve PSD.

Interval-integrated transport is source-splitting additive for an unchanged source field and consistent under coarse target integration. It does NOT make an interval-token nonlinear predictor invariant to arbitrary splitting: nonlinear updates of two different subinterval averages need not reproduce the old update. Existing SNP mean/max/logcount, span inputs and per-tree GRU also change under splitting.

The sampled-coordinate variant is pathwise invariant to artificial serialization splits when its RNG coordinates and the observed tree field at those coordinates are unchanged. Feature preparation must also be invariant, for example by canonical coalescing of adjacent identical topology/date/observation fields before interval pooling, or by using fixed-coordinate SNP neighborhoods. Altering interval-specific branch features invalidates that assumption. No claim covers biological changes in the observed field.

Required tested symmetries: internal-node renumbering; a common leaf permutation accompanied by metadata/genotype/output permutation; independent-window permutation; no dependence on global sample-ID values. Genomic order is meaningful. Adding or duplicating references changes available evidence and is not an expected invariance.

### Uncertainty and oversmoothing

Descendant overlap relates sampled sets, not proof of a persistent ancestral individual or a recombination event. Multiple cuts reduce dependence on one horizon but do not integrate a topology posterior. Observed concordance, callability and duration may gate communication; do not label these calibrated probabilities. Tree perturbation/multiple-inference tests require their own observation protocol.

Averaging is contractive and can erase short tracts or rare-class contrast. Local residuals, same-leaf routes, two passes, bounded context and small gates limit this pressure but do not prove its absence. Initially avoid unconditional smoothness penalties. Measure class recall, calibration, boundary precision/recall, short-tract recall, false short tracts and errors near window seams.

## Complexity and implementation

For B windows, T trees/window, V total native nodes/window, H leaves, width d and M cuts, complete sweeps require O(B V d^2) dense work plus O(B V d) edge work with depth dependencies. Site cost depends on observed SNP count; reuse categorical computation rather than expanding identical allele rows. Per-window state is O(B V d), routing O(B M T H), histories O(B M T H d), independent of parent length.

The transparent small-T reference may form W [B,T,T], with O(B M T^2 H d) temporal work. This avoids clade-pair and leaf-pair matrices but is not optimal. Exponential interval kernels have factorizable off-diagonal terms and admit left/right recurrences in O(B M T H d). Qualify such an optimization against the reference before claiming speed. Sampled K bounds the dense reference cost directly.

The loader must extract bounded observations without first expanding a whole parent. Offset local node indexes during batching and mask all temporal routes by window. Stable leaf columns are local to each example, not identities across parents. Cache immutable routes only, never learned states across optimizer steps. Preserve forests/polytomies; sampled-interior nodes require an explicit future policy because the frozen code assumes leaves.

## Decisive tests and limits

1. CPU: finite forward/backward, direct focal/cohort connectivity, actual PyG computation, node/leaf permutation, cut constant/mass preservation, no cross-window leakage, singleton/forest/polytomy, reject zero spans, core-only focal selection.
2. A versus B with the same TRAIN/VAL observations, metadata, focal rows, local trunk, window rule and selection protocol. Report updates and target presentations, gradients/clipping, timing and class coverage. Choose training on independent VAL rather than a fixed final pass.
3. No-cross, same-leaf only, single-cut versus multi-cut, and correspondence-shuffled cross-routes with unchanged within-tree observations. Initially prioritize B versus leaf-only rather than a combinatorial search.
4. Context-size/sampling-density tests on VAL, including short intervals, boundary recall and core seams. Dense boundary metrics need sufficiently dense support. Existing inspected TEST results cannot select this design; reserve untouched confirmation after development.
5. Cold preparation, warm step, host/device peaks and time to a fixed VAL criterion. GPU activity alone is not useful throughput or model quality.

Known negatives remain evidence: level_local_v1/v2 and tiny-partition Transport were slower; the Oct27 full-parent Transport update took 1122.875 s without validated predictive gain. That candidate replaced nonlinear cavity sweeps with path-product filters and selective affine leaf scans. B retains cavity sweeps and changes finite context plus cross-tree relation. H043-H051 historical foreign-clade/history failures make benefit uncertain and prohibit claiming that more transport is intrinsically better.

Novelty is limited to a potentially useful integration for ordered inferred genealogies, observed metadata, stable sampled identity, finite-context training and boundary resolution. Message passing, incidence pooling, kernels, scans and gates are established. Predictive and computational evidence, not the algebra alone, must establish a contribution.
