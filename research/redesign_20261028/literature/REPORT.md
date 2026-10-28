# Critical literature and formulation assessment, 28 October 2026

## Recommendation

Pursue a **finite-context, jointly trained model of complete inferred trees with persistent sampled-haplotype identities**, provided its first scientific comparison isolates correct clade alignment from ordinary same-leaf genomic processing. Treat this as a new LAI formulation and a testable biological hypothesis. Do not claim a new general graph operator, new optimal transport, first joint ordered-tree model, or state-of-the-art performance.

The proposed ingredients are individually established: graph-local and cross-graph feedback (GST), same-node temporal attention (DySAT), graph-dependent memory with irregular intervals (GraphSSM), restriction/lifting across graph resolutions (graph coarsening), measure-based attention (continuum attention), and sampled graph minibatches (GraphSAINT/Cluster-GCN). Their combination can still be a useful research contribution if it resolves a concrete genealogy problem and demonstrates a reproducible benefit. The strongest possible contribution is an explicit observation contract, a bounded training objective, precise invariances, and evidence that **correct ancestral-time-specific alignment** improves ancestry inference beyond matched alternatives.

The major new source finding is that the pinned **ARGMix author release already implements overlapping interval chunks, genetic span/delta features, minibatch training and a bidirectional GRU**. Its paper describes a local tree-subgraph model. These are distinct evidence sources. A statement that ARGMix has no joint genomic model, or that bounded ordered-tree LAI is new by itself, is unsafe.

This assessment includes no model fitting, GPU use, dataset mutation, scoring, queue changes, or Git operation. All new files are inside this literature directory on anogena. The failed fits and historical proposals remain unchanged.

## Evidence scope

This is a focused critical review, not an exhaustive priority search. No claim of reading all papers end-to-end is made.

| Source | Material directly read in this review | Main relevance |
|---|---|---|
| GST, Escrig Molina, Chen and Probst, arXiv:2606.05116v1 | Local full-text extraction: introduction, equations 1-12, experiment design, substantive results, computational-cost discussion and limitations. Public pinned gst.py read in full, never executed. | Interleaved local propagation and gated set feedback already exist. |
| ARGformer, Bonet et al., Bioinformatics 2026, btag438 | Local primary full-text XML normalized to text. Main architecture, objective, simulation, LAI and real-data narrative, conclusions read. Supplement training/efficiency section and author preprocessing/README read. Formula extraction from XML is incomplete, so no new formula-level claim depends on it. | Shared ancestral IDs across paths, masked pretraining, contrastive retrieval, independent path minibatches. |
| ARGMix, Shanks et al., bioRxiv 2026.04.05.714076v1 | Local complete extracted main narrative, methods/equations and experimental design. Pinned author joint_interval_model.py and joint_support.py relevant classes/configuration. | Local 29-node TMRCA transformer in paper; additional joint GRU in released code. |
| AncestralPaths / Irving-Pease et al., Nature 2024 | Main paper ancestry-method/application passages; supplementary note 3 feature extraction/training/comparison passages; released training.py. Not all unrelated biological supplement analyses. | Observed ancient cohorts, informative path nodes, inferred-tree training, short-tract errors. |
| Whitehouse, Ray and Schrider, MBE 2024, msae223 | Local primary full text: architecture, graph/tree GRUs, node ordering, simulated-task comparisons, efficiency and misspecification passages. | Complete marginal-tree GNN plus ordered-tree sequence learning predates this project. |
| DySAT, Sankar et al., arXiv:1812.09430v2, June 2019 workshop version | Downloaded complete PDF; introduction, model definition and structural/temporal attention equations/architecture. | Persistent-node representation across changing snapshots is established. This is the retrieved version. |
| GraphSSM, Li et al., NeurIPS 2024 / arXiv:2406.00943v2 | Local full text, sections 3.1-3.2 and equations 3-9 directly read. | Piecewise changing graphs, fixed nodes, graph-regularized memory and interval-aware dynamics. |
| Continuum Attention, Calvello et al., arXiv:2406.06486v4 | Local full text, definitions 4-5, theorem 6, remark 7, context on discretization and patched operators. | Integral attention and convergence across discretizations are established. |
| Loukas, JMLR 20, 2019 | Downloaded complete PDF; introduction, scheme 1, property 1, discussion of restricted spectral guarantees. | Pooling/lifting and projection are established; arbitrary genealogy cuts do not inherit spectral guarantees. |
| Peyre, Cuturi and Solomon, ICML 2016 | Downloaded complete PDF; motivation, coupling definitions, OT equation 2, GW equation 4 and optimization formulation. | A known coupling is not an optimized transport solution. |
| GraphSAINT, Zeng et al., ICLR 2020, arXiv:1907.04931v4 | Downloaded complete PDF; sampling, normalization propositions, layerwise unbiasedness limits, experimental protocol and runtime scope. | Fixed subgraphs per minibatch and inverse inclusion corrections are established. |
| Cluster-GCN, Chiang et al., KDD 2019, arXiv:1905.07953v2 | Downloaded complete PDF; SGD motivation, block approximation, embedding utilization, complexity and stochastic partitions. | Joint subgraphs improve computation reuse; dropping cross-block edges can change the learning problem. |

Public download URLs, byte counts and SHA256 values are in primary_sources/retrieval_receipts.json. GST source has its own receipt. Existing primary texts were reused unchanged. source_manifest.json records the local evidence. Previous review verdicts were considered internal history, not substituted for primary sources.

## Closest genealogy methods

### GST

The paper propagates within each graph, pools one graph token, attends across graph tokens, then injects graph and set context into nodes at each layer. Author GraphSetConv code has attention pooling, masked cross-graph multihead attention, node-to-set attention or broadcast, and a sigmoid gate. The exact code fusion differs from displayed paper gate equations, so an author-release baseline must identify its version.

GST does not exploit persistent leaves, genomic distances or ancestral age. These are legitimate task-specific differences. They do not make interleaving or gates new. Its computational-cost section reports substantially higher per-batch cost than staged baselines and recommends broadcast as default. Extra cross-tree computation must earn its cost in accuracy, convergence or full-training time.

### ARGMix

The paper uses 25 nearest references by TMRCA plus three inferred ancestors and the focal leaf: a 29-node subgraph. Population tokens and asymmetric age-relative TMRCA encodings capped at 1,000 generations enter query/key/value attention. Training uses 500,000 examples from five replicates, skipped trees to reduce redundancy, validation selection, and evaluation under matched and misspecified demographies. It explicitly identifies quadratic dense attention as a barrier to complete-tree attention.

Pinned repository c2481248eb0de5fda659d68f59c51e0e27becd20 contains src/joint_interval_model.py:319, JointIntervalGru. forward_chunks embeds each subgraph, appends transformed cM span/delta features, and applies GruRefiner. Configuration defaults include 1,024 rows, stride 512, batch size 16 and bidirectionality. joint_support.py:29 implements the GRU. This establishes available source functionality, not that this option produced the paper's numbers or was used in the current benchmark.

The meaningful redesign gap is **joint exchange among complete tree structures**, beyond temporal refinement of independently encoded focal subgraphs. The available-release joint GRU is an informative additional comparator when a matched condition is feasible.

### ARGformer

The paper uses ancestral-node ID tokens along marginal leaf-to-root paths. Repeated shared ancestral IDs across paths convey relationships through the training corpus. Masked-node pretraining precedes supervised contrastive training on references and retrieval/clustering. This does not establish simultaneous processing of K complete trees with per-leaf LAI supervision. Author preprocessing deduplicates marginal paths and randomly splits them; the vocabulary belongs to one ARG namespace. A model generalizing to independent chromosomes must define its own inductive mapping.

ARGformer's conclusion proposes more compressed shared-lineage tokens for scaling. A broad claim that lineage tokens are new is weak. A meaningful distinction is correspondence through observed descendant membership and dates without requiring enduring inferred internal-node IDs across independently inferred trees or chromosomes.

### AncestralPaths and Whitehouse et al.

AncestralPaths uses genealogical-neighbor distributions over five informative ancestors and ages. The supplement uses three 200 Mb training chromosomes, 50,000 examples across five admixed cohorts, and five new test chromosomes. It attributes some present-day short-tract difficulty to Relate's use of flanking information and delayed topological changes at tract edges. This supports boundary sensitivity as a biological target, not the assumption that more smoothing helps.

The released training.py inspected here is Dense/Flatten and retains eight observed ancient-cohort categories. The current audit recovered a historical CNN separately. Preserve that distinction. The audit's 92.702% CNN versus 54.802% TreeLoom accuracy reflects the configured pipelines; neither metadata parity nor convergence has yet been shown to explain the entire gap.

Whitehouse et al. apply GAT layers to complete marginal trees, a GRU to compress each tree, and a second GRU along ordered trees. They retain consistent sample ordering and include genomic location/span summaries. Targets are sequence-level population-genetic quantities rather than this per-leaf LAI objective. Their graph model can also be slower and less memory efficient than a CNN. Complete-tree and joint tree-sequence processing are therefore prior art.

## Transport algebra, short proof, and limits

Let the stable observed sampled-haplotype roster have N entries. At tree i and a fixed ancestral-age cut, B_i is an N by M_i binary membership matrix with exactly one nonempty group assignment per row. W=diag(mu_h) contains fixed positive observed-haplotype weights. Define

    D_i = B_i^T W B_i
    C_ji = B_j^T W B_i
    T_ji = D_j^(-1) C_ji.

T_ji z_i lifts source clade values to the same leaves and averages into destination groups. It needs no internal ancestor matching. For one cut and one tree pair there are at most N raw group-pair events before deduplication.

Write d_i and d_j for the diagonals of D_i and D_j. C has row sums d_j and column sums d_i. Therefore

    T_ji 1 = 1
    1^T D_j T_ji z_i = 1^T D_i z_i.

For each destination b, weights C_ba/d_jb are nonnegative and sum to one. Jensen gives

    d_jb ||sum_a C_ba z_ia / d_jb||^2 <= sum_a C_ba ||z_ia||^2.

Summing b gives ||T_ji z_i||_(D_j)^2 <= ||z_i||_(D_i)^2. Thus the raw transfer preserves constants and the leaf-weighted sum and is nonexpansive in the corresponding group-mass norm. Equivalently, D_j^(-1/2) C_ji D_i^(-1/2) overlaps two normalized indicator bases and has operator norm at most one.

These are established conditional-expectation/restriction-lifting facts, **not a new theorem of mathematical priority**. They supply a clean design constraint compared with unconstrained bilinear feature products. They do not establish that transport will cure the observed large gradients or class collapse. Such a causal explanation requires measurements.

C_ji / sum_h mu_h is a coupling with the two partition mass marginals, determined by known sample identity. It minimizes no transport cost and solves no GW objective. Coupling or mass-preserving overlap map is accurate; optimal transport is not.

Generally T_ki differs from T_kj T_ji: intermediate pooling destroys within-clade contrasts. Lift-pool on one partition is a projector, not an invertible coordinate change. At nested age cuts, projectors commute within a fixed dated tree and form familiar multiresolution averages. Across changing trees they need not commute. Neither observation establishes novelty.

Loukas's spectral results require an approximation criterion, not merely plausible partitions. Age-cut clades need not preserve the full tree Laplacian, branch geometry, arbitrary node features or LAI sufficient statistics. Retain leaf residuals and full-tree processing, or state discarded information precisely.

Uniform positive mu_h is the clearest initial contract. Reference-only zero query weights change the algebra and can leave unsupported groups. Stratified weights change the measure. Samples older than a cut need a declared representation, for example observed-sample singleton plus availability mask; they cannot be silently removed or described as extant lineages at that age. Multiple roots and sampled internal nodes require actual sample mappings.

## PSD, conservation and refinement are distinct

A clade overlap matrix is a Gram matrix of weighted descendant indicators. Its elementwise product with another PSD kernel, such as an interval-integrated genomic RBF/Laplace kernel, is a standard PSD construction. Nonnegative sums across scales are also standard. This may be coherent and useful, but is not a new-kernel claim.

Arbitrary learned directed pair gates can destroy symmetry and PSD. Row-normalized attention preserves constants but usually preserves a different stationary measure rather than the original leaf/genomic mass. Channel projections, residual MLPs and output classifiers do not conserve ancestry probabilities merely because a raw overlap map preserves sums. Limit each guarantee to its actual submodule. Fixed coupling marginals after gating would need an additional balancing construction and extra cost, without conferring biological conservation.

Genetic distance (Morgans/cM) and ancestral age (generations) are different coordinates. Neither is an inferred recombination-event posterior. A map plateau has zero genetic measure despite containing observations. A genetic-only integral must state its plateau behavior; adding physical measure is a modeling choice.

For fixed target x and piecewise constant source u(y),

    m(x) = integral k(x,y) u(y) dnu(y)

is invariant to splitting a source interval into parts with the same field and total measure. This is integral additivity, related to continuum attention. A ratio of two exact source integrals is likewise invariant when both use that unchanged measure.

This does not prove full-network invariance. Double-integrated interval kernels average over the target cell too. Splitting it changes averaged features before nonlinear updates, and an MLP of an average is not an average of the MLP. Other failures include recomputed SNP pooling, independent per-record dropout, tree-count positional encodings, record-count normalization, K-record truncation and final per-record GRUs. Define updates at fixed queries and prove each layer's property, or limit the claim to the transport primitive. Canonicalizing identical attributed observation cells can restore an input quotient, but also protects ordinary models and is not mechanism-specific novelty.

## Two bounded learning formulations

### A. Finite support with exact halos

Define a new finite-context model from the outset. Sample core supervised targets from the registered TRAIN distribution. Collect all complete inferred trees and observed SNP evidence required within a fixed genomic/genetic support. No hidden state or normalization may depend on the full chromosome. For L layers of radius r, an Lr halo suffices only when r bounds every cross-tree dependency, including dependencies of source embeddings. Compute loss on core targets alone.

This is exact minibatch evaluation of the newly defined local model. It intentionally discards information outside support and is not an exact acceleration of the old whole-chromosome recurrence. Unlimited Gaussian/Laplace support makes finite halos approximate unless explicitly truncated. Fixed K consecutive records and fixed genetic radius cannot both hold under arbitrary tree density without a capacity policy. Do not silently truncate dense regions or claim K-record refinement invariance.

A label present in several overlapping cores needs single ownership or an inclusion correction. Source genomic weighting is separate from loss weighting: a fixed SNP-target objective can retain equal target weights while messages use genetic measure. Switching to physical-span loss changes the estimand.

### B. Coordinate-sampled complete trees under strict K

If A exceeds the bound, sample K genomic coordinates from declared q within the focal support and gather the complete tree at each coordinate. With shared coordinates and unchanged observation fields, the sample is pathwise invariant to artificial record subdivision. Share computation for duplicates only when their features and required contexts are identical.

Weights proportional to 1/q(y) yield a standard Monte Carlo estimator for a linear integral. A mixture of near-focal and broader sampling can improve boundary coverage but must be chosen on TRAIN/VAL. Self-normalized learned attention is a ratio estimator; nonlinear multilayer propagation adds sampling effects. Do not claim unbiased dense-network outputs or gradients. Optimize the expected loss of the registered stochastic context model. GraphSAINT itself limits its unbiased aggregation derivation to a layerwise analysis.

This supplies a strict context-memory bound and an explicit approximation. It can miss rare informative short tracts. Measure missed-context probability, short-tract recall, sensitivity to K/q, and stochastic versus deterministic evaluation differences. The mathematics is standard sampling; the possible contribution is disciplined joint-genealogy modeling with a demonstrated accuracy-cost frontier.

## Falsifiable scientific hypothesis

**Correct ancestral-time-resolved correspondence across nearby inferred trees provides ancestry-relevant information unexplained by same-leaf temporal context, local clade geometry, capacity, cohort metadata or improved optimization. It improves classification and boundary recovery under inference noise without deleting short ancestry tracts.**

Plausibility comes from stable sampled genomes and correlated genealogical neighborhoods; several ages may distinguish recent donor affinity from older ancestry. Failure is plausible too: clades need not equal source populations, overlap need not mean the same ancestor, topology change need not mean an ancestry switch, and smoothing can increase site accuracy by deleting short true tracts. Cut choices may fit one demographic model rather than transferable biology.

The 90-update failed fit cannot be the only comparator. Establish a common legitimate observed-cohort policy and adequate TRAIN/VAL optimization before attributing recovery to transport. Already inspected TEST cannot guide tuning. A fresh untouched confirmation set is needed after iterative development.

## Decisive controls

| Question | Matched comparison | Rejection criterion |
|---|---|---|
| Is the encoder learnable? | Common metadata; independent-tree complete GNN with common target/head policy; learning, class coverage and gradients on TRAIN/VAL. | Class collapse/instability persists, or recovery is entirely metadata/optimization. |
| Is joint context useful? | Independent-tree versus same-leaf finite-context model, identical core/context sampling and selection policy. | No meaningful gain or bad accuracy-cost tradeoff. |
| Does cross-leaf alignment help? | Same-leaf versus clade-transfer models with matched capacity/budget. | Same-leaf matches or exceeds transport. |
| Is foreign state needed? | True foreign states versus partner-blind local state with the same geometry, parameters and workload. | Local nonlinear geometry explains the gain. |
| Is correct alignment needed? | Trained null: fixed parent-wide leaf permutation within observed cohort/age/reference-role strata applied only to transport, leaving encoder inputs and targets on correct identities. | Null matches true alignment. Report moved fraction and immovable strata. |
| Are multiple ages needed? | Single cut, multiple cuts, parameter-matched repeated-cut or non-genealogical grouping controls. | Benefit vanishes with matched capacity or only holds at one demographic cut. |
| Is genomic geometry useful? | Coordinate/distance support versus count-based or distance-null context with matched information. | No gain or poor length/tree-density transfer. Train a null if making a causal claim; inference-only shuffles diagnose distribution shift. |
| Is refinement exact? | Split/duplicate identical attributed intervals, rename internal IDs, equivariantly permute sample order; compare fixed-target logits/gradients. | Difference exceeds declared floating tolerance. This qualifies implementation, not biological efficacy. |
| Are halos exact? | Whole-region local-model evaluation versus core/halo chunks and several crop offsets. | Core predictions change with crop placement or need undeclared distant state. |
| Are boundaries preserved? | NLL, macro-F1, class recall, boundary precision/recall/F1, short-tract recall and false switches/Mb by tract length/cohort. | Site-accuracy gain mainly deletes short tracts or worsens boundary errors. |
| Does it scale? | Preprocessing, loader, forward/backward and validation cost; host/device peak memory; labels/update, updates and time to a fixed VAL target. | Tensor savings are offset by tree expansion, Python partition work or slower convergence. |
| Does it transfer? | Independent simulation parents, optimizer seeds, demographic and inference-error shifts, new confirmation set. | Gain disappears under modest shifts or exists only on inspected TEST. |

The chromosome/simulation parent is the biological replication unit, not every correlated window/SNP. Multiple optimizer seeds measure optimization variation. Register practical effect thresholds/stopping rules before confirmation labels. Report both information/parameter-matched conditions and reasonable validation-selected method-specific schedules; equal update counts alone do not prove fairness. Trained nulls are stronger than pathway shuffling/disabling only after training. Preserve original author baselines and name common-information adaptations separately.

## Novelty verdict and development order

This evidence rejects broad primitive-level novelty. It does not establish prior publication of the exact final formulation, and it does not prove absence. After successful experiments, defensible wording is:

> We formulate local ancestry prediction as bounded joint inference over complete inferred genealogies, using persistent sampled-haplotype identity to exchange information between ancestral-time partitions. The model is defined over genomic measure and trained on local supervised cores with explicit context. We establish the stated representation-consistency properties and test whether correctly aligned multiscale context improves ancestry and boundary recovery at a measured computational cost.

Every sentence needs implementation, proof or experiment. Until efficacy exists, improvement remains a hypothesis. Do not interpret conserved latent sums as biological conservation, overlap as detected recombination, or PSD products as new kernel theory. If alignment controls fail, preserve the bounded training improvement and drop transport-centered novelty.

Preferred order: common information policy and learnable independent-tree baseline; bounded same-leaf context; single-cut true/null transport; then multiscale extension. This avoids building a complex model whose recovery from a failed fit cannot be attributed.

## Primary sources

- GST: https://arxiv.org/abs/2606.05116v1 ; https://github.com/daenuprobst/gst-conference/blob/6469d56aa131463325058b897b1e435c6d2edcd2/src/graph_set_transformer/models/gst.py
- ARGformer: https://doi.org/10.1093/bioinformatics/btag438 ; https://github.com/AI-sandbox/ARGformer
- ARGMix: https://www.biorxiv.org/content/10.64898/2026.04.05.714076v1 ; https://github.com/AI-sandbox/argmix/blob/c2481248eb0de5fda659d68f59c51e0e27becd20/src/joint_interval_model.py
- AncestralPaths: https://doi.org/10.1038/s41586-023-06705-1 ; https://github.com/AliPearson/AncestralPaths
- Whitehouse et al.: https://doi.org/10.1093/molbev/msae223 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11600592/
- DySAT retrieved workshop version: https://arxiv.org/abs/1812.09430v2
- GraphSSM: https://papers.nips.cc/paper_files/paper/2024/hash/e5ba3d6d93213db6b1d1931c6517fe1a-Abstract-Conference.html
- Continuum Attention: https://arxiv.org/abs/2406.06486v4
- Loukas: https://jmlr.org/papers/v20/18-680.html
- Gromov-Wasserstein: https://proceedings.mlr.press/v48/peyre16.html
- GraphSAINT: https://arxiv.org/abs/1907.04931v4
- Cluster-GCN: https://doi.org/10.1145/3292500.3330925 ; https://arxiv.org/abs/1905.07953v2

Internal records directly consulted: research/comparability_audit_20261028/REPORT.md; research/reviews/H043_exact_clade_transport_design_v1/HANDOFF.md; its literature_verified.md; research/reviews/continuous_clade_transport_novelty_20261017_v1/REPORT.md; research/reviews/paired_reference_change_operator_assessment_20261017_v1/REPORT.md; research/reviews/fresh_ARG_operator_research_20261017_v1/REPORT.md. These already cover descendant overlaps, stable-leaf memory, clade diffusion and refinement-safe physical exposure. The bounded formulation must cite this internal lineage rather than rediscover it under a new name.
