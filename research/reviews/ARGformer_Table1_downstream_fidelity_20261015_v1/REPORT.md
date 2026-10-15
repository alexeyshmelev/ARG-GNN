# ARGformer Table 1 downstream fidelity audit

Status: source and public-literature audit only, 2026-10-15. No private data, truth, trees, genotypes, embeddings, probabilities or checkpoints were opened. No model/native/scientific execution, admission/accounting action, installed/live-source edit, Git operation, package installation or author contact occurred. All new files are confined to this packet.

## Finding

The supplied published paper recovers a usable **partial** Table 1 protocol: 10,000 unique marginal paths from admixed samples; contrastively trained path embeddings; PCA fitted on unadmixed references; 10 retained PCs; query projection; ancestry calls by unspecified cluster membership; and a separate nearest-reference retrieval strategy. It does **not** identify the clustering algorithm or its decision/label rule. The public author repository does not supply the missing Table 1 evaluator in any of its 12 reachable commits, its sole branch, or its sole release. This is an evidence-backed gap in the audited public sources, not a claim about private code or unreachable history.

A separate, explicitly named PCA10 + reference-fitted KMeans3 reconstruction is implementable from the same native embeddings and C3 focal registry. This packet supplies an inactive, fully disclosed specification. It supplies no measured scores and does not make that reconstruction exact paper reproduction.

The current release's normalized top-1 retrieval is a concrete author-code route. The article does not state Table 1's k, so top-1 cannot be established as either identical to or different from the original Table 1 k. Unit-normalized embeddings are supported by paper Eq. 3. Euclidean distance and cosine similarity have identical rankings for unit vectors, apart from numerical/tie behavior. The explicit **top-5** protocol in the article's real-data analyses does not establish top-5 for American Table 1.

## Exact primary evidence

Primary article: [DOI 10.1093/bioinformatics/btag438](https://doi.org/10.1093/bioinformatics/btag438), PMCID PMC13501305.1, published online 2026-08-21. Existing full JATS source and the three-page supplementary PDF/text are pinned in source_inventory_v1.json. paper_method_excerpts_v1.json preserves relevant paragraphs without OCR.

Section 2.2 states: 1,600 samples comprising 1,000 admixed and 200 each European, African and East Asian, AmericanAdmixture_4B18 from stdpopsim, one 30 million bp chromosome, flat recombination 1e-8/bp/generation and mutation 1.25e-8/bp/generation, tsinfer+tsdate inferred from simulated genotypes without phasing/genotype error, and a 90/10 pretraining split. It does not explicitly resolve haploid versus diploid use of “samples.”

Section 3.1 states: “Using $10 000$ unique marginal paths sampled from admixed individuals,” then “performing PCA on embeddings of unadmixed reference individuals and retaining the top 10 principal components” and “project each path embedding into a 10-dimensional PC space and assign an ancestry label based on its cluster membership (PCA clustering).” The following paragraph describes the separate retrieval strategy as retrieving nearest neighbors among reference-haplotype embeddings and assigning ancestry from those neighbors' labels. The enumeration is grammatically awkward: the paper writes “First” and “Second” within the PCA description, then calls retrieval “The second strategy.” It names only the Table 1 PCA and retrieval rows and supplies no further cluster definition.

Figure 2 says PCA is fitted on AFR/EAS/EUR references and admixed samples are projected, with marginal-path embeddings as points, multiple per individual. Section 3.1 and Table 1 describe contrastively trained embeddings. Section 2.1 gives a learned pooling head and L2-normalized contrastive outputs (Eq. 3), default contrastive temperature 0.05 and inverse-frequency class weights (Eq. 5). Exact Table 1 checkpoint, trained pooling configuration and run settings are not supplied.

The supplement contains real-data dataset statistics, training architecture/efficiency, Oceania UMAP, ridge probing and attention analyses. Its extracted text contains no PCA, nearest-neighbor or simulated-LAI classifier specification. Its real-data UMAP and probe settings cannot fill the American classifier gap.

Table 1 values, in EUR/EAS/AFR precision/recall order, are:

| Method | EUR P | EUR R | EAS P | EAS R | AFR P | AFR R |
|---|---:|---:|---:|---:|---:|---:|
| FLARE | 98.41 | 98.54 | 98.85 | 99.74 | 99.70 | 96.70 |
| ARGformer PCA | 98.47 | 97.91 | 98.81 | 99.07 | 97.46 | 97.81 |
| ARGformer retrieval | 98.47 | 98.00 | 98.88 | 99.04 | 97.46 | 97.90 |

These rounded values do not identify a clustering algorithm, neighbor k, sample seed or truth-coordinate rule. They carry no reported SD/replicate count.

## Public author-source route and history

Public metadata was retrieved afresh and retained here. The current sole main branch and sole v1.0.0 tag both point to f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596. The sole release has no attached assets and describes pretraining, contrastive fitting, extraction, aggregation and retrieval. No dataset/checkpoint/evaluator artifact link appears in its body or audited README links.

All 12 commits reachable from main form one complete linear chain back to the initial 2026-01-22 commit. We retained each commit-detail response, initial complete recursive tree and existing current complete tree. There is no notebook in the initial or current tree. All initial added text-source patches were checked against their declared added-line counts and are complete. Subsequent changed filenames/patches add no Table 1 scorer, classifier, simulation script or notebook. The source-history search found no PCA/cluster/KMeans implementation. Missing patches concern logos, empty init files and the dependency lockfile, not a hidden classifier source. See history_inventory_v1.json and source_search_v1.json.

The 2026-09-07 commit f3908bc adds aggregate_embeddings.py and a README PCA visualization. That example mean-aggregates unique paths by haplotype/individual, centers all supplied aggregated rows, performs an SVD and plots two PCs. It does not fit reference-only ten-PC marginal-path geometry, assign clusters or calculate Table 1 precision/recall. It was added after the article's publication and is not evidence for the original evaluator. Substituting those averaged haplotype vectors would change Table 1's local-path evaluation unit.

Concrete current native entrypoints:

| Source at pinned commit | Implemented behavior | Fidelity status |
|---|---|---|
| embeddings.py:162-171 | trained pooling head; head_norm applies L2 normalization | appropriate contrastive representation route, exact Table 1 extraction run not released |
| retrieve.py:13-16,140-148 | L2-normalize references and queries; streaming dot-product top-k; default k=1 | executable native current-release route; original Table 1 k unspecified |
| retrieve.py:78-97 | torch.topk within chunks and merged candidates | no stable lowest-index tie rule or explicit self-exclusion |
| aggregate_embeddings.py | mean local paths per haplotype, then optional mean haplotypes per individual | visualization helper, not a local-path LAI classifier |
| README.md:168-181 | centered SVD and two-PC scatter of supplied aggregated vectors | PCA visualization only |

For disjoint reference and admixed-query rosters, sample self-retrieval is absent by construction, so adding a self-exclusion filter is unnecessary. Identical truncated inputs/embeddings across distinct full paths can still create ties. Preserve the author route and record its runtime/chunk/order behavior rather than silently introducing a new tie convention.

Public issue [#1](https://github.com/AI-sandbox/ARGformer/issues/1) asks for pretrained checkpoints. The only public author reply is [“Resolved via email”](https://github.com/AI-sandbox/ARGformer/issues/1#issuecomment-5570359986). It gives no checkpoint or evaluator URL and does not establish public download availability.

The earlier README links preprint DOI 10.64898/2026.02.11.705405. Crossref and the bioRxiv DOI API confirm v1 (2026-02-13) and v2 (2026-03-18), with the same GitHub repository. Their advertised source XML endpoints, v1 full HTML and Crossref-linked Highwire text endpoint returned HTTP 403. These attempts and the successful metadata are retained. The preprint bodies were therefore not audited and are not claimed to resolve or omit additional details. No access control was bypassed.

## Remaining parameters for an exact Table 1 reproduction

SETTINGS.csv provides the machine-readable explicit/missing/current-C3 crosswalk. The irreducible downstream questions are:

1. Exact Table 1 checkpoint and embedding extraction: learned pooling implementation/configuration, layer, normalization at PCA/retrieval, and which reference path rows were used. Paper normalization and the release support head_norm, but do not bind the historical checkpoint.
2. PCA implementation: reference sampling/weighting, centering, feature scaling/whitening and solver. Reference-only fit and 10 PCs are explicit; the rest is not. “PCA” conventionally centers data, but the precise implementation was not reported.
3. Cluster model: algorithm (KMeans, mixture, hierarchical, centroid rule, etc.), number of fitted clusters, references-only versus reference+query fitting, initialization/restarts/seed/convergence, distance, and query assignment. Three ancestry labels do not prove three fitted clusters.
4. Cluster-to-ancestry mapping: reference majority vote versus one-to-one matching, weighting, empty/mixed clusters and ties. Query truth must never be used for this mapping in a faithful leakage-free reconstruction.
5. Retrieval: original k, voting or score weighting, candidate curation/balancing, metric and ties. Normalized top-1 is defined by current code; it is not retroactively established as the paper setting.
6. The 10,000 query records: sampling seed/frame, deduplication definition, any eligibility/purity filter and whether equal path weighting was used. The release extractor deduplicates complete raw leaf-to-root tuples; its max_samples=10000 extraction cap is an ordering cap, not evidence for the article's sampling rule.
7. External scoring: coordinate and true ancestry for a unique inferred path that recurs across intervals, FLARE call interpolation/support at that coordinate, missing calls, averaging across replicates and metric conventions. Standard per-class precision/recall are inferable as metric meanings, not the exact aggregation implementation.

Upstream unresolved settings also remain: simulation/catalog versions and seeds, mutation model/discrete versus continuous genome, tsinfer/tsdate versions/options/preprocessing, original sample ploidy, exact contrastive temperature and effective class weights. Released YAML temperature 0.30 differs from the paper's stated default 0.05. The existing source audit documents the release's uniform-weight fallback in contrastive config plumbing. Altering those settings would create a separate paper-equation/corrected variant, not an unchanged release run. Nothing in this downstream audit changes the frozen native v12 or H048 sources.

## Compatibility with the current American C3 reconstruction

C3 is correctly described as a reconstruction, with the intended continental ontology and method-specific observed representation. Source-only checks confirm:

| Item | Paper | Current C3 |
|---|---|---|
| Demography / ancestry task | AmericanAdmixture_4B18; EUR/EAS/AFR | same named model; ASIA maps to EAS |
| Sample counts | 1,600 “samples”: 200/source +1,000 admixed | explicit 1,600 haploid genomes, paired for phased VCFs; ploidy is a reconstruction choice |
| Length / rates | 30 Mb; recombination 1e-8; mutation 1.25e-8 | matches full native TRAIN000 and five original 30 Mb VAL parents |
| Sources / query | unadmixed references, admixed queries | AFR columns0-199, EUR200-399, ASIA400-599; ADMIX600-1599; class order EUR/EAS/AFR |
| Observation | inferred tsinfer+tsdate ARG, no phasing/genotype error | same method family and error-free phased genotypes; exact current inference versions/options and direct all-sample unary collapse are disclosed adaptations |
| Simulation truth | unspecified label-coordinate derivation | external recorded generation12 pulse migrations; truth/internal population annotations sanitized from predictor inputs |
| Native training | within-ARG paths and reference contrastive labels | full same-ARG native MLM vocabulary; only pure-reference contrastive labels; query truth excluded |
| Evaluation | 10,000 unique admixed paths | exactly 10,000 complete raw query paths selected label-blind without replacement, seed92001 for current TRAIN000; same record identities across methods |
| Path call coordinate | unreported | first occurrence interval midpoint, explicitly reconstructed |
| Score unit | path-based Table 1; full aggregation unreported | one unweighted focal record; common 3x3 confusion, per-class P/R/F1, accuracy and macro-F1 |
| Scope / generalization | within-ARG fitted representations | native TRAIN000 is engineering TRAIN, not held-out accuracy; GNN 20 × 2 Mb training is an additional disclosed design |

The current v3 materializer preserves full author path identity/order, seed42 90/10 path split, reference/query views and chosen focal identities. Existing C3 focal bridge proof establishes the same registry on short GNN TRAIN parents. Those 2 Mb parents are not substitutes for a Table 1 30 Mb native fit. Reference order must be joined from sidecars; class order is [EUR,EAS,AFR], not source-column order [AFR,EUR,ASIA]. Per-class outputs must not be transposed.

A PCA reconstruction can reuse exactly the same final native head_norm embeddings, complete reference sidecars, query record IDs and later common external evaluator. It must produce one sealed call per same focal before private truth is opened. No new simulation, model retraining, genotype use, haplotype averaging, focal resampling or native-source edit is required for that separate downstream variant. Actual CPU/RAM/storage/time admission remains root's responsibility.

## Concrete disclosed reconstruction route

reconstruction_spec.inactive.json defines **ARGformer_PCA10_reference_KMeans3_reconstruction_v1**. It is a proposal, not registered or executed, and explicitly records that clustering/settings are chosen by us. The route is:

1. Consume the same successfully completed native contrastive checkpoint's head_norm local-path vectors and exact reference/query sidecars. Keep all current complete reference path rows (9,583,977 in TRAIN000) and the same 10,000 query paths. No ancestry-dependent filtering or balancing.
2. Fit the mean and empirical covariance to reference vectors only, in float64 using stable batch-merged centered moments. No per-feature standardization, whitening, haplotype averaging or genomic-span weights. Use the top 10 eigenvectors; project references and queries with the same mean/axes.
3. Fit reference PCs only with explicit KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=17015, max_iter=300, tol=1e-4, algorithm='lloyd'). Pin the actual numerical-library/runtime versions before any execution. These are reconstruction settings, not recovered author settings.
4. Map the three cluster IDs to EUR/EAS/AFR by maximizing the total reference count agreement over all six bijections. Use no query labels. Resolve equal optima lexicographically in fixed class order. Assign queries to the nearest fitted center by squared Euclidean distance, with lower center index for an exact distance tie.
5. Seal all same focal identities, one-hot calls, source/checkpoint/PCA/cluster metadata and reference-only mapping before the existing separate external evaluator opens truth. Report a separately named row, never replace the native top-1 row or label the score exact Table 1.

The ten-dimensional reference array is about 767 MB in float64 at the current TRAIN000 reference count, excluding native embeddings, covariance work and KMeans copies. This is not a resource admission or performance promise. A streaming covariance implementation avoids a second full reference×768 float64 centered matrix; root must budget actual arrays, workspaces and runtime. It still processes all references. No hidden sampling shortcut is proposed.

## Root handoff

Keep current native runs and all frozen source/settings unchanged. Finish the native top-1 current-release reconstruction under its existing contract. If the separate PCA route is wanted, register it as a distinct reconstruction condition, pin actual versions/sources, add an independent observation-only seal mode, and use the same fixed focal registry/common external score. This audit has not implemented or launched that scorer.

For exact original Table 1, the minimum decisive missing asset is the original evaluator plus record/checkpoint/config metadata (especially cluster construction/mapping and sampled-path truth coordinates). The audited public repository route does not supply it. Public preprint bodies remain a retrieval limitation, so the bounded conclusion is “not recovered in the inspected published text/supplement and complete reachable author repository history,” not “no code exists anywhere.”
