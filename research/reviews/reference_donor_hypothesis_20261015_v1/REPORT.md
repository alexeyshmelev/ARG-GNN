# One candidate: donor identity beyond source-frequency evidence

**Retain the existing, unrun donor-identity versus within-source-remix condition. Reject a new cross-tree donor GNN or architectural-priority claim.** The earlier post_H046_donor_haplotype_condition_20261015_v1 packet already specifies this test. Reuse it if root elects to register a future experiment; do not create a duplicate condition. This audit adds its comparison with H049, explicit source boundaries and a stricter prospective criterion. No implementation, registration or execution is supplied.

## Biological case and its limit

H045 own-state gives Present182 accuracy **96.8175 ± 0.6142%**, NLL **0.092976 ± 0.020230**, and 1 kb boundary F1 **18.0557 ± 1.7761%**. H045 foreign-episode gains only **0.009568 pp** accuracy over it; H046 foreign-flank gains **0.011573 pp** over its own-state control. These are eight reused development simulations, one optimizer seed101, 30 added passes/960 body updates. They motivate testing a different route, not another anonymous-clade adapter.

The explicit site route builds a query/sample's allele-frequency/callability tuple, then mean/max pools within holds (mutation_routed_whole_tree_gnn_v1.py:281–335). H048/H049 marker emissions also reduce source-frequency evidence (marker_grid_reset_chain_gnn_v1.py:41–85). They do not explicitly maintain a query-by-reference copying posterior.

H049's stable-leaf control already carries cut-representative context through persistent leaf slots (order2_cut_memory_v1.py:386–392), then reaggregates it (:413–429). Its features can indirectly encode haplotype information through the whole-tree body. Therefore **“the GNN has no LD/donor information” is unsupported**. The unresolved question is narrower: does retaining which observed reference carried successive alleles add useful information after the existing graph predictor? No inspected error aggregate attributes Present182 errors specifically to lost donor identity. H048/H049 outcomes remain pending in the inspected decision snapshot. The H038 hold-edge recall ceiling does not establish a ceiling for newer marker-resolution decoders.

## Exact leakage-free condition

Freeze selected H045 own-state pass30 as the common four-class prior pi_qc(x). Retain all original observations: 952 haplotypes, 266 pure-source donors, 686 unknown queries, every phased SNP, inferred tree and scoring span. Donor class counts are Ana50/CHG28/WHG102/EHG86. Sample IDs are within-parent joins only, never learned embeddings; all donor state resets between independent parents. No query cohort/ancestry label, true tree, true breakpoint, imputed ancestry or predicted-error flag enters features.

Use only observed query/donor allele equality, observed allele count/callability, fixed donor source class and pi. Distances use the existing uniform map, d=1e-8*delta_bp Morgans, not the published ARGMix panel's different empirical-map recipe. Ancient donors are statistical templates; their labels and dates do not certify them as a query's ancestors.

For donor r of class c(r), set w_r=pi_c(r)/R_c(r), rho=exp(-kappa*d), and emission e_r=(1-epsilon)*1[a_query=a_r]+epsilon/k. Missing query: e=1; missing donor: e=1/k. The identity arm propagates f^-_r=rho*f_r+(1-rho)*w_r. The remix arm replaces this, immediately before each original SNP emission, by [rho*sum_(j in c(r))f_j+(1-rho)*pi_c(r)]/R_c(r). Both then apply the same emission/normalization and full forward/backward smoothing. Only the control collapses exactly to class-frequency emissions.

Read out (1-lambda)*pi+lambda*sum_(r in c)posterior_r. Fit exactly kappa, epsilon and lambda in each arm, with shared proposed starts100/Morgan, .01 and .1. This is a discriminative residual: pi and emissions reuse observations, so their product is not independent biological evidence. No donor-specific tree gate, top-k, SNP thinning or additional GNN is justified in this test.

## Equal budget and one prospective primary criterion

Two heads, three active scalars each, frozen identical backbone, same original32 TRAIN/eight VAL parents, seed101, 30 passes/960 complete-parent head updates per arm, reset AdamW and the original all686 exact-span risk. All query chunks retain full genomic carry and accumulate into one parent gradient/update. Save passes5/10/15/20/25/30 and preserve original accuracy selection/earlier ties; seal all96 arm/checkpoint/VAL products before scoring truth. The frozen backbone is a descriptive baseline, not a third equally trained head.

**Primary gate:** identity must improve equal-parent Present182 boundary F1 by at least1.00 pp over EACH of remix and the frozen baseline, while losing no more than0.05 pp mean span accuracy versus EITHER. Otherwise reject the donor-continuity utility claim at this budget. This tightens the older unregistered design's accuracy guard to include remix; it must be frozen before head outputs exist. No alternative primary endpoint or boundary-based checkpoint reselection is proposed.

Report paired per-parent differences and sample SD(ddof1), raw NLL, all seven cohort/aggregate rows and undefined counts. These eight repeatedly exposed VAL parents and one seed cannot establish confirmation or significance. Keep outer/TEST sealed.

## Originality and feasibility

| Closest established work | Consequence |
|---|---|
| Li–Stephens2003; FLARE2023, including its ancestry-by-donor forward states and keep/reset/mismatch code | Persistent reference-haplotype copying and ancestry smoothing are established. This restricted donor-state head is neither a new HMM nor an exact FLARE reproduction. |
| AncestralPaths2024; ARGMix2026; ARGformer2026 | Reference-informed genealogy ancestry and learned genomic/genealogical context already exist. A graph prior plus a known chain does not establish a new ARG-LAI architecture. |
| Rossi et al., TGN2020; the actual H049 stable-leaf arm | Persistent entity memory and dynamic graph operators are established. Genomic order is not evolutionary time; stable donor IDs alone are not a new temporal GNN. |

Sources are existing cached primary material/author code and verified primary-record receipts, not a new exhaustive priority search. FLARE updater lines114–131/163–180 explicitly retain donor-indexed forward values before ancestry sums. Possible contribution is a controlled biological utility finding conditional on the fixed genealogical predictor.

The engineering gate is substantive: exact arithmetic is O(Q*R*E), not O(Q*C*E). One FP32 forward table costs **7,299,040,000 bytes per10,000 events** for Q686/R266, before backward/autograd. No efficient exact donor scan, measured runtime or safe actual memory cap is established. Do not silently class-collapse the identity arm, reset tiles or prune donors to make it fit. Root should require a separately reviewed exact-memory implementation before registering/admitting an efficacy run.

Only this review directory was written. No fits, predictions, payload/truth/checkpoint reads, jobs, installs, guards, accounting or Git actions occurred.
