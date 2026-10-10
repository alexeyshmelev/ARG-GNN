# Census-cut exchange v1

H021, registered condition `census_cut_exchange_proto_v1_source_only`, 2026-10-10. This is a source-only prototype plus handmade CPU qualification of the proposal in [biological_kernel_design_v2.md](biological_kernel_design_v2.md). It is an unvalidated biological constraint built from inherited mathematics. No generic kernel novelty, priority, measured speedup, useful ancestry sharing or predictive gain is established. The root owns independent review, integration, experiment-ledger updates, Git and any subsequent runtime or fitting admission.

The new implementation is [src/arggnn/census_cut_exchange_v1.py](../src/arggnn/census_cut_exchange_v1.py), with [tests/test_census_cut_exchange_v1.py](../tests/test_census_cut_exchange_v1.py). It has no training or data-loading entry point. Frozen original and compact sources are imported without modification. The [author bundle](../research/reviews/census_cut_exchange_v1_author/README.md) preserves attempts, quantitative fixtures and final hash binding.

## Observed dates, roles and identities

`ObservedDatedTreeV1` accepts one complete observed tree. Native parent indexes and actual sample identities are distinct. Parents and sample maps are fixed CPU long tensors, native times are fixed CPU float64 generations, times are finite and nonnegative, and a parent cannot be younger than its child. The tree must have one root, be connected and acyclic, and contain all and only the declared sampled leaves. Sample identities are unique and nonnegative. Every persistent sample must have the same observed age across holds.

Q and pure R form an exhaustive disjoint sample roster. Scored Q is a nonempty subset of Q. The implemented policy is explicitly `context_date_policy="require_younger"`: every scored and context query must be strictly younger than census 350. An equal or older scored query fails. An equal or older context query also fails. The implementation does not fabricate singleton census ancestry for ineligible context samples. Other date policies require separate design and qualification.

The root reports pinned configuration metadata: Q120 ages 0–200, mixed context64 ages 120–237, and pure R96 ages 176–473 with 12 at or above 350. Those metadata support this declared Q184 policy but do not substitute for runtime observed-time checks. They were not recomputed from actual sample arrays in this work. `sampling_population`, hidden census labels and ancestry truth must never become model features. The generic F47 tensor interface cannot certify provenance of an external feature join. The caller remains responsible for the observed-only F47 contract.

For each eligible q, ascend to its highest observed lineage vertex whose age is at most the census. Queries sharing that vertex form a local group. Equivalently,

\[
q\sim_g q' \quad\Longleftrightarrow\quad
 t_g(\operatorname{MRCA}(q,q'))\le350.
\]

MRCA age equal to 350 qualifies, while query sample age equal to 350 does not. On a correct dated tree the relation provides sufficient evidence of a shared census lineage. On reconstructed trees it is uncertain evidence and never a certified ancestry identity. Different groups can still share ancestry. R samples, including those older than or equal to 350, stay outside these cut groups and contribute only observed donor proxy evidence.

The exported group labels are graph-local integers. `group_members` contains sets of actual Q sample IDs. Adjacent trees align only those persistent sample IDs. No cross-tree ancestral identity or ARGformer ID is inferred from an internal vertex number or group label.

## Whole-tree integration interface

`prepare_whole_tree_encoder_inputs_v1` suppresses anonymous nonroot unary vertices, preserving the root, all sampled leaves and every branching interior. It copies stored observed F47 for retained vertices, verifies its declared encoded-age column (default 6) against native times, and provides retained parents, upward/postorder and downward/preorder sweeps, ages and branch durations. Each retained duration is the native retained parent-child endpoint difference, so serial subdivision is additive. The original observed tree and features are not mutated.

`gather_whole_tree_context_v1` gathers Q/R contexts from externally encoded retained vertices through actual sample IDs. These routines define the interface for an ordinary whole-tree upward/downward neural encoder. This module does not implement that encoder or a native-bank adapter. The executable exchange layer receives already encoded whole-tree contexts. A later integration must supply the common encoder to every contrast and verify the external F47/identity joins. Leaf-only contexts are not a qualified replacement for the specified whole-tree encoder.

## Inherited partition diffusion and support retention

For one-hot local group incidence P,

\[
S_g=P_g\operatorname{diag}(P_g^T1)^{-1}P_g^T,
\qquad K_g=\delta I+\lambda(I-S_g),
\]
\[
E_g(t)=e^{-\delta t}S_g+e^{-(\delta+\lambda)t}(I-S_g).
\]

S is the ordinary orthogonal block-averaging projection. Production actions use segmented sums/gathers with Q temporary rows, without forming a Q-by-Q matrix, solving an eigenproblem or calling a dense matrix exponential. The positive-coefficient evaluation uses `exp(-t*(delta+lambda))*h + exp(-t*delta)*(-expm1(-t*lambda))*S h` to avoid subtracting nearby exponentials.

At a genuine tree boundary,

\[
\rho_{b,q}=\frac{|C_{\mathrm{left}}(q)\cap C_{\mathrm{right}}(q)|}
 {|C_{\mathrm{left}}(q)\cup C_{\mathrm{right}}(q)|},
\quad J_b=\operatorname{diag}(\rho_{b,q}^{\eta}).
\]

The same q belongs to both groups, so rho is strictly positive and at most one. Internal labels do not enter the overlap. Boundary retention describes loss of contextual support, not ancestor correspondence or a recombination posterior. A subdivision with unchanged groups gives rho=1 and J=I. Rates are global per layer, branch and latent channel. Lambda and delta are positive inverse normalized-coordinate rates. Eta is dimensionless and nonnegative in the action API. The proposed layer uses positive `initial_eta*exp(raw_eta)`.

E and J are nonnegative contractions. The inherited covariance argument applies to the completed unnormalized fixed-geometry kernel: within-hold innovation 2K preserves covariance I, and boundary innovation I-JJ^T preserves it through jumps. The reverse path is the transpose of the complete forward path. These facts do not prove PSD across different observed inputs or bound the complete normalized gated network.

## Observed donors, finite sources and physical ordering

For all pure references, the fixed observed-TMRCA map is

\[
W_{qr,s}=\operatorname{softmax}_r\left(-\frac{
 \log(1+t_{\mathrm{MRCA}(q,r)})/\log(10001)}{\sigma_s}\right),
\qquad \sigma=(0.05,0.1,0.2).
\]

The native dates yield the same declared encoded-age transform, and Torch softmax evaluates stable logits. No hidden population input is used. Layer projections are bounded with tanh. At anchor a the numerator source and its matching mass source are

\[
s_{a,s,q,h}=q_{a,q,h}+\lambda_{s,h}\sum_r W_{a,s,q,r}r_{a,r,h},
\quad m_{a,s,q,h}=1+\lambda_{s,h}\sum_r W_{a,s,q,r}.
\]

Both channels are bundled into one identical propagation action. After the finite kernel action, the branch mixture is applied to numerator and mass separately, then division occurs once. The layer uses branch softmax mixture weights. Lower-level messages accept finite nonnegative mixtures but reject nonpositive final mass. No mass or retention floor is applied. Nonfinite values and retention underflow are failures, not repaired scientific inputs. Bounded sources and positive matching mass preserve the normalized message bound.

Geometry comes from the frozen compact layout. Coordinates supplied by the caller must remain original physical bp divided by 2 Mb. The module does not renormalize a tile to unit length. The default A=32 anchors are fixed midpoint anchors, with each source multiplied by its physical anchor-cell width. Tests use A=3 solely for small independent oracles.

Exact half-open holds govern tree lookup: a boundary belongs to the right hold, the left tile endpoint is included and the final tile endpoint is excluded. Physical holds are sorted while serialized graph indexes remain intact. Events are only hold boundaries and anchors. Readouts, including duplicate positions, do not create impulses. At a boundary/anchor collision the forward sweep first applies the jump, then adds that anchor source. The forward sweep includes self once. The reverse sweep excludes self and applies transposed path factors in the reversed order. Partial-hold readouts are batched by unchanged segment. E and J generally do not commute, so a reversed tree sequence or an untransposed replay cannot replace the correct completed path.

## Executable layer and matched controls

`CensusCutExchangeLayerV1` supplies independent Q/R projections, positive lambda/delta/eta parameters, a branch mixture, message projection, sigmoid gate and residual LayerNorm update. It returns updated anchor and target contexts, allowing the inherited depth pattern. The handmade two-depth test executes every declared contrast and checks all active gradients. No optimizer or model fit runs.

| Control | Exact interpretation | Inactive dynamics parameters |
| --- | --- | --- |
| `proposed` | Observed cut block diffusion, soft sample-set retention, identical finite numerator/mass sources | None on a changing-support fixture |
| `eta0` | Same observed cut diffusion and sources with eta exactly zero, J=I | `raw_eta` |
| `permuted_groups` | One checked fixed global Q permutation of membership assignments across every tree. Actual IDs, query signals, R donor map and sources remain unchanged. Group sizes and per-boundary rho distributions are preserved. | None on the changing-support fixture |
| `raw_diagonal` | Replace K by diag(K), with diagonal `delta + lambda*(1-1/group_size)`, retaining the same finite sources, cell widths, retention and normalization | None on the changing-support fixture |
| `local` | Coordinate-local Q/R projections and inherited donor-source/mass mixture, followed by the same gate, residual norm and depth. It requires target-local R contexts and performs no genomic propagation. | `raw_damping`, `raw_eta` |

The local control is an explicitly pointwise contrast, rather than a limiting finite-anchor kernel. It shares the whole-tree encoder interface, observed features, donors, projection, gate and depth. Later fitting must match the encoder, feature roster, anchor policy, decoding, seeds, optimizer budget and predeclared selection protocol. Static supports can give an active proposed eta parameter exactly zero gradient because every J is I. That mathematical inactivity is separate from the eta0 parameter being disconnected entirely.

## Handmade CPU qualification

Only the new test file was run, with `CUDA_VISIBLE_DEVICES=` and one CPU thread, after sourcing `scripts/project_env.sh`. Pytest cache is disabled and basetemp remains inside the author directory. Attempt 1 preserved 3 failures/32 passes. Attempt 2 preserved 3 failures/33 passes: the retention replacement had not applied and the donor oracle still used float32 scale literals. These failures led to a verified single-site float32 retention cast and float64 oracle scales, without relaxing tolerances. The final attempt passed **36 cases in 4.23 s** (5.10 s including the subprocess wrapper). This test-suite duration is not a model throughput benchmark.

An independent differentiable dense oracle constructs the full physical E/J product and completes the reverse by transpose. Its dense exponential uses one unrelated constant 1-by-1 block as padding and extracts the original block, following the previously diagnosed Torch small-norm exponential issue in the compact audit. It does not reuse the production segmented exponential. Tests also forbid production `eigh`/`matrix_exp` calls.

| Comparison | Output tolerance rtol/atol | VJP tolerance rtol/atol | Largest output absolute error | Largest VJP absolute error |
| --- | --- | --- | --- | --- |
| Float64, four finite controls | 1e-10 / 1e-12 | 1e-8 / 1e-10 | 8.88e-16 | 2.22e-16 |
| Float32, four finite controls | 2e-4 / 2e-5 | 1e-3 / 1e-5 | 7.15e-7 | 1.19e-7 |

The output maximum covers raw action, numerator, matching mass and normalized output. VJPs cover query/reference sources, raw lambda, delta, eta and mixture logits wherever active. The five-point float64 directional derivative across all these sources/parameters used step 2e-4 and rtol 2e-8/atol 1e-9, with absolute error 1.05e-12. The Q4/R2/G2/A3 fixture has noncommuting projections (Frobenius commutator 0.61237) and a noncommuting projection/jump pair (0.11453). The completed toy unnormalized Gram matrix was symmetric with minimum eigenvalue 0.08016. The full per-control metrics and reproducible collector are in the author bundle.

Coverage includes strict scored/context date rejection, MRCA and reference census collisions, actual-ID Jaccard alignment, arbitrary group-label changes, Q/R permutation, unchanged-hold subdivision, serial unary insertion, all-singleton cuts, positivity, bounded normalization, correct jump ordering, graph serialization, duplicate readouts, source self counted once, pointwise locality, zero/inactive eta gradients and two-layer active/inactive parameter gradients. Invalid numeric rates, mass, geometry gradients, retention underflow, sample dates, identities, role maps and local mixtures fail closed in the tested cases.

## Complexity and remaining qualification

Each block action costs O(BQHV), with V bundled source channels (V=2 for numerator and mass). The compact event scans cost O(n_e BQHV), rather than dense O(n_e BQ²HV). Partial readouts cost O(PBQHV), with group means reused within each unchanged segment. This is a source-derived operation count, not measured efficiency. Autograd retains event states, readout activations and external encoder activations. The fixed donor bank is O(GBQR), and encoder/donor construction cost remains outside the exchange scan. No Q² projection/basis bank is needed by this module.

No actual bank, project-data or holdout array, native inference, profile, GPU/CUDA job, package installation or Git operation occurred. The author does not update the root-owned registry. Existing original, compact and running namespaces remain frozen. No actual group-size/support distribution, memory use, fitting speed, native parity or scientific outcome was measured. The neural whole-tree encoder and actual-bank joins require root integration and separate qualification. Reconstructed-date uncertainty, reference-panel dependence, singleton loss of ancestry evidence and strict context-date eligibility remain scientific limitations.

Before any fit, the root must independently review source/CPU results, record actual observed-only group/support distributions under authorization, qualify the whole-tree integration and numerical flags, freeze development/confirmation and horizon/tile/grid sensitivity, and admit matched controls. The design's prototype efficiency gate is 10x faster matched 20-epoch fitting. Eta0 ties reject turnover attribution, permuted-group ties reject ancestry-constraint attribution, and local/diagonal ties reject memory attribution. Independent parent-level NLL/calibration/accuracy confirmation remains separate and unopened here.

## Inherited theory and claim boundary

The exact prior-art equation audit is [research/reviews/biological_kernel_design_v2/README.md](../research/reviews/biological_kernel_design_v2/README.md). Sources include [GraphSSM, Eq. (5) and persistent-node alignment](https://arxiv.org/html/2406.00943), [MP-SSM, Eqs. (2–4)](https://arxiv.org/html/2505.18728), [graph-SPDE covariance/heat kernels](https://arxiv.org/html/2111.08524), [DiffPool learned incidence](https://arxiv.org/html/1806.08804), [dynamic graph kernels](https://arxiv.org/abs/1612.03615), [Zhou/Huang/Schölkopf normalized hypergraph theory](https://papers.nips.cc/paper_files/paper/2006/hash/dff8e9c2ac33381546d96deea9922999-Abstract.html), and [Relate descendant-set branch association](https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/src/anc_builder.cpp).

Partition/hypergraph diffusion, continuous-coordinate state exchange, OU/covariance reasoning, TMRCA donor pooling, whole-tree sweeps, clade overlap, finite impulses, positive rates and gated nonlinear depth are inherited. The proposed distinction is the specific observed census-horizon constraint plus sample-set support-loss retention. Its priority and biological utility are unresolved. H019's rejected unconstrained incidence is not revived as new mathematics.
