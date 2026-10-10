# Positive-feature lineage kernel operator: prior-art audit

Status: literature and mathematical formulation audit, 2026-10-10. This is a proposed operator, not an implemented or validated contribution. The audit identifies direct prior art but is not a claim that the complete literature has been exhausted. Cached public PDFs and extracted equations are under `.runtime/cache/prior-art-audit`. Version URLs, byte counts and SHA256 hashes are retained in `docs/operator_audit_sources.json`.

## Assessment

Positive feature maps, normalized key-value sufficient statistics, recurrent exponential retention, continuous attention integrals and structural-kernel attention are established. Rebranding their combination as the first linear, continuous, order-sensitive or graph-aware attention operator is not supportable. A narrower contribution could be an exact interval-integrated structural correspondence operator for piecewise graph-valued signals, with explicit assumptions, sparse identity handling and empirical evidence against direct alternatives.

## Closest equation-level comparisons

| Reference | Verified mechanism | Boundary for the proposal |
|---|---|---|
| Katharopoulos, Vyas, Pappas and Fleuret, [Transformers are RNNs](https://arxiv.org/abs/2006.16236v3), ICML 2020, Sections 3.2–3.3, Eqs. 10–12 | `S_i=sum phi(K_j)V_j^T`, `Z_i=sum phi(K_j)`, output `phi(Q_i)^T S_i / phi(Q_i)^T Z_i`. Section 3.2.1 explicitly notes that exact exponential/softmax kernel features are infinite-dimensional | Positive content features, associative normalized moment attention and linear token complexity are already present |
| Choromanski et al., [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794v4), ICLR 2021 | FAVOR+ uses positive random features to approximate softmax attention in linear time and space | Positive finite features are not novel. Exact evaluation of a chosen finite-feature kernel must be distinguished from approximate softmax |
| Sun et al., [Retentive Network](https://arxiv.org/abs/2307.08621v4), 2023, Eqs. 1–6 | Full-matrix state expression followed by diagonalization. Eq. 6 gives `S_n=gamma S_(n-1)+K_n^T V_n`, with parallel and chunkwise equivalents | Exponentially decayed key-value moments and recurrence/attention equivalence are established. RetNet itself does not enforce positive normalized attention |
| Yang et al., [Gated Linear Attention](https://arxiv.org/abs/2312.06635v6), ICML 2024, Section 4 and Table 1 | `S_t=G_t elementwise S_(t-1)+k_t^T v_t`, with input-dependent matrix gates. Its comparison table includes Mamba, Mamba-2, gated retention and other matrix-memory RNNs | Input-conditioned decay is also established. A gate applied once per serialized record generally breaks subdivision consistency unless converted into a continuous rate |
| Dao and Gu, [Transformers are SSMs](https://arxiv.org/abs/2405.21060v1), ICML 2024 | Structured state-space duality relates attention and SSMs through semiseparable matrices | Attention/SSM duality and efficient structured scans are not new properties |
| Calvello, Kovachki, Levine and Stuart, [Continuum Attention for Neural Operators](https://arxiv.org/abs/2406.06486v4), Definitions 4–5 and 10–11, Theorems 6 and 12 | Defines normalized softmax attention as an integral/expectation on function spaces, including cross-attention. Discrete attention is connected to Monte Carlo approximation | Continuous-coordinate normalized attention and resolution-independent function-space formulation are established. Our finite-feature graph operator would be a particular tractable kernel, not a new definition of continuum attention |
| Chen et al., [ContiFormer](https://arxiv.org/abs/2402.10635v1), NeurIPS 2023, Eqs. 1–3 | ODE-evolved keys/values, continuous query interpolation, and time-integrated query/key inner products | Continuous-time query/key interactions and integral attention have prior art. Its score integral and trajectory construction differ from the proposed piecewise-constant KV moment scan |
| Mialon, Chen, Selosse and Mairal, [GraphiT](https://arxiv.org/abs/2106.05667), 2021, Section 4.2.2, Eq. 5 | Multiplies a content exponential kernel by a nonnegative graph-structure kernel before row normalization. It explicitly describes the resulting product as kernel smoothing | Multiplying content and structural similarities inside normalized attention is established. A clade-incidence Gram kernel is a domain-specific structural choice |
| Rampášek et al., [GraphGPS](https://arxiv.org/abs/2205.12454v4), NeurIPS 2022 | Combines local message passing, positional/structural encodings and global attention. Explicitly supports Performer and other linear-attention alternatives | Local GNN plus scalable global attention is established. Structural coupling that preserves efficient evaluation needs a precise factorization |
| Li et al., [GraphSSM](https://arxiv.org/abs/2406.00943v2), NeurIPS 2024, Eqs. 5 and 8, Appendix B.2 | Piecewise graph-conditioned forcing, exact zero-order-hold state evolution, and changing-node-set memory alignment | Interval evolution, ZOH subdivision composition and identity alignment must be compared directly. The existing project implementation is an explicitly documented adaptation |

The relevant continuum reference is **Continuum Attention for Neural Operators**, arXiv 2406.06486, rather than SparseContinuousAttention or Infinite-former.

## A precise kernel and exact interval calculation

Let interval g be `[a_g,b_g)`. A graph-local encoder supplies source values `v_gi` and nonnegative finite features `psi_gi` for every node i. Fix a nonnegative query feature vector `psi_q(x)` at coordinate x and a positive measure weight `w_gi`. These must describe the underlying graph-valued signal rather than its serialized partition. For example, a product structural kernel can be written

\[
\kappa(q,i)=\phi(q)^T\phi(i)
\left(\alpha+\beta c_q^Tc_i+\eta\,\mathbf1[\mathrm{id}_q=\mathrm{id}_i\ge0]\right),
\quad \alpha,\beta,\eta\ge0,
\]

where `phi` is nonnegative and c is a normalized nonnegative incidence vector. This product has a tensor-product feature factorization. An additive kernel, `phi(q)^T phi(i)+beta c_q^T c_i+eta equality`, has a different direct-sum factorization and different complexity. The implementation and manuscript must choose one formula explicitly. The current `LineageCrossAttention` implementation uses an exponential content score with additive leaf-equality and Jaccard-clade-overlap biases before softmax. Its kernel is therefore not the finite-feature clade-dot kernel written here. Replacing it with the proposed factorization changes the kernel, rather than exactly linearizing the existing softmax/Jaccard operator. Positive definiteness alone does not guarantee nonnegative attention weights. The incidence dot product above is nonnegative because the incidence features themselves are nonnegative.

For a scalar nonnegative rate lambda, a symmetric coordinate kernel is `exp(-lambda |x-s|)`. Define

\[
F_g=\sum_iw_{gi}\psi_{gi}v_{gi}^{T},\qquad
z_g=\sum_iw_{gi}\psi_{gi}.
\]

Left-side moments satisfy `dS/dx=-lambda S+F_g` and `dZ/dx=-lambda Z+z_g` on the current hold. For a hold length d,

\[
S(x+d)=e^{-\lambda d}S(x)
+\frac{1-e^{-\lambda d}}{\lambda}F_g.
\]

The same equation applies to Z. At lambda zero, use the continuous limit d. Evaluate the factor using `expm1` or a stable exprel series. Right-side moments are scanned in a reverse coordinate with positive distances. A query inside an interval needs the exact left and right partial holds. Combining their disjoint integrals gives

\[
y_q(x)=\frac{\psi_q(x)^T(S_{\rm left}(x)+S_{\rm right}(x))}
{\psi_q(x)^T(Z_{\rm left}(x)+Z_{\rm right}(x))}.
\]

The containing interval must not be counted twice in full. A positive content term and nonzero source mass can guarantee a positive denominator. A numerical epsilon is a documented modification of normalized attention, not a proof that a possibly zero denominator is harmless.

This calculation is exact for the **chosen finite-feature kernel and piecewise-constant source density**, in real arithmetic. It is not exact softmax, not exact integration of arbitrary continuously varying encoder features, and not a universal approximation theorem inherited automatically from the continuum paper. Different feature-wise or graph-dependent rates need their own precisely stated path kernel and recurrence.

## Subdivision property and its assumptions

For constant source F, let `T_d(S)=exp(-lambda d)S+exprel(-lambda d)dF`. Then `T_(d2)(T_(d1)(S))=T_(d1+d2)(S)`. The equivalent interval integral is additive. Therefore splitting an unchanged graph into identical subinterval records preserves the output at each fixed query coordinate when:

- Graph-local features, source weights, query features and rates do not depend on interval width, midpoint, graph ordinal or total record count.
- Source contributions are integrated by physical interval length, rather than added once per record.
- Queries retain their physical coordinates. Partial-interval contributions are evaluated exactly.
- Random masks are disabled or coupled to the underlying graph signal. Independent dropout on duplicate records is not exactly invariant.
- Normalization is applied to integrated moments, without introducing record-count-dependent normalization or nonlinear per-record state jumps.
- Every stacked layer continues to represent the same underlying signal. Reusing an interval-end contextual state as a constant key for a subsequent layer can introduce subdivision dependence.

This is a standard semigroup/integration consequence. It can be a useful model contract and a meaningful implementation test, but must not be called a newly discovered theorem. Midpoint sampling of an exponential coordinate kernel generally fails this contract.

## Identity, clade and complexity qualifications

An integer sample ID is a correspondence key, not a numeric learned feature. Equality must be false for anonymous IDs, so all internal nodes with ID -1 do not become one artificial persistent entity. Equality supports corresponding leaves. It does not select anonymous ancestors by itself.

A query-side clade feature must be explicitly defined. If `GraphSequence.clades` records only descendant **reference** landmarks, a target leaf outside that landmark set has the zero vector. Directly taking `c_query dot c_source` then selects no ancestors. To select the target's observed ancestral lineage, supply observed graph-derived target-membership incidence or another well-defined query-clade representation. This is a feature/formulation decision and may change storage and complexity. In the current project mechanism, attention targets every target-tree node. Target leaves receive leaf-equality bias, ancestral target nodes receive clade-overlap context, and context feeds back through target-tree edges. Thus a zero landmark-clade vector on a query leaf limits direct ancestor selection from that leaf feature, but does not nullify all-node attention or the indirect ancestral route. Do not claim direct lineage selection from a zero leaf clade. No truth ancestry label may enter this route.

Let N be the total packed source-node count, E packed native edges, G intervals, M fixed leaf-coordinate queries, D value width, and R the effective finite feature dimension. A dense moment implementation costs approximately `O(N R D + G R D + M R D)` beyond the local encoder, with ordinary training storage potentially `O(G R D)` for saved states. Interval lookup/sorting contributes `O(G log G + M log G)` unless an ordered sweep removes it. These statements hold with R and D fixed. The local encoder still contributes its own node/edge cost.

If content rank is r and structural incidence dimension is C, an additive factorization uses roughly `R=r+C+L` for L leaf identities. A product kernel uses roughly `R=r(1+C+L)`. Full leaf one-hot features and full descendant incidence therefore cannot be treated as constant rank as sample count grows. Multiplying dense node-by-identity matrices can hide quadratic work.

Leaf equality can instead maintain identity-keyed moments. With sparse updates, cost is proportional to actual leaf occurrences and queries, including the content rank for a product kernel. A lazy-decay implementation can avoid touching inactive identity reservoirs at every graph. Under this project's complete-leaf contract, every queried leaf occurs in every graph, so `G L <= N` already bounds identity occurrences. This does not eliminate the cost of dense all-ancestor clade incidence. Its number of nonzeros can itself be quadratic in the number of leaves for highly unbalanced trees. Fixed reference landmarks, sparse incidence or a tree-specific dynamic program require separately documented tradeoffs.

Define the source measure carefully. Summing equally weighted nodes makes graphs with more ancestral nodes contribute more mass. Per-graph normalized weights can remove that mass effect, but change the operator. Both preserve interval subdivision if unchanged graphs retain the same within-graph weights. Neither automatically gives invariance to adding or duplicating internal vertices.

## Recommended hypothesis and gates

A supportable hypothesis is that **exact interval mass plus observed structural correspondence improves fixed-coordinate inference while avoiding a materialized query-source attention matrix**. Claim task-specific factorization and a useful representation contract if the implementation and measurements support them. Do not claim the first linear graph attention, first continuous attention, first exponential KV recurrence, or first partial-node alignment.

Before expanding implementation, define the kernel, query clade semantics and source measure. Compare content-only exact KV scans, equality-only correspondence, clade-only correspondence, their combination, zero-decay duration pooling, midpoint quadrature, matched late temporal attention, the supra-graph control and documented GraphSSM adaptation. Property checks should cover duplicate-interval splitting at fixed queries, source-node permutations, stable-ID renaming, serialization reordering and substantive coordinate-order rearrangement. Performance and scaling claims need actual recorded results with feature rank, value width, leaf count, total nodes, parameters, memory and runtime reported.
