# Experimental fixed Kron sample interface

## Status and scope

This is a fixed observed-input reduction primitive and a small CPU flow reference, with no novelty or ancestry-performance claim. Kron reduction, Schur complements, resistor response matrices, harmonic extension and spectral heat equations are established. A separate prior-art audit assesses any later learned reduced-graph encoder plus coupled interval-memory proposal. The adapter itself is canonical linear algebra, not an architectural novelty claim.

The implementation is isolated in `src/arggnn/kron_interface.py`. No neural encoder, training runner, model registry or protocol is changed. All reduction computations use CPU float64 and observed inputs. Reference panel roles may enter a later reference bank, but query ancestry labels remain external. Persistent sample IDs match boundary slots only, with no learned arbitrary-ID embedding.

## Resistance model and static identities

For a simple undirected observed graph, give each edge a strictly positive finite resistance `r_e`, with conductance `w_e=1/r_e`. The weighted Laplacian is `L=B diag(w_e) B^T`, where `B` is an oriented incidence matrix. Orientation does not affect `L`. Bidirected serialization of one physical edge is deduplicated. Parallel physical edges must first be combined by summing conductances.

For an observed marginal tree, the proposed resistance is `r_e=alpha*(t_parent-t_child)`, with one declared global positive scale `alpha` and unencoded observed node ages. This is a modeling choice, not a biological law. The sequence wrapper decodes ages from `log1p(generations)/log1p(time_scale)`, then differences adjacent ages. It never adds log-encoded branch features as if they were physical durations. The wrapper verifies a simple connected tree with one older parent per nonroot node and observed sample nodes at leaves.

Zero-length branches are rejected. A zero edge is a short circuit and needs an explicit node-contraction policy with sample/boundary handling. Adding epsilon to every resistance changes the network and destroys exact series-subdivision consistency. Extremely small positive resistances may overflow reciprocal conductances or make elimination ill-conditioned. The fixed primitive rejects these cases rather than regularizing silently. Loss of distinct ages through an upstream float32 encoding must be reported and resolved at the observed-data precision stage.

Order boundary sample nodes first, with variable internal nodes second:

```
L = [[L_BB, L_BI],
     [L_IB, L_II]]
C = L_BB - L_BI solve(L_II, L_IB)
x_I = -solve(L_II, L_IB) x_B
```

Every connected component containing interior nodes must touch the boundary. This makes `L_II` positive definite for a positive-conductance graph. The implementation uses a Cholesky solve, never an explicit inverse. The response matrix `C` is symmetric positive semidefinite, has zero row sums and nonpositive off-diagonal entries, up to numerical error.

For the harmonic lift `x=[x_B,x_I]`, interior currents vanish and boundary currents equal `C*x_B`. The full network energy `x^T L x / 2` equals `x_B^T C x_B / 2`. This lift minimizes full energy subject to the boundary voltages. The same identities hold for multiple value channels by summing channel energies.

Inserting a nonboundary unary node into an edge with resistances `r_1,r_2>0` and `r_1+r_2=r` preserves the response matrix. When resistance is a fixed scale times physical duration, inserting an observed age strictly between child and parent gives exactly this series law. Equality is to floating-point tolerance in the implementation. It does not cover adding new observed samples, changing reference roles, adding zero edges or changing the underlying observed tree metric.

For a geographic graph with all nodes declared boundary, no elimination occurs and `C=L` in the declared node order. This generic special case is tested. No genealogy or tree assumption is imposed by `resistance_laplacian` or `kron_reduce`, only by the observed-tree wrapper.

## Leaf inputs and identity matching

`precompute_leaf_interfaces` reduces onto **all** persistent observed sample leaves, sorted by their matching identities, and requires the same sample set in every marginal tree. It supports changing internal node counts. The output bank contains responses `[G,B,B]`, selected leaf features `[G,B,F]`, boundary packed indices, unchanged intervals and conditioning diagnostics.

The raw parent-branch column is mandatory to declare and is always excluded from leaf features. A unary insertion shortens the leaf's immediate parent branch, so retaining that column in a learned leaf input would invalidate subdivision consistency even when `C` is unchanged. Caller-selected feature columns must be audited independently. Observed reference/query roles, sample age and fixed observed TMRCA/reference summaries remain suitable when their values are invariant to non-sample unary insertion. Original-graph degree, hop distance, immediate-parent features and internal-node-count summaries are unsuitable for that guarantee. Dropping one column does not certify arbitrary remaining features.

A proposed leaf GNN would use weights `-C_ij` for `i!=j` and these audited leaf features. It must be permutation equivariant, with sample identities used only to align trees and query/reference order. Feeding the original variable internal graph to a local GNN would create an additional non-invariant input path and needs a separate analysis.

## Agreed bank/cache schema

`LeafInterfaceBank` is a frozen schema with `response[G,B,B]` holding raw float64 `C`, sorted `sample_ids[B]`, `boundary_nodes[G,B]`, `leaf_x[G,B,F]`, selected `feature_columns` and diagnostics. It also retains `intervals`, original `query_node_ids` order, `positions`, `target_graph` and optional `window_bounds`. `features` is an audit alias for `leaf_x`. The tree wrapper excludes the raw parent-branch column. The generic `precompute_graph_interfaces` wrapper instead accepts explicitly supplied edge resistances and boundary IDs, with all persistent nodes as its default boundary and no tree-specific feature assumption.

`with_query_reference_cache(bank, reference_ids, query_ids=None)` returns a replaced frozen bank with `query_cache`. Query IDs default to the original bank order and an explicit order must match it. Reference IDs are explicit observed identities, and the partition is exhaustive. The cache contains `normalized_response`, `normalization_scale`, `cqq`, `cqr`, `eigenvalues`, `eigenvectors`, query/reference indices and IDs, normalized `interval_weights`, and `bag_cqq`, `bag_eigenvalues`, `bag_eigenvectors`.

The default full-boundary normalization is `C_normalized = C / mean(diag(C))` per graph. An exactly zero response uses scale 1. Raw `C` and its scale are retained. This scalar is invariant to boundary permutation and nonboundary unary subdivision, since it depends only on the fixed boundary response. It removes the absolute resistance scale from the normalized operator. A later model can explicitly use the retained scalar as an invariant observed feature if that information is needed. The `normalization='none'` option retains physical response units.

The bag cache uses the normalized `Cqq` average with weights `interval_width / total_covered_width`. Its eigensystem is valid for exact homogeneous-average dynamics when channel rates and damping are global across the tile. Graph-dependent channel rates preserve each local eigensystem for ordered flow, but the average of `lambda_g*Cqq_g` generally has a different eigensystem from the average of `Cqq_g`. The cached bag must not be described as that exact average generator if those rates vary by graph.

Bank/cache `.to(device,dtype)` transfers tensors for a later consumer. Bank transfer preserves the original floating dtype of `intervals`, `positions` and `window_bounds`, even when matrices and leaf features are cast. Form genomic differences in coordinate precision before casting durations to model precision. Reduction and eigen-caching occur on CPU float64 before any lower-precision transfer. `.as_dict()` and `.from_dict()` use schema-version-1 plain tensor/metadata dictionaries that roundtrip with `torch.load(weights_only=True)`. Frozen dataclasses prevent field reassignment, while tensor mutation remains possible and is forbidden by the cache-use convention. Parent-owned preparation/loading must bind data/protocol/source hashes and verify provenance. The primitive schema loader checks fields/version but does not replace that validation.

Eigenvectors are used only inside the exact spectral action, never as arbitrary neural input features. Eigenvector signs and orthogonal basis choices within a repeated eigenspace cancel in that action when rates are scalar per channel. This is necessary for the permutation/refinement interpretation.

## Optional coupled query dynamics

Declare an exhaustive disjoint query/reference boundary partition. The primitive rejects silently omitted boundary leaves. If an additional observed sample is neither query nor reference, specify a separate boundary condition or explicitly eliminate it before using the two-block formula. With the declared order, obtain `Cqq[Q,Q]` and `Cqr[Q,R]`.

For query memory `Z[Q,H]`, a piecewise constant reference bank `Rbank[R,H]` and learned query forcing `Fq[Q,H]`, define

```
A = -lambda*Cqq - damping*I
F = -lambda*Cqr*Rbank + Fq
dZ/dx = A*Z + F
```

Positive scalar `lambda` and positive `damping` make `A` symmetric negative definite when `Cqq` is positive semidefinite. Homogeneous differences contract at least as `exp(-damping*distance)`. A forcing-driven trajectory can grow in norm. The augmented affine generator need not itself be dissipative.

This exchanges existing temporal memory between query leaves through off-diagonal terms of `A`. It differs operationally from the earlier per-leaf semigroup, where each query's dynamic state evolves independently. Reference leaves provide a clamped local forcing bank in this construction. They do not carry dynamic memory unless a later model explicitly adds it.

The audit utility `coupled_affine_transition` computes the exact zero-order-hold flow through a `[Q+H,Q+H]` augmented matrix exponential. For the matrix state, augment `[Z;I_H]` under generator `[[A,F],[0,0]]`. It returns the homogeneous `[Q,Q]` transition and integrated `[Q,H]` offset. Positive durations compose exactly for an unchanged generator, up to floating-point error. This generic dense routine is a verification reference, not an optimized training architecture.

**Static response/energy equivalence does not identify reduced dynamics with projected full-tree heat flow.** Eliminated internal nodes are assumed to equilibrate instantaneously in the static construction. Giving them finite dynamic states generally introduces additional memory and modes. The module claims no strong lumpability or exact full-tree diffusion reduction. A three-node unequal-resistance path counterexample, initialized with the harmonic interior lift, is included in the tests.

## Recommended spectral model API and cost

For a later learned model, cache the declared normalized `Cqq = U diag(mu) U^T` on CPU for each fixed graph and query partition. Keep reference leaves in `Cqr`. Use positive **scalar per-channel** rates `lambda_h` and damping `d_h`:

```
A_h = -lambda_h*Cqq - d_h*I
k[q,h] = lambda_h*mu[q] + d_h
E[q,h] = exp(-distance*k[q,h])
J[q,h] = -expm1(-distance*k[q,h]) / k[q,h]
Z_new = U * (E elementwise (U^T*Z_old)
             + J elementwise (U^T*F))
```

The `expm1` form avoids subtracting nearly equal exponentials for short intervals. Positive damping keeps the denominator positive, including a zero eigenvalue of `Cqq`. Cache the eigensystem once. Graph-content conditioning may change a positive scalar rate/damping for each channel without changing `U`. Per-node rates generally change the eigensystem and do not justify this fixed-basis cost claim.

Each exact online state/constant-force action costs `O(Q^2*H)`, plus `O(Q*R*H)` to form reference forcing. Eigensystem caching costs `O(Q^3)` per graph and stores `O(Q^2)`. Dense full-leaf reduction costs roughly `O(I^3+I^2*B+I*B^2)` and response storage `O(B^2)` per graph. The current diagnostic eigenvalue calculations add dense cubic work at precompute time. Matrix-state storage is `Q*H` per scan direction. A cached dense response at `B=32/64` has 8/32 KiB per graph in float64, before features, eigenvectors and diagnostics.

Recommended, not yet implemented, neural stages:

1. CPU `precompute_leaf_interfaces(seq, observed_metadata)` with input/source hashes and audited feature selection.
2. Explicit query/reference partition and cached `Cqq` eigensystem.
3. An independent reduced-leaf graph encoder generating query/reference local banks from `C` and invariant observed leaf features.
4. Graph-local scalar channel rates, damping and query forcing. Generators must not depend on record count, tree midpoint, previous memory or segmentation width for the interval-refinement guarantee.
5. Positive-duration forward/reverse exact spectral actions at fixed coordinates and fixed tile boundaries, followed by symmetric readout if chromosome reversal invariance is required.

This is a candidate architecture recommendation. Training accuracy, calibration, ancestry switch localization, inferred-tree robustness, optimized GPU throughput and novelty remain unvalidated. Comparisons should share observed input availability and feature exclusion, and disclose encoder/capacity differences. Strong nearest-TMRCA and reference-summary controls remain necessary.

## Public primitive API

- `canonical_resistance_edges(N, edge_index, resistance)`: simple undirected deduplication with duplicate-weight checks.
- `resistance_laplacian(N, edge_index, resistance)`: CPU float64 weighted graph Laplacian.
- `observed_branch_laplacian(node_times, edge_index, resistance_scale=...)`: resistance proportional to unencoded age differences.
- `kron_reduce(L, boundary_nodes, max_condition=1e12, relative_tolerance=1e-9)`: response, harmonic lift and diagnostics.
- `KronReduction.harmonic_lift(values)` and `.boundary_current(values)`: static audit operations.
- `precompute_leaf_interfaces(seq, time_column=..., parent_branch_column=..., time_scale=10000, resistance_scale=..., feature_columns=...)`: observed sample interface bank, with the parent-branch feature excluded.
- `precompute_graph_interfaces(seq, edge_resistance, boundary_ids=None, feature_columns=...)`: generic graph boundary bank, with all persistent nodes as default boundary.
- `with_query_reference_cache(bank, reference_ids, query_ids=None, normalization='mean_boundary_diagonal')`: normalized local/bag query eigensystems and explicit Q/R blocks.
- `LeafInterfaceBank` and `QueryInterfaceCache`: frozen fields, coordinate-preserving transfer and tensor-safe dictionary roundtrip.
- `query_reference_partition(C, sample_ids, query_ids, reference_ids)`: explicit exhaustive query/reference blocks, including batched responses.
- `coupled_query_dynamics(Cqq, Cqr, reference_bank, coupling_rate=..., damping=..., learned_query_forcing=...)`: optional CPU affine dynamics. Contains no trainable module, but supplied forcing/rate tensors may retain gradients.
- `coupled_affine_transition(A, F, distance).apply(Z)`: generic exact CPU coupled-flow reference.

Reduction diagnostics record the interior spectral condition number, relative linear-solve residual, response symmetry/row-sum errors, minimum response eigenvalue and largest off-diagonal entry. Row-sum errors are reported against both the original Laplacian and the reduced response, because source-relative backward stability alone does not guarantee high relative accuracy for a very weak boundary response. A declared condition threshold rejects dangerous solves. No jitter, eigenvalue clipping, pseudoinverse fallback or diagonal adjustment is hidden in the reduction. The optional dynamics utility also rejects rate/forcing overflow and an actually non-dissipative numerical `A`, including damping too small to dominate permitted response roundoff.

Tests verify analytical star response, bidirected deduplication, positive-resistance assumptions, unary series subdivision, node relabeling, arbitrary boundary order, harmonic current/energy equality, the harmonic energy minimum, all-boundary identity, conditioning rejection, parent-branch exclusion, variable-size observed trees, persistent sample matching, exhaustive query/reference partitions, cross-query memory exchange, affine semigroup/zero-time identities, finite supplied-tensor gradients and the cached eigenbasis formula. These are implementation checks, not evidence for learned ancestry performance.

## Recorded bounded CPU probe

The final bank/cache schema passed **16 CPU tests** on 2026-10-10 (`pytest tests/test_kron_interface.py -q`, 3.01 seconds). These include unequal-response/unequal-width bag averages, eigensystem reconstruction, branch exclusion, metadata/query ordering, coordinate-preserving transfer, frozen field assignment, tensor-safe serialization and the generic all-boundary case.

The final-source probe ran at 2026-10-10T02:51:35.437366+00:00. It read only the prepared observed input `data/ancient_common_input_pilot_v2/train/rep000/prepared.pt`: **61,236 nodes, 972 trees, 32 persistent sample leaves, Q=8, R=24, 128 scoring coordinates and 48 original prepared columns**. Dropping the declared raw parent-branch column leaves 47 selected leaf columns. The wrapper used observed encoded-time column 6, parent-branch column 7, `time_scale=10000` and `resistance_scale=1/10000`. It decoded no zero branch entries. The smallest positive duration was 2.175191 generations.

Python 3.11.17, PyTorch 2.6.0+cu124, CPU and one PyTorch thread were used with `CUDA_VISIBLE_DEVICES=''`. Initialization used seed 42. Each timing has one untimed warmup and three measured repetitions. There were no gradients, labels, optimizer updates, GPU contexts, local neural encoder or sequential chromosome scan. Input loading and full input-contract validation were outside the timings. Tree-bank timings include grouping, resistance/Laplacian construction, reductions, diagnostics, feature selection and metadata copying with `validate=False` after the input was validated. Query-cache timings include full-boundary normalization, partitioning, local and bag eigen-caching.

| Actual 972-tree operation | Median seconds |
| --- | ---: |
| Full observed leaf bank precompute | 1.988793 |
| Normalized Q/R and local/bag eigensystem cache | 0.013917 |
| Batched generic affine transition construction, Q8/H8 | 0.012855 |
| Batched cached spectral state/constant-force action, Q8/H8 | 0.000807 |

The bank's tensors occupy **19.924 MB** before the query cache, including **7.963 MB** for raw responses. The bank plus local/bag query cache occupies **30.455 MB** in float64. These are tensor payloads, not process peak-memory measurements.

Maximum interior condition number was **24494.743**. Maximum relative solve residual was `3.772e-17`, and maximum response-relative row-sum residual was `1.234e-15`. The smallest normalized `Cqq` eigenvalue was `0.010466707` in this declared Q/R partition.

The state-action probe used eight channels, common scalar coupling rate 1, damping 0.1 and each tree's actual positive interval width. Float64 spectral and generic affine actions differed by at most `8.882e-15`. Casting cached matrices/eigenpairs, state/forcing and the already-formed duration to float32 changed this single action by at most `9.018e-7` relative to float64. Outputs were finite in both precisions. This does not establish complete-scan or training precision.

Balanced binary-tree feasibility probes used unit edge resistances, full response normalization, eight state channels and one interval of length 0.01. Single-call timing is sensitive to overhead, so these records do not rank the two sizes by throughput.

| Leaves / queries / references | Full-tree nodes | Median Kron seconds | Median cached state action seconds | Raw response bytes |
| --- | ---: | ---: | ---: | ---: |
| 32 / 8 / 24 | 63 | 0.000648 | 0.000032 | 8192 |
| 64 / 16 / 48 | 127 | 0.000934 | 0.000031 | 32768 |

The synthetic spectral/affine absolute differences were at most `2.924e-12`. These are engineering consistency/feasibility observations only. They do not compare complete architectures, optimization speed or ancestry predictions.

Input SHA256: `92d1f4e438de6fe4792096a0af096064cd8ae632b7cb9d8ea77925ff6e9daa6a`. Final primitive source SHA256: `69ead9015c3d6edda5208b3e1807da46bb9fc0e650ba9852da93bd98b9a21f5c`. The complete raw probe record is retained here:

```json
{
  "timestamp_utc": "2026-10-10T02:51:35.437366+00:00",
  "source": "data/ancient_common_input_pilot_v2/train/rep000/prepared.pt",
  "source_sha256": "92d1f4e438de6fe4792096a0af096064cd8ae632b7cb9d8ea77925ff6e9daa6a",
  "implementation_sha256": "69ead9015c3d6edda5208b3e1807da46bb9fc0e650ba9852da93bd98b9a21f5c",
  "python": "3.11.17",
  "torch": "2.6.0+cu124",
  "threads": 1,
  "device": "cpu",
  "shape": {
    "nodes": 61236,
    "graphs": 972,
    "queries": 8,
    "positions": 128,
    "input_dim": 48
  },
  "protocol": "CPU float64 fixed reduction/eigh; seed42; one warmup plus three timed repetitions; no gradients, labels, optimizer or GPU; input load/validation excluded; independent interval actions, no encoder or sequential chromosome scan",
  "normalization": "raw C divided by mean full-boundary diagonal per graph; raw C and scales retained",
  "decoded_branch_audit": {
    "zero_bidirected_entries": 0,
    "min_positive_generations": 2.175191459947655,
    "max_generations": 82417.22989558199
  },
  "actual_leaf_precompute": {
    "seconds": [
      1.9887928469106555,
      2.013192276470363,
      1.96442567743361
    ],
    "boundary_samples": 32,
    "selected_features": 47,
    "response_tensor_bytes": 7962624,
    "bank_tensor_bytes_without_query_cache": 19924480,
    "max_interior_condition": 24494.743052703227,
    "max_solve_residual": 3.7715015430121013e-17,
    "max_row_residual_to_source": 8.723878692860999e-17,
    "max_row_residual_to_response": 1.2338351585055559e-15
  },
  "actual_query_cache": {
    "seconds": [
      0.013917407020926476,
      0.013146383687853813,
      0.015791344456374645
    ],
    "tensor_bytes": 10530304,
    "bank_plus_cache_tensor_bytes": 30454784,
    "Q": 8,
    "R": 24,
    "minimum_cqq_eigenvalue": 0.01046670697592689,
    "normalization_scale_min": 8.983856460678222,
    "normalization_scale_max": 12.850167160282858
  },
  "actual_query_flow": {
    "Q": 8,
    "R": 24,
    "channels": 8,
    "batched_affine_transition_seconds": [
      0.018024523742496967,
      0.012854582630097866,
      0.01282513327896595
    ],
    "cached_spectral_action_seconds": [
      0.0008208239451050758,
      0.0008069733157753944,
      0.0007816031575202942
    ],
    "max_spectral_vs_affine_error": 8.881784197001252e-15,
    "max_float32_action_vs_float64_error": 9.018381090797334e-7,
    "all_finite": true
  },
  "synthetic_cases": [
    {
      "nodes": 63,
      "boundary_leaves": 32,
      "Q": 8,
      "R": 24,
      "channels": 8,
      "resistance": "one per balanced binary-tree edge; full C normalized by mean diagonal",
      "kron_seconds": [
        0.0006553120911121368,
        0.0006466032937169075,
        0.0006478326395153999
      ],
      "affine_transition_seconds": [
        0.00030203256756067276,
        0.00028883107006549835,
        0.0003091413527727127
      ],
      "spectral_action_seconds": [
        0.00003429967910051346,
        0.0000319797545671463,
        0.00003117043524980545
      ],
      "response_tensor_bytes": 8192,
      "interior_condition": 13.252550697550653,
      "max_spectral_vs_affine_error": 2.397193554770638e-12
    },
    {
      "nodes": 127,
      "boundary_leaves": 64,
      "Q": 16,
      "R": 48,
      "channels": 8,
      "resistance": "one per balanced binary-tree edge; full C normalized by mean diagonal",
      "kron_seconds": [
        0.0010603638365864754,
        0.0009339842945337296,
        0.0009132726117968559
      ],
      "affine_transition_seconds": [
        0.0002789003774523735,
        0.0002655414864420891,
        0.00026297103613615036
      ],
      "spectral_action_seconds": [
        0.00003231968730688095,
        0.000029940158128738403,
        0.00003146938979625702
      ],
      "response_tensor_bytes": 32768,
      "interior_condition": 15.695921171679142,
      "max_spectral_vs_affine_error": 2.9238833576528123e-12
    }
  ]
}
```

## Analytic single-boundary degeneracy

A connected admitted network with exactly one boundary vertex has identically
zero Dirichlet response. Finite-precision Schur subtraction may instead leave a
tiny scalar residual whose sign varies. The primitive first checks its ordinary
backward-error, symmetry, PSD and conductance conditions, then sets this
analytically known one-boundary response to exact zero. Diagnostics retain the
raw Schur residual. Mean-diagonal normalization subsequently uses its declared
zero-response scale1. This is a specific analytic degenerate-case policy, not
epsilon flooring or broad clipping of small multi-boundary responses.

The regression fixtures include a .2/.3-resistance chain and a .7-resistance
single edge, which previously produced positive and negative scalar residuals.

The same analytic policy is applied independently to connected components
containing exactly one retained boundary vertex. Other components' nonzero
responses are preserved. Component detection uses strictly negative Laplacian
off-diagonals, with no magnitude threshold. Disconnected and mixed-boundary
regression fixtures prevent roundoff from becoming artificial conductance.
