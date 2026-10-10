# Experimental coupled query memory on invariant leaf responses

## Status and motivation

`src/arggnn/boundary_memory.py` implements a learned boundary-response GNN followed by an exact piecewise-constant coupled query-memory model. This is a candidate architecture with CPU implementation checks. There is no novelty, trained ancestry-performance, projected full-tree dynamics or strong-lumpability claim. Kron response identities, spectral diffusion and affine zero-order-hold integration are established mathematics. The fixed geometry and cache contract are documented in `docs/kron_interface.md`.

An established genealogical fact motivates one possible biological use: two query haplotypes whose true local TMRCA is below the target census age share the same census ancestor and source label at that coordinate. When such local kinship appears, coupled query memory can combine uncertain evidence accumulated earlier by different queries. A separate per-query dynamic state cannot directly exchange that stored evidence. This is a hypothesis about a useful inductive bias, not a theorem of improved ancestry inference. No census-age threshold or additional kinship gate is implemented. Inferred topology/time errors require empirical robustness checks. The current Q8/Q16 implementation probes do not establish behavior at Q32/Q64, where more cross-query relationships may be available.

This is not the first temporal node-exchange mechanism. In particular, interleaved GraphSSM graph updates can propagate histories between nodes. Comparisons must include those mechanisms and distinguish the invariant input interface, source-bank independence and exact flow from unsupported novelty claims.

## Agreed fixed bank interface

The model consumes a `LeafInterfaceBank` directly. Root-owned preparation/loading supplies the exact clipped tile and keeps target labels separate. Required fields include:

| Bank/cache field | Use |
| --- | --- |
| `leaf_x[G,B,F]` | Audited invariant observed leaf features |
| `response[G,B,B]` | Raw fixed Kron response retained for provenance |
| `sample_ids`, `boundary_nodes`, `feature_columns` | Observed matching/provenance, never ID embeddings |
| `intervals[G,2]`, `positions[P]`, `target_graph[P]` | Original physical coordinates and graph selection |
| `query_node_ids[Q]`, optional `window_bounds` | External output-column order and supplied tile bounds |
| `query_cache.normalized_response[G,B,B]` | Learned GNN conductance graph |
| `cqq[G,Q,Q]`, `cqr[G,Q,R]` | Exhaustive disjoint query/reference boundary partition |
| `eigenvalues[G,Q]`, `eigenvectors[G,Q,Q]` | Fixed normalized `Cqq` spectral basis |
| `interval_weights[G]` | Clipped interval width divided by total tile width |
| `bag_cqq`, `bag_eigenvalues`, `bag_eigenvectors` | Fixed homogeneous-bag operator cache |

Query order must match `bank.query_node_ids`. Reference IDs are explicit observed panel identities, and every boundary sample must belong to Q or R. Eigen-caching and reductions happen on CPU float64. The model never computes an eigensystem or augmented matrix exponential online.

For trees, raw parent-branch input is obligatorily omitted by `precompute_leaf_interfaces`. In the existing 48-feature prepared ancestry input, excluding column 7 leaves 47 leaf features. Sample age, observed roles and within-tree reference/TMRCA summaries are acceptable only when independently audited as invariant to non-sample unary insertion. Retaining alternative immediate-parent, node-degree or internal-count features would invalidate the full-input guarantee. The model does not return to the original variable ancestor graph through another encoder.

Normalize raw response by `s_g=mean(diag(C_g))` over the **full fixed boundary**, setting `s_g=1` for an exactly zero response. The model requires the cache's `mean_boundary_diagonal` normalization and rejects silently supplied physical-unit caches. This removes absolute common resistance scale from the neural/dynamic operators. Raw response and scales remain in the bank/provenance. The learned model does not use those scales as extra features. Boundary permutations and nonboundary unary subdivision preserve the scale because they preserve the response.

Bank transfer with `.to(device,dtype)` preserves the original floating dtype of intervals, scoring positions and window bounds while casting features and cached matrices. Form coordinate differences first, then cast durations to memory dtype. The model requires the bank to be moved/cast once before forward. It does not repeat CPU cache preparation inside an epoch.

## Independent learned source banks

The GNN receives only invariant leaf features and normalized full-boundary response. For off-diagonal entries, conductance adjacency is `W_ij=-C_ij`, with zero diagonal. A layer takes the incoming weighted-neighbor mean, concatenates its information through separate self/neighbor linear maps, applies GELU and a residual LayerNorm update. A zero-degree row has zero neighbor contribution. The input projection and every update are shared across boundary nodes and graphs.

Each graph is encoded independently. Neither interval width, midpoint, graph-record count, scoring coordinate, numerical sample ID nor temporal state enters this encoder. Its output is `H_g[B,D]`, with query/reference rows selected by matching indexes. No learned arbitrary ID embeddings or eigenvector-as-feature inputs exist.

Two linear heads create bounded signals:

```
q_g = tanh(query_projection(H_g[query]))       # [Q,L]
r_g = tanh(reference_projection(H_g[ref]))    # [R,L]
b_g[:,h] = q_g[:,h] - lambda_h*Cqr_g*r_g[:,h]
```

Reference signals are graph-local clamped forcing, not persistent dynamic reference memory. Query ancestry labels never enter the bank, encoder, forcing or generator. The source bank is independent of all temporal states. Changing full versus diagonal dynamics preserves these exact learned local features and forcing when parameters are shared.

For all-boundary geographic forecasting, R may be empty and the reference product is zero. The universal model still allocates its reference projection, whose parameters then have zero gradients and are reported as inactive in the bank-specific configuration manifest. This is disclosed rather than called a fully active matched-parameter model.

## Exact coupled interval dynamics

For each learned latent channel h, rates are **global across the tile**:

```
lambda_h = softplus(raw_coupling_h) > 0
delta_h  = softplus(raw_damping_h) > 0
A_g,h = -lambda_h*Cqq_g - delta_h*I
dZ[:,h]/dx = A_g,h*Z[:,h] + b_g[:,h]
```

There is no positive floor added to these rates. Nonfinite/nonpositive floating-point rates or an actually nonpositive decay due to cache roundoff are rejected. No spectral clipping, hidden diagonal jitter or matrix-dependent learned node rates are used. `Cqq` is positive semidefinite mathematically, so positive damping makes this modeled homogeneous flow strictly contracting. A forcing-driven state can grow, and the forced affine system is not described as a norm contraction around zero.

With cached `Cqq=U diag(mu) U.T`, channel decay is `k[q,h]=lambda_h*mu[q]+delta_h`. For unchanged forcing over distance d:

```
Z_new = U @ (
    exp(-d*k) * (U.T @ Z_old)
    + (-expm1(-d*k)/k) * (U.T @ b)
)
```

`expm1` avoids subtraction of nearly equal exponentials on narrow intervals. Strict damping handles zero eigenvalues without a pseudoinverse. A zero-time action returns the input state exactly, including floating-point bases. Cached eigenvector signs and orthogonal choices within repeated eigenspaces cancel in the action because rates are scalar per channel. Gradients flow through GNN/source heads, forcing, coupling and damping, with fixed observed geometry/eigenbasis.

The ordered model sorts by physical interval start, evolves from a zero state at the supplied left tile endpoint, and applies partial actions at each exact scoring coordinate. Bidirectional ARG mode also scans in reversed order with **positive** durations from a zero state at the right endpoint. It never inverts a dissipative flow. The default readout combines the local query embedding with `forward+backward` and `(forward-backward)^2`, making reflection exchange the two histories without changing logits **away from exact marginal-tree breakpoints**. Intervals use the declared half-open `[left,right)` convention. At a breakpoint, reflection changes which local containing-tree representation is selected, so complete logits need not be reflection invariant even though the ordered memory itself is continuous. The output is `[P,Q,K]`.

## Controls and causal scope

| Constructor setting | What changes |
| --- | --- |
| `coupling="full"`, `aggregation="ordered"` | Coupled `Cqq` flow through ordered intervals |
| `coupling="diagonal"` | Replace `Cqq` by its diagonal; use identical source banks and `Cqr` forcing |
| `coupling="gain_matched_diagonal"` | Independent query states with rate `diag(K)` and force `diag(K)*e`, retaining the full static equilibrium `e=K^-1F` |
| `coupling="gain_leak_diagonal"` | Independent query states with rate `delta+lambda*rowsum(Cqq)` and force `rate*e`, retaining full equilibrium and uniform-state reference leakage |
| `aggregation="homogeneous_bag"` | Use one width-weighted average `Cqq` and one width-weighted learned forcing over the same domain |
| `aggregation="equilibrium"` | Read each target graph's static `Z*=(-A)^(-1)b` through its cached spectral basis; no temporal scan |

Full, raw diagonal, gain-matched diagonal and gain-leak diagonal controls allocate exactly the same parameter schema. The diagonal control removes **temporal query-memory exchange**, while preserving cross-query information already present in the graph-local GNN sources. Those are different information routes and must not be conflated.

The two gain controls isolate persistent-state coupling more closely than the raw diagonal control. For each graph/channel, let `K=lambda*Cqq+delta*I`, `e=K^(-1)F` and choose a positive diagonal rate D. Their dynamics are `dZ/dx=-D*Z+D*e`. The full model and both gain controls have exactly the same static equilibrium. Their instantaneous forcing still mixes evidence across queries through the cached full spectral inverse. Holding that forcing fixed, each query's output derivative with respect to another query's initial state is exactly zero. The gain controls must not be described as removing every cross-query information route.

`gain_matched_diagonal` uses `D=diag(K)`. `gain_leak_diagonal` uses `D_q=delta+lambda*sum_j Cqq[q,j]`, matching the full operator's leakage on a uniform query state. This avoids treating all self-degree as independent leakage when off-diagonal terms would cancel it. Query-block row sums equal reference-grounding conductance mathematically and are nonnegative. The implementation rejects a negative row larger than `4*Q*eps*sum(abs(row))` in magnitude, where eps is the current matrix dtype's machine epsilon. Only negative rows inside that explicit floating-point bound are set to zero. No positive floor or silent material clipping is added, and positive damping is still enforced.

For homogeneous-bag gain controls, the equilibrium and independent rate are built from the **averaged full generator and original averaged forcing**, rather than averaging already transformed local gain forces. Thus full and gain controls retain the same homogeneous-bag equilibrium. For static-equilibrium aggregation, both gain settings return the same full equilibrium and have the same reduced head shape as the full equilibrium control.

The homogeneous bag uses cached eigenpairs of `sum_g w_g*Cqq_g`, with `w_g=width_g/total_width`. Because coupling/damping are global per channel, this is exactly the width-average affine generator, and forcing is averaged online after its learned heads. It discards source arrangement while retaining the same query locations, tile endpoints and local readout. Graph-varying rates would invalidate this fixed averaged eigensystem and are not implemented.

The equilibrium control uses a single context in its head and disables the reverse branch, avoiding permanently zero reverse-difference features. Its readout therefore has fewer parameters than a bidirectional temporal head. Report this capacity difference. It depends on the current observed graph, not preceding graph arrangement or elapsed span.

`bidirectional=False` provides one-sided ordered flow, and `num_classes=1` provides scalar regression directly. The ordered one-sided implementation cannot read later independent source evidence at an earlier scoring coordinate, as tested. The homogeneous bag reads the complete supplied window even if its final scan is one-sided, so it is an offline control and must not be called causal. Forecasting inputs/labels must still be aligned using only observations available by the forecast origin. Root owns that protocol and training runner.

## Representation guarantees and limitations

Inserting a nonboundary unary ancestor with additive positive physical branch resistance preserves the fixed Kron response. An unobserved dangling root cap has no boundary current and likewise preserves it. If every selected leaf feature is unchanged, the full learned source bank, normalization and modeled generator/forcing are unchanged.

Splitting an unchanged genomic interval into identical graph copies preserves the ordered memory by the affine semigroup law. The source bank does not depend on the new record count or split point. Combined unary/root-cap and genomic subdivision was tested on **all selected input features**, cached responses and final predictions. Bag averaging preserves total interval mass, and static equilibrium remains unchanged. The claim is conditional on invariant input features and fixed supplied tile endpoints, with floating-point tolerance. It does not cover adding observed samples, changing metric/roles, zero branches or changing where states are reset.

Node-index permutations, nonmonotone numerical ID renaming, graph serialization and external query-column permutations have the expected invariant/equivariant outputs. Order sensitivity means physically moving graph contents, rather than merely changing their stored row order. The ordered control can change under that substantive rearrangement; the homogeneous-bag control preserves held-target outputs when the graph/width multiset is unchanged.

Kron static energy/current equivalence is **not** an exact projection of finite-state ancestor dynamics. Eliminated interior vertices need not equilibrate instantaneously in a physical process. The model specifies a new boundary diffusion, with no strong-lumpability or full-tree heat-flow claim. Resistance proportional to branch duration, full-boundary normalization, learned sources and genomic diffusion rates are explicit modeling choices. Their biological validity and inferred-tree robustness need experiments.

## Runner-facing API

```python
from arggnn.kron_interface import precompute_leaf_interfaces, with_query_reference_cache
from arggnn.boundary_memory import BoundaryMemoryModel

# CPU preparation phase, once for each exact clipped tile.
bank = precompute_leaf_interfaces(
    prepared_tile.sequence, time_column=6, parent_branch_column=7,
    time_scale=10000.0, resistance_scale=1/10000,
)
bank = with_query_reference_cache(bank, reference_ids=observed_reference_ids)
model = BoundaryMemoryModel(
    input_dim=bank.leaf_x.shape[-1], num_classes=4,
    hidden_dim=32, num_layers=2, latent_dim=8,
    coupling="full", aggregation="ordered",
    bidirectional=True, symmetric_readout=True,
    initial_coupling=20.0, initial_damping=1.0,
)
model = model.to(device)
bank = bank.to(device, dtype=next(model.parameters()).dtype)
logits = model(bank)  # [P,Q,4]; labels remain in a separate case/tile record.
provenance = model.configuration_manifest(bank)
```

The model itself must be on the requested device before transferring the bank. No GPU run occurred in this subtask. `source_bank(bank)` returns `BoundarySourceBank(local,query_signal,reference_signal,forcing)`. `encode_context(bank)` returns `BoundaryMemoryContext(local,forward,backward)`. `configuration_manifest(bank=None)` records constructor choices, parameter count, exact interface/source hashes, normalization, source independence, channel-rate definition, coordinate/model dtypes and inactive reference parameters when applicable. `head` is a linear output map, and `num_outputs` reflects its actual width after a regression-head replacement.

Root's proposed `BoundaryTileCase(bank,labels,parent_id,split,tile_id,bounds,provenance)` is sufficient. The runner must preserve parent grouping and external query order, select checkpoints by parent-validation metrics, and bind all source/protocol/cache hashes. A clipped window may reuse a graph's response/eigenpairs but needs **tile-specific** interval weights and bag eigensystem, computed once during CPU preparation. No cache is silently reconstructed or repaired by forward.

Dense leaf-source encoding costs approximately `O(G*B^2*D)` per message layer. Each spectral action costs `O(Q^2*L)` plus linear spectral factors, and reference forcing costs `O(G*Q*R*L)`. This first ordered scan is sequential over G records. Decay, full-span exp/expm1 factors, spectral forcing projections and integrated forcing offsets are batched once per bank. The recurrence uses those coefficients without per-hold tensor-to-Python boolean checks. Sorted graph order is copied once to CPU integer metadata, avoiding implicit CUDA scalar-index synchronization at every hold. Partial target actions are batched. Public standalone action helpers retain their validation. The scan is still sequential and is not an optimized parallel-scan kernel. Q8/Q16 scalar and classification contracts are tested, with no complete-model GPU throughput or memory claim. Cache-size and primitive timings in `docs/kron_interface.md` do not measure this learned encoder/scan.

## Verification record

The combined command `python -m pytest -q tests/test_kron_interface.py tests/test_boundary_memory.py` passed **88 tests in 4.47 seconds** on CPU with `CUDA_VISIBLE_DEVICES=''` and one PyTorch thread. Nineteen tests belong to the current fixed interface and sixty-nine to the memory consumer. No package install, registry/config edit, training run, GPU job or Git operation was performed by this implementation subtask.

The memory suite verifies exact spectral versus dense affine reference actions, semigroup and exact zero-time behavior, finite parameter/source/rate gradients, scalar Q8/Q16 all-boundary geographic outputs, explicit normalization enforcement, all four coupling choices and all aggregation controls, query/node/ID/storage invariance, reflection away from breakpoints and explicit half-open breakpoint behavior, causal one-sided source exclusion, static equilibrium stationarity, and combined tree/interval subdivision.

One isolated mechanism test puts forcing into query 1 in an early **zero-response, uncoupled** interval, then applies a coupled later interval with zero forcing. Query 0 receives a positive later memory only in the full model and remains exactly zero in the diagonal control. A separate learned-source gradient test changes past observed input for query 1 while future forcing stays fixed. The late target-query gradient is nonzero with full coupling and exactly zero with diagonal coupling. These establish the implemented route of temporal exchange, not ancestry accuracy.

Tests also verify gain-controlled equilibria against the full model, independent initial-state derivatives, instantaneous source mixing, dense affine correctness and leakage-roundoff rejection. All coupling modes pass node/nonmonotone-ID/storage/query invariance and dual subdivision. Unsorted scoring positions preserve their original output order for the parent-level scatter. A breakpoint test confirms the right-hand local graph, continuous reflected forward/backward memory and the possible change in local/full logits.

A test replaces `torch.linalg.eigh` and `torch.matrix_exp` with failing stubs after cache preparation for every coupling mode. Forward still succeeds in float32 while coordinates remain float64. `docs/boundary_memory_provenance.json` records the verification commands, source/test/interface hashes, environment and example constructor/parameter manifests. Root owns actual dataset cache preparation, scientific experiments and subsequent integration.


The batched scan is checked against the validated public action routine for every coupling mode. The number of decay/factor validation calls stays constant when another unchanged hold is added. A direct test makes `Tensor.__bool__` fail inside a prepared full-hold action and the action still succeeds.

## Bounded actual-input CPU timing

A label-free forward probe used `data/ancient_common_input_pilot_v2/train/rep000/prepared.pt`, SHA256 `92d1f4e438de6fe4792096a0af096064cd8ae632b7cb9d8ea77925ff6e9daa6a`. The full observed bank has 972 marginal graphs, B32, Q8, R24 and 47 selected leaf features. Fixed geometry/eigenpairs were prepared outside the measured forward, then cast to float32 while coordinates stayed float64. The model used width 32, two local GNN layers, eight memory channels and seed 101. Each control had one untimed warmup and three timed no-gradient forwards on one CPU thread. No labels, optimizer or GPU were used.

| Ordered bidirectional control | Parameters | Median CPU forward seconds |
| --- | ---: | ---: |
| Full, before coefficient batching | 6,564 | 0.41443 |
| Full, after batching | 6,564 | 0.06894 |
| Raw diagonal | 6,564 | 0.05067 |
| Gain-matched diagonal | 6,564 | 0.05235 |
| Gain-leak diagonal | 6,564 | 0.05305 |

Every forward returned finite `[128,8,4]` logits. The full-control medians show about a sixfold CPU improvement in this bounded engineering probe. It is not a GPU or training throughput benchmark. Pre/post aggregate output-sum and squared-sum differences were `2.9124e-5` and `1.6510e-5`, respectively, with float32 arithmetic. These checksums are not an elementwise equality proof. Equivalence to the public validated action is separately tested in float64. Complete precision/performance comparisons remain root-owned experiments.

At the same constructor width/depth/feature/output counts, equilibrium and one-sided controls have 6,532 parameters because their head consumes one memory context rather than two. Full/raw diagonal/gain controls with the same aggregation/readout have identical parameter counts. Complete raw timing repetitions and exact consumer source hashes are retained in `docs/boundary_memory_provenance.json`.
