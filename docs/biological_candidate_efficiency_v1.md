# Biological candidate efficiency audit v1

Source-only engineering audit, 2026-10-10. Root owns integration, experiments and scheduling. This audit adds documentation only. It does not edit the frozen candidate or any imported module/configuration, open data/holdout arrays, launch CUDA work, install packages, use Git or perform a storage scan.

The current dense age/plain candidate can be accelerated without changing its mathematical model. Keep the same frozen observed geometry, fixed anchors, nonlinear anchor sources, channel rates, mixture, matching mass normalization and gated decoder. Remove readouts from the sequential propagation schedule, retain spectral state within each unchanged hold, and batch exact partial-hold readouts. Combine both scan directions in that hold's eigenbasis before one physical reconstruction. A source operation count suggests approximately 34.4 billion basis MACs per 250 kb tile become approximately 9.83 billion, including boundary conversions and source projections, at the example dimensions below. This is a mathematical operation estimate, not a measured speedup or bottleneck attribution. Dense reconstruction, nonlinear decoding, backward activations, bank loading and transfer still matter.

## Evidence and scope

The frozen `src/arggnn/biological_candidate.py` SHA256 is `6b3a25d0e59026cab7189032eb41221feb7d1bdf69e15f6c737a672dfb711dd3`, checked against the root-supplied value. The audit's compact provenance and process snapshots are in [source_audit_receipt.json](../research/reviews/biological_candidate_efficiency_v1/source_audit_receipt.json).

| Source location | Observed behavior | Engineering implication |
| --- | --- | --- |
| `biological_candidate.py:332-401` | Physical coordinates use original bp / 2,000,000, with 250 kb clipping, Q184 dynamics, observed R96 and Q120 scored queries | Preserve geometry, coverage, identity order and coordinate normalization |
| `biological_candidate.py:497-518` | Sealed full banks preserve every age/plain cache field and verify lossless serialization | Keep original sealed artifacts, derive a separately bound execution view |
| `biological_candidate.py:549-581` | Every `load_bank` hashes and deserializes the full bundle, converts the prepared object to CPU float32, validates identities, then transfers it | Repeated I/O, conversion and unused-field transfer are plausible costs |
| `anchor_kernel.py:69-116` | `_layout` unions hold endpoints, fixed anchors and every target position | Readout density increases the sequential propagation event count |
| `anchor_kernel.py:182-193,251-279` | Two directions apply two basis transforms per segment and gather `eigenvectors[segment_graph]` | Repeated bases and per-event matrix dispatch are source-visible costs |
| `anchor_kernel.py:332-349` | Numerator and matching mass share the scan, scales mix before division, no mass floor | Optimize both components together and preserve validation before division |
| `anchor_kernel.py:389-412` | Encoder acts on all observed boundary leaves, anchors and targets receive pointwise gated updates in each layer, only anchors create the next layer's sources | Q184 anchor dynamics remain essential, while Q120 target gates can be selected early |
| `kron_interface.py:246-251,287-301`, `anchor_kernel.py:62-65` | Device conversion traverses full cache fields and all operator geometries | A reviewed minimal execution schema can omit unused fields from device residency |
| `biological_candidate.py:788-799` and `anchor_kernel.py:244-260,345-347,396` | Device booleans and scalar extraction validate several stages, including each parameter gradient | Consolidate reductions at required fail boundaries, never remove checks |

The new census-cut/turnover operator in [biological_kernel_design_v2.md](biological_kernel_design_v2.md) changes the scientific model. It is separate from this faithful dense-candidate engineering plan. Segmented partition diffusion, altered anchors, low-rank bases, replacing Cqq/Cqr, dropping eigenmodes, changed coordinate units, quantization, numerical floors and changed supervision are outside this audit's faithful scope. Engineering optimization is not ML novelty.

Compact process observations were permitted by root. An initial `ps` snapshot showed all four frozen PIDs, with approximately 102–188% aggregate process CPU and 2.2–2.4 GB RSS. A subsequent snapshot no longer listed PID 938070 and listed the other three at approximately 102–104% CPU. Root was notified. The remaining processes' `/proc/PID/io` at command timestamp `2026-10-10T18:38:55Z` reported `rchar` of 374,230,381,847, 364,937,822,129 and 290,767,817,918 bytes for PIDs 937999, 938040 and 938092, respectively. These are aggregate characters returned by process reads, not physical disk traffic, not bytes attributable specifically to banks, and not a timing breakdown. Their corresponding `read_bytes` were 749,076,480, 732,008,448 and 667,996,160. These counters and source inspection do not establish that I/O, Python or matrix arithmetic is the measured bottleneck. No profiler, GPU-utilization measurement, log-output evaluation or predictive result was collected by this audit.

## Faithful compact scan

For geometry g, scale s and channel h, the frozen query operator has the supplied fixed eigensystem

\[
 C_{g,s}=U_{g,s}\operatorname{diag}(c_{g,s})U_{g,s}^{T},\qquad
 d_{g,s,h}=\lambda_{s,h}c_{g,s}+\delta_{s,h},
\]
\[
 E_{g,s,h}(t)=U_{g,s}\operatorname{diag}(e^{-t d_{g,s,h}})U_{g,s}^{T}.
\]

V=2 bundles numerator and matching mass. All channel rates/damping remain learned globally per layer/scale/channel, exactly as now. Geometry is fixed and receives no learned eigendecomposition or eigenpair gradient.

1. Build the compact CPU schedule from exact float64 tree endpoints and the same 32 fixed midpoint anchors. Retain the original target vector/order, duplicate targets, half-open graph mapping, anchor-output mapping and weights. Readouts are observations, never impulses. Consecutive compact events bound a source-free subhold with one supplied graph operator.
2. At entry to a hold, maintain `z = U_g.T @ state` with shape [S,Q,H,V]. Within that hold, propagate by elementwise `exp(-delta * d_g) * z`. At an anchor, add its projected weighted source `U_g.T @ impulse`; project that impulse once per layer and reuse it in both directions. Different channels/branches still have distinct decay factors and states.
3. At a true graph change, preserve physical state. Use `z_new = U_new.T @ (U_old @ z_old)`, with the appropriate traversal direction. Two state-basis matrix actions suffice. Do not precompute dense adjacent-basis products, whose Q-cubed preparation and G Q-squared storage may overwhelm the saved work. Operators from different holds generally do not commute.
4. For a compact subhold [l,r), store forward state F_l including a source at l and backward state B_r including a source at r. The same physical state can be represented at the boundary in the left or right hold basis. For any readout x in that subhold,

\[
 Y_{s,h}(x)=U_{g,s}\left[
 e^{-(x-l)d_{g,s,h}}z^{F}_{l,s,h}
 +e^{-(r-x)d_{g,s,h}}z^{B}_{r,s,h}\right].
\]

Here B_r contains anchors at or to the right of r. There are no anchors strictly inside (l,r). F_l contains anchors at or to the left of l. Thus an anchor/readout at l is counted once by F_l, with the reverse term excluding that anchor. Assign a readout at a tree boundary to the subhold starting there, as required by the half-open containing-graph convention. The tile's right endpoint remains excluded. The final B state is zero. The frozen rule remains inclusive forward plus exclusive reverse.
5. Batch each subhold's readouts, multiply both exponential factors in the common spectral representation, add, and reconstruct Y once total. Apply the same scale softmax, then the same numerator/mass division and gated target update. Record/gather coincident anchor outputs separately through the exact existing maps. Anchors need all Q184 rows for subsequent nonlinear sources. Target states never feed back into anchors.

Combining directions before reconstruction is valid because reconstruction is linear and both partial actions use the same U in that subhold. It does not permit reordering physical holds, averaging operators, assigning an impulse to a neighboring tree, or commuting exponentials across tree boundaries. Global readout/anchor coincidence must be represented once and gathered by both consumers.

Use bounded chunks of readouts, grouped by subhold, rather than a tensor containing a repeated Q-by-Q basis per readout. A practical implementation can first use one batched matrix multiply per subhold and then group/pad subholds with similar readout counts if dispatch overhead justifies it. The sequential scan has O(G+A) events. Output construction still has O((P+A) S Q-squared H V) arithmetic, which is unavoidable for this dense reconstruction plan.

## Source operation and memory estimates

The example uses Q=184, S=3, H=8, V=2, L=2, P=2500 target centers, A=32 anchors and G=116 holds. G is the inherited illustrative estimate, not a measured tile count in this audit. Coincidences and clipped tiles change the exact counts. Assume distinct targets, anchors and interior boundaries: N=P+A+G-1=2647 propagated segments and M=P+A=2532 unique evaluations. One full basis transform costs S Q-squared H V=1,625,088 MACs.

| Forward work per tile | Source estimate |
| --- | ---: |
| Current, two transforms per segment per direction per layer: 4 L N S Q² H V | 34,412,863,488 MACs |
| Compact, one combined reconstruction per evaluation per layer: L M S Q² H V | 8,229,445,632 MACs |
| Compact, two transforms at each boundary per direction: 4 L (G-1) S Q² H V | 1,495,080,960 MACs |
| Compact, anchor impulse projection reused by both directions: L A S Q² H V | 104,005,632 MACs |
| Compact subtotal | 9,828,532,224 MACs |

This is approximately 3.50 times fewer basis MACs, not a 3.50 times fitting speed prediction. Endpoint projection overhead, source maps, exponential factors, encoder, decoder, losses, optimizer, bank validation/loading and backward costs are excluded. The illustrative count omits projection of identically zero outer states. It conservatively reconstructs all Q184 rows at every evaluation. Without combining directions before reconstruction, the readout contribution doubles to 16.46 billion MACs.

Float32 `operators.eigenvectors[segment_graph]` has approximately 1,075,401,984 logical bytes at these dimensions. Unique per-hold [G,S,Q,Q] bases have 47,127,552 bytes. This is a 22.8-fold reduction for that repeated-basis tensor only, not total memory. Readout raw values [M,S,Q,H,V] alone still have about 89,450,496 bytes, and current per-segment decay factors have about 46,756,608 bytes. Batch chunks should bound partial-readout factor/state workspaces. A differentiation implementation can retain compact event spectral states, then optionally recompute batched readouts during backward. Activation checkpointing is a separate measured tradeoff and must reproduce gradients.

The plain geometry's scales share a fixed supplied basis but have distinct learned rates. Reuse that basis through a broadcast view. Across geometries, share storage only when required arrays are bitwise equal with a provenance certificate. Identical Cqq does not alone certify identical numerical eigenvectors, especially for repeated eigenvalues or sign choices. Preserve the supplied eigenpairs for the first implementation, including their original order and signs.

## Compact execution bank and dispatch changes

A new execution view can contain the same converted `leaf_x`, plain `normalized_response` needed by the shared boundary encoder, query/reference index vectors, exact coordinate/layout metadata, and the selected geometry's fixed eigenvalues/eigenvectors and Cqr source maps. For full coupling, Cqq is not used in the neural action after its offline validation. Target local features still come from the same graph encoder. Diagonal/gain-matched controls need their declared Cqq diagonals and transformations. The matched GRU must retain its own required fields. A candidate-only schema must not silently break these controls.

Cqr is consumed at anchors only, so an execution view can retain an exact gather for the anchor graphs and the matching Cqr row sums. Keep source arithmetic and reduction order equivalent to the reference before considering fused reductions. Keep identity/role/feature validation on the trusted CPU representation. Transfer only required immutable fields to the device. Do not drop any fields from the original sealed bank or change its serialization receipt. A derived view needs its own namespace, original bank/config/source hashes, selected-field fingerprints, dtype/shape/order metadata, and an exact conversion/gather receipt. Build one shared immutable geometry view per tile/model requirement rather than independently copying aliases. Do not share learned encoder activations between differently parameterized models.

A verified in-memory cache or separately sealed compact file can eliminate repeated full deserialization and conversion. Current `checked()` deliberately rehashes the full file on each use. Moving this check to admission changes when subsequent artifact mutation is detected. Root must define the immutable-snapshot/pinned-object contract, validate hashes before admission, reject changed bound artifacts at required checkpoints and preserve final source/provenance checks. Removing revalidation without that contract is not a faithful fail-closed change. No prepared bank was opened for this audit, and no compact cache size or load latency was measured.

Selecting Q120 before target gates/head preserves their pointwise result because gate, linear, LayerNorm and head act independently on each target row and only anchors supply future sources. This saves target decoder work and activations. Retain Q184 anchor dynamics, full observed encoder inputs and all current numerical checks. The conservative first implementation reconstructs and validates all Q184 message rows before selecting target rows. Reconstructing only Q120 would additionally require a reviewed replacement for current finiteness/positive-mass checks on omitted mixed64 rows. It is not included in the conservative MAC subtotal.

Consolidate device finiteness/positivity checks into small flag reductions at each required fail boundary. Source/rate/decay checks, pre-division mixed mass checks, finite loss before backward and finite gradients before clipping/step remain distinct causal stages. Convert one packed flag set at each boundary, then diagnose individual failed flags only on failure. Never execute an optimizer step after an invalid source, denominator, loss or gradient. Preserve lease/title/runtime checks. Tile loss values can be retained as detached device scalars and copied together at the existing epoch reporting boundary, preserving their order and numerical values, instead of synchronizing separately per tile. Measure each change independently.

## Numerical restrictions

The compact scan is faithful in real arithmetic. It cannot promise bitwise equality to the current float32 routine. The latter repeatedly applies approximate U U-transpose at readout-only events and multiplies rounded short-interval exponentials. The compact scan removes those projections and computes direct partial-hold exponentials. The two forms can accumulate different roundoff and different rate gradients. Compute coordinate differences in original float64 before casting the duration, as the existing code does. Do not subtract cast absolute coordinates or normalize every 250 kb tile to unit length. Do not add an epsilon, clamp mass, clip eigenvalues, change rate parameterization, use TF32/mixed precision or change deterministic runtime policy to obtain parity.

An unchanged-hold subdivision is algebraically invisible but can change the frozen float32 outputs. If CPU/CUDA qualification shows drift beyond the reviewed output/gradient budget, this optimization is not admitted for the frozen execution semantics. Keep the original routine as a reference/fallback and disclose any unavoidable finite-precision difference. A numerical pass on tiny handmade fixtures alone does not certify long training trajectory identity. Small gradient changes may alter early-stopping/checkpoint choices.

## Handmade output and gradient qualification

No parity/gradient test was run in this source-only audit. The following plan is an admission requirement for implementation, not a claim that parity passed.

Use small explicitly constructed CPU float64 fixtures with Q=3, R=2, S=2, H=2 and V=2. Suitable noncommuting grounded M-matrix holds are

\[
 C_0=\begin{bmatrix}2&-1&0\\-1&3&-1\\0&-1&2\end{bmatrix},\quad
 C_1=\begin{bmatrix}3&0&-1\\0&2&-0.5\\-1&-0.5&2.5\end{bmatrix},
\]
\[
 C^{QR}_0=\begin{bmatrix}-.6&-.4\\-.3&-.7\\-.8&-.2\end{bmatrix},\quad
 C^{QR}_1=\begin{bmatrix}-1.2&-.8\\-.5&-1\\-.4&-.6\end{bmatrix}.
\]

Their query row sums match `-Cqr @ 1`. If a full boundary response is needed, use these Cqq/Cqr blocks and a diagonal reference block of `-Cqr.T @ 1`, yielding a symmetric weighted-Laplacian response. Explicitly verify that C0 C1 differs from C1 C0. Supply eigenpairs computed once in CPU float64 only for the fixture. Choose unequal positive rates/damping for every scale/channel, nonzero bounded query/reference signals, distinct scale mixtures and original fixed midpoint anchor weights.

The independent oracle must not copy the optimized recurrence. For each anchor/readout/channel/scale, enumerate the physical path and compose dense `torch.matrix_exp(-dt * (rate*Cqq + damping*I))` actions in chronological order. For an anchor to the right of a readout, transpose the complete forward path product. Sum weighted anchor impulses, including coincident self terms once. Apply the identical source/mass construction, scale mixture and division. Compare this direct finite-anchor sum with both the frozen scan and the compact implementation.

Required fixture families are:

- Noncommuting holds, reversed operator order and unsorted serialized graph order. Sorting physical intervals must preserve graph IDs. Deliberately commuting scalar/singleton cases cannot establish ordering correctness.
- Readouts at the tile left endpoint, at true graph boundaries, exactly at anchors, on both sides of an anchor/boundary, with duplicates and unsorted original target order. Reject the tile right endpoint. Include a coincident anchor/boundary/readout, using right-half-open source geometry.
- Several anchors in a hold, holds with no anchors, one graph, one anchor, two coincident consumers and many inserted readout-only subdivisions. At least one test should insert thousands of observations into a constant hold to expose float32 projection/rounding accumulation.
- Sample-ID permutations with consistent query/reference permutations, source impulses of both signs, repeated eigenspaces and sign/permutation-equivalent bases. Compare physical outputs rather than spectral coordinates. Fixed bank gradients are out of scope.
- Scalar, raw-diagonal and gain-matched-diagonal controls using their exact stated sources. The gain transform must apply to numerator and mass. These controls test common schedule/mapping behavior without substituting them for full coupling.
- Positive-mass/bounded-message fixtures and deliberately invalid NaN/Inf, nonpositive rates/damping, nonpositive or underflowed mass, malformed intervals/maps and changed provenance. Fail before division or optimizer mutation as the reference contract requires.

Compare raw per-scale numerator and mass, mixed numerator/mass, normalized messages, anchor updates, target updates and final Q120 logits. Compare vector-Jacobian products for query/reference sources, raw log coupling, raw damping, mixture logits, encoder/projection/gate/norm/head parameters and source weights. A nonuniform seeded cotangent over outputs avoids the cancellations of a simple sum. Use scalar objectives with different output weights, not only a constant-source test that normalizes to a trivial value. Check source/rate/damping gradients against the independent dense oracle and five-point directional finite differences. Perform one identical AdamW step including the current parameter groups and clipping, comparing parameters, optimizer states and gradient norms.

Provisional small-fixture acceptance targets are float64 output `rtol=1e-10, atol=1e-12` and gradient `rtol=1e-8, atol=1e-10`. Provisional float32 reference-comparison targets are output `rtol=2e-4, atol=2e-5` and gradient `rtol=1e-3, atol=1e-5`, with absolute error used near zero. These are proposed budgets, not observed tolerances. Root should freeze scale-aware budgets and rejection criteria before evaluating actual development-bank parity. Report maximum absolute/relative errors by quantity, fixture, direction, dtype and parameter family. Do not silently relax limits after failures or use relative errors on near-zero gradients. Genuine mass/sign/provenance failures always reject regardless of closeness.

## Implementation admission and measured efficiency

1. Root assigns fresh module/test/config namespaces without modifying dependencies of the four frozen jobs. Keep the exact current implementation/hash as the baseline and preserve the original sealed banks. Register the engineering repeat/new condition in the ledger/decisions before any substantive experiment, linking H020 or the relevant frozen attribution hypothesis.
2. Implement compact CPU layout and kernel action first. Qualify against the handmade direct oracle and the frozen routine, including gradient and fail-closed tests. Run no data/holdout access during fixture qualification. Add the derived execution view only after action parity passes, then qualify conversion/gather parity independently.
3. Root reviews numerical budgets, provenance timing, deterministic runtime and full-field validation obligations. Only root admits any later development-bank parity run or CUDA qualification through the existing shared-lease/title/memory-reservation launcher. Preserve the two-physical-GPU limit and frozen job allocations.
4. After admission, measure synchronized stage timings and peak allocated/reserved memory for hash/read, deserialize/convert, transfer, encoder, sources, compact scan, readout reconstruction, gates/head, backward and optimizer. Use warm-up plus repeated timed trials, record exact G/P/event counts, runtime/config/source hashes and deterministic settings. CPU counters alone cannot substitute for this profile. Do not use unsynchronized CUDA wall-clock timing.
5. Compare an identical input/model/weight snapshot and compute the same Q120 loss with the same Q184 anchors. Test the compact schedule, compact bank and decoder selection as separate conditions before combining them. Record total forward/backward step time and total fitting wall time, not only one kernel. A change is accepted for efficiency only after parity and a prespecified measured speed/memory gate pass. The design-v2 tenfold prototype gate refers to its new model, and must not be asserted from this dense-candidate 3.5-fold arithmetic estimate.
6. Do not infer predictive superiority or a new biological contribution from engineering parity/speed. Full refitting or confirmation is a separately registered decision with parent-level protocol and selection criteria frozen before outcomes are read. Root decides whether small float32 drift requires a new execution-semantic condition.
