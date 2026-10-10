# Compact spectral finite-anchor action v1

Source-only engineering implementation, 2026-10-10. The fresh module [anchor_spectral_compact_v1.py](../src/arggnn/anchor_spectral_compact_v1.py) implements the current dense candidate's finite-anchor kernel with a compact schedule. The [handmade CPU suite](../tests/test_anchor_spectral_compact_v1.py) passes **36 tests**, in 7.27 seconds of pytest reporting time on the existing project runtime. This duration is unit-test execution, not an efficiency benchmark. Author receipts, failed-run evidence, independent-oracle diagnosis and quantitative parity measurements are in [the author bundle](../research/reviews/anchor_spectral_compact_v1_author/README.md).

No frozen candidate dependency or configuration was modified. No actual data or holdout arrays were opened. No CUDA job, installation, Git operation or storage scan was performed. Root owns integration, decision-ledger registration and any subsequent runtime/model qualification. This is algebraically faithful engineering, with measured small-fixture numerical differences. It is not a bitwise-equivalence, predictive-performance, biological novelty or measured-speedup claim.

## Public interface and execution fields

- `prepare_compact_anchor_layout_v1(intervals, positions, num_anchors=32, target_graph=None, window_bounds=None)` creates CPU metadata from the exact float64 physical holds and readouts. The same equal-cell midpoint anchors and width weights define the finite source measure. Readouts must lie in the half-open tile. The original target order and duplicates survive via `target_output`.
- `compact_layout_from_anchor_v1(native_layout)` derives and checks the same metadata from an existing frozen `AnchorKernelLayout`, before device transfer. It checks native durations, physical graph runs, anchors, weights, containing graph IDs, evaluation/output maps and source-event identity. It does not replace the native sealed layout or bank.
- `CompactSpectralGeometryV1.from_operators(native_operators, include_diagonal=False)` creates a lightweight per-hold view that aliases the supplied eigenvalues, eigenvectors and Cqr. Full coupling needs no stored Cqq. Diagonal and gain-matched controls require `include_diagonal=True`. No eigensolve, eigenmode truncation, reconstruction, learned geometry, quantization or modification of supplied arrays occurs.
- `anchor_spectral_compact_action_v1(layout, geometry, sources, rates, damping, coupling="full", readout_chunk_size=128)` maps bundled [A,S,Q,H,V] sources to [M,S,Q,H,V] unique sorted evaluations. V=2 can carry numerator and matching ones mass. All supplied rates and damping retain the frozen routine's strictly positive domain.
- `compact_anchor_messages_v1(...)` constructs the current `q - rate*Cqr*r` sources and matching `1 - rate*Cqr*1` masses, scans them together, mixes scales using the supplied native softmax mixture, validates finite strictly positive mixed mass, and divides without floors. It returns the existing `AnchorLayerMessages` type and both anchor/target messages.

Geometry and signal tensors must have the same floating dtype/device. Eigenpairs, Cqr and optional diagonal are fixed and must not require gradients. Source signals, rates, damping and mixtures remain differentiable. Coordinate metadata retains float64 during `.to(device)`; physical durations and partial-readout offsets are formed before casting to the neural dtype. Host tuples govern the compact event loop, avoiding per-event device scalar extraction. Required numerical guards remain active. Offline PSD/orthogonality/role/source-provenance validation still belongs to the original reviewed bank admission. The new view does not replace that admission.

## Algebra and boundary conventions

For a supplied fixed hold eigensystem C=U diag(c) U-transpose, retain z=U-transpose h and channel decay d=rate*c+damping. Between compact events, multiply z by exp(-duration*d). Compact events contain all physical hold endpoints and fixed anchors. They do not contain readout-only positions. Project each weighted anchor impulse once per layer and reuse it in both directions. At a genuine graph boundary, preserve physical state by applying U_new-transpose (U_old z_old). No product of noncommuting hold operators is reordered or averaged.

For each compact subhold [l,r), F_l includes the impulse at l and B_r includes the impulse at r. They are expressed in that subhold's basis. At readout x,

\[
Y(x)=U\left[e^{-(x-l)d}z^F_l+e^{-(r-x)d}z^B_r\right].
\]

The forward term includes an anchor at the readout. The reverse term excludes it. A coincident anchor/boundary/readout belongs to the right-hand half-open containing graph. During the reverse scan, add that anchor in the right-hand basis before changing to the left-hand basis. An anchor is always strictly inside the physical tile, and the tile right endpoint is not a permitted readout. Coincident anchor/target consumers gather one evaluation result. Duplicate targets gather the same result in their original order.

Both directions combine before reconstruction because they use the same hold basis at a given readout. The implementation packs each [B,S,Q,H,V] chunk as [S,Q,B*H*V], then applies one [S,Q,Q] basis per scale. It does not index bases by readouts or broadcast them into a [P,S,Q,Q] materialization. Chunking bounds temporary readout factor/state workspaces. Autograd still retains required output and factor history, so this is not a constant-memory backward implementation.

The repeated-basis allocation and sequential scan length are removed from readout density. The dense mathematical model still requires basis reconstruction at every requested evaluation. At the illustrative dimensions in [the source efficiency audit](biological_candidate_efficiency_v1.md), the combined-readout plan estimates about 9.83 billion basis MACs versus 34.4 billion for the current schedule. Those estimates exclude substantial remaining work and are not measured fitting speed or memory results. The action contains no timing or profiler instrumentation.

## Handmade numerical qualification

The suite uses explicit Q3/R2 grounded noncommuting matrices, fixed CPU eigensystems, S2/H2 channels, native midpoint anchors and bounded handmade source signals. It compares:

- Full, scalar, raw-diagonal and gain-matched-diagonal coupling, in float64 and float32, against the original action and an independent dense finite-anchor path sum. Gain matching applies to both numerator and mass.
- Raw per-scale results, mixed numerator/mass, normalized messages and query/reference/raw-log-rate/raw-log-damping/mixture-logit vector-Jacobian products. A nonuniform output cotangent plus a log-mass term prevents trivial gradient cancellations.
- Boundary/anchor/readout collisions, duplicated and unsorted target readouts, the tile left endpoint, rejection of the right endpoint, unsorted graph serialization, noncommuting order reversal, holds with no anchors and several anchors within one hold.
- Zero Cqq/Cqr geometry with an arbitrary fixed orthogonal basis. This is zero graph coupling while preserving strictly positive rates, as the original API requires. It yields querywise damping and zero rate gradients. Rates equal to zero remain rejected.
- Consistent query/reference sample permutations, identical-geometry hold subdivision, and insertion of 1,025 readouts into one unchanged hold. The latter has four compact segments versus 1,028 original segments. The reconstruction helper is instrumented to confirm chunks contain at most 17 readouts and receive only one [S,Q,Q] basis.
- Five-point directional derivatives across source signals, raw rates/damping and mixture logits. Two layers of the original gate/LayerNorm/head, identical parameter groups, active gradient clipping, one AdamW update, and optimizer-state parity in both dtypes.
- Invalid coordinate coverage/mapping, NaN/Inf signals/rates, nonpositive rates, invalid mass/underflow, invalid chunk size, altered native source measure and attempted learned eigenpair/source-map/control-diagonal geometry. Failures occur before normalization or optimizer mutation.

The dense oracle enumerates each anchor-to-readout physical path, composes dense exponentials in chronological order, transposes the complete forward product for a right-hand anchor, and sums its weighted contribution. It has no spectral state scan. Rate derivatives pass through dense `torch.matrix_exp`; gain matching uses dense `torch.linalg.solve`.

The first run had four float64 failures and 29 passes. The compact action agreed with the frozen routine. An independently constructed diagonal diagnostic found local Torch 2.6 `matrix_exp` errors up to 1.4953e-10 against scalar `exp` in a small-norm regime. SciPy `expm` agreed with the scalar result. The oracle now embeds its matrix in a block diagonal matrix with a constant unrelated 1x1 block and extracts the top-left exponential. This preserves the target exponential mathematically and avoids the observed low-norm accuracy regime. Its closed-form value/derivative and noncommuting-matrix values are separately checked against scalar `exp` and SciPy. The failed-run log and diagnostic are preserved. Acceptance tolerances were not relaxed. After this oracle correction, 34 cases passed. Two additional fixed-geometry failure cases then passed in the final 36-case suite.

| Quantity | float64 rtol / atol | float32 rtol / atol |
| --- | --- | --- |
| Raw outputs, numerator, mass, normalized messages | 1e-10 / 1e-12 | 2e-4 / 2e-5 |
| VJPs and optimizer-step comparisons | 1e-8 / 1e-10 | 1e-3 / 1e-5 |
| Five-point directional derivative | 2e-8 / 1e-9 | not used |

Absolute tolerance controls near-zero quantities. The recorded acceptance-unit error is absolute difference divided by `atol + rtol*abs(reference)`. The largest value across the ten reported main/dense-readout metric rows is 0.118820, below one. This is a finite fixture qualification, not a guarantee over all geometry or training trajectories.

For the full noncommuting/collision fixture, maximum raw differences from the original/dense oracle were 1.33e-15 / 2.22e-15 in float64 and 2.38e-7 / 4.77e-7 in float32. Matching mixed mass differed from the original by 8.88e-16 and 2.38e-7, respectively. Full-fixture raw-log-rate VJP differences from the original were 1.11e-16 and 8.20e-8. Complete metrics for all controls and parameter families are retained in `parity_metrics.json`.

The 1,025-readout float32 fixture exposes accumulated rounding in the frozen scan. Its maximum raw difference is 1.96e-5. Gradients of the sum of squares across all outputs differ by up to 0.0427 for raw log rates and 0.0470 for raw log damping, on an objective/gradient scale that grows with readout count. Their largest tolerance-normalized errors are 0.0197 and 0.0345. Reporting only tiny-fixture errors would hide this density dependence. The compact scan is faithful in real arithmetic and qualified to the listed fixture tolerances. It is not bitwise equivalent. Actual Q184 long-hold/CUDA checks remain required before scientific adoption.

## Versioned candidate adapter

The module is not wired into the live candidate. A new reviewed adapter can prepare compact metadata and the selected geometry view while the frozen prepared object is still on CPU:

```python
compact_layout = compact_layout_from_anchor_v1(prepared.layout)
geometry_view = CompactSpectralGeometryV1.from_operators(
    prepared.operators[geometry_name],
    include_diagonal=coupling in {"diagonal", "gain_matched_diagonal"},
)
compact_layout = compact_layout.to(device)
geometry_view = geometry_view.to(device, dtype=model_dtype)
```

The new adapter should retain the current plain-boundary encoder and the same local initial anchor/target states. For each existing `AnchorKernelLayer`, supply its unchanged tanh query/reference projections, `layer.rates`, `layer.damping` and `layer.mixture` to `compact_anchor_messages_v1`, then use `layer.update` for anchors and targets. Keep all Q184 anchor states and all R96 reference sources. Use the unchanged head and the same Q120 supervision/output map. The two-layer fixture exercises this pattern with identical state-dict names, parameter groups and optimizer updates.

A later lightweight candidate bank may transfer only the original required encoder arrays/identity indexes, compact layout and selected spectral geometry. Original full-bank sealing, source hashes, conversion/gather fingerprints, role checks and identity/coordinate joins must remain bound. This implementation provides an in-memory geometry view, not a new on-disk cache format or immutable-artifact admission policy. Do not remove per-use hashing until root reviews the replacement mutation-detection contract. The new adapter/config/runner must have fresh source fingerprints and execution namespace, leaving active frozen jobs untouched.

Conservative integration validates all Q184 message rows and retains all target updates. Selecting only Q120 target gates/head is a separate faithful pointwise optimization with its own qualification. Reconstructing only Q120 changes validation obligations for omitted mixed64 rows and is not implemented here.

## Later admission and reproducibility

Root should integrate the author receipt as a source-only engineering condition linked to H020, with explicit later qualification reason: the frozen scan contains readout-dependent rounded projections, and this fresh implementation changes floating-point operation order while preserving the finite-anchor model. No new fit or model-outcome comparison is registered or authorized by this receipt.

Before a development-bank/CUDA comparison, freeze the adapter source/config, admissible numerical budgets, device/dtype/determinism policy, integrity timing, current source geometry and full Q184 validation obligations. Use a fresh reviewed namespace and the existing root-owned shared-lease/process-title/memory-reservation launcher. Record exact graph/readout/event counts, output/mass/gradient parity and clipped optimizer-step parity. Profile synchronized hash/load/convert/transfer/encoder/source/scan/readout/decoder/backward/optimizer stages and total step/fitting time. A passing handmade CPU suite does not establish measured acceleration, CUDA parity, selected-checkpoint identity, early-stopping identity or predictive superiority.

Run the authorized handmade suite only from the server project root after sourcing the project environment:

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .runtime/env/bin/python -m pytest -q tests/test_anchor_spectral_compact_v1.py \
  --basetemp=research/reviews/anchor_spectral_compact_v1_author/pytest_tmp -p no:cacheprovider
```

The reproducible metric collector in the author bundle uses only these same explicit fixtures. It does not open project data arrays. CPU test and metric output are evidence for engineering correctness within stated tolerances, separate from source operation estimates and future measured efficiency.
