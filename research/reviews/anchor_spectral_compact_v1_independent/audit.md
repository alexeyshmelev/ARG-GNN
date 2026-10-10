# Independent compact spectral action v1 audit

**PASS for the pinned source-only, trusted-bank, handmade CPU engineering scope.** This does not admit a candidate adapter, actual Q184 development-bank use, CUDA execution, training trajectory equivalence, or a speed/memory claim. No implementation edit is requested by this audit. Root owns integration, source registration, installation, Git and scheduling.

Audit performed on anogena in /workspace-SR003.nfs2/ARG-GNN. Read AGENTS.md and used scripts/project_env.sh before the existing .runtime/env/bin/python. Every execution explicitly hid CUDA and used one CPU thread. All new files are in this independent directory. No observed/holdout arrays, model outcomes, GPU calls/jobs, profiler, installation, Git action, storage census or Mac project writes were used.

## Exact audited pins

| File | SHA256 |
| --- | --- |
| src/arggnn/anchor_spectral_compact_v1.py | 35cd11e92499a997dd9623f937797f2c482886698e2a79b48d0a3eb8aa87bf9c |
| tests/test_anchor_spectral_compact_v1.py | 4b5394572ab3bef233652a9322711eb13bbd1d80330107a1abdc93a4c971701a |
| docs/anchor_spectral_compact_v1.md | 736d88dd1f42ebd9cd1bb06781802703e09b77c5a3f0a212604cb4459e34804f |
| src/arggnn/biological_candidate.py | 6b3a25d0e59026cab7189032eb41221feb7d1bdf69e15f6c737a672dfb711dd3 |
| src/arggnn/anchor_kernel.py | 2b8cc1c1726f746f961c0d63a4870e806a1a9d3d6e1fa6724dc0bda39967b083 |

AGENTS.md, project_env.sh, kron_interface.py, age_interface.py and boundary_memory.py also matched the author receipt's frozen hashes. Final hash verification is recorded in independent_receipt.json. The original current candidate remains imported against its frozen anchor kernel, with no wiring to the new module.

## Source conclusions

The original anchor scan, anchor_kernel.py:251-279, keeps physical state, unions readouts into its event schedule, gathers one basis per segment and performs U-transpose/factor/U at every propagated event in both directions. The new module changes numerical operation order while preserving the finite-source action in exact arithmetic.

- Layout preparation at new-module lines 60-125 keeps exact float64 positive contiguous holds, sorts physical order while preserving serialized graph IDs, uses the original equal-cell midpoints/widths, rejects the tile's right endpoint, and assigns a coincident anchor/readout to the right-hand half-open graph. Coordinates contain endpoints and anchors only. Unique sorted evaluations plus anchor_output/target_output preserve duplicates and original consumer order. Readouts never add impulses. Derivation at lines 128-184 checks native physical runs, durations, anchors/weights/maps and source-event identity. Restriction: native graph IDs must be 0 through G-1 and each appear in one contiguous run, matching the original tile format.
- Projection at lines 306-309 happens once per weighted anchor source and is reused in both directions. State is retained in each hold's eigenbasis. Lines 316-324 change basis only when the serialized graph ID changes, via U_new-transpose U_old acting on state. No dense adjacent-basis product is precomputed. No noncommuting operators are reordered or averaged.
- In the reverse scan, lines 330-338 add an endpoint anchor in the previously active right-hand basis before converting into the left-hand basis. In a segment [l,r), stored F_l includes the impulse at l and stored B_r includes the impulse at r. Readout factors at lines 345-350 therefore count a coincident anchor once through F_l, while excluding it from the reverse contribution. This holds when every anchor collides with a graph boundary, including paths traversing holds without anchors.
- Both directions are added before reconstruction because they are represented in the same basis at that readout. Lines 224-234 pack [B,S,Q,H,V] into [S,Q,B*H*V] and multiply one [S,Q,Q] basis. There is no [P,S,Q,Q] basis gather/materialization in this readout path. Chunking bounds per-call readout factors and states, but outputs/chunks and autograd history still scale with evaluation count. This is not constant-memory backward.
- Lines 378-388 preserve q-rate*Cqr*r and 1-rate*Cqr*1 sources, bundle numerator and mass into V, mix branches before division, and reject nonfinite/nonpositive mixed mass without a floor. Gain matching at lines 293-305 applies D*K_full^-1 to every bundled component and retains rate/damping derivatives. Source, raw-rate, raw-damping, query/reference and mixture gradients were compared with independent differentiable formulas.
- Full geometry views extract supplied eigenpairs/Cqr without eigensolve, truncation, geometry learning or altered source maps. Diagonal/gain-matched controls use the supplied Cqq diagonal only when explicitly included. Their numerical domain retains finite strictly positive rates/damping/decays. Fixed-geometry requires_grad rejection is active.

## CPU evidence and independent oracle

The original 36-test suite reproduced: **36 passed in 7.16s**. The final independent suite reports **25 passed in 6.00s**. These are pytest durations, not efficiency benchmarks. Logs are author_reproduction.txt and independent_tests_final.txt. A preserved intermediate independent run passed 24 cases before adding the view-admission restriction case and improving permutation coverage.

The independent oracle enumerates each anchor/readout physical path and traverses hold intervals in the actual travel direction. It uses the closed symmetric 2 × 2 exponential
exp(-tK)=exp(-tm)[cosh(t rho)I-sinh(t rho)(K-mI)/rho]
with m=tr(K)/2 and rho=sqrt(((K00-K11)/2)^2+K01^2). It does not use a spectral scan, supplied eigenpairs or torch.matrix_exp. Gain transformation uses a direct 2 × 2 inverse. Values and derivatives therefore exercise a distinct numerical implementation from both scans and the author's dense oracle.

Fixtures include six deliberately unsorted grounded noncommuting holds with all four anchors on graph boundaries, holds without anchors, three scales/channels/bundled components, all four controls, duplicate/unsorted readouts, original/nonuniform source cotangents, mass/normalization/ones-signal checks, large dyadic shifted physical coordinates, independently changed orthogonal bases within equal degenerate geometry, eigenpair sign/order gauges, actual query/reference permutation comparisons, and singleton packing with no targets or references. Author coverage additionally provides five-point directional derivatives, two unchanged gated layers, active gradient clipping and AdamW state/parameter parity, dense readout insertion, malformed metadata and numeric failure cases.

The final independent_metrics.json contains 171 numerical comparisons. Maximum tolerance-normalized error is 0.003817127551883459, maximum absolute error 1.430511474609375e-6. The maximum float64 acceptance error is 0.0002773806010795337. All comparisons use the author's unchanged budgets: float64 values 1e-10/1e-12 and gradients 1e-8/1e-10, float32 values 2e-4/2e-5 and gradients 1e-3/1e-5. Absolute tolerance governs near-zero values.

The author metric collector was copied with exactly its output directory changed to this directory. Reproduced parity_metrics.json is byte-identical to the author metric artifact: ten rows, maximum acceptance error 0.11882004141807556. In the 1,025-readout float32 hold, the compact/native raw difference is 1.9609928131103516e-5, raw-log-rate gradient difference 0.042724609375 and raw-log-damping difference 0.0469970703125. Corresponding acceptance errors are 0.1188200, 0.0196657 and0.0345297. Large absolute gradient differences here must stay visible, since the summed objective grows with readout count.

## Four initial failures and oracle correction

The preserved initial log shows the same four tolerance pairs as the current close helper. Initial and oracle-corrected attempt receipts bind the same implementation SHA ff07b3129103fd33ff790bcb1daf42504c33c32d5dcf4e632c0d5dd95d4f2163. The final implementation has a different pinned hash after added fixed-geometry guards.

Independently substituting ordinary torch.matrix_exp for the author's padded oracle on the final source reproduces all four logged failure names: float64 full/diagonal/gain-matched main comparisons and the float64 no-anchor-holds comparison. Compact/native comparisons still pass before the oracle failure in those test functions. A standalone diagonal closed-form case reproduces ordinary matrix_exp error 1.4953172033926876e-10, matching the preserved diagnostic. The padded block result and its derivative agree with direct scalar exp at strict tolerances. The unchanged original suite also checks the padded noncommuting values against SciPy.

A constant independent 1 × 1 block preserves the target exponential mathematically and forces a different numerical algorithm norm regime. The correction is justified by this independent value/derivative evidence, with no tolerance relaxation. Historical source files for the first two test hashes are not archived in the bundle, so this audit establishes their printed tolerance evidence, equal implementation receipts and reproducible failure diagnosis, rather than asserting an unavailable full historical source diff.

## Material limitations and next qualification

The view at lines 201-213 is an extraction/transfer helper, not sealed geometry admission. It aliases mutable arrays, does not verify PSD/orthogonality/source-role provenance, and the dataclass does not make tensors immutable. A control diagonal is a strided Cqq view and retains the full original CPU Cqq storage until a real dtype/device copy occurs. Full coupling has no Cqq field in the view. A caller retaining the native full prepared bank still retains its full bank memory. All these restrictions are consistent with the source-only documentation, and were independently tested.

The frozen candidate's load_bank at lines 549-581 rehashes the full file, loads the original complete schema, converts all prepared arrays to CPU float32, checks Q184/R96/280 × 47 dimensions, exact identities/roles/features/readout joins, then transfers the full prepared bank. None of those operations was changed. The geometry view alone cannot establish lower total residency or loading time. Per-use hash timing must remain until root defines and reviews a replacement mutation-detection contract. Candidate adaptation and a minimal execution-bank/cache schema need fresh fingerprints and conversion/gather receipts.

Before actual development-bank or CUDA admission, root should:
1. Register the source-only engineering condition and later qualification reason against H020, freeze a separate adapter/config/runner namespace, and retain the frozen candidate/source bank as reference.
2. Preserve the plain boundary encoder, all observed 280 inputs, Q184 anchor states, R96 clamped sources, identical projections/rates/mixtures/gates/LayerNorm/head, supervision restricted to actual Q120, full Q184 finiteness/mass checks and existing role/identity/coordinate/provenance admission. Q120-only reconstruction is not implemented. A later early target-gate selection is a separate condition.
3. Freeze scale-aware numerical budgets, underflow/positivity rejection, deterministic CUDA/dtype/TF32 policy and integrity-check timing before evaluating current development banks. Include real hold/event/readout counts, longest holds, dense readout density, condition/rate ranges, all intended geometries and controls.
4. Compare raw per-scale bundles, mixed numerator/mass/normalized Q184 messages, anchor/target states, Q120 logits/loss, source/raw-rate/raw-damping/mixture/encoder/gate/norm/head VJPs, clipping norms and one identical AdamW parameter/state update on the same snapshot. Test CPU conversion/gather separately from CUDA action. Reject budget violations without after-the-fact relaxation.
5. Use only the root-owned admitted shared-lease/memory-reservation/process-title launcher, within eligible free devices and the two-physical-GPU limit. Preserve the three frozen fits on 2/3. This audit did not query or affect them.
6. After parity admission, profile synchronized hash/read, deserialize/convert, transfer, encoder, source, scan, reconstruction, decoder, backward, optimizer and total step/fitting time with warmup/repeated trials and peak allocated/reserved memory. Compare the schedule, execution bank and decoder conditions separately. Arithmetic estimates, pytest duration and CPU parity cannot certify speed, CUDA parity, early-stopping/selected-checkpoint identity, biological novelty or predictive benefit.

Reproduce using scripts/project_env.sh and CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 with the existing runtime. Pass -p no:cacheprovider and a --basetemp inside this directory. independent_receipt.json binds exact source/artifact hashes and full commands.
