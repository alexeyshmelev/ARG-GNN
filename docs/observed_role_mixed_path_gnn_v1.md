# H035: matched observed-role mixed paths

H035 fits only two controls, signed_difference and path_sum, using the unchanged frozen observed-role mixed-path prototype. Both start from the same seed 101 state, allocate 5056 scalars (3470 trainable, 1586 inherited frozen unused), share the complete observed geometry, H030 source/encoder/decoder and sparse path maps, and receive 160 optimizer updates. The prototype SHA256 is 1cd64f32a69490cf720278af8112248ec1d3b93d8d8cddeb7904fd3428966053. No prototype edit, added arm or recurrent reservoir is part of this experiment.

## Role correspondence and parallel paths

Preparation retains complete native observed trees and analytic 350 census cuts, with younger references at their H030 cuts and older references at actual sampling nodes. The same 96 pure observed source labels provide class-balanced positive log evidence, normalized separately by age family, with explicit empty support and no older-ANA pseudocount. Query/context ancestry labels never enter the model.

Exactly identical adjacent labelled metric trees coalesce. Across remaining adjacent holds, T maps an actual sampled identity or a unique exact descendant-roster bitset plus exact observed node age. Source and target keys must each be unique. Ambiguous and absent roles reset to zero. This is an observed structural correspondence, not a true ancestor identity. Native ancestor IDs are not the alignment key, and numeric descendant bit values never enter neural features.

A is the shared learned undirected geometry-weighted tree mean with unit self loops. For each adjacent direction, the two compositions are:

signed_difference  = 0.5*(A_new*T-T*A_old)
path_sum  = 0.5*(A_new*T+T*A_old)

The same nonlinear H030 encoder provides up+outside values. Left and right path streams are computed in parallel from that field, transformed by a shared path_value map, and decoded with the H030 static gates and linear three-stream whole-tree decoder. Existing left/right distance decay, bounded ell and same-age query centering are retained. The bias-free zero-initialized context head adds the centered graph context to the frozen H025 logits. Inherited carrier and unused kernel/edge maps remain frozen; no retained state crosses a hold or tile.

H030 query census-cut readouts are retained here. This differs from H033's actual-native-query harmonic anchors. Both H035 controls have the same readout policy. Exact adjacent tree coalescing is qualified; general unary-subdivision invariance of the nonlinear GNN is not claimed.

## Diagnostics and interpretation

geometry.json records per-tile and row-weighted aggregate actual match numerators, candidate source/target row denominators, ambiguity counts and fractions. Guaranteed exact sampled-identity pairs are separated from internal-role pairs, so overall agreement cannot hide sparse internal matches. ambiguous_rows_reset counts both endpoint occurrences for shared keys that are not one-to-one. Missing roles are also unmatched but are not reclassified as ambiguity. No tile boundary is joined.

Every TRAIN and VAL forward records actual left/right path L2 and RMS, combined path L2, centered-context L2 and ell. No additional path calculation is added to obtain those magnitudes. The first actual TRAIN tile additionally records the signed-to-sum norm ratio before each arm's first update, when their weights still match exactly. Later magnitudes use the separately fitted arm weights and are not a cancellation comparison under identical parameters.

Sparse role matching, small signed-path activations, absent gradients or a tie with path_sum must remain visible. Differences of tiny numerical size do not establish novelty. Tree message passing, sparse alignment, mixed path sums/differences and commutators are established. This is a utility experiment with no novel primitive, posterior, causal edit attribution or ancestral population-state claim. A signed-path utility claim requires improvement over matched path_sum and the frozen baseline on both development parents, with boundary/short-tract/class-recall checks.

## Frozen scientific protocol

Use only seed 101, FP32, the original four observed TRAIN/VAL parents, 16 TRAIN tiles and 16 VAL tiles, original 100bp center-bin support and two development VAL parents. Reuse the selected immutable H025 genomic-smoothing epoch10 skip and selected H026 native-reference descriptive comparator without refitting either. TRAIN teacher predictions are in-sample rather than out-of-fold.

Run 10 epochs and 160 updates per arm. AdamW learning rate 0.001 and weight decay 0.0001 follow the H030 setting; raw_cost and raw_ell have zero decay. Only parameters with requires_grad are optimized. Clip gradient norm at 1. Both arms use an exact copied initialization and shared deterministic tile order. The first actual training update records forward/backward time and per-arm CUDA allocated/reserved peaks and counts within 160.

Target-free geometry is sealed before TRAIN exact-span C4 mass is opened for loss. Both arms' predictions on both VAL parents are sealed before the unchanged biological scorer opens epoch VAL truth. Select the earliest minimum equal-parent Present64 raw-probability NLL over all 10 epochs; report accuracy from that checkpoint. No accuracy reselection, floors, posthoc renormalization, bin dropping, seed repeats or test parents are introduced. The two reused development parents limit interpretation.

## Root input and execution bindings

Input schema is observed_role_mixed_path_gnn_input_v1. The root-admitted active seed 101 CUDA input uses the same refs and observed_parents contract as H033: config, runtime, storage, frozen_baseline_source/config/completed/checkpoint, emitter_bindings, target_bindings, strong_reference_completed/config/checkpoint, plus the same four observed parent records. source_sha256 must pin the new driver and the unchanged prototype. Frozen H026 refs are checked against the configuration.

The actual scripts/launch_reserved_gpu_full_storage_v1.py must be an exact reservation consumer binding. Configuration helper refs pin it, src/arggnn/project_storage_full_reservation_v1.py, immutable V6, immutable V9 core and all reused scientific sources. The driver uses validate_storage_v9, PilotOutputV9 and budget_check_v9 from the full-reservation adapter; there is no old process checker or legacy retry adapter. Retain the strict original full-storage charge, compact 2GB output cap, bounded temporary native-tree writes and 32MB CUDA-qualification write allowance.

Root owns registration, reservation, admission and GPU launch. The author source-only qualification opens no real arrays or checkpoint/model artifacts, runs no fitting or optimizer step and initializes no GPU context. Tiny fixture correctness does not qualify real-tile runtime, memory, predictive utility or scientific priority.
