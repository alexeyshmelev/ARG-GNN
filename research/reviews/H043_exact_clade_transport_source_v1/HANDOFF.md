# H043 exact changed-clade source prototype

Status: compact source and handmade CPU qualification complete. No experiment registration, training runner, actual data qualification, prediction, GPU context, admission or resource guard configuration is provided. Root decides future integration. The completed design audit in ../H043_exact_clade_transport_design_v1 remains unchanged.

Main source: src/arggnn/anonymous_exact_clade_transport_v1.py.

SHA256: 441751a252f111fdf0d8af4e70e93e5a923b7bad7829f3aed7d5060d65727304.

## Public API

- build_clade_transport_arrays_v1(parent, sample_index, *, max_payload_bytes=DEFAULT_PAYLOAD_CAP) accepts explicit CPU structural arrays. Tiny panels n>=3 are supported only for source fixtures. It returns compact arrays plus construction metadata. Stable sample-column identity is an upstream reader contract.
- AnonymousExactCladeTransportV1.from_verified_arrays(arrays, meta) validates compact shapes/dtypes, SHA pins, counts, endpoint flags, ordering/deduplication, cardinality and byte bounds, then constructs CPU tensors. .to(device) and .tensor_bytes() mirror existing compact payload APIs.
- build_clade_transport_v1(bank, *, max_payload_bytes=...) requires the exact original CPU MutationRoutedWholeTreeInputsV1 with all952 sampled haplotypes and686query columns. It hashes topology/sample/query indices in metadata. This full-bank entry has not been run on real data.
- RankEightExactCladeTransportV1(upward, transport, *, arm="true_partner" or "own_state") produces an FP32 latent residual. No label/target reader is accepted.
- AnonymousExactCladeTransportGNNV1(..., arm=...) inherits the actual H042 Torch/PyG body and original readout. Call forward_holds(bank, transport=...), forward_train_holds(bank, transport=...) or forward(bank, transport=...). Width32/full952 and all original body constraints remain.
- load_h038_transport_warmstart_v1(model, payload) reuses the original teacher-free H038 warm-start validator and returns new operator/arm metadata. No actual checkpoint was read in this task. Root must reset optimizer state identically under a registered future protocol.

## Exact construction and retained flags

The extractor computes descendant bitsets and node sizes in one postorder per tree. Memoized root discovery follows each parent link at most once and rejects cycles. For each boundary it compares proper-clade sets and uses one root-to-leaf traversal on each side to carry the lowest changed representative. There is no separate ancestor climb for every leaf.

One event is emitted for each affected stable leaf, then local endpoint pairs are deduplicated with integer support. Both-null events are omitted; one-sided events use explicit -1 sentinels. Exact intersection counts and support-normalized weights are computed while adjacent-tree masks are available. Only compact arrays and integer audit counts are returned. No bitset, clade-set dictionary or persistent ancestral-node identity escapes.

All H042 gained/lost flags are retained, including changed ancestors with zero nearest-leaf support. The handcrafted six-leaf counterexample has four changed clades on each side but only three represented endpoints per side; the hidden ancestor flags remain1. Every tested flag array and complete flag metadata object equals the unchanged H042 builder's output.

"Exact" describes observed descendant-set comparison and assignment. It does not imply full changed-clade endpoint coverage, ARG-event recovery, ancestral homology, optimal transport or biological mass conservation.

## Matched layer and original initialization

The layer contains exactly the original four bias-free U/V/W32->8 and O8->32 projections:1,024 weights. The original H042 local rank term is retained exactly. A geometry/constant term and two signed directional messages implement the audited specification. Null endpoints are clamped only to safe local gather positions and masked to zero before use; absent partners remain zero in both arms.

Both arms compute the same node projections, geometry, nonlinear messages and scatter updates. Only the partner gather index changes: true counterpart versus recipient-own state when a counterpart exists. Both have30,788 total model parameters and the same module/state keys. Reusing H042's already initialized projection modules preserves every initial H042 parameter and the caller's CPU RNG state, without extra random draws. The output projection remains zero-initialized.

The constant geometry coordinate yields nonzero rank features for zero/equal node states; zero O intentionally makes the initial output injection zero. Subsequent learned cancellation remains possible. This source promises neither universal nonzero messages nor an accuracy gain.

Event blocks default to32,768 and use non-reentrant Torch checkpointing during gradient-enabled training. U/V/W are projected once per full node bank, not once per event. The inherited second whole-tree sweep, sample readout, span features, genomic GRU and classifier are reused. No training/evaluation orchestration is added.

## Handmade CPU evidence

fixture_check.py passed under existing torch2.6.0+cu124/PyG2.7.0, CUDA_VISIBLE_DEVICES empty, one CPU thread. It used only literal3/6/64-leaf trees and synthetic FP32 latent states.

Checks cover:

- exact one-sided events/support and normalized weights0.5 and1/sqrt(2); polytomy support2 collapses to weight1;
- hidden-clade flags, unary duplicate collapse, a single hold and complete equality to H042 flag output/metadata;
- arbitrary rooted-node ID relabelling and joint stable-leaf column permutation; nonzero layer-output relabelling equivariance;
- a64-leaf comb:254vertices,252memoized parent-link steps,254nearest-traversal visits, rather than per-leaf ancestor climbing;
- cap rejection without subsampling, cycle rejection and payload SHA drift rejection;
- nonzero zero-state rank features, exact zero-output initialization and intended first-step O-only gradient behavior;
- finite FP32 input/U/V/W/O gradients in both arms, one projection call per matrix per bank, and different foreign/own outputs;
- checkpointed block forward/backward equivalence;
- actual inherited PyG MessagePassing modules, equal30,788/1,024parameter counts and identical H042 initial state/RNG.

The full model was instantiated to inspect parameters/PyG modules, not run for predictions. No handmade score, scientific result, real observation, private target, real checkpoint or H040 VAL result was used. fixture_receipt.json pins the actual source and fixture hashes.

Reproduce the bounded source validation only:

    env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .runtime/env/bin/python -B research/reviews/H043_exact_clade_transport_source_v1/fixture_check.py

The fixture keeps temporary/codegen caches inside this review folder's tmp/cache directories. Those directories are disposable validation caches; source and JSON/Markdown receipts are the review artifacts.

## Cost and integration limits

Compact tensors use exactly6N+24P bytes: uint8[N,2] flags, int32[N] node sizes, five int32 event fields and one FP32 weight. P<=n(T-1). The default compact cap is1GiB minus64MiB; it reserves only compact-event/block room and is not a guarantee that the body/projections fit a GPU. Parent-link/root indexing and adjacency are CPU temporaries. Exact integer-bitset work still scales with bitset word width; only adjacent-tree masks are retained.

Construction currently uses bounded per-boundary NumPy chunks followed by concatenation, so peak CPU event-buffer memory can approach twice the final event bytes. Metadata has O(T) audit/count lists outside tensor_bytes. Future cache serialization, complete metadata bounds, bank/source hashes, all CPU/GPU buffers and backward memory require root qualification and reserved retained scopes. No throughput, real event density, cache hit rate, wall time or GPU fit was measured.

Root should bind the compact cache metadata's exact topology/sample hashes to the actual original observation bank and immutable source closure before use. The array constructor's hashes are source/data-coherence checks, not independent proof of biological provenance. Keep the existing all952/all686 support and H042 flags. Exceeding a cap rejects; it never drops leaves, clades or boundaries.

Future eligible physical GPUs remain4/5 only. No hardware action was taken. Root's next decision is whether to register the two matched arms after reviewing this source evidence and the existing experiments.
