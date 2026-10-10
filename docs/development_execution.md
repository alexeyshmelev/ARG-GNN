# Deterministic fresh development and selector interoperability

Fresh development applies one strict numerical policy after completed-source admission and before CUDA initialization. The new development_execution helper fixes CUBLAS_WORKSPACE_CONFIG to :4096:8, enables deterministic algorithms with warn_only=False, disables cuDNN benchmarking, enables cuDNN deterministic execution and disables CUDA matmul/cuDNN TF32. Float32 matrix multiplication uses highest precision. Existing pilot, preparation and model implementations are unchanged.

The helper refuses an already initialized CUDA context and checks its effective runtime settings. Each bound protocol records deterministic_execution_policy, and each fitting run records the applied policy. Selected epoch NLL and checkpoint-reloaded NLL are preserved separately with an explicit replay delta. A nonzero replay difference fails the run. This contract applies to the recorded software and hardware. It is not a cross-platform bitwise guarantee.

The candidate now emits the generic selector's complete family/namespace/seed/dependency header. Checkpoints bind seed, exact protocol bytes, completion, boundary manifest and implementation. Raw validation logits preserve source-qualified ordered parent IDs, positions/query order and the same bindings. Bank-transfer or fitting exceptions retain a failure.json receipt. The model, optimization grid, loss and earliest-minimum selection rule are unchanged.

The four common candidate/anchor-GRU/anchor-Transformer/reference-smoother configs now include factual input_scope and cross_tree_aggregation strings. Sparse-anchor models, the candidate's all-hold geometry path and all-record controls retain different information access. Shared parent provenance does not make these scopes equal.

Current non-executable source proposals are configs/argmix_development_search_plan_v2.json and configs/allrecord_development_search_plan_v2.json. Their v1 files remain historical snapshots. The raw binder defaults to v2. No executable fresh config is emitted until genuine preparation completion and full source verification.

The focused CPU integration suite passed151 tests in136.64s, covering execution settings, completion/source locks, candidate fitting/provenance/failures, temporal ordering, raw ARGMix and all-record controls. Three standalone policy tests also passed. No fresh data were fitted, and no CUDA support or throughput measurement follows from these CPU checks. A reserved-device forward/backward/checkpoint-replay admission smoke remains necessary before the fresh family search. Root schedules it once an eligible slot is free within the two-GPU cap.

The independent integrity reaudit under research/reviews/integrity_fix_reaudit_v1 remains a hash-bound historical source snapshot before this provenance/numerical-policy extension. The new source hashes and CPU evidence are recorded separately in development_execution_v1.json.

## Runtime audit repair v2

The v1 source receipt and its matching bytes are preserved in the independent audit original_snapshot directory. The current v2 repair rejects policy drift after validation and replay, caller mutation of captured settings or control dictionaries, and exact test/confirmation path components before any read. Canonical and prior-art runners register their settings with the composed captured-source snapshot before device admission. The root CPU integration suite passed 392 tests in 294.18 seconds. These synthetic checks do not establish CUDA admission, ancestry performance, completed fresh preparation or final selection. Current bytes and the test log are bound by development_execution_v2.json. The independent audit retains its original accepted reproductions and separate repaired replays.
