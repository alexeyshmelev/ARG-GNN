# Native ARGformer r002 continuation

The r001 queue stopped before native model/probe execution because root runtime_GPU2 qualified_GPU_UUID included GPU-, while the original lease and actual CUDA receipt use a bare UUID. The corrected proposal uses 976a5cd3-df12-2535-fa51-e3ea063688ba and remains inactive. Root must create an active copy and pass its exact path/SHA to the runner.

The runner preserves native source/config, original stage proposals/registrations/source review, all280 transductive MLM and full4096 probe. It reuses the four existing20GB reservations. Old runtime_GPU2.json and every r001 artifact stay immutable because reservation consumer bindings pin the old packet. No ledger writes occur.

Each parent runs its own probe then fit, then CPU retrieval; retrieval may overlap the next parent's GPU stages. TRAIN000 can yield real accuracy before all four parents finish. Fresh r002 output/log paths come from unchanged admission_bindings. short_tmp is recomputed from that function and is intentionally the same approved per-parent scope across runs.

Inspect without launch:
    .runtime/env/bin/python -B research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v4/run_parent_continuation.py

Root launch after activating a corrected runtime packet:
    .runtime/env/bin/python -B research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v4/run_parent_continuation.py --execute --runtime ROOT_ACTIVE_CORRECTED_RUNTIME_PATH --expected-runtime-sha256 ROOT_ACTIVE_RUNTIME_SHA

The existing project GPU launcher selects physical GPU2 exclusively and enforces the existing process/storage limits. The supervisor records stage starts/terminals and owns child watchdogs. GPU3 workers are unaffected.
