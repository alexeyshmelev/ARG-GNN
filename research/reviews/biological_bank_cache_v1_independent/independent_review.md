# Independent review: biological bank cache v1

Verdict: qualified for source and fresh handmade CPU checks only. No concrete source/handmade blocker was found in the sealed bundle. This review is not an actual-device profile, a fit admission, or evidence of speed or accuracy improvement.

Sealed inputs:
- Source: src/arggnn/biological_bank_cache_v1.py, SHA256 1582c493c54e9b75dfa65c910fd6cec47fa11622ca4ae8bbb30564391a82caec
- Config: configs/biological_bank_cache_v1.json, SHA256 24a74c529fc154fe8059c7ec6c3523fc688b443529ccdd00889d89a9509bfb45
- Author receipt: research/reviews/biological_bank_cache_v1_author/author_receipt.json, SHA256 179e609087eb5f6617e969e96019180c793cb2d3e8aa7d056188f860f9a87b8a

The source replaces both biological_mechanism_controls_v1.load_bank and biological_candidate.load_bank inside installed_cache_v1. The latter is the global used by validation prediction. Both aliases restore on success and exception. Nested or preexisting wrappers fail closed. All prepared fields come from the unchanged original loader and its original .to operation. The wrapper performs no feature, partition, data, architecture, loss, optimizer, or training-loop transformation.

A full frozen.checked(tile['bank']) SHA256 runs before either CPU or GPU hits. A changed trailing file byte fails even with a resident entry. File identity checks additionally reject same-byte inode replacement. Cache keys bind parent/tile identities, complete tile metadata and bank reference, scientific config, parent split, dynamics/reference/query roles, inferred node IDs, and the original coordinate grid. The original loader still validates the serialized schema, complete field provenance, original IDs/features, float32 features/operators and float64 coordinates.

resident_bytes_v1 counts unique tensor storages. transfer_upper_bytes_v1 conservatively accounts every tensor occurrence and the logical dense size of expanded tensors. The source stores whole CPU entries and whole GPU LRU entries, evicts GPU children when their CPU parent is evicted, and uses the original uncached transfer for entries that do not fit. Shipped retained-tensor caps are exactly 64,000,000,000 CPU bytes and 32,000,000,000 GPU bytes. These are retained-cache tensor limits, with explicit separate root headroom for decoded/uncached banks, caller-retained objects, model/optimizer/autograd/runtime, Python objects and allocator-reserved memory.

Admission binds source/config/review/scientific references and a fresh memory reservation. Profile and fit require separate phase records, and fit additionally requires the bound measured profile/parity gate. Original runner arguments are forwarded verbatim to the original parser and main. Existing source/config validation, storage checks and model/runtime admission remain in that original runner. Root remains responsible for generating and validating these capabilities and the actual device lease.

Independent evidence:
- Fresh one-graph handmade fixed bank: boundary 280, dynamics 184, references 96, actual query flags 120, selected features 47, branches 3, anchors 32 and two readout positions. This is approximately 4.38 MB of resident tensors, not a scaling fixture.
- All 11 check groups passed. They cover complete nested field/dtype/device/ID/feature/position equality, SHA hit behavior, CPU cache/LRU/oversize, storage accounting, simulated GPU LRU/oversize/CPU cascade, role/config/schema/dtype/device failures, tensor mutation/error cleanup, immutable file identities, and capability phase/source/memory/review/gate rejection.
- All seven unchanged original default models passed exact complete-field/logit/all-parameter-VJP/clipped-AdamW-state/RNG parity on CPU with a disposable target-free mean-logit objective.
- Two original-runner spy executions passed verbatim CLI forwarding, alias/sys.argv restoration, and receipt writing after cleanup, including a deliberate runner exception. Admission and original runner were mocked for these two dispatch tests only.
- The GPU transfer and LRU checks explicitly simulated CUDA transfers on CPU. They establish branch bookkeeping, not actual CUDA dtype/device/numerical parity.
- The author receipt hash, its 18 listed artifact hashes, and 26 original source pins all match.
- Negative fixtures and logs remain in this directory. There was no unexpected failed test in the independent attempt.

All work ran over BatchMode SSH on anogena in this repository, with scripts/project_env.sh sourced and .runtime/env/bin/python. CUDA visibility was explicitly empty. No real banks or ancestry targets were opened, no device/live-ledger queries ran, and no actual profile, fit, GPU launch, installation, Git, scheduling, or source/config edit was performed.

Before any cached fit, root still needs the separate actual-device target-free loading/parity/memory profile and its measured gate. Full-file hashing remains in measured timings. Single-tile timing cannot establish whole-loop utilization. PyTorch ordinary mutations are guarded by version/storage metadata; arbitrary .data or external-storage writes are outside the documented read-only consumer contract.

