# Official ARGMix optimizer variation v1

This is a blocked source-only plan for training seeds 101, 202 and 303 on the same completed biological ARGMix condition. No override, fit, inference, GPU work, package installation or Git action is implemented/executed here. The completed seed50 run remains the baseline and must not be repeated.

The repeat reason is optimizer/RNG variation of the existing biological condition. It is not new biology, extra independent parent replication, or an April-paper numerical reproduction. Model initialization, native shuffled minibatches and dropout are stochastic consequences of the training RNG. Their native algorithms and all scientific settings remain fixed.

## Current source blocker

The pinned release is [c2481248eb0de5fda659d68f59c51e0e27becd20](https://github.com/AI-sandbox/argmix/tree/c2481248eb0de5fda659d68f59c51e0e27becd20), author source tree SHA256 4fbcc41801027983a9654920d143f053d5b74ffea020b751a13757c0ae327ae8. All 86 source files and their ordinary execution-copy counterparts match. The frozen owned module remains SHA256 4a6c44dfd81ba3f75e74c5287a4b55776bb9fe1612732c457748643cd75d7b1f.

Native [src/train_joint_model.py:58-64](https://github.com/AI-sandbox/argmix/blob/c2481248eb0de5fda659d68f59c51e0e27becd20/src/train_joint_model.py#L58) exposes only --config, --train-pt, --val-pt and --checkpoint-output. At [line305](https://github.com/AI-sandbox/argmix/blob/c2481248eb0de5fda659d68f59c51e0e27becd20/src/train_joint_model.py#L305), main unconditionally calls seed_everything(50) before building the model. Calling seed_everything(101) in a parent first would be overwritten.

The public helper accepts a seed and seeds Python, NumPy, Torch and visible CUDA RNGs, setting native deterministic CuDNN options. The native entrypoint does not expose a configurable argument for this helper. The frozen owned bridge likewise enforces source_seed=50, initializes its reference model at seed50, records source_seed50, and dispatches the fixed author CLI.

joint_model.train.sample_seed=0 is separate archive/subgraph-sampling bookkeeping. In this prebuilt-archive trainer it is read/logged. It is not the model/optimizer RNG control and must remain zero. Do not alter it, repeat seed50 under a new label, pre-seed the caller and claim variation, or silently intercept native execution.

configs/official_argmix_optimizer_variation_v1.json is an unimplemented amendment plan. It is not an accepted input to the current frozen executor and grants no runtime/storage admission.

## Explicit future entrypoint-only RNG option

A separately versioned driver may be proposed for independent review. Its label must be **author implementation plus disclosed entrypoint training RNG override**. It must never be described as unchanged native-source execution.

The minimal proposed runtime semantic change is to intercept the one exact imported seed_everything(50) binding at the native trainer entrypoint, forwarding its argument to the requested seed101/202/303 through the unchanged native helper, then restoring the binding in a finally block. Assert the pinned trainer AST still contains exactly that single literal50 call and no seed CLI. Reject unexpected calls, extra reseeding, an unknown requested seed, or source drift. Do not replace the helper's RNG/CuDNN semantics.

This preserves the native trainer body, model, AdamW objective, masking, batch order algorithm, dropout algorithm, validation and selection. The binding changes execution semantics and therefore needs explicit source/config/provenance review even if author file bytes stay unchanged. No such driver or binding override is implemented by this task.

The frozen stock_train wrapper cannot be reused blindly: its initial-state capture and fitted source_seed fields are still50. The new driver must capture each requested seed's actual initial state, unused-classifier state, selected state and native selected epoch/loss, and record the applied RNG override. The unused classifier must remain unchanged relative to that seed's own initialization. It must preserve the source's checkpoint and weight-change checks. It must produce fresh fit/inference receipts under the amendment, not rewrite the baseline50 receipt.

The source alteration is scientifically justified solely by the intended optimizer contrast. Every non-RNG scientific and numerical rule must remain identical. Root must separately authorize implementation before any actual execution.

## Exact same prepared products and scientific contract

Reuse the reviewed v3 compatibility amendment and the existing original prepared products in place. No bulk preparation or archive copy is requested.

| Binding | SHA256 |
| --- | --- |
| Base reviewed v3 config | 25829cdb811da918db4cb7fad5ee607a44610bd1f04bc40dbe2f5564a6d58eb2 |
| Original native_prepared.json | 2c15bd908ae89b81e1a8c9d07809e82c46d6b1b41a2031ac93013dc54534961c |
| Original observed_frames.json | b8de4b503bfebcf7eb71b6861879826f24731deb37aabe68b5283f6f07f07531 |
| Completed seed50 compact summary | 29b7122455dd692875e9ee641ad591cb69e33d60629e3aad532af1dd7ea7ede3 |
| Completed seed50 stock_fit.json | d148a4b9c95d7d590e7346b3af1d10f9a211bd86aa89ff64a3cb3c6b6254b02b |

The inherited train.pt and val.pt are each 161466044 bytes by stat, with the existing manifest's SHA256 references 8985b513565000bd4add27d9d2ba93281200c958ec471866bb9e3a0ec1a09c59 and 4c560fe9affa63d0c930d07faa68300e131dcd34318cea05e2db232f79072147. This task did not open or independently rehash array contents.

Keep the same four observed Relate parents, original support, original panel dates/haplotype ordering, C4 order ANA/CHG/WHG/EHG, 96 pure references, 64 mixed unlabeled references and 120 queries. Keep two TRAIN and two VAL parents, no TEST. TRAIN gradients and native VAL-only checkpoint selection retain their original label scopes.

Keep int16 TMRCA quantization, native nearest-reference selection/eligibility, train drop_top_m_nearest=1 and VAL/inference=0, chunk1024/stride512 plus exact-end overlap, native model dimensions, GRU/transformer architecture, objective and loss weighting. Keep smoothed masked unweighted native VAL batch-mean selection, 20-epoch cap and patience2. Keep native base export, joint refinement and arithmetic overlap averaging of chunk-softmax probabilities. Keep full common physical support for all 120 queries and the unavailable/no-denominator-reduction policy.

Report all planned seeds using each run's unchanged native checkpoint selection. Do not select a cross-seed winner using external physical-span metrics. Seed50 is existing descriptive evidence, not a new replicate. Variation across optimizer seeds cannot increase the number of independent biological parents or establish superiority from this small development pilot.

## Admission and minimal meaningful review

Existing native build and CPU FFI receipts can be reused only if their exact runtime/native-library/binary hashes and admitted scientific bindings still match. The old GPU qualification remains factual evidence of the unchanged model/Rust kernels under its original v3 source/config. It alone cannot certify a new RNG entrypoint's execution/provenance. A composite amendment admission must bind the new driver, config, seed, original guards and any required new seed-specific fixture evidence, without relaxing old checks.

Before implementation admission, an independent AST/diff audit must verify unchanged author trainer/model/loss/data/quantization/reference/overlap/refinement/selection code and the one explicitly allowed entrypoint RNG change. Pure handmade/mocked checks should establish one native seed call is forwarded exactly once for each 101/202/303 and restores the binding on success/failure, reject unknown/multiple calls, verify seed-correct initial-state/provenance recording, and reject stale admissions. This does not require real tensors or model fitting.

All existing build/CPU/GPU qualification, module/config/source inventory, typed stage, borrowed prepared-product, library mapping, actual CUDA UUID lease, titled process, NVTOP, immutable input/output and final weight/selection/seal/replay checks remain required. Root owns scheduling and integration. Use fresh per-seed directories under experiments/results/official_argmix_optimizer_variation_v1. No worker is launched by this plan.

## Conditional storage reservation

Sizes were obtained by stat only from the completed seed50 directories, including metadata/logs/cache files.

| Completed reference | Bytes |
| --- | ---: |
| Fit | 4008383 |
| Inference and external summary artifacts | 102527732 |
| Fit + inference for one seed | 106536115 |
| Three analogous retained outputs | 319608345 |
| Existing GPU qualification | 7217239 |

Compression and metadata string lengths may change with weights, so these observations are not future upper bounds. The proposed enforceable cap is 1000000000 retained bytes per seed, or 3000000000 total. Reserve 5000000000 bytes once for the one active launcher/scratch headroom, 50000000 for new amendment qualification, and 10000000 for source/plans/manifests. Total conditional additional reservation: **8060000000 bytes (8.06 decimal GB)**. Execute seeds serially, at most one active seed job. No source/runtime reinstallation or duplicated prepared archives is included.

Root reported a conservative projection of 439875000000 bytes including existing reservations and the 8GB CPU runtime/handmade reserve, with further admission limited to 10100000000 bytes. Adding these caps would project 447935000000 bytes, below 450000000000, with 2065000000 bytes before the warning threshold. This arithmetic uses root's reported projection, not a fresh live filesystem measurement. All reservations/external reserves must be reconciled through the root-owned ledger before actual admission, with exact atomic bindings and per-stage local accounting. Stop on any cap or projected 450GB breach, preserve partial outputs, and retain the 500GB hard limit. Do not borrow the CPU runtime reservation for actual observed inference.

The active storage reservation is **zero** while the seed API/implementation is blocked. The plan is not a reservation, approval or claim that the existing fixed50 launcher already enforces these new per-seed caps. Root must implement/audit the amendment's guards and confirm current headroom before launching.

## Author audit scope

The author audit records source AST/CLI evidence, unchanged 86-file author and execution-copy inventory, exact small config/prepared/fit/qualification/summary bindings, and stat-only sizing. One inference receipt JSON was read for runtime/build pointer metadata before its size was known (30411278 bytes); no referenced HDF5/NPZ/tensor/checkpoint file content was opened. Subsequent sizing used stat only. No native import, model construction, optimizer step, inference, scoring-array access, GPU query, package installation or Git occurred.

The deliverable is the exact source blocker plus a separately labelled option requiring future implementation and independent review. It preserves the completed seed50 run and all original data/model/runtime products.
