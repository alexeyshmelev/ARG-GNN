# Executable persistent-prefetch current980 C3 GNN successor

Source-only packet, 15 October 2026. This is a separate executable root route for the unchanged **20 independent2Mb TRAIN parents,30epochs,600 complete-parent FP32 updates and6,000,000 focal exposures**. Native MLM is not a dependency. The original current980 packet, its installed canonical shim, original trainer/hooks and every frozen provider file remain unchanged.

The only source outside this new packet added here is `src/arggnn/american_c3_gnn_prefetch_current980_exec_v1.py`. It has exact byte parity with `canonical_dispatch.staged.py`, SHA256 `a57f4dd122a0cf612763c16b06afab20fce76c0067cbb3a298f39cc43ab071c1`. No existing source file was edited.

## Actual routes

`ROOT_COMMANDS.sh` supplies default-inert help plus explicit `bind`, `admit STAGE`, `launch STAGE` and `close STAGE APPROVAL SHA`. The Python driver has equivalent flags. The actual canonical module is `arggnn.american_c3_gnn_prefetch_current980_exec_v1`, accepted through the unchanged GPU45 launcher. CPU routes enter `worker_entry_v1.py` directly.

GPU stages: train and predict_val_000 through predict_val_004. CPU stages: plan_val_000 through plan_val_004, matrix and score. There is no TEST or resume route. Fresh outputs use `experiments/results/C3_GNN_prefetch_current980_STAGE_v1`; packets use this directory's `admitted_STAGE_v1`; TMP uses `.runtime/tmp/c3gnnprefetch980/STAGE`. Original outputs and the live/completed CPU plan are preserved.

Root must externally hold `research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock` for the entire admit or close command. Neither wrapper nor driver takes that cooperative outer lock. Launch uses the existing dedicated-session root Popen supervisor, original GPU launcher, original lease/title/UUID/runtime checks, current980 reservation and writer. Reserve/close call the original unwrapped guard. Only read-only launch/worker boundary checks use pinned memo2. Hot resource callbacks have no journal/accounting/storage scans.

## Late-bound requests and caps

Copy the desired `request_STAGE.inactive.json` and `caps.inactive.json` into fresh root-owned actual files. Fill actual references and keep active=false. Set request.caps_template_ref to the actual filled inactive caps path/SHA. TRAIN also names an inactive scientific_training_config_ref; its content must match the original reviewed training config except CPU_observation_cache_bytes, which is operational and may be raised to fit a real block. No other scientific field can change.

All root reservation/product/TMP/aux/log/RSS/deadline bounds are intentionally null in the inactive caps and must be supplied. Positive root caps retain CPU RSS<=512,000,000,000B, CUDA reserved<=50,000,000,000B and deadline<=86,400s. Product+TMP+aux+packet caps must fit the reservation. GPU4or5 and actual UUID are separately bound; CPU phases require null device/UUID.

TRAIN requires exactly six `observation_prefetch` fields:

| Field | Binding rule |
|---|---|
| num_workers | Explicit integer2or4 |
| max_block_bytes | Covers the declared feature planning bound of every actual frozen plan |
| worker_rss_cap_bytes | Positive integer greater than max_block_bytes |
| aggregate_rss_cap_bytes | Covers main+all workers and the conservative planning sum; <=root max_cpu_rss_bytes |
| task_timeout_seconds | Explicit number >0 and <=3600 |
| drain_timeout_seconds | Explicit number >0 and <=30 |

The cache must cover an entire block. Required planning inequality:
`cache_bytes + (num_workers+2)*max_block_bytes + num_workers*worker_rss_cap_bytes <= aggregate_rss_cap_bytes`.

Other stages require observation_prefetch=null. Their original VAL loaders remain0workers. No bound is filled from an unbound auxiliary file or silently chosen default. Actual caps/config refs enter request, admitted source/input closure, operational config and model protocol. Inactive placeholders cannot execute.

Conditional commands from repository root:

~~~sh
source scripts/project_env.sh
bash research/reviews/C3_GNN_prefetch_current980_root_20261015_v1/ROOT_COMMANDS.sh bind ACTUAL_FILLED_REQUEST_PATH ACTUAL_SHA
flock research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock bash research/reviews/C3_GNN_prefetch_current980_root_20261015_v1/ROOT_COMMANDS.sh admit train
bash research/reviews/C3_GNN_prefetch_current980_root_20261015_v1/ROOT_COMMANDS.sh launch train
~~~

These are executable future root actions, not actions performed by this packet.

## Persistent pool and exact science boundary

`persistent_workers_v1.py` and `persistent_provider_v1.py` are separately owned operational successors. The original provider/core/bootstrap/API and their actual tiny transport receipt are frozen. One lazy spawn pool of2or4 workers persists across20 observation seals and600 complete-parent updates. It never restarts or retries after failure.

Every parent session binds exact registered parent ID, immutable plan ref, full actual optional observation seal, block count and a fresh context serial/hash to every worker. Every read request/reply authenticates that context, parent, plan, physical block and task serial. At most one task/result per worker is in flight. Demand delivery remains exact indexed access, including reverse/repeated/nonmonotone checkpoint demands. Speculative results are validated before discard.

Each CPU worker runs the original `SealedObservationProviderV1` and its unchanged Dataset/_read checks. The parent class subclasses the frozen prefetch provider and inherits its read/get/seal/hash logic. All original feature hashes, source checks, every tree/sample/focal and exact-once10000-focal observation seals remain mandatory. All20 observation seals still precede the first private TRAIN resolver.

At each original provider.release boundary, all pending replies are checked/discarded, every worker clears its active Dataset/mmap/cache state and acknowledges the exact context. Worker processes stay alive. The next parent cannot bind before that barrier. CPU cache state clears through the original release implementation.

`trainer_prefetch_v1.py` copies the reviewed optional-factory trainer, relocates imports and binds its own actual source path. Its original complete-parent step-call AST is identical. All eleven other trainer function ASTs remain identical, including checkpoint/progress/RNG/optimizer validators. `execution_hooks_prefetch_v1.py` imports that copy explicitly and preserves original target joins, model, recurrent scans, losses, seeds,30epochs,600updates and scheduled epochs5/10/15/20/25/30. There is no graph regrouping, reduced corpus, shortcut, mixed precision, C4 initialization or changed metric.

`metadata_contract_v1.py` relocates the exact registry-validator function AST into a Torch-free metadata module. This prevents root bind/admit validation from indirectly importing Torch through the target module. Worker entry installs the original authenticated abstract-IPC bootstrap before Torch/PyG imports, after actual device/lease/title checks for GPU work. Spawned readers hide CUDA and install the same bootstrap before Torch.

`tracking_factory_v1.py` deep-copies the persistent factory's actual execution binding and adds its own source. Worker entry retains that factory and calls finalize in an outer finally with the actual training outcome, including hook/failure-writer exceptions. Success requires620 checked parent releases and one final pool drain with actual zero exits. `observation_prefetch_all_worker_drains.json` contains all parent-release metadata, actual final PID/start identities/exits, forced/discarded counts and global drain. Missing/failed drain aborts; no fallback occurs.

Only one pool means2or4 worker identities, rather than1240/2480 repeatedly spawned workers. The original256-PID guard roster limit remains unchanged. The outer supervisor combines actually observed process identities with the actual final worker drain manifest and still drains helper/adopted processes. An explicit bounded post-main-wait grace permits legitimate multiprocessing helpers to finish before classifying a drain failure. Every admitted original storage scope remains fully measured.

## Original VAL route and closed old plans

Prediction still uses every original30Mb tree/site/node/sample and the native10000 focal identities, unchanged forward/reverse scans, six scheduled checkpoints and0-worker VAL loaders. Matrix requires all30 genuine seals. Separate score reaches original5VAL truth only after that barrier and selects greatest mean per-parent macroF1, earlier epoch on exact tie. TEST remains deferred.

A predict request now includes `bindings.plan_stage_evidence` with exactly route, completion_ref, actual_wait_ref and close_receipt_ref. Route is current980_zero_worker or prefetch_current980. Validation requires exact stage/output paths, driver pin, genuine terminal0/drain, the exact observation-only result and plan ref, an actual current980 close receipt, root terminal consumption and capability revocation. Thus the completed old VAL000 observation plan can be reused only after root has genuinely closed it. No old GNN training or native MLM completion becomes a new model dependency.

## Verification and limits

Actual checks: all10 packet Python files compile; three default-inert entrypoints return0 without importing Torch/PyG; root contract/hooks import remains Torch-free; all50 transitive source refs resolve; original step call and eleven other trainer function ASTs are identical; original registry-validator AST is identical; all54 pre-existing regular source/template pins checked remain unchanged; the new canonical shim matches staged bytes. Inactive templates retain null actual refs/caps. Details are in source_checks.json and source_inventory.json.

A tightly bounded read-only peer source review found no blocking context/physical-index/lifecycle API issue and confirmed the required outer-finalize/manifest/fresh-run/supervision caveats. It ran no jobs/tests and edited no source.

The original tiny Torch/PyG fixture qualified the abstract shared-tensor transport mechanism. **The new persistent/context lifecycle, actual full Dataset transport, full600-update fit, aggregate memory, gradient equivalence, throughput and speedup remain unexecuted/unqualified.** No extra test campaign was run. RSS checks sum historical parent/worker peaks conservatively; releasing dataset state does not lower those peaks.

No bind, admission, payload/model/optimizer/GPU execution, accounting mutation or Git action occurred. Root owns actual metadata/cap review, admission, execution, direct-wait consumption, close certificates, ledger/reporting and Git.
