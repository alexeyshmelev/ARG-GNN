# Independent source review: native VAL000 GPU0 monitor route

2026-10-16. **No source blockers found for this operational component.** Future namespace, failure handover, source/direct-binding integration and root admission remain separate requirements. This review grants no launch or retry authority.

Reviewed handoff: research/reviews/native_VAL000_GPU0_monitor_operational_source_20261016_v1/source_handoff.v1.json, SHA256 e3aedf427998341e35091aae332751b9ee64f3fee940f730393b5a82f560802a.

Reviewed actual source: native_gpu0_monitor_route_v1.py in that directory, SHA256 ce23a4c3e97dd02854a53c25dc3b9aa3762db191d51a513e55c1c839adc3481e. The handoff has the exact filename. There is no separate combined_native_gpu0_monitor_route_v1.py.

## Independently verified source route

- The unchanged pool-native runtime loads the original root driver privately. wrap_original_runtime_v1 intercepts only that exact source record, then patch_private_driver_v1 changes its one launcher-script expression.
- Reconstructing the original driver launch function with that exact replacement produces the project Python command followed by this adapter, --route-role launcher, --, and the original GPU/job/storage/source arguments and original entry declaration.
- Reconstructing the launcher main function yields this adapter with --route-role supervisor and the unchanged original module arguments. The original module/input/storage/GPU0/lease validations execute before that spawn.
- The supervisor bootstrap loads the original supervisor under the exact name supervisor_v1, patches only its native-child script expression, then calls the original entry. That entry imports the already loaded supervisor_v1.
- The generated child command retains the admitted native executable, -B, -s, all original child-input path/SHA arguments, and now inserts this adapter's --route-role child. The original child main is still the actual scientific entry.

Each of the three replacement anchors occurs exactly once in the bound original function. All reconstructed functions parse and compile without execution.

## Monitor identity and retained guards

Adapter lines118-123 install the actual current module as sys.modules["gpu_monitor_v8"] and reject a distinct pre-existing alias. Launcher, supervisor and native-child roles install that alias before original entry/context imports. Both parent and native-child context factories use their unchanged _fast_guard import of gpu_monitor_v8.OwnedGpuMemory.

OwnedGpuMemory constructor lines17-37 requires integer physicalGPU0 and the exact canonical GPU0 UUID before loading NVML. Original actual UUID/index verification remains. Independent AST comparison confirms ProcessInfo ABI and the _ok, query and owned_bytes methods are identical to the historical monitor. Fresh-query bounds/retries, per-process owner restriction, owned PID/start/title validation and memory summation remain intact. No GPU4/5 query path was added.

The original driver, launcher, supervisor, child, context and runtime source hashes all match the handoff. Their disk files remain unchanged. Only private function namespaces are adapted. Their storage, expiry, title/lease, memory, actual wait and owned drain operations are retained. The capability reference added by the adapter survives the original environment-copy paths into both subprocesses. Each role authenticates it before dispatch.

## Scientific and execution scope

Original input SHA256 52242fe04f8f944dccbc81bf12940ad311d0d80f300e3cdd099be5e1748bc5d2, storage SHA256 49d4097f8ceb367f1b8bd7a99649b0ae90db1045515953a0059b2e42bbf3eb81 and caps SHA256 2f9763d36a58e45f6d0f06b78c2bead18041e2fc88aaff084845414ae99d1b7e independently match their actual bytes. The scientific charge remains7516192768bytes, with expiry2026-10-18T23:41:32.476768+00:00.

The original child preserves full native corpus authentication, tokenizer107144, original author objective/configuration, fresh optimizer, seed17, batch512, BF16, sequence length128, five epochs and227990updates. It requires genuine Trainer and checkpoint optimizer counters at completion. No numerical training logic is changed by the monitor patch.

This adapter does not permit failed-path reuse. The unchanged driver still rejects existing output, worker_aux and TMP scopes, and the supervisor/native child retain their fresh-path checks. Therefore it cannot simply rerun the failed admission as currently stored. Root must provide the separately reviewed fresh namespace and honest failure lineage while retaining original charge, expiry and artifacts.

No current C3/controller observation or change, project-module import, native/NVML/process/lease probe, numerical run, qualification campaign, scientific payload read, reservation, launch, signal, installation or Git action occurred during this review. The review is source and bounded existing metadata only. No runtime success, training speed or ancestry accuracy follows from it.
