# H043 root context, CLI and admission-driver source proposal

This is an inactive future-source proposal. No live guard was imported/configured, no accounting state or journal was read, no admission or lease was requested, and no process/context/driver/CLI or GPU was executed. No raw observations, checkpoints, labels, prediction arrays or metric files were read. The only observed evidence read was root's explicitly approved counts-only manifest, SHA5ea2eedbb44fe34bf89083e15387533b8886cf0230e10f72bf30241c1893e9f8.

The existing finalized H043 scientific API and prototype remain unchanged. H040/H042 sources and all current guards, journals and launchers remain unchanged. H043 is a future candidate independent of H040/H042 VAL scores, with threshold0.001 already prospectively fixed and the original checkpoint selection preserved.

## Compact source reuse

- root_common_v1.py imports and reuses the exact pinned H042 pure root helpers for canonical refs, metadata reads/saves, policy environment, deduplication and current guard loading. Importing the H042 module itself is inert.
- load_guard_v1 defaults to consumer=False. Root reserve/close operations retain unmemoized originalv4 functions, and reject an already wrapped v4 module. Only launch consumer checks and the future worker call consumer=True to install finalized memo2 against exact originalv4 function identities. The original credit-v3 authority, source and journals remain unchanged.
- context_factory_v1.py reuses the H042 reader, source closure, active logical-job/lease/title, configure_fit, guarded writer and lightweight resource patterns.
- root_driver.py reuses the H042 reservation/launcher/direct wait and finalized frozen-evaluator supervisor pattern: bounded read1 stdout capture in RAM, actual wait in finally, and terminal-only guarded log writing. It asks root for an explicit physical4 or5 at admission. Both H043 arms run in one scientific worker on that one GPU.
- src/arggnn/H043_GPU45_transport_worker_v1.py adds the thin active-capability CLI missing from the scientific API. The old review-folder worker_cli.py is retained as superseded source evidence and is not launched or reserved. Default invocation is inert. It checks the complete pinned source closure, loads the pinned root context and calls the unchanged scientific run_context.
- No H040 predecessor, completion or score is required. Evaluation requires the actual H043 paired training completion and matching successful direct launcher wait/launch identity.

Future root entry points are explicit --admit train/evaluate --gpu4/5, followed by --launch train/evaluate. Evaluation additionally takes exact path/SHA refs for train completion, train launch and train actual wait. This handoff has not run any entry point. The inactive input previews contain null GPU/runtime/storage fields and cannot execute.

## Scientific and provenance binding

The config preserves the finalized H043 API's exact original32 TRAIN/eight VAL roster, H038 aligned seed101 teacher-free warm initializer,30passes/960updates per arm and5/10/15/20/25/30 checkpoints. The actual warm completion/checkpoint refs remain in the config, source manifest and scientific validator. No warm bytes were read during proposal authoring.

The context independently requires original H038 template registration/generation/cache and all40 observation-entry/source refs. Fresh phase target bindings must exactly equal the matching original H038 TRAIN or VAL subset, with explicit TRAIN_included/VAL_included/outer_included flags and original generation receipts. No outer bindings are accepted.

The57-file runtime source manifest extends the frozen H042 source closure with finalized H043 mechanism/API, actual CLI/context/driver/common sources, memo2 and originalv4, both capacity source generations, actual v9 consumer marker, marker repair helper/authority, current credit-v3 guard, policy and GPU45 launcher. Exact transitive metadata and direct reservation bindings are checked. The existing64-direct-binding limit stays in place. Count metadata is a direct reservation binding. Root still owns current resource admission, exact actual launch/wait evidence, scheduling and Git integration.

Current authority pins:
- credit-v3: ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef.
- policy: a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85.
- actual GPU45 launcher: d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d.
- finalized memo2: c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364.
- originalv4: a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47.

The existing GPU45 launcher remains unchanged and uses its original guard behavior. Memo2 is installed only for checks in the new future launch supervisor and worker processes, before their consumer checks/threads. Root admission loads the guard with consumer=False and does not install memo2. No reserve or close mutation uses memo2. It is not installed into an existing live process or the separate unchanged launcher process.

## Proposed caps and count-derived bounds

New GB values below are explicit decimal bytes. The original H042 GPU residency cap is retained exactly.

- CPU observation banks16,000,000,000 bytes.
- CPU target mass4,000,000,000 bytes.
- RAM transport cache8,000,000,000 bytes.
- Complete per-parent transport payload1,000,000,000 bytes.
- GPU parent residency8,589,934,592 bytes.
- GPU transport1,000,000,000 bytes.
- Frozen RAM predictions8,000,000,000 bytes.
- Existing root RSS cap512,000,000,000 bytes and CUDA reservation cap50,000,000,000 bytes.
- Separate fresh output/auxiliary scope cap200,000,000 bytes per phase, including a4,000,000-byte retained launcher log.

The payload cap remains below the scientific source's1,006,632,960-byte limit. No science was changed to fit a cap.

Root's count metadata reports max native holds1026 and max nodes1,952,478, with952 haplotypes. The686-query count is explicitly from the fixed original protocol, not a new measurement. The maximum complete compact transport upper bound is35,134,068 bytes using6N+24*952*(T-1). All32 upper bounds together fit the proposed8GB transport cache. All TRAIN target masses total335,108,256 bytes by T*686*4classes*4bytes. The all96 native prediction matrix count-bound fits8GB.

metadata_forecast.json contains per-parent calculations and these scope bounds:

TRAIN:
- All32 observed recipe JSON upper bound9,164,082 bytes.
- Twelve checkpoint archive allowance12,582,912 bytes.
- Retained log4,000,000 bytes.
- Fixed seal/metric/history/filesystem allowance16,777,216 bytes.
- Total42,524,210 bytes.

Evaluation:
- All8 recipe JSON upper bound2,328,266 bytes.
- No new checkpoints.
- Retained log4,000,000 bytes.
- Fixed seal/metric/history/filesystem allowance16,777,216 bytes.
- Total23,105,482 bytes.

Each recipe forecast uses the exact nine boundary audit key names with integer lengths bounded by declared N/T/952, the three per-tree flag audit lists, and32KiB fixed hash/geometry/proof metadata. Raw event/flag/prediction arrays are never serialized. The1MiB checkpoint allowance covers FP32 model weights, independent AdamW moments/steps and archive/provenance overhead for30,788 parameters. The fixed16MiB allowance covers fixed-count96 seals,12 primary score packets, two selected diagnostic packets over eight parents/seven populations, histories/setup/completion and filesystem overhead. Selected diagnostics omit per-query reports as in the unchanged original scorer.

No actual count-derived bound exceeds200MB. Both future admission and context recompute the forecast and fail before a build if the declared bound exceeds200MB. They do not increase reservations or modify science automatically.

## before_transport_build and continuous qualification

The context supplies before_transport_build(pid, bank, declared_raw_bounds). It:
1. Runs the original root runtime/lease guard.
2. Recomputes the raw bound from the actual original CPU bank dimensions.
3. Requires equality with the scientific API's declaration and exact frozen per-parent root count metadata.
4. Compares the complete6N+24P upper bound with payload, CPU cache and GPU payload caps.
5. Reads current-process Linux RSS and requires current RSS plus an explicit complete construction planning allowance to stay within512GB.

The scratch planning allowance is2058N+128h+48P+6N, covering a conservative Python/container allowance and simultaneous event chunks/concatenation. It is a planning allowance, not an allocator/RSS theorem. The original peak-RSS and CUDA guard remains authoritative. Every build/rebuild checks again, and any mismatch stops without subsampling or replay.

No standalone FP64 equivalence run or pilot campaign was added. The first real full-parent FP32 steps remain optimizer step1 of each arm's continuous30-pass run, as implemented in the finalized scientific API. Full-parent forward/backward and GPU memory qualification remain pending. Evaluation remains a separate future phase with all96 immutable RAM seals before any VAL truth.

## Mechanical qualification and next integration

source_check.py passed nine stdlib-only checks plus a targeted original-prefix preservation check: all source compiles, exact57-file source closure and scientific config, count-derived output/raw/cache/target/RAM caps, all40 dimension-only prebuild checks plus rejection of cap/count/RSS drift, inactive inputs, one-GPU/direct-wait/memo/prebuild source contracts, unmemoized default admission versus explicit consumer-only memo, finally-wait/terminal-log ordering with no live-child scope scans or log writes, and the exact frozen launcher arggnn-module/source-path/direct-binding contract. No Torch or credit-v3 module was imported. Pure preparation checks used dimension-only stand-ins and injected RSS values, not real graph arrays or process reads.

Root must review/integrate this source proposal, choose/schedule4 or5, retain current storage headroom and exact registered H043 provenance, then separately admit the full training scope before any real build. Guard/lease/launcher/context/runtime behavior is unexecuted in this package. The source proposal makes no accuracy, runtime, GPU-fit, significance or novelty claim.


## Pre-admission operational review fixes

Root identified three operational defects before any admission. The scientific API, protocol, counts, threshold and design were unchanged.

The supervisor now retains at most4,000,000 stdout bytes in RAM while draining all child output. No guarded charged-scope scans or launch.log writes occur during child execution. The actual proc.wait and stdout close are in finally. Only after that wait does the supervisor perform a guarded bounded log write and save actual_wait.json. A terminal logging error is recorded while retaining the actual child return code. A supervisor error is separately recorded, causes a nonzero supervisor result when the child returned0, and is rejected by later TRAIN lineage validation. The receipt explicitly makes no descendant-drain/no-live-writer claim.

The guard loader now defaults to consumer=False, returns the original guard without installing memo2, and checks that originalv4 function module identities remain unwrapped before reserve mutations. admit explicitly uses consumer=False. Launch checks and worker checks explicitly use consumer=True. No close/reserve operation was added to those consumer paths.

The unchanged frozen launcher accepts only canonical arggnn modules and requires its exact src/<module>.py source in direct consumer bindings. The active proposed CLI is now arggnn.H043_GPU45_transport_worker_v1, mapped to src/arggnn/H043_GPU45_transport_worker_v1.py. Its exact source is included in all actual CLI refs, the frozen source closure and the mandatory direct reservation bindings. Source checks verify this exact launcher mapping and argv contract. The prior review-folder CLI is superseded.

pre_admission_review_before_refs.json and pre_admission_supervisor_memo_fix.diff preserve compact pre-fix review evidence. pre_admission_review_fix_receipt.json records the final operational fixes and current source refs. These remain unexecuted source changes.


Final prospective prefix-pin fix: launch retains the original reservation storage reference and its immutable-prefix initial_ledger_pins. It does not inspect the latest ledger, rebase pins or write launch_storage.json. The narrow AST/compile/hash check passed. The previous nine source checks were not rerun because science and their operational paths were unchanged.
