# H049 bounded preparation writer — source only

This packet replaces repeated global history replay on **future H049 CPU preparation saves** with a local, bounded namespace check. It changes bookkeeping and publication only. Original reserve/admission, original unwrapped full launch validation, genuine root wait/drain, and original terminal/close validation remain required. No current source, allocation, writer or live process was changed.

The [284-line writer](bounded_phase_writer_v1.py) uses standard-library checks and imports no budget/guard API. [H049_prepare_layout_v1.py](H049_prepare_layout_v1.py) derives exact limits from the existing count plan and config. [H049_prepare_context_v1.py](H049_prepare_context_v1.py) is an explicit prepare-only context successor; [context_integration.diff](context_integration.diff) shows its narrow delta. It calls the original guard loader with consumer=False, uses the original full validate_storage_v9 once, and supplies its actual result to the local writer. No memo or guard function is replaced.

The context refuses revoked prepare_v1. A future root input must name the actual repaired producer, match cfg.preparation_producer_ref, and carry matching preparation_execution_version>=2. The repaired producer and codec retain all scientific validation and metadata, including actual codec/source/input refs and lossless roundtrip checks. This packet does not repair or claim resolution of the separate scientific .max failure.

## Exact phase bound

The source metadata forecast gives:

| Fully retained component | Conservative bytes |
|---|---:|
| 40 NPZ + 40 recipe JSON files, allocation rounding and 8,192-byte per-file counters | 239,931,392 |
| Combined original fixed metadata cap | 524,288 |
| Output, auxiliary and empty read-only runtime directories | 49,152 |
| Fixed root lifetime lease file | 16,384 |
| Original root log/wait tail, held unavailable during phase | 1,654,784 |
| One zero-byte publication placeholder | 8,192 |
| **Whole phase peak** | **242,184,192** |
| Existing reviewed output cap | 242,221,056 |
| Remaining margin | 36,864 |

[Counted forecast](counted_phase_forecast.json) uses existing count/config metadata only. It is not a runtime preparation result. A fresh exact plan must be generated against the actually repaired source closure and validated by the proposed context. No larger cap or active-scope credit is assumed.

The exact success set is 82 files: started.json, 40 ordered parent NPZ/JSON pairs, and completed.json. failed.json is the mutually exclusive terminal alternative. There are 83 allowed names, no overwrite, discard or restart. Each NPZ permits exactly the six original MEMBERS, with original compact dtypes and count-derived dimensions. Original codec decoding and lossless roundtrip remain unchanged. The original combined fixed-metadata cap is enforced in aggregate, not multiplied by the number of metadata files.

## Why skipping repeated global replay is safe only in this phase

1. The original full immutable reservation remains globally charged at its entire allowance until terminal close. Future root admissions still use the unchanged global guard, so other jobs cannot consume this phase's unused allowance. No measurement, local counter or partial completion creates active credit.
2. The actual pinned root source performs original full launch validation of source/runtime/input/output/reservation bindings. The worker's context validates the original scientific closure and count metadata, then the local writer rechecks the complete source map and small immutable input/config/runtime receipts on every save. Bulk observations still use the original authenticated scientific reader; no input authentication is bypassed.
3. The root and worker hold one exact inherited exclusive phase descriptor. Its immutable bytes bind the actual reservation digest, pre-admission plan hash, job, root PID/start identity, nonce and expiry. Root's actual writer-tree wait/drain and an independent exclusive-lock check are mandatory before the original terminal/close guard. Root must not revoke, close or credit that reservation while the phase descriptor remains held. This lifecycle condition is part of the new root operational contract, not a capability of an old current input.
4. There is one writing process, no active external scientific reader, and no other filesystem writer in these scopes. The CPU scientific route writes NPZ buffers in RAM. Every named TMP/cache route points to one empty read-only auxiliary directory; bytecode writes are disabled. Aux contains only the fixed lifetime lease and that empty directory during the phase. The root retains its bounded log in RAM and writes its original serialized log only after actual worker drain. An unexpected file, inode change, aux growth or TMP content blocks further writes.
5. Before each save, bounded local inventory covers the entire output and sole auxiliary scope. All completed files retain their inode/size/block signature. New data must fit its exact payload cap, aggregate group cap, actual filesystem free bytes, whole scope allowance, placeholder allowance and reserved terminal tail. Actual allocated blocks are checked after writing. This retains the original contractual allocation-bound model; it is not a kernel quota against an uncooperative writer.
6. Source/expiry/process/lease validation happens before mutation and again before publication. Errors poison the writer and preserve partial artifacts. They do not authorize a replay or automatic cleanup.

## Publication on this NFS

Scratch is created with O_EXCL|O_NOFOLLOW, written completely and fsynced. Publication first creates the final name with O_EXCL|O_NOFOLLOW, verifies that fresh empty placeholder's exact inode, then POSIX-renames the scratch over that owned placeholder and fsyncs the directory. Final inode must equal the captured fsynced scratch inode. Existing final files are never adopted or overwritten. All visible files remain single-linked, including during publication, so unchanged full scope readers do not encounter an invalid hardlink namespace.

The placeholder may be visible while save is in flight. Complete-byte visibility is guaranteed when save returns. The admitted one-writer/closed-reader phase is therefore an essential condition. This is not a general atomic no-replace primitive against unrelated writers. NFS returned EINVAL for renameat2(RENAME_NOREPLACE); [that failed fixture](handmade_test_failure_v2.json) is preserved. The earlier hardlink prototype was superseded before any activation.

## Exact root integration

Use a fresh future operational admission, preserving all current inputs and sources.

1. Bind writer, layout, context and [root lifecycle helper](root_phase_lifecycle_v1.py) in the future source manifest and original reservation consumer closure. Generate the plan before reservation, with exact count-derived fields, reviewed original caps and actual repaired source hashes. Root review must explicitly authorize the bounded phase. The plan must be activated before its SHA enters the original reserve call. Do not mutate a bound plan afterward.
2. The plan binds reservation key, output, auxiliary path and full byte cap. After original reserve succeeds, the root lifetime lease binds the actual complete reservation digest and plan hash. This order avoids a reservation/plan/lease hash cycle.
3. Keep the original full prewrite guard before auxiliary creation, with at least 49,152 planned bytes for aux, read-only runtime directory and lease. Pass its actual result into establish_after_original_full_guard_v1. The helper returns the held descriptor, lease_ref and exact private runtime environment. It performs no guard call or admission.
4. Add bounded_phase_plan_ref and bounded_phase_lease_ref to the fresh launch input. Use the approved actual repaired CPU producer directly with its existing --execute/--input/--expected-input-sha256 route and the new root_context_module_ref. Pass the lease FD explicitly through Popen(pass_fds=(phase_fd,)). Keep original CPU/RSS/deadline controls and all scientific reader validation. The new context independently performs the original unwrapped full launch validation.
5. Root keeps its log in RAM. During preparation, each out.save uses local source/expiry/lease/complete-scope checks and exact JSON/NPZ serialization. The 40-parent algorithm, compact codec, observation checks and roundtrip comparisons do not change.
6. After actual producer wait and complete owned-writer drain, use gate_after_actual_wait_and_drain_v1. It closes only root's exact inherited lifetime handle and acquires an independent exclusive descriptor; a surviving inherited holder blocks. Keep that descriptor while calling the original full terminal check, writing original bounded log/wait receipts and invoking the original authorized close flow. No guard is wrapped or weakened. If an unchanged guard refuses, retain the charged failure state; do not skip it.
7. Preserve actual execution version, producer/config/input/source manifest/codec refs and full preparation completion/wait lineage for the downstream GPU owner. No GPU dispatcher clone is needed for this bookkeeping change.

The local engine has no global reserve, close, rollback, credit or production-activation API. This is limited to fixed CPU preparation. It must not be extended to GPU qualification, native trainers, compiler caches, dynamic TMP output, concurrent writers or rolling checkpoints without a separately reviewed complete bound.

## Verification and remaining limits

The same one focused handmade cap/atomicity fixture was revised only when publication changed: initial hardlink prototype passed, renameat2 failed with EINVAL, and the final owned-placeholder primitive passed seven checks. [Final receipt](handmade_test_receipt_v3.json) covers successful exact bytes/hash, payload and group cap rejection without files, duplicate preservation, unexpected NPZ member rejection, existing-final preservation and exact fsynced single-link inode publication. Tiny fixture files and failed attempts are preserved. It does not test scientific reconstruction, the .max repair, all40 preparation, full launch/close lifecycle or performance.

All Python sources compile without execution. No production guard was imported or called during this task. No job, GPU probe, admission, canonical installation, accounting mutation or Git operation was performed. Independent source review remains pending in the root workflow.
