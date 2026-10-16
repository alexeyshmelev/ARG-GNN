# Deferred C3 complete-state CPU migration

This is executable source for a future root-owned migration, not an execution or migration result. Old C3/H050 jobs, checkpoints and every older packet remain untouched.

The runtime privately imports the exact reviewed H050_one_FP32_CPU_forward_qualification_source_20261016_v1/ordinary_root_cpu_v1.py. It reuses its ordinary 980GB admission, seeded owned identities, monitor handshake, CPU environment, aggregate2GiB RSS soft limit,300second lifetime including admission, actual waits, owned drain, bounded logs and temporary scopes. The generated adaptations change the contract, scientific callback,8MiB cap and result checks. The process-monitor and reserve/storage algorithms remain identical.

The only scientific entry point is the already reviewed original checkpoint_migration_validator_v1.migrate_complete_boundary_v1. Its callback persists separate copies with the unchanged configured guard.PilotOutputV9.save(name,value,kind="torch"). It performs zero model forward calls and zero optimizer updates. No numerical qualification is rerun.

## Capacity is derived from actual bytes

request_metadata_v1.make_root_request_v1 accepts the actual post-terminal intent and a list of exact original checkpoint refs with their actual byte lengths. It returns JSON and writes nothing. At most five scheduled candidates plus a distinct selected boundary are allowed. A boundary already present in the scheduled list is copied once.

For each copy with actual source length L, the proposed serialized ceiling is ceil4096(L+128KiB), plus the original8KiB writer charge. Add a fixed1MiB allowance for ordinary control products, original128KiB admission metadata, logs, result, waits, completion and256KiB TMP/cache. The sum must fit8MiB. Six1MiB inputs yield8,175,616bytes. Larger actual inputs can be rejected before admission, with no automatic cap expansion. The1MiB fixed allowance is a planning bound, not an extra storage reservation.

The original prospective guard reserves8MiB for output plus auxiliary scope. It accounts its bounded admission metadata through the unchanged per-event allowance. The count-derived bound conservatively includes those metadata bytes as well. Its output guard and monitor continue measuring the complete actual output/auxiliary scope. There is no historical charge reuse or release.

Before scientific migration the monitored worker restats every original input and rejects a changed actual length. Before each original guarded torch save, the callback serializes in bounded CPU memory and checks the actual new length against its declared ceiling and cumulative copy budget. The original writer repeats serialization and retains its guard, free-space and exclusive-create checks. Serialized-size disagreement or insufficient actual capacity fails with artifacts preserved.

The validator's original maximum checkpoint read limit of2GiB remains a read ceiling. It does not require reserving2GiB of output.

## Exact root sequence

1. Finish genuine H050 TRAIN and consume its logical success. Root then owns any original C3 stop/drain and durable-boundary selection. No command in this packet stops or signals those jobs.
2. Seal the existing validator's actual stopped-boundary evidence, actual migration intent, qualified bounds and ordinary FP32 qualification. Use the already passed result from C3_one_new_H8Q2_FP32_CPU_root_v2, with its actual tool65969exit0 evidence. Do not rerun it. The intent must include every actual scheduled candidate, including its own entry if the selected boundary is scheduled. If later work is knowingly discarded, use the original explicit rollback authorization and disclose it. Checkpoint names alone do not prove no later updates.
3. Obtain actual stable file lengths of only those stopped exact checkpoint refs through root's existing bounded CPU preparation. Do not guess them from the current checkpoint or grant this source author payload access.
4. Load this packet's request_metadata_v1.py through an exact source hash. Invoke the deferred builder below, then save its result through root's ordinary exclusive bounded metadata writer outside actual_admission_v1. That directory remains absent until execution.

    request = metadata.make_root_request_v1(
        ACTUAL_STATIC_CONTRACT_REF,
        ACTUAL_POST_TERMINAL_INTENT_REF,
        [
            {"checkpoint_ref": ACTUAL_ORIGINAL_CHECKPOINT_REF, "actual_bytes": ACTUAL_STABLE_LENGTH},
        ],
        root_execute_authorized=True,
    )

Supply every other distinct actual candidate as another list entry, with no invented values. Root persists the returned request and records its actual path/SHA using its existing writer.

5. Root alone runs the real outer tool command with the actual request path/SHA:

    CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B research/reviews/C3_full_state_migration_CPU_runtime_source_20261016_v1/ordinary_migration_cpu_v1.py --execute --request ACTUAL_REQUEST_PATH --expected-request-sha256 ACTUAL_REQUEST_SHA256

6. Consume the genuine new outer tool terminal0. The worker writes the original validator-returned dict to result.json, with actual_migration_tool_terminal_consumed=false and exit code null. The root monitor's completed.json confirms its real worker wait/drain only. It does not manufacture the external terminal.
7. Only after actual outer0, root may create a separate sealed manifest copied from that original result, setting the consumed flag and actual exit0 and linking genuine tool evidence. Preserve the unconsumed result and every old checkpoint. This packet does not authorize scientific continuation. Root separately constructs the resumed job with the existing full-state runtime and original scientific controls.

## Scope and review

Fixed new output: experiments/results/C3_full_state_migration_CPU_root_v1.
Fixed auxiliary scope: the same name with _aux.
The existing single-use ordinary admission directory is inside this packet. Existing directories or a previous attempt block replay.

The static source basis has44refs. Actual intent/evidence refs are added, then the final ordinary runtime/request/config/runtime/closure union is checked against the original64binding ceiling before admission metadata. The original check also remains at actual admission. Checkpoints are data refs authenticated by the original intent and original validator, not hidden extra source bindings.

The source inspection compiles all three new files and the generated worker/monitor, verifies the original monitor and reserve/storage bodies, and checks synthetic count-plan acceptance/rejections. These are source/metadata checks, not scientific migration, training or accuracy evidence. Independent review and real root admission remain required.

No live process, lease or GPU probes, checkpoint stat/read/hash, scientific array reads, installation, guard mutation, capacity admission, job stop/launch or Git operation were performed by the source author.
