# Wave supervision v2: avoid live-writer stat snapshots

This operational successor fixes the same supervisor/worker race observed in the full30Mb C3 v5 run. It preserves the original wave driver, all40 scientific configs,20 IDs/seeds, registry, producer modules, inference/unary policy, runtimes, caps and condition ID. No launch, admission, guard configuration, accounting state, scientific data read or GPU operation occurred here.

## Confirmed race

The v1 wave supervisor called budget_check_v9(boundary=True) immediately after Popen through its guarded launch.json write and again for every log chunk. The owned reservation includes output and worker_aux. While the child writes output or temporary/cache files, an independent stat-stable full-scope scan can detect a legitimate change and raise. The supervisor then terminates a healthy child. The full30Mb occurrence establishes the operational failure; this source audit identifies the same calls in the wave driver.

## Narrow fix

Before launch, normal guarded boundary checks remain. While the child can write:

- Launch metadata is saved only in the compact root packet outside both worker-owned scopes, under the overlay's charged bookkeeping allowance.
- Stdout/stderr is retained in a bounded16MiB RAM buffer. There is no supervisor owned-scope boundary scan or worker-scope write.
- Timeout, expiry, log-cap and sampled cache/temp limits remain active. The cache sampler tolerates entries disappearing during a live sample instead of demanding a stat-stable snapshot.

After proc.wait, the worker's isolated process group must return ESRCH to killpg0. Only then does the supervisor guard, write/fsync the bounded log to reserved worker_aux, and guard again. Actual wait evidence retains the direct return code, complete supported owned roster, group drain, log/hash and capture errors.

A capture or launch-metadata failure terminates and directly waits the owned child. A terminal log/guard failure retains the actual return code in compact root evidence rather than losing it. If a group is not drained, the supervisor performs no worker-scope scan or log write; it preserves at most32KiB of log-prefix triage in compact root metadata, records failure and requires root follow-up. It never claims a full durable log or completed credit in that case.

The existing scientific workers' own guarded writes are unchanged. No boundary=False planned-write bypass, relaxed storage contract, source/config mutation or extra credit is introduced. Up to four live buffers retain64MiB of payload total, plus Python overhead. Logs become durable at terminal drain; live launch metadata still exposes the exact worker/supervisor PIDs.

## Root interface and fresh scopes

Use:

    .runtime/env/bin/python -B research/reviews/American_C3_short_train2mb_wave_supervision_v2/root_wave_driver.py --run-parents american_c3_short_train_000 american_c3_short_train_001 --policy research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json --policy-sha a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85

The APIs and individual --admit/--launch simulate|infer modes remain unchanged. Initial root batches may contain two parents. All runs remain CPU-only; GPU eligibility stays physical4/5 for other root-owned work.

New root packets are under this supervision_v2 directory's admitted/PARENT/STAGE. New product outputs are experiments/results/American_C3_short_train2mb_wave_v2/PARENT/{simulation,inference}. scope_routing_v2.json is the explicit operational map; the unchanged original registry continues to define scientific identity/config refs. Consumers should join actual preparation completion refs and this routing map for product paths.

The same four-slot scheduler lock namespace is shared with v1, preserving the cross-process parent limit and inherited lock-FD behavior. New quota keys use-v2. Active old-v1 allocations count toward the limit, and any original-v1 reservation for the same parent requires explicit root audit rather than silently duplicating it. This successor assumes root had not yet admitted the v1 parent wave, as stated in the task.

Each parent still has two separate exact stage reservations,1GB simulation and2GB inference. No fifth open wave-parent allocation or60GB bulk reservation is allowed. Root owns complete supervisor/group drain observations, capability revocation and all later completed-stage credits.

## Source verification

source_qualification_receipt.json records successful compilation, no undefined globals, static verification that the live-capture function contains no owned-scope boundary check or scope write, exact placement of durable log writes after group drain, unchanged40 config hashes and bounded log handling. supervision_fix.diff shows the operational changes. No no-op/model/data/inference test was run.

Root should review this successor and freeze its source before admission. The v1 sources/receipts remain intact for audit.
