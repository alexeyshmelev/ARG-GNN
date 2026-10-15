# Controlled C3 focal/plan continuation

Ready for root's fresh execution. The original001–013 coordinator is not rerun. Its genuine stopped result is pinned at `C3_focal_plan_fanout_root_source_20261015_v1/run_v1/fanout_result.json`, SHA `f72432733a5bf3630d4501e1e8f7192038d859d90466d99281fad2a9334bc88f`: root stop signal15, all direct children waited, no admitted-without-launch entries, and genuine successful001–005 worker wait0/drain/completions. Root reported its outer attempt stopped/drained; this packet performs no fresh PID probe.

## Actual readiness

The metadata inventory was observed during this source task; execution rechecks it before creating a run/admitting:

| Parents | State | Original path/focal seeds |
|---|---|---|
| TRAIN000–005 |Completed10k focal/plan products verified; reused and excluded from every selectable range|92201–92206|
| TRAIN006–013 |Inference wait0/drain, completion and observed manifest ready; focal packets/outputs absent|92207–92214|
| TRAIN014–015 |Same genuine readiness, including014 recovery; focal packets/outputs absent|92215–92216|
| TRAIN016–019 |Not ready: original infer/actual_wait.json absent; focal packets/outputs absent|92217–92220|

TRAIN006's exact original focal admission packet and output were both absent. `availability_actual_v1.json` records all current metadata refs/hashes and reusable completions. No payload was opened. A later genuine completion of016–019 changes readiness; none is assumed or prefilled.

## Root commands

From `/workspace-SR003.nfs2/ARG-GNN`, currently ready006–015:

```sh
.runtime/env/bin/python -B research/reviews/C3_focal_plan_continuation_memo2_20261015_v1/continue_v1.py --execute --first 6 --last 15 --max-workers 2 --run-name run006_015_v1
```

Root can choose `--max-workers 1` to reduce concurrent checks further. At most2 is enforced by CLI choices and pool/submission limits. Admission uses the existing shared wave admission.lock only while its admission child runs; producer execution runs outside that lock. Do not wrap the entire coordinator in another independent acquisition of the same lock.

After original016–019 preparation genuinely completes, use a new run packet:

```sh
.runtime/env/bin/python -B research/reviews/C3_focal_plan_continuation_memo2_20261015_v1/continue_v1.py --execute --first 16 --last 19 --max-workers 2 --run-name run016_019_v1
```

That command currently fails readiness before creating a run or admitting work. Without `--execute`, the same range arguments print source/metadata readiness only. Default range6–19 is also metadata-only unless explicitly executed. A selected range must be ordered and within6–19; completed000–005 cannot be selected.

Once all20 genuine products exist, the unchanged original bridge's `--collect-plans` can collect them:

```sh
.runtime/env/bin/python -B research/reviews/C3_next_plan_root_handoff_20261015_v1/focal_plan_bridge_v1.py --collect-plans
```

No automatic replay, resume, overwrite or credit occurs. Any selected existing admission packet, partial output or changed readiness blocks before new admissions. A failed/stopped run preserves actual waits, completed parents, skipped/admitted-without-launch lineage and logs. Root must review partial admissions separately; a fresh range/run is not permission to replay them.

## Science and guard reuse

The frozen `focal_plan_bridge_v1.py` is reused byte-for-byte, SHA `b2e7fcfa1d301b01cda62a99408986e89cd75fe14509dc4188bd04d0cf84317e`. It already uses memo2 only in consumer=True execution (guard_module54–67; execute170), while admission uses consumer=False and requires original unwrapped v4. No bridge/producers/seed registry/runtime/scientific source is copied or edited, so new completions keep the same source_ref and original all20 collector remains valid.

Original short2Mb1600-haplotype R600/Q1000 inferred parents, complete trees/samples, native first-appearance full-path ID order, independent registry focal seeds,10,000 no-replacement label-blind query focals, midpoint rule and existing PyG block-plan construction remain exact. Original33,554,432-byte per-parent output cap,512,000,000,000-byte process RSS cap,7200-second deadline,100,000-tree and1024-node full-path hard bounds, three-hour admission expiry and every resource/writer/guard boundary remain unchanged. No MLM training dependency, native corpus, query ancestry-truth access, ancestry score or changed scientific condition is introduced.

Memo2 stays transaction-local inside each original guard mutex, with original exit snapshot validation and no cross-operation cache. Reserve/close use original unwrapped functions; this coordinator performs no closes or automatic credits. At most two focal producers and serial admissions reduce concurrent root/consumer demand; no measured speedup or guaranteed mutex waiting time is claimed. Current guard still decides each actual admission against980GB; there is no bulk reservation.

## Stop and evidence behavior

Thirteen original helper ASTs, including child PID/start/group/session identity, bounded64KiB stdout capture, minimal direct wait before postprocessing, exact owned-group ESRCH, monitored error propagation and parent path/env helpers, are unchanged. Original STOP-on-error/SIGINT/SIGTERM behavior stops additional admissions and drains already-running owned children; it does not kill/restart science automatically. The new result explicitly reports selected-parent completion, reused000–005 refs, prior stopped-fanout ref, admitted-without-launch rows and uncompleted selected parents, without claiming all20 ready from a subset.

The new direct entrypoint bootstraps repository root/src/scripts in sys.path before imports; children retain the original project-only PYTHONPATH. The real source-only CLI passed with PYTHONPATH unset. `source_static_receipt_v1.json` records compile/hash/AST/metadata checks. `source_only_cli_plan_v1.json` is the actual metadata-only CLI output. No guard import/configure/state/measurement, admission, PID probe, signal, science/payload/GPU run, live edit or Git operation was performed.
