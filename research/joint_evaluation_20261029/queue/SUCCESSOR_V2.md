# Final campaign handoff with explicit current-model comparison

Use this version for the first handoff. V1 was prepared and tested, never launched. V2 retains its idle-boundary, genuine wait, output replay refusal, GPU priority, shared GPU3 lock and storage guards.

The queue preserves all 252 original task IDs and adds one external task, joint_argmix.prediction_barrier. Thirty-six original native stages remain aliases of the two already-running native sidecars. Eleven never-launched legacy ARGMix TreeLoom stages are explicitly marked superseded, each with its replacement task ID and scientifically_complete:false. These statuses are included in counts and terminal detection. The original failed American TreeLoom attempt and its blocked descendants remain unchanged.

The existing native paired prediction barrier changes only its second method path and dependency: the new joint model's barrier replaces the old untrained TreeLoom barrier. All eight original target-label producers, the native ARGMix score, their class order, target support and metric functions remain unchanged. The new external barrier requires its exact task ID and model ID, a genuine successful actual_wait.json, an empty process session, its exact result path, and the unchanged native verifier's complete eight-parent barrier checks. No future wait is invented while that artifact is absent.

The successor has 253 task definitions. The575 focused checks passed, including failure preservation, explicit supersession, launch-race handling, GPU lock exclusivity, metadata overcommit rejection and full production cache accounting. No test signalled any process or launched a worker. The task source-pin audit is successor_v2_source_checks.json.

## Root launch

On anogena, in `/workspace-SR003.nfs2/ARG-GNN`:

```bash
source scripts/project_env.sh
export PYTHONDONTWRITEBYTECODE=1
.runtime/env/bin/python -B -m research.joint_evaluation_20261029.queue.successor_v2_handoff --config research/joint_evaluation_20261029/queue/successor_v2_config.json --inspect
mkdir -p research/joint_evaluation_20261029/queue/successor_records_v2
nohup .runtime/env/bin/python -B -s -u -m research.joint_evaluation_20261029.queue.successor_v2_handoff --config research/joint_evaluation_20261029/queue/successor_v2_config.json > research/joint_evaluation_20261029/queue/successor_records_v2/controller.log 2>&1 < /dev/null &
```

Final reviewed config SHA-256: e3cf7982043993957774d8f6b7c4c1d72047a1f214de67782f564ff4885721ff. Record the returned watcher PID and start ticks once. It preserves the live original worker and waits for genuine TEST002 prediction and retirement plus the exact TEST003 claim block. It freezes only an already-idle controller, rechecks for races, resumes it on a race, and terminates only the verified idle controller. An existing handoff/state is a hard replay refusal.

## Root GPU priority

All successor GPU tasks wait until root publishes successor_v2_gpu3_release.json with schema root_joint_models_gpu_release_v1, successor_config_ref equal to the final V2 config, and completed_models mapping american_joint, argmix_joint and european_four_joint to their actual completed chain wait references. Each wait must contain actual_wait:true, actual_returncode:0, owned_process_group_absent:true and a drained PID session. Root's European direct-four entry represents its entire GPU chain. CPU stages do not depend on this gate.

While native ARGformer is unfinished, successor GPU work can use only physical GPU3. Root model workers and successor GPU3 workers share `.runtime/benchmarking/joint_root_successor_gpu3.lock`, held for the worker lifetime and passed to the child. Foreign RAM reservation is768GiB, with each owned task and64GiB host headroom additional.

## Conditional production cache upgrade

The current root ARGMix allocation is8GiB. Before production cache writes, root must admit24GiB cache plus1GiB results. This is a17GiB increase, fully charged by the helper below:

```bash
.runtime/env/bin/python -B -m research.joint_evaluation_20261029.queue.argmix_production_admission --config research/joint_evaluation_20261029/queue/successor_v2_config.json
.runtime/env/bin/python -B -m research.joint_evaluation_20261029.queue.argmix_production_admission --config research/joint_evaluation_20261029/queue/successor_v2_config.json --prepare-proposal
```

The helper writes only queue-local immutable proposal and ledger files. It does not authorize cache execution or write the root admission path. A proposal may honestly show admitted:false while waiting for natural native retirement and verified completion. Root may publish an admitted proposal with root_confirmed:true only when its complete ledger is at most1,200,000,000,000 bytes.

The root admission path is research/joint_evaluation_20261029/root/argmix_joint_production_admission_v1.json. Its schema is joint_argmix_cache_admission_v1, with exact protocol_ref, prior_allocation_ref, successor_config_ref, cache/results scopes,24GiB/1GiB byte caps,64GiB RAM, combined_reserved_bytes, owner_ceiling_bytes and actual_ledger_receipt_ref. The successor authenticates that immutable ledger and begins charging25GiB as soon as the authorized file exists. Production cache code requires the same root receipt. No current or future execution may silently retain the8GiB charge after the upgrade.

The full original128GiB native ARGMix scope and all outstanding native ARGformer task caps remain reserved independently. The successor enforces its256MiB metadata allowance during active monitoring and before admission. No new cleanup operation is introduced.
