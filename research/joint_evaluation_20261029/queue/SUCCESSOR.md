# Campaign successor at the verified idle boundary

This additive controller preserves the original 252 task IDs. Thirty-five ARGformer tasks and the one ARGMix native TRAIN002 task become external aliases to their existing owners. Their successful results require the pinned sidecar plan, the exact output completion path, a matching immutable actual_wait.json, successful real return code, and an empty worker session. The successor does not claim a local wait for an external result. The old failed TreeLoom row and every unrelated blocked descendant remain unchanged.

The root launches one watcher on anogena. Its initial discovery only reads the original controller identity, state and descendants. It waits until original TEST002 prediction and retirement have genuine successful waits, original TEST003 preparation is blocked by the owned claim, and the controller has no running row or live descendant. Only then does it use a pidfd to freeze that exact controller. It repeats the state and descendant checks while frozen. A launch race receives SIGCONT, with no termination. A verified idle controller receives SIGTERM and SIGCONT; no worker is signalled. All historical worker sessions must be empty. The final original state is sealed unchanged before the new controller takes the original global scheduler lock.

The watcher has no worker deadline. The successor removes the arbitrary controller and CLI wall deadlines for its never-launched tasks. Future native Relate calls use the same original CLI and original native run_parallel code, with subprocess timeouts removed and genuine subprocess.run waits retained. All scientific arguments, original simulation seeds, inputs, output paths, byte limits and RAM limits remain identical. The existing native retention stages remain in their original dependency graph. No replay of an existing pending output is permitted.

The budget charges the original authenticated inventory and noncampaign allowances, the full 128 GiB original ARGMix scope, the full 64 GiB American model allowance, the full 8 GiB new ARGMix model allowance, CNN 640 MiB, both native sidecar metadata caps, original retained outputs and logs, and successor metadata. Every unfinished ARGformer task retains its full output cap. A completed task releases unused reservation only after its actual wait authenticates; its retained output remains charged. This replaces the old 78.9 GB campaign growth envelope with explicit scopes. The successor admits no new task until the total is at most 1,200,000,000,000 bytes. It separately checks active plus candidate growth against the original 128 GiB ARGMix cap.

The latest initial observation was about 1.209 TB, so immediate successor work was not admitted. Original TEST002 retirement and genuine sidecar completions reduce that conservative commitment automatically. No file deletion is performed by the accounting code. The unchanged scientific retention tasks execute only at their original dependency boundaries.

Foreign RAM reservation is 768 GiB: up to 512 GiB for the two native ARGformer lanes and 64 GiB each for CNN, the American joint model the separate ARGMix joint model, and the European direct-four joint model. The inherited 64 GiB host headroom and each owned task's full remaining reservation are additional. The successor owns at most two workers. Every GPU task also waits for the root priority release described below. While the native ARGformer queue is unfinished only GPU3 is eligible. GPU3 uses the shared exclusive flock for the full owned task lifetime; the worker inherits its descriptor.

## Root launch after final independent review

Run on anogena, in `/workspace-SR003.nfs2/ARG-GNN`. Inspect is read-only and sends no signals. Confirm the exact config hash recorded in the review before starting the watcher once.

```bash
source scripts/project_env.sh
export PYTHONDONTWRITEBYTECODE=1
.runtime/env/bin/python -B -m research.joint_evaluation_20261029.queue.successor_handoff --config research/joint_evaluation_20261029/queue/successor_config.json --inspect
mkdir -p research/joint_evaluation_20261029/queue/successor_records_v1
nohup .runtime/env/bin/python -B -s -u -m research.joint_evaluation_20261029.queue.successor_handoff --config research/joint_evaluation_20261029/queue/successor_config.json > research/joint_evaluation_20261029/queue/successor_records_v1/controller.log 2>&1 < /dev/null &
```

Record the returned PID and process start ticks. While waiting, inspect watch_status.json. At handoff, preserve handoff.json, original_state_sealed.json, the final plan.json and adoption_audit.json. Once active, state.json records current closed-writer accounting and authentic alias evidence. Never repeat a launch after uncertain delivery; inspect the returned PID and existing records. An existing handoff or successor state is a hard replay refusal.

Verification: successor_tests.json contains 560 passing checks including launch-race recovery, zero signals while busy, pidfd identity, failed-history preservation, authentic native sidecar receipt adoption, and resource-overcommit rejection. successor_source_checks.json authenticates 523 unique task source references. Tests sent zero signals, launched zero workers, created no claims and changed no live state. successor_scheduler.diff is the precise delta from the frozen original scheduler. Sources are all ordinary plaintext; no encoded command or decoded executable staging is used.

## Root model GPU priority release

Root fits acquire `.runtime/benchmarking/joint_root_successor_gpu3.lock` with exclusive flock before spawning a GPU3 worker, keep it until the real successful or failed wait and full session drain, and pass the descriptor to the worker. The successor uses this exact same path. Before all three root GPU chains finish, `successor_gpu3_release.json` remains absent; this blocks successor GPU tasks while leaving CPU work available.

After all three root GPU chains complete, root writes this ordinary JSON document at `research/joint_evaluation_20261029/queue/successor_gpu3_release.json`, filling the exact reviewed config and real wait references:

```json
{
  "schema": "root_joint_models_gpu_release_v1",
  "successor_config_ref": {"path": "research/joint_evaluation_20261029/queue/successor_config.json", "sha256": "c8a548d092c285b1379bf6d383e296c2c8506d799b5c6d9aa38306534372aa78"},
  "completed_models": {
    "american_joint": {"path": "ACTUAL_AMERICAN_WAIT_PATH", "sha256": "ACTUAL_SHA256"},
    "argmix_joint": {"path": "ACTUAL_ARGMIX_WAIT_PATH", "sha256": "ACTUAL_SHA256"},
    "european_four_joint": {"path": "ACTUAL_EUROPEAN_FOUR_CHAIN_WAIT_PATH", "sha256": "ACTUAL_SHA256"}
  }
}
```

The placeholders are documentation only. The validator requires all three existing receipts to contain `actual_wait: true`, `actual_returncode: 0`, `owned_process_group_absent: true`, and the recorded PID's empty process session. It authenticates every referenced file. A wrong config, missing real wait, failed fit or surviving session cannot release the gate.

The256MiB successor metadata allowance is enforced during active monitoring and before each admission. Root is preparing a separate explicit supersession of eleven never-launched legacy ARGMix TreeLoom tasks. This version preserves their pending history and keeps all GPU work behind the root gate; root must adopt the reviewed supersession before publishing the release document. The nativeCPU chain can advance independently.
