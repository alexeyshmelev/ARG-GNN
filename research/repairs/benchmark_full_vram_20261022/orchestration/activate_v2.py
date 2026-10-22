"""Launch a separately approved full-device queue without interrupting healthy work."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import subprocess
import time
from . import storage_workflow_v2 as facade

queue = facade.queue
c = queue.c
HERE = Path(__file__).parent
ACTUAL = HERE / 'actual_v2'


def activate():
    preparation = c.read_json(ACTUAL / 'preparation_receipt.json')
    reference = preparation['plan_ref']
    plan = queue.load_plan(reference)
    review = c.read_json(ACTUAL / 'final_review.json')
    queue.need(review.get('status') == 'passed' and review.get('plan_ref') == reference,
               'Independent review of the exact full-device plan is required')
    queue.need(not (ACTUAL / 'launch_intent.json').exists(), 'Reconcile any previous launch intent, never replay it')
    records = c.contained_path(plan['records_directory'])
    records.mkdir(exist_ok=True)
    queue.need(not (records / 'state.json').exists(), 'Fresh queue state required')
    with queue.shared_gpu.CAPACITY.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        policy = c.read_json(c.POLICY_PATH)
        queue.need(policy.get('full_vram_workflow_drain_ref') == plan['operational_successor_of']
                   and policy.get('gpu_memory_policy') == 'full_device', 'Retain the explicit user permission and future-dispatch drain')
        queue.need(policy.get('approved_full_vram_workflow_ref') == plan['operational_successor_of'], 'Exact drained previous authority required')
        previous = c.read_ref(plan['operational_successor_of'])
        previous_state = c.read_json(c.contained_path(previous['records_directory']) / 'state.json')
        queue.need(previous_state.get('run_status') == 'drained_for_operational_update'
                   and not any(row['status'] in ('running', 'unknown') for row in previous_state['tasks'].values()), 'Never withdraw an approval used by a live worker')
        queue.save(ACTUAL / 'policy_before_activation.json', policy, exclusive=True)
        before = dict(policy)
        policy['preserved_full_vram_v1_workflow_ref'] = before['approved_full_vram_workflow_ref']
        policy['approved_full_vram_workflow_ref'] = reference
        policy['full_vram_v2_workflow_approved_at_utc'] = datetime.now(timezone.utc).isoformat()
        policy['full_vram_v2_workflow_release_reason'] = 'Explicit user full VRAM permission on GPUs 2, 3 and 5. Preserve completed science and healthy C4/native workers. Resume six-update American checkpoint and retry genuinely failed AP fit without artificial GPU memory limits.'
        queue.need(all(policy[key] == value for key, value in before.items() if key != 'approved_full_vram_workflow_ref'), 'Every live original approval must remain intact; drained v1 stays in history')
        queue.save(c.POLICY_PATH, policy)
        queue.load_plan(reference, approved=True)
        state = queue.initial_state(plan, reference)
        projection = queue.storage_projection(plan, state)
        queue.need(projection < plan['storage_ceiling_bytes'], 'Full conservative storage accounting must fit')
        queue.save(ACTUAL / 'policy_after_activation.json', policy, exclusive=True)
    command = [str(c.ROOT / '.runtime/env/bin/python'), '-B', '-s', '-m', queue.MODULE,
               '--plan-ref', json.dumps(reference, sort_keys=True), '--execute']
    intent = dict(plan_ref=reference, command=command, launcher_ref=c.file_ref(__file__),
        cwd=str(c.ROOT), launched_at_utc=datetime.now(timezone.utc).isoformat(),
        healthy_workers_preserved=plan['full_vram_live_predecessors'],
        all_original_approvals_preserved=True, signals_sent=0, storage_projection_bytes=projection,
        independent_final_review_ref=c.file_ref(ACTUAL / 'final_review.json'))
    queue.save(ACTUAL / 'launch_intent.json', intent, exclusive=True)
    with (records / 'controller.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=c.ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    identity = None
    for _ in range(10):
        identity = queue.shared_gpu.identity(process.pid)
        if identity is not None:
            break
        time.sleep(0.1)
    receipt = dict(intent, controller_pid=process.pid, controller_identity=identity,
                   log_path=str((records / 'controller.log').relative_to(c.ROOT)), returncode_at_receipt=process.poll())
    queue.save(ACTUAL / 'launch_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, indent=2), flush=True)

if __name__ == '__main__':
    activate()
