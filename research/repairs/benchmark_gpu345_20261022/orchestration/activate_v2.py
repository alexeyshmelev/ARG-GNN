"""Launch the separately approved GPU3/4/5 queue without interrupting healthy work."""
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
                   and policy.get('gpu_memory_policy') == 'full_device'
                   and policy.get('currently_authorized_physical_gpus') == [3,4,5], 'Retain the explicit user permission and future-dispatch drain')
        queue.need('approved_gpu345_workflow_ref' not in policy, 'Do not replace an existing live full-device authority')
        queue.save(ACTUAL / 'policy_before_activation.json', policy, exclusive=True)
        before = dict(policy)
        policy['approved_gpu345_workflow_ref'] = reference
        policy['gpu345_workflow_approved_at_utc'] = datetime.now(timezone.utc).isoformat()
        policy['gpu345_workflow_release_reason'] = 'User explicitly moved GPU2 work toGPU4. Only GPUs3/4/5 may execute new work with full VRAM. Preserve completed science, latest American checkpoint and healthy GPU3/5 workers. Stop future legacy dispatch and never modify unrelated processes.'
        queue.need(all(policy[key] == value for key, value in before.items()), 'Every original approval must remain intact')
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
