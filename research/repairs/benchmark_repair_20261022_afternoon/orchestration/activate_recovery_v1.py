"""Activate and launch exactly one reviewed full recovery queue, preserving old jobs."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from research.repairs.benchmark_repair_20261022_afternoon.runtime import readable_workflow_v1 as queue

c=queue.c
HERE=Path(__file__).parent
ACTUAL=HERE/'actual_v1'
PLAN='configs/benchmarks/executable_workflow_20261022_afternoon_v1.json'


def activate():
    preparation=c.read_json(ACTUAL/'preparation_receipt.json')
    reference=preparation['plan_ref']
    queue.need(c.file_ref(PLAN)==reference,'Exact prepared recovery plan required')
    validation=c.read_json(HERE.parent/'runtime/validation_v1.json')
    for ref in validation['operational_source_refs']:
        queue.need(c.file_ref(ref['path'])==ref,'Reviewed operational source changed')
    plan=queue.load_plan(reference)
    queue.need(not (ACTUAL/'launch_intent.json').exists(),'Launch intent exists, inspect receipt and PID instead of replay')
    records=c.contained_path(plan['records_directory'])
    records.mkdir(parents=True,exist_ok=True)
    queue.need(not (records/'state.json').exists(),'Fresh controller state required')
    with queue.shared_gpu.CAPACITY.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        policy=c.read_json(c.POLICY_PATH)
        before=dict(policy)
        queue.need(policy['parallel_resource_overlay_drain_ref']==policy['approved_parallel_resource_overlay_ref'],
            'Preserve old overlay future-dispatch drain')
        queue.save(ACTUAL/'policy_before_activation.json',policy,exclusive=True)
        policy['approved_recovery_workflow_ref']=reference
        policy['recovery_workflow_approved_at_utc']=datetime.now(timezone.utc).isoformat()
        policy['recovery_workflow_release_reason']='User directed fix and continue. Full349-task successor adopts236 genuine successes, preserves four healthy jobs, retries only failed American fit at64GiB cap with original600 updates, and retains all historical artifacts. Seven operational CPU tests and independent source review passed.'
        for key,value in before.items():
            queue.need(policy[key]==value,'Existing worker authority changed: '+key)
        queue.save(c.POLICY_PATH,policy)
        queue.load_plan(reference,approved=True)
        state=queue.initial_state(plan,reference)
        projection=queue.storage_projection(plan,state)
        queue.need(projection<980000000000,'Storage ceiling exceeded')
        queue.save(ACTUAL/'policy_after_activation.json',policy,exclusive=True)
    command=[str(c.ROOT/'.runtime/env/bin/python'),'-B','-s','-m',queue.MODULE,
        '--plan-ref',json.dumps(reference,sort_keys=True),'--execute']
    intent=dict(plan_ref=reference,command=command,cwd=str(c.ROOT),
        launcher_source_ref=c.file_ref(__file__),launched_at_utc=datetime.now(timezone.utc).isoformat(),
        preserved_old_policy_fields=True,storage_projection_bytes=projection,
        signals_sent=0,unrelated_processes_modified=False)
    queue.save(ACTUAL/'launch_intent.json',intent,exclusive=True)
    with (records/'controller.log').open('xb') as stream:
        process=subprocess.Popen(command,cwd=c.ROOT,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    identity=None
    for _ in range(10):
        identity=queue.shared_gpu.identity(process.pid)
        if identity is not None:break
        time.sleep(0.1)
    receipt=dict(intent,controller_pid=process.pid,controller_identity=identity,
        log_path=str((records/'controller.log').relative_to(c.ROOT)),
        process_returncode_at_receipt=process.poll())
    queue.save(ACTUAL/'launch_receipt.json',receipt,exclusive=True)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':activate()
