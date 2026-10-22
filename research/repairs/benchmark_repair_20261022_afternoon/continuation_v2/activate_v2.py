"""Launch the reviewed full continuation, preserving all existing scientific workers."""
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time
from research.repairs.benchmark_repair_20261022_afternoon.runtime_v2 import readable_workflow_v1 as queue

c=queue.c
HERE=Path(__file__).parent
ACTUAL=HERE/'actual'


def activate():
    preparation=c.read_json(ACTUAL/'preparation_receipt.json')
    reference=preparation['plan_ref']
    plan=queue.load_plan(reference)
    manifest=c.read_json(HERE/'runtime_manifest.json')
    for row in manifest['source_lineage']:
        queue.need(c.file_ref(row['original_ref']['path'])==row['original_ref'],'Original live source changed')
        queue.need(c.file_ref(row['successor_ref']['path'])==row['successor_ref'],'Reviewed new source changed')
    queue.need(c.file_ref(HERE.parent/'european/runtime_v2_source_review_v1.json')['sha256']=='5cf676271d4bd402093c03424146cdebcee96f2beb79364d9f9c7ec14e8e745d','Independent gateway review required')
    queue.need(not (ACTUAL/'launch_intent.json').exists(),'Existing launch intent requires reconciliation, never replay')
    records=c.contained_path(plan['records_directory']);records.mkdir(exist_ok=True)
    queue.need(not (records/'state.json').exists(),'Fresh current queue state required')
    with queue.shared_gpu.CAPACITY.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        policy=c.read_json(c.POLICY_PATH)
        before=dict(policy)
        queue.need(policy.get('recovery_workflow_drain_ref')==policy['approved_recovery_workflow_ref'],'Only future previous dispatch must be drained')
        queue.save(ACTUAL/'policy_before_activation.json',policy,exclusive=True)
        policy['approved_recovery_continuation_workflow_ref']=reference
        policy['recovery_continuation_workflow_approved_at_utc']=datetime.now(timezone.utc).isoformat()
        policy['recovery_continuation_release_reason']='User requested verification and previously authorized necessary fixes. Preserve237 successes and all3healthy workers. Retry failed C4 only at64GiB cap/72GiBfree, identical22qualifiedsources and original90updates. Independent source and metadata checks passed.'
        queue.need(all(policy[key]==value for key,value in before.items()),'Existing authority must stay intact')
        queue.save(c.POLICY_PATH,policy)
        queue.load_plan(reference,approved=True)
        state=queue.initial_state(plan,reference)
        projected=queue.storage_projection(plan,state)
        c4=next(t for t in plan['tasks'] if t['id']=='C4__all_train__c4_treeloom_fit')
        queue.need(projected+c4['max_output_bytes']<980000000000,'Retry storage admission must fit')
        queue.need(queue.external_resources(plan)[1]+c4['max_rss_bytes']<=plan['max_parallel_rss_bytes'],'Full RAM reservations must fit')
        queue.save(ACTUAL/'policy_after_activation.json',policy,exclusive=True)
    command=[str(c.ROOT/'.runtime/env/bin/python'),'-B','-s','-m',queue.MODULE,
        '--plan-ref',json.dumps(reference,sort_keys=True),'--execute']
    intent=dict(plan_ref=reference,command=command,launcher_ref=c.file_ref(__file__),
        cwd=str(c.ROOT),launched_at_utc=datetime.now(timezone.utc).isoformat(),
        all_original_approvals_preserved=True,signals_sent=0,storage_projection_bytes=projected)
    queue.save(ACTUAL/'launch_intent.json',intent,exclusive=True)
    with (records/'controller.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=c.ROOT,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    identity=None
    for _ in range(10):
        identity=queue.shared_gpu.identity(process.pid)
        if identity is not None:break
        time.sleep(0.1)
    receipt=dict(intent,controller_pid=process.pid,controller_identity=identity,
        log_path=str((records/'controller.log').relative_to(c.ROOT)),returncode_at_receipt=process.poll())
    queue.save(ACTUAL/'launch_receipt.json',receipt,exclusive=True)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':activate()
