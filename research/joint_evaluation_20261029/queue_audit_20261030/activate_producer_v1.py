"""Root-only, one-time transfer of untouched producer tasks; no process signals."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue_audit_20261030.common import probe_scheduler_v1 as engine
from research.joint_evaluation_20261029.queue_audit_20261030.critical_path.producer_v1 import support_v1 as support
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import producer_owner_v1 as owner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--review',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    plan_ref=r.ref(args.plan);plan=engine.validate(r.read(plan_ref))
    review_ref=r.ref(args.review);review=r.read(review_ref)
    support.need(review['passed'] is True and review['plan_ref']==plan_ref and
                 review['launcher_ref']==r.ref(__file__), 'Exact independent source/activation review required')
    spec,state=support.validate_preclaim(plan)
    directory=r.path(spec['producer_directory'])
    activation_path=directory/'activation_v1.json'
    registry_path=r.path(owner.REGISTRY)
    event_path=directory/'activation_events_v1.jsonl'
    log_path=directory/'controller_v1.log'
    claims_path=r.path(spec['claims_receipt_path'])
    for path in [activation_path,registry_path,event_path,log_path,claims_path,directory/'records_v1/state.json']:
        support.need(not path.exists(), 'Preserve existing activation; no automatic replay: '+str(path))
    rows={'tasks':{task['id']:{'status':'pending'} for task in plan['tasks']}}
    first=next(task for task in plan['tasks'] if task['id']==spec['claim_task_ids'][0])
    floor=support.completion_reservation(plan,rows,first)
    accounting=support.admission_accounting(plan)
    support.need(floor['admitted'] and accounting['admitted'], 'Complete parent capacity required')
    command=[str(r.ROOT/'.runtime/env/bin/python3.11'),'-B','-m',
             'research.joint_evaluation_20261029.queue_audit_20261030.critical_path.producer_v1.scheduler_v1',
             '--plan',str(r.path(args.plan))]
    if not args.apply:
        print(json.dumps(dict(valid=True,plan_ref=plan_ref,completion_floor=floor,accounting_total_bytes=accounting['total_bytes'],command=command,signals_sent=[])))
        return
    with (directory/'activation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        support.validate_preclaim(plan)
        with event_path.open('x') as events:
            def event(action,**values):
                events.write(json.dumps(dict(action=action,time_unix=time.time(),**values),sort_keys=True)+'\n')
                events.flush();os.fsync(events.fileno())
            event('preclaim_verified',plan_ref=plan_ref,review_ref=review_ref,completion_floor=floor,
                  original_controller=spec['original_controller'],guard_pid=spec['guard_pid'])
            tasks={task['id']:task for task in plan['tasks']}
            for name in spec['claim_task_ids']:
                support.validate_preclaim(plan,allow_claimed=True)
                task=tasks[name];output=r.path(task['output_directory'])
                output.mkdir(parents=True,exist_ok=False)
                with (output/support.MARKER).open('x') as stream:
                    json.dump(support.marker_value(plan,task),stream,sort_keys=True,indent=2)
                    stream.write('\n');stream.flush();os.fsync(stream.fileno())
                event('pending_generation_claimed',task_id=name,marker_ref=r.ref(output/support.MARKER))
            support.validate_preclaim(plan,allow_claimed=True)
            claims=support.verify_claims(plan,empty=True)
            claims_ref=r.save(claims_path,dict(schema='native_relate_producer_claims_v1',root_confirmed=True,
                plan_ref=plan_ref,manifest_ref=plan['producer_manifest_ref'],claims=claims,review_ref=review_ref))
            event('claims_bound',claims_ref=claims_ref)
            env=os.environ.copy()
            env.update(PYTHONPATH=str(r.ROOT/'src')+':'+str(r.ROOT),PYTHONDONTWRITEBYTECODE='1',
                       PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            with log_path.open('xb') as log:
                child=subprocess.Popen(command,cwd=r.ROOT,env=env,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ticks=engine.process_start_ticks(child.pid)
            activation=dict(schema='native_relate_producer_root_activation_v1',root_confirmed=True,pid=child.pid,
                process_start_ticks=ticks,command=command,plan_ref=plan_ref,manifest_ref=plan['producer_manifest_ref'],
                claims_ref=claims_ref,review_ref=review_ref,started_unix=time.time(),signals_sent=[],
                original_controller_untouched=True)
            activation_ref=r.save(activation_path,activation)
            event('producer_started',activation_ref=activation_ref)
            state_path=r.path(plan['records_directory'])/'state.json'
            deadline=time.monotonic()+30
            while not state_path.is_file() and child.poll() is None and time.monotonic()<deadline:
                time.sleep(0.5)
            support.need(child.poll() is None and state_path.is_file(), 'Producer startup incomplete; inspect existing attempt, never replay')
            state=json.loads(state_path.read_text())
            support.need(state['controller_pid']==child.pid and state['plan_ref']==plan_ref, 'Actual producer identity differs')
            support.need(engine.process_start_ticks(child.pid)==ticks, 'Producer identity changed')
            registry=dict(schema='native_relate_producer_active_owner_v1',root_confirmed=True,
                original_plan_ref=spec['original_plan_ref'],producer_plan_ref=plan_ref,
                producer_manifest_ref=plan['producer_manifest_ref'],state_path=str(state_path.relative_to(r.ROOT)),
                activation_ref=activation_ref,owned_task_ids=spec['owned_task_ids'])
            registry_ref=r.save(registry_path,registry)
            owner.load_active_owner(required=True)
            event('registered_actual_owner',registry_ref=registry_ref)
            print(json.dumps(dict(activation_ref=activation_ref,registry_ref=registry_ref,pid=child.pid,signals_sent=[])))


if __name__=='__main__':main()
