"""Reviewed resource-only wrapper around the sealed owned-worker migration.

The original migration performs process identity validation, checkpoint capture,
history preservation and repair construction. This wrapper retains that request
and creates a second immutable request changing only the two RAM-cap fields.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import time
from arggnn.benchmarking.paper_campaign import processes, runtime as r, repairs
from . import migrate as original

GiB=2**30
MAXIMUM=592*GiB
PRIOR_MAXIMUM=768*GiB
WORK=original.WORK/'resource592'
REQUEST=WORK/'repair_request.json'
REVIEW=r.ROOT/'research/preprocessing_optimization_20261026/review/resource592_migration_review.json'
RELATE='paper_argmix_published_recon_train_001.native_relate'
PINS=[
 {'path':'research/preprocessing_optimization_20261026/integration/migrate.py','sha256':'940bdc4514c284c6b4a9fbe623e95f59bfc43f85fde7fea81e5a1bd0a61ab724'},
 {'path':'research/preprocessing_optimization_20261026/integration/release.json','sha256':'7fe628a46be4ff61b7f955610d9959eb738d2e4aef908973b937dcfe59774944'},
 {'path':'research/preprocessing_optimization_20261026/review/resource_envelope_assessment.json','sha256':'ff5853607025bcb7b8ba3de63061de7a1bed81f5e739b225911ed2458bffa38a'},
 {'path':'src/arggnn/benchmarking/paper_campaign/scheduler.py','sha256':'494a1347d7dc5f6a06824d712736e2e0663da8a39c70574ca113eeb4dc8ec456'},
 {'path':'research/repairs/TreeLoom_lazy_compact_prefetch_20261024_v1/pool.py','sha256':'f527f761b610e3dda4eb7ccbbc422979040b36976500dd04bcb2e5e41eb85bf9'}]
need=original.require


def authenticate(require_review=True):
    from .treeloom_resume import authenticate_release
    for ref in PINS:
        need(r.ref(ref['path'])==ref,'Pinned operational input changed: '+ref['path'])
    authenticate_release(PINS[1])
    need(r.read(PINS[2])['status']=='pass','Passed RAM-envelope assessment required')
    if require_review:
        review=json.loads(REVIEW.read_text())
        need(review['status']=='pass' and review['operational_source_ref']==r.ref(__file__)
             and review['pinned_inputs']==PINS,'Current independent operational review required')


def check_memory_fields(task,expected=PRIOR_MAXIMUM):
    a=task['call']['spec']['arguments']
    need(task['id']==original.TASK and task['max_rss_bytes']==expected
         and a['max_rss_bytes']==expected,'Exact own task and trainer RAM caps required')
    need(a['observation_cache_bytes']==352*GiB
         and a['preparation_headroom_bytes']==128*GiB
         and a['training_headroom_bytes']==128*GiB
         and a['max_truth_uncompressed_bytes']==8*GiB,
         'Original all20 cache, preparation, training and truth reserves must remain unchanged')


def amend_request(request):
    need(len(request['replacement_tasks'])==1,'Only the own TreeLoom task may be amended')
    old=request['replacement_tasks'][0]
    check_memory_fields(old)
    need(old['call']['module']=='research.preprocessing_optimization_20261026.integration.treeloom_resume',
         'Original qualified checkpoint continuation required')
    result=copy.deepcopy(request)
    task=result['replacement_tasks'][0]
    task['max_rss_bytes']=MAXIMUM
    task['call']['spec']['arguments']['max_rss_bytes']=MAXIMUM
    check_memory_fields(task,MAXIMUM)
    # Verify the complete task differs at precisely these two operational fields.
    restored=copy.deepcopy(task)
    restored['max_rss_bytes']=PRIOR_MAXIMUM
    restored['call']['spec']['arguments']['max_rss_bytes']=PRIOR_MAXIMUM
    need(restored==old,'Non-resource task amendment is forbidden')
    result['request_id']='exact_preprocessing_20261026_resource592'
    result['reason']=request['reason']+' Task and trainer RAM caps are592GiB, retaining cache352/preparation128/training128GiB and every scientific argument.'
    return result


def anonymous_release_lower_bound(leader):
    """Conservative private-anonymous reclaim estimate for this owned session."""
    total=0
    for member in processes.session_members(leader):
        fields={line.split(':',1)[0]:int(line.split()[1])*1024
                for line in Path('/proc',str(member['pid']),'smaps_rollup').read_text().splitlines()
                if ':' in line and line.split(':',1)[0] in ('Anonymous','Shared_Clean','Shared_Dirty')}
        need(set(fields)=={'Anonymous','Shared_Clean','Shared_Dirty'},'Complete anonymous memory accounting required')
        total+=max(0,fields['Anonymous']-fields['Shared_Clean']-fields['Shared_Dirty'])
    return total


def admission_arithmetic(available,reclaim,active_growth,contender_caps):
    required=MAXIMUM+sum(active_growth)+sum(contender_caps)+64*GiB
    projected=available+reclaim
    return dict(available_bytes=available,owned_anonymous_release_estimate_bytes=reclaim,
        projected_available_bytes=projected,required_bytes=required,
        margin_bytes=projected-required,admissible_projection=projected>=required,
        mandatory_host_headroom_bytes=64*GiB)


def snapshot(before_stop):
    plan,state,task=original.current()
    check_memory_fields(task)
    own=state['tasks'][original.TASK]
    need((before_stop and own['status']=='running') or
         (not before_stop and own['status']=='failed' and own.get('actual_wait') is True
          and own.get('owned_process_group_absent') is True),'Correct original migration phase required')
    active=[]
    identities={}
    for item in plan['tasks']:
        row=state['tasks'][item['id']]
        if row['status']!='running':continue
        identities[item['id']]=(row['pid'],row['process_start_ticks'],row['attempt'])
        if item['id']==original.TASK:continue
        measured=processes.pss(row['pid'])
        # Scheduler may use its last observation until the next tick. Reserve
        # against the smaller observation rather than overstating free growth.
        accounted=min(measured,row.get('observed_pss_bytes',0))
        active.append(dict(task_id=item['id'],cap_bytes=item['max_rss_bytes'],
            measured_pss_bytes=measured,accounted_pss_bytes=accounted,
            remaining_growth_bytes=max(0,item['max_rss_bytes']-accounted)))
    need(not plan.get('existing_workers'),'Additional inherited workers require separate resource review')
    ready=[]
    for item in plan['tasks']:
        row=state['tasks'][item['id']]
        if item['id']==original.TASK or row['status']!='pending':continue
        if all(state['tasks'][dep]['status']=='succeeded' for dep in item['dependencies']):
            ready.append(dict(task_id=item['id'],kind=item['kind'],cap_bytes=item['max_rss_bytes'],
                preferred_gpus=item.get('preferred_gpus',[])))
    # Include the known pending256GiB Relate even if a predecessor is completing.
    relate=next(item for item in plan['tasks'] if item['id']==RELATE)
    if state['tasks'][RELATE]['status']=='pending' and not any(x['task_id']==RELATE for x in ready):
        need(relate['max_rss_bytes']==256*GiB,'Known Relate reservation changed')
        ready.append(dict(task_id=RELATE,kind=relate['kind'],cap_bytes=relate['max_rss_bytes'],
            preferred_gpus=relate.get('preferred_gpus',[]),anticipated=True))
    # Reserve every currently ready competitor, including GPU tasks. This is
    # intentionally stricter than hardware-slot feasibility and changes no queue.
    reclaim=anonymous_release_lower_bound(own['pid']) if before_stop else 0
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:')))*1024
    later_plan,later,later_task=original.current()
    need(later.get('plan_revision',0)==state.get('plan_revision',0) and later_task==task,
         'Plan changed during read-only admission snapshot')
    need({key:(row['pid'],row['process_start_ticks'],row['attempt'])
          for key,row in later['tasks'].items() if row['status']=='running'}==identities,
         'Active worker identities changed during admission snapshot')
    arithmetic=admission_arithmetic(available,reclaim,
        [x['remaining_growth_bytes'] for x in active],[x['cap_bytes'] for x in ready])
    return dict(schema='resource592_admission_snapshot_v1',observed_unix=time.time(),
        phase='before_owned_stop_projection' if before_stop else 'before_repair_submission',
        plan_ref=state['plan_ref'],plan_revision=state.get('plan_revision',0),
        own_task=original.TASK,own_attempt=own['attempt'],active_others=active,
        ready_or_anticipated_competitors=ready,**arithmetic,
        instantaneous_admission_guaranteed=False,
        note='A dynamic conservative snapshot. The unchanged scheduler rechecks actual RAM and GPU admission. Other task progress can still cause normal waiting.')


def save_snapshot(value):
    return original.exclusive(WORK/('admission_'+str(time.time_ns())+'.json'),value)


def stop():
    authenticate()
    value=snapshot(True)
    save_snapshot(value)
    need(value['admissible_projection'],'Insufficient projected RAM after owned exit and ready competitors')
    original.stop()


def amend():
    authenticate()
    source=original.WORK/'repair_request.json'
    request=json.loads(source.read_text())
    amended=amend_request(request)
    original.exclusive(REQUEST,amended)
    original.exclusive(WORK/'amendment_receipt.json',dict(status='pass',
        original_request_ref=r.ref(source),amended_request_ref=r.ref(REQUEST),
        operational_source_ref=r.ref(__file__),operational_review_ref=r.ref(REVIEW),
        pinned_inputs=PINS,only_changed_task_fields=['max_rss_bytes','call.spec.arguments.max_rss_bytes'],
        old_maximum_bytes=PRIOR_MAXIMUM,new_maximum_bytes=MAXIMUM,
        numerical_source_changed=False,scientific_arguments_changed=False))


def prepare():
    authenticate()
    original.prepare()
    amend()


def submit():
    authenticate()
    value=snapshot(False)
    save_snapshot(value)
    need(value['admissible_projection'],'Fresh actual RAM admission requires waiting before submission')
    plan,state,task=original.current()
    request=json.loads(REQUEST.read_text())
    need(request==amend_request(json.loads((original.WORK/'repair_request.json').read_text())),
         'Amended request changed beyond exact reviewed RAM fields')
    directory=r.path(plan['records_directory'])
    body=directory/'repair_requests'/(request['request_id']+'.request.json')
    pending=directory/'repair_requests'/(request['request_id']+'.ref.json')
    need(not body.exists() and not pending.exists(),'Repair must be submitted exactly once')
    body_ref=original.exclusive(body,request)
    staged=WORK/(request['request_id']+'.ref.json')
    original.exclusive(staged,body_ref)
    repairs._prepare(plan,state,directory,staged)
    os.link(staged,pending)
    print('Reviewed resource592 repair submitted through unchanged controller API.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inspect','stop','prepare','amend','submit'])
    args=parser.parse_args()
    if args.action=='inspect':
        authenticate(require_review=False)
        print(json.dumps(snapshot(True),indent=2))
    else:
        globals()[args.action]()
