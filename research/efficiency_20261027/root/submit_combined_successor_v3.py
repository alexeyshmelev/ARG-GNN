"""Prepare and submit only the captured TreeLoom branch through queue repair.

No signals, controller restart or manual state edit. Source and history are
validated through the successor builder and the existing scheduler API.
"""
import argparse
import json
import os
from arggnn.benchmarking.paper_campaign import runtime as r,repairs
from research.efficiency_20261027.integration_candidate.prepare_task_combined_v5 import build_task
from research.efficiency_20261027.root import capture_boundary_combined_v3 as capture

WORK=capture.WORK
REQUEST_ID='treeloom_memory_selected_adaptive2_20261027'
need=capture.need
save=capture.save


def require_review(path):
    value=json.loads(r.path(path).read_text())
    need(value.get('status')=='pass' and value.get('operational_source_ref')==r.ref(__file__),
         'Passing review of this exact submission helper required')
    return r.ref(path)


def prepare(args):
    review_ref=require_review(args.review)
    release_ref=capture.authenticate_release(args.release)
    complete=json.loads((WORK/'capture_completed.json').read_text())
    need(complete['status']=='completed' and complete['release_ref']==release_ref,
         'Qualified immutable capture must precede repair construction')
    plan,state,old,row=capture.current('failed')
    need(row.get('actual_wait') is True and row.get('owned_process_group_absent') is True
         and not r.path(old['output_directory']).exists(),
         'Reaped predecessor and absent archived output required')
    from research.efficiency_20261027.root.transition_admission_v2 import assess
    before=json.loads((WORK/'before_stop.json').read_text())
    admission=assess(old,before['peer_directories'],before_stop=False)
    need(admission['admitted'],admission['reason'])
    replacement=build_task(transition_ref=complete['transition_ref'],
        execution_release_ref=release_ref,preserved_history_ref=complete['preserved_history_ref'],
        attempt_id='memory_selected_adaptive2_attempt8')
    need(replacement['id']==capture.TASK and replacement['max_output_bytes']==old['max_output_bytes'],
         'Only the original TreeLoom task with unchanged output cap may be replaced')
    closure={capture.TASK}
    while True:
        enlarged=closure|{t['id'] for t in plan['tasks'] if set(t['dependencies'])&closure}
        if enlarged==closure: break
        closure=enlarged
    reset=[]
    for task in plan['tasks']:
        name=task['id']
        if name in closure-{capture.TASK} and state['tasks'][name]['status']=='blocked':
            need(not r.path(task['output_directory']).exists(),
                 'A blocked dependent with output requires separate preservation')
            reset.append(name)
    archive=r.read(complete['archive_manifest_ref'])
    request=dict(schema=repairs.SCHEMA,request_id=REQUEST_ID,
        expected_plan_ref=state['plan_ref'],expected_plan_revision=state['plan_revision'],
        expected_attempts={name:state['tasks'][name].get('attempt',1) for name in [capture.TASK]+reset},
        reset_task_ids=reset,replacement_tasks=[replacement],
        archived_bytes=archive['archived_bytes'],archive_manifest_ref=complete['archive_manifest_ref'],
        reason='Qualified resident/explicit execution selected before forward using actual memory, with qualified concurrent preparation. Preserve model, all optimizer/RNG state, original600-update order, inputs and labels. Restore byte-identical history inside the admitted worker. Author competitors and simulations unchanged.')
    save(WORK/'repair_request.json',request)
    save(WORK/'repair_preparation.json',dict(operational_source_ref=r.ref(__file__),
        operational_review_ref=review_ref,request_ref=r.ref(WORK/'repair_request.json'),
        captured_updates=complete['complete_parent_updates'],reset_descendants=reset,
        admission=admission,scheduler_submission_performed=False))
    print(json.dumps(dict(status='prepared_only',reset_descendants=len(reset),
        update_to_resume=complete['complete_parent_updates']+1)))


def submit(args):
    require_review(args.review)
    release_ref=capture.authenticate_release(args.release)
    plan,state,task,row=capture.current('failed')
    complete=json.loads((WORK/'capture_completed.json').read_text())
    need(complete['release_ref']==release_ref,'Original captured release required')
    request=json.loads((WORK/'repair_request.json').read_text())
    expected=build_task(transition_ref=complete['transition_ref'],execution_release_ref=release_ref,
        preserved_history_ref=complete['preserved_history_ref'],attempt_id='memory_selected_adaptive2_attempt8')
    need(request['replacement_tasks']==[expected],'Prepared exact successor changed')
    directory=r.path(plan['records_directory'])
    body=directory/'repair_requests'/(REQUEST_ID+'.request.json')
    pending=directory/'repair_requests'/(REQUEST_ID+'.ref.json')
    need(not body.exists() and not pending.exists(),
         'Existing dispatch must be inspected, never resubmitted')
    from research.efficiency_20261027.root.transition_admission_v2 import assess
    before=json.loads((WORK/'before_stop.json').read_text())
    admission=assess(task,before['peer_directories'],before_stop=False)
    need(admission['admitted'],admission['reason'])
    save(WORK/'final_submission_admission.json',admission)
    body_ref=save(body,request)
    staged=WORK/(REQUEST_ID+'.ref.json')
    save(staged,body_ref)
    repairs._prepare(plan,state,directory,staged)
    os.link(staged,pending)
    print(json.dumps(dict(status='submitted_once',request_id=REQUEST_ID,
        controller_restarted=False,manual_state_edits=False)))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['prepare','submit'])
    parser.add_argument('--release',required=True)
    parser.add_argument('--review',required=True)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args)
    else: submit(args)


if __name__=='__main__':main()
