"""Repair only failed attempt6's pre-restore runtime class check.

No signals, no native history overwrite, and no controller state edits. The exact
original268-update capture remains the source of the successor. Failed attempt6
output is retained separately. Both actions require exact independent review.
"""
import argparse
import json
import os
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r,repairs,processes
from research.preprocessing_optimization_20261026.integration import migrate as prior
from research.efficiency_20261027.root import capture_boundary_v2 as capture

WORK=capture.WORK/'runtime_class_repair'
REQUEST_ID='treeloom_explicit512_runtime_class_20261027'
need=capture.need
save=capture.save


def authenticate(args):
    from research.efficiency_20261027.integration_candidate.entrypoint_v3 import validate_release
    review=json.loads(r.path(args.review).read_text())
    need(review.get('status')=='pass' and review.get('operational_source_ref')==r.ref(__file__),
         'Passing exact operational-source review required')
    release_ref=r.ref(args.release)
    validate_release(release_ref)
    plan,state,task=prior.current()
    row=state['tasks'][capture.TASK]
    need(row['status']=='failed' and row['attempt']==6 and row['pid']==1162519
         and row['process_start_ticks']=='574318983' and row.get('actual_wait') is True
         and row.get('actual_returncode')==1 and row.get('owned_process_group_absent') is True,
         'Exact failed, reaped attempt6 only')
    need(not Path('/proc',str(row['pid'])).exists() and not processes.session_members(row['pid']),
         'No old worker or owned session may remain')
    need(task['call']['module']=='research.efficiency_20261027.integration_candidate.restore_entrypoint_v2',
         'Only the identified pre-restore wrapper failure is in scope')
    log=r.path(row['log_path'])
    need(log.stat().st_size<2**20 and 'ValueError: Original campaign resource checker required' in log.read_text(),
         'Known class-identity error required, not arbitrary task failure')
    need(not prior.NATIVE.exists(),'A restored native history requires a separate recovery review')
    return plan,state,task,row,release_ref


def prepare(args):
    from research.efficiency_20261027.integration_candidate.prepare_task_v3 import build_task
    plan,state,old,row,release_ref=authenticate(args)
    complete=json.loads((capture.WORK/'capture_completed.json').read_text())
    need(complete['complete_parent_updates']==268,'Preserve the original268-update capture')
    replacement=build_task(transition_ref=complete['transition_ref'],execution_release_ref=release_ref,
        preserved_history_ref=complete['preserved_history_ref'],attempt_id='explicit512_attempt7')
    for key in ('max_rss_bytes','max_output_bytes','max_seconds','minimum_free_gpu_bytes','output_directory','preferred_gpus'):
        need(replacement[key]==old[key],'Original resource/scope envelope must remain exact: '+key)
    output=r.path(old['output_directory'])
    need(output==prior.NATIVE.parent and output.is_dir() and not output.is_symlink(),
         'Exact failed outer output required')
    files=[]
    for p in output.rglob('*'):
        need(not p.is_symlink() and (p.is_file() or p.is_dir()),'Ordinary failed output only')
        if p.is_file(): files.append(dict(relative_path=str(p.relative_to(output)),reference=r.ref(p)))
    size=r.scope_bytes(output)
    need(size<4*2**20,'Only the small pre-restore failure is in scope')
    archive=WORK/'failed_attempt6_outer'
    need(not archive.exists(),'Never replace preserved failed evidence')
    save(WORK/'before_archive.json',dict(task=old,row=row,files=files,
        release_ref=release_ref,observed_state_ref=r.ref(prior.STATE),archived_bytes=size))
    authenticate(args)
    output.rename(archive)
    for item in files:
        need(r.ref(archive/item['relative_path'])['sha256']==item['reference']['sha256'],
             'Failed output changed during rename')
    archive_ref=save(WORK/'archive_manifest.json',dict(archived_bytes=size,
        archives=[dict(original=old['output_directory'],archived=str(archive.relative_to(r.ROOT)),bytes=size)],
        failure_scope='Pre-restore runtime class identity, no new checkpoint or CUDA work'))
    closure={capture.TASK}
    while True:
        larger=closure|{t['id'] for t in plan['tasks'] if set(t['dependencies'])&closure}
        if larger==closure:break
        closure=larger
    reset=[]
    for t in plan['tasks']:
        if t['id'] in closure-{capture.TASK} and state['tasks'][t['id']]['status']=='blocked':
            need(not r.path(t['output_directory']).exists(),'Dependent output requires its own preservation')
            reset.append(t['id'])
    request=dict(schema=repairs.SCHEMA,request_id=REQUEST_ID,
        expected_plan_ref=state['plan_ref'],expected_plan_revision=state['plan_revision'],
        expected_attempts={name:state['tasks'][name].get('attempt',1) for name in [capture.TASK]+reset},
        reset_task_ids=reset,replacement_tasks=[replacement],archived_bytes=size,
        archive_manifest_ref=archive_ref,
        reason='Repair pre-restore runtime callback class identity under python -m. New subprocess regression covers actual launch. Same qualified CPU512 executor and original268-update checkpoint, all scientific inputs/model/schedule unchanged.')
    save(WORK/'repair_request.json',request)
    print(json.dumps(dict(status='prepared_only',preserved_updates=268,failed_output_bytes=size)))


def submit(args):
    from research.efficiency_20261027.integration_candidate.prepare_task_v3 import build_task
    plan,state,old,row,release_ref=authenticate(args)
    need(not r.path(old['output_directory']).exists(),'Failed output must already be archived')
    complete=json.loads((capture.WORK/'capture_completed.json').read_text())
    request=json.loads((WORK/'repair_request.json').read_text())
    expected=build_task(transition_ref=complete['transition_ref'],execution_release_ref=release_ref,
        preserved_history_ref=complete['preserved_history_ref'],attempt_id='explicit512_attempt7')
    need(request['replacement_tasks']==[expected],'Only exact rebuilt successor may be dispatched')
    directory=r.path(plan['records_directory'])
    body=directory/'repair_requests'/(REQUEST_ID+'.request.json')
    pending=directory/'repair_requests'/(REQUEST_ID+'.ref.json')
    need(not body.exists() and not pending.exists(),'Inspect existing dispatch rather than resubmit')
    body_ref=save(body,request)
    staged=WORK/(REQUEST_ID+'.ref.json')
    save(staged,body_ref)
    repairs._prepare(plan,state,directory,staged)
    os.link(staged,pending)
    print(json.dumps(dict(status='submitted_once',preserved_updates=268,controller_restarted=False)))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('action',choices=['prepare','submit'])
    p.add_argument('--release',required=True)
    p.add_argument('--review',required=True)
    args=p.parse_args()
    if args.action=='prepare':prepare(args)
    else:submit(args)


if __name__=='__main__':main()
