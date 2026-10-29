"""Explicit exact-scope sidecar ownership; never edits the original controller."""
import json
import os
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r

MARKER='SIDECAR_CLAIM.json'

def need(value,message):
    if not value:raise ValueError(message)


def verify_storage(plan):
    ledger=r.read(plan['storage_ledger_ref'])
    need(ledger['schema']=='parallel_queue_storage_v1' and ledger['writer_closure_confirmed'] is True,
         'ROOT must explicitly confirm complete post-inventory writer closure')
    need(ledger['ceiling_bytes']==plan['storage_ceiling_bytes']==1_200_000_000_000,'Exact owner project ceiling')
    for source in ledger['evidence_refs']:r.read(source)
    total=sum(row['reserved_bytes'] for row in ledger['base_allowances'])
    need(total+plan.get('other_queue_disk_reserved_bytes',0)==plan['reserved_existing_project_bytes'],'Exact ledger baseline')
    need(plan['reserved_existing_project_bytes']+sum(t['max_output_bytes'] for t in plan['tasks'])+plan['controller_metadata_allowance_bytes']<=ledger['ceiling_bytes'],
         'Every selected complete task must fit at its original disk cap')
    return ledger


def verify_original(plan):
    state=json.loads(r.path(plan['original_state_path']).read_text())
    expected=plan['original_identity']
    need(state['plan_ref']==expected['plan_ref'] and state.get('plan_revision',0)==expected['plan_revision'],
         'Original plan/revision changed; regenerate claim proposal')
    need(state['controller_pid']==expected['pid'],'Original controller identity changed')
    stat=Path('/proc')/str(expected['pid'])/'stat'
    need(stat.exists() and stat.read_text().rsplit(')',1)[1].split()[19]==expected['start_ticks'],
         'Original controller must still be the reviewed live process')
    guard=state['tasks'][plan['guard_task_id']]
    need(guard['status']=='running' and guard['pid']==plan['guard_pid']
         and guard['process_start_ticks']==plan['guard_start_ticks'],
         'Guard predecessor changed; no atomic-claim race may be assumed')
    child=Path('/proc')/str(guard['pid'])/'stat'
    need(child.exists() and child.read_text().rsplit(')',1)[1].split()[19]==guard['process_start_ticks'],
         'Reviewed guard worker must remain alive')
    for task in plan['tasks']:
        need(state['tasks'][task['id']]['status']=='pending','Only unchanged pending tasks can be claimed')
    return state


def marker_value(plan,task):
    return dict(schema='exact_scope_sidecar_claim_v1',owner=plan['claim_id'],task_id=task['id'],
                original_plan_ref=plan['original_identity']['plan_ref'],output_directory=task['output_directory'])


def claim_all(plan):
    """Called only by explicitly launched sidecar under its dedicated lock."""
    verify_storage(plan);before=verify_original(plan)
    roots=[t for t in plan['tasks'] if t['id'] in plan['claim_task_ids']]
    need(len(roots)==7 and all(t['id'].endswith('_prepare') for t in roots),'Seven explicit preparation gates')
    for task in plan['tasks']:
        need(not r.path(task['output_directory']).exists(),'All sidecar output scopes must initially be absent')
    claimed=[]
    for task in roots:
        verify_original(plan)
        output=r.path(task['output_directory']);output.parent.mkdir(parents=True,exist_ok=True)
        output.mkdir(exist_ok=False)
        marker=output/MARKER
        with marker.open('x') as stream:
            json.dump(marker_value(plan,task),stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        claimed.append(r.ref(marker))
    verify_original(plan)
    r.save(r.path(plan['records_directory'])/'claims.json',dict(schema='sidecar_claim_receipt_v1',
        claims=claimed,original_counts=before['counts'],original_controller_untouched=True,
        original_pending_rows_unmodified=True,no_process_signals=True))


def permit_claimed_output(plan,task,row):
    if task['id'] not in plan['claim_task_ids'] or row['status']!='pending':return False
    output=r.path(task['output_directory'])
    need(output.is_dir() and not output.is_symlink(),'Ordinary claimed directory required')
    need(sorted(p.name for p in output.iterdir())==[MARKER],'Claim root must contain only its marker before first launch')
    need(json.loads((output/MARKER).read_text())==marker_value(plan,task),'Exact sidecar claim ownership')
    return True
