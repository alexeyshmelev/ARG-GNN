"""Repair inherited external-scratch accounting before the GPU345 queue is released."""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import json
from . import storage_workflow_v2 as facade
from research.repairs.benchmark_resume_20261021.native import plan_paths_v1 as paths

queue=facade.queue
c=queue.c
HERE=Path(__file__).parent
ACTUAL=HERE/'actual_v2'
OUTPUT='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_gpu345_v2'
PLAN='configs/benchmarks/executable_workflow_20261022_gpu345_v2.json'


def amend():
    preparation=c.read_json(HERE/'actual_v1/preparation_receipt.json')
    original_ref=preparation['plan_ref']
    original=c.read_ref(original_ref)
    policy=c.read_json(c.POLICY_PATH)
    queue.need('approved_gpu345_workflow_ref' not in policy
               and not (HERE/'actual_v1/launch_intent.json').exists(), 'The first migration plan must remain genuinely unlaunched')
    plan=deepcopy(original)
    snapshot=c.read_ref(plan['adopted_state_snapshot_ref'])
    by_id={task['id']:task for task in plan['tasks']}
    adjustments=[]
    for task in plan['tasks']:
        identifier=task['id']
        if identifier in plan['adopted_completed_task_fingerprints'] or task['resource']=='external':
            continue
        if task.get('remeasure_succeeded_tasks'):
            owners=task['remeasure_succeeded_tasks']
            kept=[owner for owner in owners if by_id[owner]['resource']!='external'
                  and snapshot['tasks'][owner].get('external') is not True]
            removed=[owner for owner in owners if owner not in kept]
            if removed:
                adjustments.append(dict(task_id=identifier,local_owners=kept,external_owners=removed))
                task['remeasure_succeeded_tasks']=kept
        if task.get('factory_module')!=paths.COMPETITORS:
            new_output=OUTPUT+'/other/'+identifier
            queue.need(not c.contained_path(task['output_directory']).exists()
                       and not c.contained_path(new_output).exists(), 'Both unreleased and corrected scopes must be fresh')
            task['output_directory']=new_output
    plan,path_receipt=paths.repair_plan_paths(plan,snapshot,OUTPUT)
    plan['retention_owner_rebinding']=paths.rebind_owner_paths(plan,snapshot)
    addition=c.read_json(HERE.parent/'storage/plan_addition_v1.json')
    plan[addition['plan_field_name']]=deepcopy(addition['plan_field_value'])
    owner=addition['plan_field_value']['owner_task_id']
    retirement=addition['plan_field_value']['retirement_task_id']
    queue.need(adjustments==[dict(task_id=retirement,
        local_owners=[item for item in by_id[retirement]['remeasure_succeeded_tasks'] if item!=owner],
        external_owners=[owner])], 'Only the exact inherited external embedding owner may leave local remeasurement')
    plan['external_retirement_remeasurement_adjustments']=adjustments
    plan['records_directory']='.runtime/benchmarking/workflow_20261022_gpu345_v2'
    plan['prelaunch_revision_of']=original_ref
    plan['prelaunch_revision_reason']='Local remeasurement must not rewrite an external producer. Earn its storage reduction only after authenticated successful scratch retirement and exact current scope measurement.'
    ACTUAL.mkdir(exist_ok=False)
    plan['GPU345_retirement_path_rebinding_ref']=queue.save(ACTUAL/'path_rebinding.json',path_receipt,exclusive=True)
    refs={ref['path']:ref for ref in plan['sources']}
    for ref in addition['additional_source_refs']:refs[ref['path']]=ref
    addition_ref=c.file_ref(HERE.parent/'storage/plan_addition_v1.json')
    refs[addition_ref['path']]=addition_ref
    for folder in [HERE,HERE.parent/'storage']:
        for path in folder.glob('*.py'):
            ref=c.file_ref(path);refs[ref['path']]=ref
    plan['sources']=[refs[key] for key in sorted(refs)]
    reference=queue.save(PLAN,plan,exclusive=True)
    queue.load_plan(reference)
    state=queue.initial_state(plan,reference)
    queue.need(facade.retirement.completed_storage_reduction(plan,state)==0,'No unearned external retirement credit at release')
    projection=queue.storage_projection(plan,state)
    queue.need(projection<plan['storage_ceiling_bytes'],'Preserve all conservative initial storage charges')
    receipt=dict(plan_ref=reference,unlaunched_predecessor_ref=original_ref,
        source_state_snapshot_ref=plan['adopted_state_snapshot_ref'],tasks=len(plan['tasks']),
        counts=dict(Counter(entry['status'] for entry in state['tasks'].values())),
        healthy_workers_preserved=plan['full_vram_live_predecessors'],
        storage_projection_bytes=projection,initial_external_retirement_credit_bytes=0,
        local_remeasurement_changes=adjustments,original_AM_updates_restored=plan['GPU4_migration_completed_updates'],
        remaining_AM_updates=600-plan['GPU4_migration_completed_updates'],full_gpu_memory=True,
        authorized_physical_gpus=[3,4,5],scientific_settings_changed=False,scientific_launches=0,signals_sent=0)
    queue.save(ACTUAL/'preparation_receipt.json',receipt,exclusive=True)
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':
    amend()
