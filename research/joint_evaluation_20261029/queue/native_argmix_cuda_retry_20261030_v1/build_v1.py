"""Prepare one additive native CUDA retry; no launch, signal, or historical rewrite."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from arggnn.benchmarking.paper_campaign import runtime as r
BASE='research/joint_evaluation_20261029/queue'
HERE=BASE+'/native_argmix_cuda_retry_20261030_v1'
MODULE='research.joint_evaluation_20261029.queue.native_argmix_cuda_retry_20261030_v1'
TRAIN='paper_argmix_argmix.train'


def write_new(path,value):
    with r.path(path).open('x') as stream:
        if isinstance(value,str):stream.write(value)
        else:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    return r.ref(path)


original_source=r.ref(BASE+'/successor_recovery_scheduler_v1.py')
assert original_source['sha256']=='3544eb7972623311e48433fac78391d75351594281676f849225cce30019b628'
source=r.path(original_source['path']).read_text()
changes=[
('from research.joint_evaluation_20261029.queue import successor_v2_support as support',
 'from '+MODULE+' import support_v1 as support'),
("lock = r.path('.runtime/benchmarking/paper_campaign_global.lock').open('a+')",
 "lock = r.path('"+HERE+"/sidecar_owner.lock').open('a+')"),
("state['handoff_ref'] = plan['handoff_ref']",
 "state['operational_retry_manifest_ref'] = plan['retry_manifest_ref']"),
("accounting = storage_credit.commitment(plan, plan['storage_credit_ref'])",
 "accounting = support.admission_accounting(plan)"),
("remaining_growth = plan['foreign_ram_reserve_bytes'] + sum(",
 "remaining_growth = plan['foreign_ram_reserve_bytes'] + support.foreign_owner_growth(plan) + sum("),
("def main():\n    parser = argparse.ArgumentParser(description=__doc__)",
 "def main():\n    import setproctitle\n    setproctitle.setproctitle('AI Auto Research Pipeline Test')\n    parser = argparse.ArgumentParser(description=__doc__)")]
for old,new in changes:
    assert source.count(old)==1,old
    source=source.replace(old,new)
scheduler_ref=write_new(HERE+'/scheduler_v1.py',source)
original_plan_ref=r.ref(BASE+'/successor_records_v2/plan.json')
assert original_plan_ref['sha256']=='544d73bc30ff7e237850c27fc69172f4947644c2571860179369f16f0a686043'
original=r.read(original_plan_ref)
state_path=BASE+'/successor_records_v2/state.json'
state=json.loads(r.path(state_path).read_text())
assert state['plan_ref']==original_plan_ref and state['controller_pid']==2031398
by_id={task['id']:task for task in original['tasks']}
closure={TRAIN}
while True:
    found={task['id'] for task in original['tasks'] if closure.intersection(task['dependencies'])}
    if found<=closure:break
    closure|=found
owned={name for name in closure if state['tasks'][name]['status']!='superseded'}
assert len(owned)==20
preserved={name:state['tasks'][name] for name in sorted(owned)}
assert preserved[TRAIN]['status']=='failed' and preserved[TRAIN]['actual_returncode']==1
assert all(row['status']=='blocked' and 'pid' not in row for name,row in preserved.items() if name!=TRAIN)
retry_output=by_id[TRAIN]['output_directory']+'/retry_attempt5'
wrapper_ref=r.ref(HERE+'/cuda_cli_v1.py')
support_ref=r.ref(HERE+'/support_v1.py')
source_refs=[original_source,scheduler_ref,support_ref,wrapper_ref,r.ref(__file__)]
smoke_ref=r.ref('research/joint_evaluation_20261029/review_finish/recovery_20261030/native_argmix_cuda_smoke_run_v1/completed.json')
manifest=dict(schema='native_argmix_failed_fit_cuda_retry_sidecar_v1',root_authorized=True,
    created_unix=time.time(),retry_task_id=TRAIN,retry_output_directory=retry_output,
    wrapper_module=MODULE+'.cuda_cli_v1',wrapper_source_ref=wrapper_ref,
    original_plan_ref=original_plan_ref,original_state_path=state_path,original_controller_pid=2031398,
    preserved_original_rows=preserved,
    failed_actual_wait_ref=r.ref(by_id[TRAIN]['output_directory']+'/actual_wait.json'),
    failed_log_ref=r.ref(preserved[TRAIN]['log_path']),operational_source_refs=source_refs,
    cuda_equivalence_smoke_ref=smoke_ref,repair_directory=HERE,metadata_max_bytes=16*2**20,
    no_original_state_rewrite=True,no_historical_wait_rewrite=True,no_existing_process_signal=True,
    duplicate_ownership_prevented_by='Exact immutable original failed/blocked rows and exclusive dedicated sidecar lock',
    inherited_GPU3_mutex_path='.runtime/benchmarking/joint_root_successor_gpu3.lock',
    unchanged_original_scientific_source_refs=True,unchanged_training_recipe=True,
    unchanged_selection_budget=True,unchanged_task_resource_limits=True,
    downstream_paths_preserved=True,TreeLoom_dependency_integration='Original product paths and colocated actual waits; no controller restart or state overlay',
    final_audit_integration='Prediction selected_model_ref follows retry_attempt5/completed.json and adjacent genuine wait',
    automatic_retries_authorized=False)
manifest_ref=write_new(HERE+'/manifest_v1.json',manifest)
external={dep for name in owned for dep in by_id[name]['dependencies']}-owned
tasks=[]
for task in original['tasks']:
    name=task['id']
    if name in owned:
        task=copy.deepcopy(task)
        if name==TRAIN:
            task['output_directory']=retry_output
            task['argv'][3]=MODULE+'.cuda_cli_v1'
            task['argv'][task['argv'].index('--output')+1]=retry_output
        elif name.endswith('.argmix_prediction'):
            task['argv'][task['argv'].index('--selected-model')+1]=retry_output+'/completed.json'
        tasks.append(task)
    elif name in external:
        tasks.append(copy.deepcopy(task) if task['kind']=='external' else dict(id=name,kind='external',dependencies=[],
            sidecar_plan_ref=original_plan_ref,state_path=state_path,source_task_id=name))
plan=copy.deepcopy(original)
plan.update(tasks=tasks,records_directory=HERE+'/records_v1',retry_manifest_ref=manifest_ref,
            controller_source_refs=original['controller_source_refs']+source_refs,
            controller_metadata_allowance_bytes=16*2**20,max_parallel_tasks=1,max_cpu_tasks=1,max_jobs_per_gpu=1,
            pipeline_storage_floors=[])
plan.pop('handoff_ref',None)
plan_ref=write_new(HERE+'/plan_v1.json',plan)
write_new(HERE+'/preparation_receipt_v1.json',dict(prepared=True,plan_ref=plan_ref,manifest_ref=manifest_ref,
    owned_task_count=len(owned),external_dependency_count=len(external),scheduler_source_ref=scheduler_ref,
    source_changes=[dict(before=old,after=new) for old,new in changes],launches=0,signals=0,
    original_state_or_source_or_wait_mutations=0))
print(json.dumps(dict(plan_ref=plan_ref,manifest_ref=manifest_ref,scheduler_ref=scheduler_ref,
                     support_ref=support_ref,wrapper_ref=wrapper_ref,owned_tasks=len(owned),external_dependencies=len(external)),indent=2))
