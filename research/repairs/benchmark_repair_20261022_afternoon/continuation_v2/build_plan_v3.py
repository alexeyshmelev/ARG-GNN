"""Carry successful and active work forward and retry only the failed European fit."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from research.repairs.benchmark_repair_20261022_afternoon.continuation_v2 import storage_workflow_v3 as facade
queue=facade.queue
from research.repairs.benchmark_resume_20261021.native import plan_paths_v1 as paths

c=queue.c
HERE=Path(__file__).parent
ACTUAL=HERE/'actual_v3'
OUTPUT='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_afternoon_v3'
PLAN='configs/benchmarks/executable_workflow_20261022_afternoon_v3.json'
RECORDS='.runtime/benchmarking/workflow_20261022_afternoon_v3'


def build():
    policy=c.read_json(c.POLICY_PATH)
    old_ref=policy['approved_recovery_workflow_ref']
    queue.need(policy.get('recovery_workflow_drain_ref')==old_ref,'Future old dispatch must be drained')
    old=c.read_ref(old_ref)
    source_state_path=c.contained_path(old['records_directory'])/'state.json'
    state=c.read_json(source_state_path)
    queue.need(state['plan_ref']==old_ref,'Actual approved predecessor state required')
    ACTUAL.mkdir(exist_ok=False)
    snapshot_ref=queue.save(ACTUAL/'source_state_snapshot.json',state,exclusive=True)
    plan=deepcopy(old)
    plan['adopted_state_snapshot_ref']=snapshot_ref
    by_id={task['id']:task for task in old['tasks']}
    plan['adopted_completed_task_fingerprints']={identifier:c.fingerprint(by_id[identifier])
        for identifier,entry in state['tasks'].items() if entry['status']=='succeeded'}
    active_ids={identifier for identifier,entry in state['tasks'].items() if entry['status']=='running'}
    queue.need(active_ids=={'AM_treeloom_fit'},'Exact healthy local American worker required')
    bindings=deepcopy(plan['workflow_external_predecessors'])
    task=by_id['AM_treeloom_fit'];entry=state['tasks']['AM_treeloom_fit']
    supervisor=queue.shared_gpu.identity(entry['pid'])
    queue.need(supervisor is not None,'Healthy American supervisor must remain live')
    identity_ref=c.file_ref(c.contained_path(task['output_directory'])/'supervision_identity.json')
    identity=c.read_ref(identity_ref)['child_identity']
    binding=dict(source_plan_ref=old_ref,source_state_path=str(source_state_path.relative_to(c.ROOT)),
        source_state_snapshot_ref=snapshot_ref,source_task_id=task['id'],source_task_fingerprint=c.fingerprint(task),
        original_job_ref=entry['job_ref'],original_supervision_identity_ref=identity_ref,
        original_supervisor_pid=entry['pid'],original_supervisor_start_ticks=supervisor['start_ticks'],
        owned_process_group_id=identity['pid'],owned_group_leader_start_ticks=identity['start_ticks'],
        owned_group_leader_session=identity['session'],max_rss_bytes=task['max_rss_bytes'],
        max_output_bytes=task['max_output_bytes'],resource=task['resource'],output_directory=task['output_directory'])
    queue.support.predecessor._source(binding)
    queue.need(not any(b['source_task_id']==task['id'] for b in bindings),'No duplicate American binding')
    bindings.append(binding)
    failed=[b for b in bindings if b['source_task_id']=='C4__all_train__c4_treeloom_fit']
    queue.need(len(failed)==1,'Exact failed C4 binding required')
    failed=failed[0]
    _,failure=queue.support.predecessor._source(failed)
    terminal=queue.support.predecessor._drained(failed,failure)
    queue.need(failure['status']=='failed' and terminal is not None and failure['actual_controller_wait']==1
        and terminal[1]['actual_direct_returncode']==1,'Genuine failed C4 waits and owned drain required')
    bindings.remove(failed)
    retention=queue.save(ACTUAL/'retained_failed_C4.json',dict(original_binding=failed,
        actual_source_state_ref=c.file_ref(failed['source_state_path']),actual_wait_ref=terminal[0],
        actual_retained_bytes=failure['retained_bytes'],actual_retention_charge_bytes=failure['retained_bytes'],old_scope_future_writes_revoked=True,
        files_deleted=False,new_credit_bytes=0),exclusive=True)
    queue.need(queue.scope_bytes(c.contained_path(failed['output_directory']))==failure['retained_bytes'],'Preserve the exact drained failure scope at its current measured charge')
    plan['external_storage'].append(dict(cap_bytes=failure['retained_bytes'],scopes=[failed['output_directory']],
        completion_evidence=[],evidence_ref=retention,reason='Preserve actual188416-byte drained C4 failure charge already used by original predecessor helper. No candidate-only full cap was admitted. All files retained and future old-scope writes revoked'))
    plan['workflow_external_predecessors']=bindings
    patch=c.read_json(HERE.parent/'european/c4_task_resource_patch_v2.json')
    replacement=deepcopy(patch['replacement_task'])
    source_task=c.read_ref(failed['original_job_ref'])['task']
    changed={key for key in source_task.keys()|replacement.keys() if source_task.get(key)!=replacement.get(key)}
    queue.need(changed=={'max_cuda_reserved_bytes','minimum_free_gpu_bytes','output_directory'},'Only resource caps and fresh scope may change')
    queue.need(patch['resume_keyword_must_not_be_added'] is True,'No unsupported European resume argument')
    new_tasks=[];scope_changes=[]
    for original in old['tasks']:
        identifier=original['id'];entry=state['tasks'][identifier]
        if identifier=='AM_treeloom_fit':
            task=dict(id=identifier,dependencies=deepcopy(original['dependencies']),resource='external',workflow_predecessor=binding)
        elif identifier==replacement['id']:
            task=deepcopy(replacement)
        else:
            task=deepcopy(original)
            if entry['status']!='succeeded' and task['resource']!='external':
                queue.need(not entry.get('job_ref') and not entry.get('reservation_bytes'),'Reconcile any launched future task before rebasing')
                if task.get('factory_module')!=paths.COMPETITORS:
                    before=task['output_directory'];after=OUTPUT+'/other/'+identifier
                    queue.need(not c.contained_path(before).exists() and not c.contained_path(after).exists(),'Fresh future scope required')
                    task['output_directory']=after
                    scope_changes.append(dict(task_id=identifier,before=before,after=after))
        new_tasks.append(task)
    addition=c.read_json(HERE.parent/'storage_ap_train001_v1/plan_addition_v1.json')
    archive_task=deepcopy(addition['task'])
    plan[addition['plan_field_name']]=deepcopy(addition['plan_field_value'])
    state=deepcopy(state)
    state['tasks'][archive_task['id']]=dict(status='pending',external=False)
    retire='AM_argformer_american_validation_002_retire_native_scratch'
    for task in new_tasks:
        if task['id']==replacement['id']:
            task['dependencies'].append(retire)
        if task['id']=='AM_argformer_american_validation_002_argformer_embeddings':
            task['dependencies'].append(archive_task['id'])
    plan['tasks']=[archive_task]+new_tasks
    plan,path_receipt=paths.repair_plan_paths(plan,state,OUTPUT)
    plan['retention_owner_rebinding']=paths.rebind_owner_paths(plan,state)
    plan['tasks']=[archive_task,next(t for t in plan['tasks'] if t['id']==replacement['id'])]+[t for t in plan['tasks'] if t['id'] not in {replacement['id'],archive_task['id']}]
    plan['records_directory']=RECORDS
    plan['continuation_approval_key']='approved_recovery_continuation_workflow_ref'
    plan['operational_successor_of']=old_ref
    plan['execution_authority']='Separate continuation authority preserves active v1 American worker and old overlay approvals'
    plan['task_family_dispatch_order']='Lossless external archive, VAL002 native embeddings/retrieval/retirement, then C4 retry. Operational capacity dependencies preserve every scientific setting'
    plan['operational_capacity_dependencies']={replacement['id']:[retire],'AM_argformer_american_validation_002_argformer_embeddings':[archive_task['id']]}
    plan['recovery_resource_changes']=dict(C4_allocator_GiB=68,C4_minimum_free_GiB=78,C4_host_RSS_GiB=512,
        C4_original_epochs=30,C4_original_full_parent_updates=90,qualified_sources_unchanged=22,
        max_parallel_RSS_GiB=1792,host_margin_GiB=64)
    plan['recovery_paths_evidence_ref']=queue.save(ACTUAL/'path_rebinding.json',dict(generic=scope_changes,native=path_receipt),exclusive=True)
    refs={ref['path']:ref for ref in old['sources']}
    for path in sorted((HERE.parent/'runtime_v2').glob('*.py')):refs[c.file_ref(path)['path']]=c.file_ref(path)
    for path in [Path(__file__),HERE/'runtime_manifest.json',HERE.parent/'european/c4_task_resource_patch_v2.json',HERE.parent/'european/c4_failure_receipt_v1.json']:
        refs[c.file_ref(path)['path']]=c.file_ref(path)
    for ref in addition['additional_source_refs']:refs[ref['path']]=ref
    for path in [Path(facade.__file__),HERE.parent/'storage_ap_train001_v1/plan_addition_v1.json']:
        refs[c.file_ref(path)['path']]=c.file_ref(path)
    plan['sources']=[refs[key] for key in sorted(refs)]
    reference=queue.save(PLAN,plan,exclusive=True)
    queue.load_plan(reference)
    initial=queue.initial_state(plan,reference)
    projection=queue.storage_projection(plan,initial)
    cpu,rss=queue.external_resources(plan)
    queue.need(projection+archive_task['max_output_bytes']<980000000000,'Archive must fit before any storage credit')
    queue.need(queue.storage_projection(plan,initial)==facade._BASE_PROJECTION(plan,initial),'Initial supplemental storage credit must be exactly zero')
    queue.need(rss+replacement['max_rss_bytes']<=plan['max_parallel_rss_bytes'],'C4 retry admission must fit full RAM reservations')
    receipt=dict(plan_ref=reference,source_state_snapshot_ref=snapshot_ref,tasks=len(plan['tasks']),
        counts=dict(Counter(row['status'] for row in initial['tasks'].values())),healthy_local_jobs_preserved=sorted(active_ids),
        external_full_RSS_bytes=rss,storage_projection_bytes=projection,storage_with_C4_retry_bytes=projection+replacement['max_output_bytes'],
        old_models_data_sources_and_training_budgets_preserved=True,supplemental_archive_task=archive_task['id'],
        archive_active_projection_bytes=projection+archive_task['max_output_bytes'],initial_supplemental_credit_bytes=0,
        C4_scientific_dependencies_retained=True,operational_capacity_dependencies=plan['operational_capacity_dependencies'],
        scientific_launches=0,policy_changes=0)
    queue.save(ACTUAL/'preparation_receipt.json',receipt,exclusive=True)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':build()
