"""Metadata-only exact failed-partial inventory. Never compress, delete or launch."""
import copy
import json
from pathlib import Path
from . import closed_archives_v1 as old
from . import closed_archives_v2 as new

ROOT = old.ROOT

def main():
    config_ref = old.reference(ROOT,'research/repairs/benchmark_recovery_20261020/storage/compression_plan_v3.json')
    config = old.read(ROOT,config_ref)
    state_path = old.path(ROOT,config['records_directory'])/'state.json'
    state_ref = old.reference(ROOT,state_path)
    state = old.read(ROOT,state_ref)
    first = config['recovery_plan']['tasks'][0]
    entry = state['tasks'][first['id']]
    output = old.path(ROOT,first['output_directory'])
    attempt = dict(config_ref=config_ref,state_ref=state_ref,task_id=first['id'],entry=entry,
        job_ref=entry['job_ref'],wait_ref=old.reference(ROOT,output/'actual_wait.json'),
        output_directory=first['output_directory'],retained_bytes=old.scope_bytes(output),
        supervisor_identity=dict(pid=3664137,start_ticks=512666377))
    new._failed_attempt(attempt,ROOT)
    original_ref = old.reference(ROOT,old.INVENTORY)
    original = old.read(ROOT,original_ref)
    prior = {}
    for sidecar in sorted((output/'native').glob('file_*.json')):
        value = json.loads(sidecar.read_text())
        old.need(value['state']=='archive_committed' and value['decompressed_hash_and_size_verified'] is True,
                 'Only genuine fully committed prior sidecars can be reused')
        source = old.path(ROOT,value['original_path'])
        archive = old.path(ROOT,value['archive_path'])
        key = str(source.relative_to(ROOT))
        old.need(key not in prior and not source.exists() and archive.is_relative_to(output/'native'),
                 'One exact committed missing original with retained gzip required')
        prior[key] = dict(original_path=key,sidecar_ref=old.reference(ROOT,sidecar),entry=value,
            archive_identity=old.record(archive)['identity'],attempt_output_directory=first['output_directory'])
    targets,used = [],set()
    for target in original['targets']:
        rows = []
        for row in target['candidates']:
            item = old.path(ROOT,row['path'])
            if row['path'] in prior:
                old.need(target['key']==first['closed_storage_recovery_target'],'Prior archive must belong to exact first failed target')
                rows.append(prior[row['path']]);used.add(row['path'])
            else:
                old.need(old.record(item)['identity']==row['identity'],'Every other original must retain exact identity')
        for row in target['protected_files']:
            old.need(old.record(old.path(ROOT,row['path']))['identity']==row['identity'],'Protected evidence changed')
        targets.append(dict(key=target['key'],original=copy.deepcopy(target),
            start_scope_bytes=old.scope_bytes(old.path(ROOT,target['scope'])),prior_archives=rows))
    old.need(used==set(prior) and len(prior)==114,'All exact114 prior committed archives must be covered once')
    inventory = dict(schema=new.SCHEMA+'_inventory',original_inventory_ref=original_ref,
        source_plan_ref=config['source_plan_ref'],initial_credit_bytes=0,
        preserved_failed_attempts=[attempt],targets=targets,
        no_historical_failure_relabelled=True,all_other_targets_unchanged=True,
        fresh_full_prior_gzip_verification_required_at_runtime=True,
        old_standalone_config_approval_must_be_removed_before_successor_writes=True)
    path = old.path(ROOT,new.INVENTORY)
    with path.open('x') as handle:
        json.dump(inventory,handle,indent=2,sort_keys=True);handle.write('\n')
    reference = old.reference(ROOT,path)
    for target in targets:
        new.load_target(reference,target['key'])
    base = old.read(ROOT,config['source_plan_ref'])
    plan,evidence = new.add_recovery_tasks(base,
        'experiments/results/benchmark_campaign_20261018_workflow/recovery_20261020/storage_v2')
    artifact = dict(schema='versioned_closed_storage_recovery_tasks_v2',inventory_ref=reference,
        source_plan_ref=config['source_plan_ref'],recovery_helper_module=new.MODULE,
        tasks=plan['tasks'][:12],closed_storage_recovery=plan['closed_storage_recovery'],
        preserved_failed_external_storage=[v for v in plan['external_storage'] if v.get('scopes')==[first['output_directory']]],
        source_refs=[old.reference(ROOT,name) for name in new.NEW_SOURCES],evidence=evidence,
        required_policy_remove_approved_closed_storage_recovery_refs=[config_ref],
        actual_failed_wait_preserved=True,initial_credit_bytes=0,execution_started=False)
    artifact_path = path.with_name('recovery_tasks_v2.json')
    with artifact_path.open('x') as handle:
        json.dump(artifact,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps(dict(inventory_ref=reference,tasks_ref=old.reference(ROOT,artifact_path),
        prior_sidecars=len(prior),failed_attempt_retained_bytes=attempt['retained_bytes'],tasks=len(artifact['tasks'])),sort_keys=True))

if __name__=='__main__':main()
