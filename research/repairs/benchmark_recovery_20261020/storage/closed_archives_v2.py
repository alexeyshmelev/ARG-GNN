"""Lossless versioned completion after a genuinely failed partial archive.

ROOT activates the single successor queue. Original failures and all completed
partial archives remain immutable and separately charged. No automatic replay.
"""
from __future__ import annotations
import copy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
from . import closed_archives_v1 as old

ROOT = old.ROOT
SOURCE = 'research/repairs/benchmark_recovery_20261020/storage/closed_archives_v2.py'
INVENTORY = 'research/repairs/benchmark_recovery_20261020/storage/closed_archive_inventory_v2.json'
SHIM = 'src/arggnn/benchmarking/storage_recovery_20261020_v2.py'
MODULE = 'arggnn.benchmarking.storage_recovery_20261020_v2'
SCHEMA = 'exact_closed_european_archive_20261020_v2'
NEW_SOURCES = (SOURCE, INVENTORY, SHIM) + old.NEW_SOURCES
need, path, reference, read, record = old.need, old.path, old.reference, old.read, old.record
codec = old.codec

def live_scope_bytes(value):
    """Active writers may atomically replace names; only vanished entries skip."""
    value = path(ROOT,value)
    total = 0
    def walk_error(exc):
        if not isinstance(exc,(FileNotFoundError,ProcessLookupError)):
            raise exc
    for base,dirs,files in os.walk(value,followlinks=False,onerror=walk_error):
        for name in dirs+files:
            item = Path(base)/name
            try:
                info = item.lstat()
                need(not stat.S_ISLNK(info.st_mode),'Recovery output cannot contain symlinks')
            except (FileNotFoundError,ProcessLookupError):
                continue
            total += max(info.st_size,info.st_blocks*512)+4096
    return total

def _failed_attempt(attempt,repo):
    config = read(repo,attempt['config_ref'])
    job = read(repo,attempt['job_ref'])
    wait = read(repo,attempt['wait_ref'])
    state = read(repo,attempt['state_ref'])
    task = next(t for t in config['recovery_plan']['tasks'] if t['id']==attempt['task_id'])
    entry = state['tasks'][attempt['task_id']]
    need(state['plan_ref']==attempt['config_ref'] and job['plan_ref']==attempt['config_ref']
         and job['task']==task and entry==attempt['entry']
         and path(repo,task['output_directory'])==path(repo,attempt['output_directory'])
         and entry['status']=='failed' and type(entry.get('actual_controller_wait')) is int
         and entry['actual_controller_wait']!=0 and entry.get('owned_process_group_absent') is True
         and wait.get('job_ref')==attempt['job_ref'] and wait.get('actual_direct_wait') is True
         and wait.get('actual_direct_returncode')==entry['actual_controller_wait']
         and wait.get('owned_process_group_absent') is True,
         'Exact genuinely failed archive wait and unchanged failure lineage required')
    need(old.scope_bytes(path(repo,attempt['output_directory']))==attempt['retained_bytes'],
         'Preserved failed archive output changed')
    expected = attempt['supervisor_identity']
    proc = Path('/proc')/str(expected['pid'])/'stat'
    try:
        raw = proc.read_text()
        ticks = int(raw[raw.rfind(')')+2:].split()[19])
        need(ticks!=expected['start_ticks'],'Original archive controller is still live')
    except (FileNotFoundError,ProcessLookupError):
        pass
    return attempt

def load_target(inventory_ref,target_key,repo=ROOT):
    inventory = read(repo,inventory_ref)
    need(inventory['schema']==SCHEMA+'_inventory' and inventory['initial_credit_bytes']==0,
         'Exact zero-credit versioned recovery inventory required')
    target = next(row for row in inventory['targets'] if row['key']==target_key)
    original = old.load_target(inventory['original_inventory_ref'],target_key,repo)
    need(target['original']==original,'Original closed scope and complete candidate set must remain unchanged')
    attempts = {row['output_directory']:_failed_attempt(row,repo) for row in inventory['preserved_failed_attempts']}
    originals = {row['path']:row for row in original['candidates']}
    seen = set()
    for prior in target['prior_archives']:
        need(prior['original_path'] in originals and prior['original_path'] not in seen,
             'One exact prior archive per inventoried original required')
        seen.add(prior['original_path'])
        need(prior['attempt_output_directory'] in attempts,'Prior archive must have an explicitly charged failed owner')
        value = read(repo,prior['sidecar_ref'],2**20)
        archive = path(repo,value['archive_path'])
        need(value==prior['entry'] and value['state']=='archive_committed'
             and value.get('decompressed_hash_and_size_verified') is True
             and path(repo,value['original_path'])==path(repo,prior['original_path'])
             and value['original_bytes']==originals[prior['original_path']]['identity'][2]
             and archive.is_relative_to(path(repo,prior['attempt_output_directory'])/'native')
             and record(archive)['identity']==prior['archive_identity']
             and archive.stat().st_size==value['archive_bytes'],
             'Exact durable verified prior gzip and sidecar required')
    return inventory,target

def _check_start(target,repo):
    original = target['original']
    owner = path(repo,original['scope'])
    need(old.scope_bytes(owner)==target['start_scope_bytes'],'Closed scope changed after versioned inventory')
    prior = {row['original_path']:row for row in target['prior_archives']}
    for row in original['candidates']:
        item = path(repo,row['path'])
        if row['path'] in prior:
            need(not item.exists(),'A committed prior raw path must remain absent')
        else:
            need(record(item)['identity']==row['identity'],'Remaining original candidate changed')
    for row in original['protected_files']:
        need(record(path(repo,row['path']))['identity']==row['identity'],'Protected science or original evidence changed')

def _old_authority_revoked(inventory,repo):
    policy = json.loads(path(repo,'configs/benchmarks/execution_policy.json').read_text())
    for attempt in inventory['preserved_failed_attempts']:
        need(attempt['config_ref'] not in policy.get('approved_closed_storage_recovery_refs',[]),
             'ROOT must remove the failed standalone batch approval before successor writes')

def factory(*,inventory_ref,target_key,output_directory):
    inventory,target = load_target(inventory_ref,target_key)
    _check_start(target,ROOT)
    spec = dict(schema=SCHEMA,inventory_ref=inventory_ref,target_key=target_key,output_directory=output_directory)
    return dict(module=MODULE,function='run',python='.runtime/env/bin/python',spec=spec,
                source_refs=[reference(ROOT,name) for name in NEW_SOURCES])

def _verify_prior(prior,check):
    value = prior['entry']
    archive = path(ROOT,value['archive_path'])
    need(record(archive)['identity']==prior['archive_identity'],'Prior archive identity changed')
    archive_hash = hashlib.sha256()
    with archive.open('rb') as handle:
        for block in iter(lambda:handle.read(8*2**20),b''):
            check('verify_preserved_gzip_bytes')
            archive_hash.update(block)
    restored_hash,restored_bytes = hashlib.sha256(),0
    with gzip.open(archive,'rb') as handle:
        for block in iter(lambda:handle.read(8*2**20),b''):
            check('verify_preserved_gzip_content')
            restored_hash.update(block)
            restored_bytes += len(block)
    need(archive_hash.hexdigest()==value['archive_sha256']
         and restored_hash.hexdigest()==value['original_sha256'] and restored_bytes==value['original_bytes']
         and record(archive)['identity']==prior['archive_identity'],
         'Fresh full preserved gzip digest, restored digest and size verification required')
    return dict(value,archive_identity=prior['archive_identity'],prior_sidecar_ref=prior['sidecar_ref'],
                preserved_failed_attempt_output=prior['attempt_output_directory'],fresh_repair_gzip_verification=True)

def run(spec,*,resource_check,execution_check=None,bounded_runner=None):
    need(spec.get('schema')==SCHEMA and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Exact CPU-only versioned recovery required')
    inventory,target = load_target(spec['inventory_ref'],spec['target_key'])
    _check_start(target,ROOT)
    output = path(ROOT,spec['output_directory'])
    owner = path(ROOT,target['original']['scope'])
    need(not output.is_relative_to(owner) and not owner.is_relative_to(output),'New separately charged repair scope required')
    output.mkdir(parents=True,exist_ok=False)
    original = target['original']
    check_execution = execution_check or (lambda:None)
    def authority():
        check_execution()
        _old_authority_revoked(inventory,ROOT)
    authority()
    lockdir = path(ROOT,'.runtime/locks')
    lockdir.mkdir(parents=True,exist_ok=True)
    key = hashlib.sha256(str(owner).encode()).hexdigest()
    descriptor = os.open(lockdir/('closed_recovery_'+key+'.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    entries = []
    with os.fdopen(descriptor,'r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        codec._save(output/'repair_inventory.json',target)
        prior = {row['original_path']:row for row in target['prior_archives']}
        for index,row in enumerate(original['candidates']):
            authority()
            if row['path'] in prior:
                entry = _verify_prior(prior[row['path']],resource_check)
                codec._save(output/('prior_file_%05d.json'%index),entry)
            else:
                source = path(ROOT,row['path'])
                need(record(source)['identity']==row['identity'],'Remaining candidate changed before compression')
                allowance = min(old.MAX_TEMP,original['max_output_bytes']-live_scope_bytes(output)-old.METADATA)
                need(allowance>=2**20,'No admitted temporary output allowance remains')
                def before_unlink():
                    authority()
                    wait = read(ROOT,original['producer']['wait_ref'],2**20)
                    need(wait.get('actual_direct_wait') is True and wait.get('owned_process_group_absent') is True,
                         'Original genuine producer drain evidence changed')
                entry = codec.archive_one(source,output/('file_%05d.json'%index),allowance,resource_check,before_unlink)
                if entry['state']=='archive_committed':
                    entry['archive_identity'] = record(path(ROOT,entry['archive_path']))['identity']
            entries.append(entry)
        for row in original['protected_files']:
            need(record(path(ROOT,row['path']))['identity']==row['identity'],'Protected science or evidence changed')
        after = old.scope_bytes(owner)
        need(after<=target['start_scope_bytes'],'Closed original scope grew during repair')
        manifest = dict(schema=SCHEMA,inventory_ref=spec['inventory_ref'],target_key=target['key'],
            owner_scope=original['scope'],archive_output_directory=str(output),entries=entries,
            original_wait_ref=original['producer']['wait_ref'],original_status_preserved=original['producer']['entry']['status'],
            before_scope_bytes=original['before_scope_bytes'],repair_start_scope_bytes=target['start_scope_bytes'],after_scope_bytes=after,
            original_scope_reduction_bytes=original['before_scope_bytes']-after,
            every_original_byte_restorable=True,protected_science_and_evidence_unchanged=True,
            raw_replacement_after_verified_durable_archive=True,
            archives_in_new_or_exact_separately_charged_failed_scopes=True,
            preserved_failed_attempts=inventory['preserved_failed_attempts'],old_failure_relabelled=False)
        codec._save(output/'restore_manifest.json',manifest)
    return reference(ROOT,output/'restore_manifest.json')

def add_recovery_tasks(plan,output_root,repo=ROOT):
    need(plan['storage_ceiling_bytes']==980_000_000_000,'Preserve strict 980 GB ceiling')
    inventory_ref = reference(repo,INVENTORY)
    inventory = read(repo,inventory_ref)
    result = copy.deepcopy(plan)
    tasks = []
    for row in inventory['targets']:
        _,target = load_target(inventory_ref,row['key'],repo)
        original = target['original']
        tasks.append(dict(id='STORAGE_RECOVERY_V2__'+target['key'],dependencies=[] if not tasks else [tasks[-1]['id']],resource='cpu',
            max_rss_bytes=2**30,max_output_bytes=original['max_output_bytes'],max_seconds=86400,
            output_directory=str(Path(output_root)/'closed_storage'/('STORAGE_RECOVERY_V2__'+target['key'])),
            factory_module=MODULE,factory_function='factory',
            factory_arguments=dict(inventory_ref=inventory_ref,target_key=target['key']),
            closed_storage_recovery_target=target['key']))
    need(not {t['id'] for t in tasks}&{t['id'] for t in result['tasks']},'No automatic versioned recovery replay')
    need(not any(t.get('closed_storage_recovery_target') for t in result['tasks']),
         'Remove old standalone dispatch descriptors, preserving their frozen failure records')
    result['tasks'] = tasks+result['tasks']
    result['closed_storage_recovery'] = dict(schema=SCHEMA,recovery_helper_module=MODULE,inventory_ref=inventory_ref,
        initial_credit_bytes=0,old_scope_writer_revocation_required=True,
        preserved_failed_attempts=inventory['preserved_failed_attempts'])
    for attempt in inventory['preserved_failed_attempts']:
        matches = [v for v in result['external_storage'] if v.get('scopes')==[attempt['output_directory']]]
        need(len(matches)<=1,'Failed archive scope must be charged once')
        if not matches:
            result['external_storage'].append(dict(scopes=[attempt['output_directory']],cap_bytes=attempt['retained_bytes'],
                completion_evidence=[],actual_wait_ref=attempt['wait_ref'],retention='Preserved genuine failed partial archive; no old-task credit'))
        else:
            need(matches[0]['cap_bytes']==attempt['retained_bytes'],'Keep exact preserved partial archive charge')
    return result,dict(tasks_added=len(tasks),initial_credit_bytes=0,preserved_failed_output_bytes=sum(v['retained_bytes'] for v in inventory['preserved_failed_attempts']))

def completed_storage_reduction(plan,state,repo=ROOT):
    policy = plan.get('closed_storage_recovery')
    if policy is None:
        return 0
    need(policy['schema']==SCHEMA and policy['recovery_helper_module']==MODULE and policy['initial_credit_bytes']==0,
         'Exact versioned zero-initial-credit recovery policy required')
    inventory = read(repo,policy['inventory_ref'])
    need(policy['preserved_failed_attempts']==inventory['preserved_failed_attempts'],'Preserved failure bindings changed')
    for attempt in inventory['preserved_failed_attempts']:
        _failed_attempt(attempt,repo)
        matches = [v for v in plan['external_storage'] if v.get('scopes')==[attempt['output_directory']]]
        need(len(matches)==1 and matches[0]['cap_bytes']==attempt['retained_bytes'],
             'Every reused prior archive must remain separately charged exactly once')
    total,seen = 0,set()
    for task in plan['tasks']:
        if task.get('factory_module')!=MODULE:
            continue
        entry = state['tasks'][task['id']]
        if entry['status']!='succeeded':
            continue
        _,target = load_target(policy['inventory_ref'],task['closed_storage_recovery_target'],repo)
        original = target['original']
        need(original['scope'] not in seen,'No duplicate closed original scope credit')
        seen.add(original['scope'])
        need(entry.get('actual_controller_wait')==0 and entry.get('owned_process_group_absent') is True,
             'New repair requires genuine successful controller wait and drain')
        job = read(repo,entry['job_ref'])
        output = path(repo,task['output_directory'])/'native'
        expected_spec = dict(schema=SCHEMA,inventory_ref=policy['inventory_ref'],target_key=target['key'],output_directory=str(output))
        need(job['task']==task and job['call']['module']==MODULE and job['call']['function']=='run'
             and job['call']['spec']==expected_spec,'Exact new repair task and scientific descriptor required')
        wait = read(repo,reference(repo,path(repo,task['output_directory'])/'actual_wait.json'),2**20)
        need(wait.get('job_ref')==entry['job_ref'] and wait.get('actual_direct_wait') is True
             and wait.get('actual_direct_returncode')==0 and wait.get('owned_process_group_absent') is True,
             'New successful actual direct wait and absence required')
        envelope = read(repo,entry['result_ref'])
        need(envelope.get('job_ref')==entry['job_ref'] and path(repo,envelope['value']['path'])==output/'restore_manifest.json',
             'Exact new complete restoration manifest envelope required')
        manifest = read(repo,envelope['value'])
        need(manifest['schema']==SCHEMA and manifest['inventory_ref']==policy['inventory_ref']
             and manifest['target_key']==target['key'] and manifest['owner_scope']==original['scope']
             and manifest['original_wait_ref']==original['producer']['wait_ref']
             and manifest['every_original_byte_restorable'] is True
             and manifest['protected_science_and_evidence_unchanged'] is True
             and manifest['raw_replacement_after_verified_durable_archive'] is True
             and manifest['archives_in_new_or_exact_separately_charged_failed_scopes'] is True
             and manifest['preserved_failed_attempts']==inventory['preserved_failed_attempts']
             and manifest['old_failure_relabelled'] is False,'Complete versioned lossless restoration proof required')
        originals = {row['path']:row for row in original['candidates']}
        prior = {row['original_path']:row for row in target['prior_archives']}
        need(len(manifest['entries'])==len(originals),'Every original candidate requires one terminal restoration entry')
        used = set()
        for row in manifest['entries']:
            original_path = path(repo,row['original_path'])
            key = str(original_path.relative_to(Path(repo).resolve()))
            need(key in originals and key not in used and row['decompressed_hash_and_size_verified'] is True
                 and row['original_bytes']==originals[key]['identity'][2],'One exact verified original per restoration entry')
            used.add(key)
            if row['state']=='archive_committed':
                archive = path(repo,row['archive_path'])
                need(not original_path.exists() and record(archive)['identity']==row['archive_identity']
                     and archive.stat().st_size==row['archive_bytes'],'Current exact committed gzip required')
                if key in prior:
                    frozen = prior[key]
                    need(archive==path(repo,frozen['entry']['archive_path'])
                         and row.get('prior_sidecar_ref')==frozen['sidecar_ref']
                         and row.get('fresh_repair_gzip_verification') is True
                         and row['original_sha256']==frozen['entry']['original_sha256']
                         and row['archive_sha256']==frozen['entry']['archive_sha256'],
                         'Prior gzip must be freshly verified and remain in its charged failed scope')
                else:
                    need(archive.is_relative_to(output),'New gzip must remain inside the new repair output')
            else:
                need(key not in prior and row['state']=='original_kept_no_storage_gain' and row['credited_bytes']==0
                     and record(original_path)['identity']==originals[key]['identity'],'Kept original receives no speculative credit')
        binding = original['accounting']
        if binding['kind']=='external_storage':
            matches = [v for v in plan['external_storage'] if v['scopes']==binding['group_scopes']]
            need(len(matches)==1 and matches[0]['cap_bytes']==binding['group_cap_bytes'],'Original external charge changed')
        else:
            matches = [v for v in plan['workflow_external_predecessors'] if v['source_task_id']==binding['source_task_id']]
            need(binding['kind']=='workflow_predecessor' and len(matches)==1
                 and matches[0]['original_job_ref']==original['producer']['job_ref'],'Original predecessor charge changed')
        current = old.scope_bytes(path(repo,original['scope']))
        need(current==manifest['after_scope_bytes'] and manifest['before_scope_bytes']==original['before_scope_bytes']
             and 0<=current<=target['start_scope_bytes']
             and manifest['original_scope_reduction_bytes']==original['before_scope_bytes']-current,
             'Exact current original-scope reduction required')
        for row in original['protected_files']:
            need(record(path(repo,row['path']))['identity']==row['identity'],'Protected original changed')
        total += original['before_scope_bytes']-current
    return total
