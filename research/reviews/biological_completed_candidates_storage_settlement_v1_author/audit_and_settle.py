"""Exact two completed candidate storage slots: source-only, root inspect, root apply."""
from pathlib import Path
import argparse, contextlib, copy, datetime, fcntl, hashlib, importlib.util, json, math, os, stat
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
AUTHOR=ROOT/'research/reviews/biological_completed_candidates_storage_settlement_v1_author'
SCRIPT=AUTHOR/'audit_and_settle.py'
CONFIG=AUTHOR/'config.json'
FIXED_CONFIG={'kind': 'biological_completed_candidates_storage_settlement_v1', 'targets': [{'key': 'biological_gpu2_v1/biological-candidate-seed202', 'seed': 202, 'old_bytes': 29281048576, 'new_bytes': 15000000000, 'output': 'results/biological_candidate_pilot_v1_seed202', 'retained_pool_log': '.runtime/gpup/biological_gpu2_v1/jobs/biological-candidate-seed202.log', 'registry': '.runtime/gpup/biological_gpu2_v1/registry.json', 'prior_stat_passes': [{'path': 'research/reviews/biological_completed_candidates_storage_inventory_v1/seed202_stat_inventory_pass1.json', 'sha256': '31bb803476540f04982e72b134c2e8d47cc15dcbda485e81fba7aa3c397c7fce'}, {'path': 'research/reviews/biological_completed_candidates_storage_inventory_v1/seed202_stat_inventory_pass2.json', 'sha256': '0f345c4550939885080932a25bb89f0166b2e503137a3223c8b9d48bedf35d76'}]}, {'key': 'biological_gpu3_v1/biological-candidate-seed303', 'seed': 303, 'old_bytes': 29281048576, 'new_bytes': 15000000000, 'output': 'results/biological_candidate_pilot_v1_seed303', 'retained_pool_log': '.runtime/gpup/biological_gpu3_v1/jobs/biological-candidate-seed303.log', 'registry': '.runtime/gpup/biological_gpu3_v1/registry.json', 'prior_stat_passes': [{'path': 'research/reviews/biological_completed_candidates_storage_inventory_v1/seed303_stat_inventory_pass1.json', 'sha256': 'aa47fb3a4da443137d902233517a094c94c168387bfcc7d6816253a7dbca2aa6'}, {'path': 'research/reviews/biological_completed_candidates_storage_inventory_v1/seed303_stat_inventory_pass2.json', 'sha256': '0b79854f5bfad3dd3d78504920a3706e901750d48e24e29a9b5046afea60d59f'}]}], 'prior_inventory': {'path': 'research/reviews/biological_completed_candidates_storage_inventory_v1/inventory_receipt.json', 'sha256': 'f39cf36c362ec3a040d70a4503b881a6bb988402960eaf0739197596f7775cfd'}, 'prior_metadata': [{'path': 'research/reviews/biological_completed_candidates_storage_inventory_v1/metadata_before.json', 'sha256': '3ee49323c9857e05734d30033e97186de5f90d4a76b3b20460c1d99337f2af13'}, {'path': 'research/reviews/biological_completed_candidates_storage_inventory_v1/metadata_after.json', 'sha256': 'afe8b6fbf243c0f50459835de098065540c0b0f288f1b9ae91cb508a57712cda'}], 'prior_GST_procedure': {'path': 'research/reviews/biological_gst_completed_storage_settlement_v1/storage_revision_v2/audit_and_settle.py', 'sha256': '9508b4c2f4e21d9a2d2d1e3b6a47e42c6296773dee43233da53e1633173e9066'}, 'prior_GST_independent_review': {'path': 'research/reviews/biological_gst_storage_settlement_v1_independent/review_receipt_v2.json', 'sha256': 'e6ebd15ef989918abf74545bf43a02beeca8cae057a8a94f1bc50f5074c66bcb'}, 'pool_source': {'path': 'src/arggnn/gpu_pool_v1.py', 'sha256': 'ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1'}, 'candidate_source': {'path': 'src/arggnn/biological_candidate.py', 'sha256': '6b3a25d0e59026cab7189032eb41221feb7d1bdf69e15f6c737a672dfb711dd3'}, 'candidate_config': {'path': 'configs/biological_candidate_pilot_v1.json', 'sha256': '4151e7e156db3f188512bcc738b43597407d7707bc77c420509165c2e9bfc2f1'}, 'candidate_source_sha256': {'src/arggnn/biological_candidate.py': '6b3a25d0e59026cab7189032eb41221feb7d1bdf69e15f6c737a672dfb711dd3', 'scripts/train_biological_candidate_v1.py': '6b14c76991c0db5ce660a760f9290e8c15ecb0afed1fbcd1c17dd7f908bc3394', 'scripts/train_ancestralpaths_biological_v1.py': '826a1e8c3053cf63b8374361f793903970dd75d44cbcd536755e9aaa3b985056', 'src/arggnn/ancestralpaths_native_bridge.py': 'fb69904e3004ca7de6f16736e542949fb1b3695edb104cdf601187a7f8d33f2c', 'src/arggnn/anchor_kernel.py': '2b8cc1c1726f746f961c0d63a4870e806a1a9d3d6e1fa6724dc0bda39967b083', 'src/arggnn/anchor_temporal_baselines.py': '3e88095eb466ea25db5f69553dfa31e6e002fa55ce4d31c4ddb17cb73115518b', 'src/arggnn/boundary_memory.py': 'a056cd10d1c848d7dabdc1daf646548876fd90e8f489667c46a374ac28cd11dc', 'src/arggnn/kron_interface.py': '09ec6286c185e7aa2b13d76dcbe8fea24b3a1c785a6619ac9bc5464e0de66d69', 'src/arggnn/age_interface.py': '15f9615a84a230386d8f71f5a317a46051c395f75408c79a6213f1b07f101f3d', 'src/arggnn/reference_encoder.py': '039f3b25383ea92f243090c2eaf592df590768fea7abf41d15fd7c34a7715138', 'src/arggnn/genealogy_baselines.py': '388e84c183413a79db8e0e75c86f1ce202b4bb9999473b729be045d7459dc9f8', 'src/arggnn/models.py': '8429bd02a40950a3dc4d76046ab4b31b366871499d95e295a16ade730f567df5', 'src/arggnn/data.py': 'fb4e4b3ad8744731e6d2c5966cc2679f9f2d59dfd079facc280d69fbc61ef654', 'scripts/infer_biological_paths_relate_v1.py': 'a63a79b36a422f7346f2a35206f41b35fb440053edcfe4e080755c649ed737e9', 'configs/biological_paths_relate_v1.json': '7da0bae298c0927fd0fce6a6ee320f0c51fc459b901de9c6f0d1e225cae05de9', 'docs/biological_paths_relate_v1.md': '285cf216c7018835bff152acc3b4756f81ef65556eaddb4691c0d1f14a163ecd', 'scripts/launch_gpu.py': 'f07324053d8e649e27d66cbca9799bb6ea237c92c247853e2d45e6d440a2c094', 'src/arggnn/native_argformer_exec.py': '84ed405379ac56511c2f4722c130e8b884226018bbf8e2949ea2bb4dceeb99d0', 'scripts/qualify_official_argformer_runtime.py': '8fe652f87848f4a296e3d88555998fb388f11ae5c04d0858095e70c34b10b31c', 'src/arggnn/development_execution.py': '4a2f87320615eb655a7700b6bbff4a7d293b1b16749468afa46f0de2d12e276f', 'src/arggnn/resources.py': 'ffc7ac9ce1ebb19b6d7030b9db643d8de9a07179318a9bb1e593779fd3f6d111', '.runtime/env/bin/nvtop': '5a7691ba86c1c950b934ebfd72887b822747965ae0581d4638e1f126e0ba29cf'}, 'models': ['candidate_age_d2', 'candidate_plain_d2', 'matched_gru_d2'], 'config_sha256': '4151e7e156db3f188512bcc738b43597407d7707bc77c420509165c2e9bfc2f1', 'observed_manifest_sha256': '6872222127c96b3defbfb1b9edb83ba1d5b11b0b60e6888956369bc507cefe28', 'prepared_manifest_sha256': '3c0a3320fbc6dd9d421a2acfad398961c4912cdbc49d111faf02108ee4f8596d', 'snapshot': {'path': 'experiments/results/biological_shared_pool_storage_admission_v1/snapshot.json', 'sha256': '0826fcedefa40dd261999c4916b85ae11f28628e5cee9046b2e4f6bf868fa10d'}, 'before_ledger_sha256': 'ddfdd905b645f48fbd7fcb7cafb3fc2c86bee801cc923abe37ba81d2b1d09e26', 'ledger_path': '.runtime/gpup/storage_reservations.json', 'scheduler_lock': '.runtime/locks/scheduler.lock', 'baseline_bytes': 167336640512, 'external_reserve_bytes': 130000000000, 'warning_bytes': 450000000000, 'hard_limit_bytes': 500000000000, 'inspection_dir': 'research/reviews/biological_completed_candidates_storage_settlement_v1_author/root_inspection', 'apply_dir': 'research/reviews/biological_completed_candidates_storage_settlement_v1_author/root_apply', 'operator_frozen_output_policy': 'Completed candidate outputs and retained pool logs are contractually frozen. Any new scope writes require renewed accounting before authorization. New fits use fresh versioned output roots and reservations. Legitimate ledger writers honor scheduler.lock and pinned terminal registries are trustworthy. Known saved identity absence and stable inventories do not prove arbitrary unknown writers absent. Future unrelated side effects remain covered by the unchanged external reserve.', 'unknown_writer_absence_claimed': False, 'no_product_files_deleted': True, 'only_two_reserved_bytes_deltas': True}

def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def exact(left,right):
    return encoded(left)==encoded(right)
def confined(root,value):
    p=Path(value);p=p if p.is_absolute() else root/p
    if p==root or root not in p.parents or '..' in p.parts or p.resolve()!=p:
        raise ValueError('canonical non-symlink project scope required')
    return p

def stat_fields(info):
    return dict(device=info.st_dev,inode=info.st_ino,mode=info.st_mode,size_bytes=info.st_size,
                allocated_bytes=info.st_blocks*512,mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns,
                link_count=info.st_nlink,uid=info.st_uid,gid=info.st_gid,
                retained_charge_bytes=max(info.st_size,info.st_blocks*512))
def opaque_file(root,path,metadata=False):
    p=confined(root,path);before=p.lstat()
    if not stat.S_ISREG(before.st_mode):raise ValueError('canonical regular file required')
    if metadata and before.st_size>16*1024*1024:raise ValueError('metadata size bound exceeded')
    digest=hashlib.sha256();chunks=[]
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        if stat_fields(os.fstat(stream.fileno()))!=stat_fields(before):raise ValueError('file identity changed before read')
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
            if metadata:chunks.append(block)
        after=os.fstat(stream.fileno())
    if stat_fields(before)!=stat_fields(after) or stat_fields(after)!=stat_fields(p.lstat()):
        raise ValueError('file stat changed while hashing')
    return {'sha256':digest.hexdigest(),**stat_fields(after)},b''.join(chunks)
def sha(path):
    return opaque_file(ROOT,path)[0]['sha256']
def reference(path):
    p=confined(ROOT,path);return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def parse_json(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def bound_json(root,ref,expected=None):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('exact path/hash reference required')
    digest=ref['sha256']
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('exact SHA256 required')
    p=confined(root,ref['path'])
    if p.suffix!='.json' or (expected is not None and p!=confined(root,expected)):raise ValueError('exact JSON scope required')
    info,raw=opaque_file(root,p,metadata=True)
    if info['sha256']!=digest:raise ValueError('bound metadata bytes changed')
    return parse_json(raw)
def write_new(root,directory,name,value):
    p=confined(root,directory/name)
    if directory not in p.parents:raise ValueError('owned audit scope required')
    with p.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    info,_=opaque_file(root,p);return {'path':str(p.relative_to(root)),'sha256':info['sha256']}
def verify_sources(c):
    for field in ['prior_GST_procedure','prior_GST_independent_review','pool_source','candidate_source','candidate_config']:
        if sha(c[field]['path'])!=c[field]['sha256']:raise ValueError('reviewed source reference changed '+field)
    candidate=bound_json(ROOT,c['candidate_config'])
    if not exact(candidate['source_sha256'],c['candidate_source_sha256']) or not exact(candidate['models'],c['models']):raise ValueError('candidate scientific source/config bindings differ')
def policy():
    c=parse_json(CONFIG.read_bytes())
    if not exact(c,FIXED_CONFIG):raise ValueError('exact sealed two-candidate policy required')
    return c

def prior_evidence(root,c):
    receipt=bound_json(root,c['prior_inventory'])
    if (receipt['schema']!='completed_candidate_storage_inventory_receipt_v1' or receipt['status']!='preliminary_stat_inventory_complete'
        or receipt['settlement_applied'] is not False or receipt['settlement_authorized'] is not False
        or receipt['current_ledger_sha256']!=c['before_ledger_sha256'] or receipt['before_ledger_sha256']!=c['before_ledger_sha256']):
        raise ValueError('exact prior stat-only inventory/ledger binding required')
    for ref in receipt['artifact_refs']:
        ref={k:ref[k] for k in ['path','sha256']};bound_json(root,ref)
    metadata=[bound_json(root,ref) for ref in c['prior_metadata']]
    targets={}
    for target in c['targets']:
        key=target['key'];rows=[next(x for x in m['candidates'] if x['key']==key) for m in metadata]
        if not exact(rows[0]['terminal_job'],rows[1]['terminal_job']) or any(m['ledger_sha256']!=c['before_ledger_sha256'] for m in metadata):raise ValueError('prior terminal/ledger evidence drift')
        passes=[bound_json(root,ref) for ref in target['prior_stat_passes']]
        if not exact(passes[0]['entries'],passes[1]['entries']):raise ValueError('prior stat membership drift')
        for inv in passes:
            if inv['key']!=key or inv['seed']!=target['seed'] or inv['output']!=str(confined(root,target['output'])) or inv['retained_pool_log']!=str(confined(root,target['retained_pool_log'])):raise ValueError('prior inventory target scope drift')
        targets[key]={'terminal_job':rows[1]['terminal_job'],'prior_stat_entries':passes[1]['entries']}
    return targets

def common_bindings(c):
    return {'root_authorizer':'root_agent','active':True,'procedure':reference(SCRIPT),'config':reference(CONFIG),
            'prior_inventory':c['prior_inventory'],'expected_before_ledger_sha256':c['before_ledger_sha256'],
            'scope_keys':[t['key'] for t in c['targets']],'old_bytes_each':29281048576,'retained_bytes_each':15000000000,
            'operator_frozen_output_policy':c['operator_frozen_output_policy'],'operator_freeze_confirmed':True,
            'unknown_writer_absence_claimed':False,'delete_product_files_authorized':False,
            'controller_mutation_authorized':False,'scientific_array_decode_authorized':False,'GPU_authorized':False}
def validate_authority(root,c,mode,authority,inspection=None,review=None):
    expected={'kind':'biological_completed_candidates_storage_settlement_v1_root_'+mode+'_admission',**common_bindings(c)}
    if any(not exact(authority.get(k),v) for k,v in expected.items()):raise ValueError('exact root admission/frozen-output contract required')
    if mode=='apply':
        expected_inspection=root/c['inspection_dir']/'inspection.json'
        inspected=bound_json(root,authority['inspection'],expected_inspection)
        if inspection is None or not exact(inspected,inspection):raise ValueError('exact inspected scope required')
        fields={'kind':'biological_completed_candidates_storage_settlement_v1_independent_review','review_role':'independent_reviewer','verdict':'PASS',
                'procedure':expected['procedure'],'config':expected['config'],'inspection':authority['inspection'],
                'prior_inventory':c['prior_inventory'],'expected_before_ledger_sha256':c['before_ledger_sha256'],
                'scope_keys':expected['scope_keys'],'operator_frozen_output_contract_accepted':True,'actual_apply_performed':False}
        if review is None or any(not exact(review.get(k),v) for k,v in fields.items()):raise ValueError('fresh exact independent inspected-source PASS required')
    return expected

def terminal(root,c,target,prior,module):
    registry_path=confined(root,target['registry']);regstat,raw=opaque_file(root,registry_path,metadata=True);registry=parse_json(raw)
    matches=[j for j in registry['jobs'] if j.get('logical_job')==target['key'].split('/',1)[1]]
    if len(matches)!=1 or registry['safety_error'] is not None:raise ValueError('unique trustworthy terminal registry job required')
    job=matches[0]
    if not exact(job,prior['terminal_job']):raise ValueError('terminal candidate job differs from sealed identity')
    requested=job['requested']
    if (job['state']!='succeeded' or type(job['exit_code']) is not int or job['exit_code']!=0
        or requested['module']!='arggnn.biological_candidate' or requested['output']!=str(confined(root,target['output']))
        or type(requested['seed']) is not int or requested['seed']!=target['seed'] or requested['storage_bytes']!=target['old_bytes']
        or job['module_source_sha256']!=c['candidate_source']['sha256'] or job['log_path']!=str(confined(root,target['retained_pool_log']))):raise ValueError('exact completed candidate job provenance required')
    identities=job['process_identities']
    if not identities or not any(s['pid']==job['pid'] and s['start_ticks']==job['leader_start_ticks'] for s in identities):raise ValueError('saved leader and descendant identity coverage required')
    process_checks=[]
    for saved in identities:
        if type(saved['pid']) is not int or saved['pid']<=0 or type(saved['start_ticks']) is not int or saved['start_ticks']<=0:raise ValueError('positive exact saved PID/start identity required')
        try:current=module.proc_identity(saved['pid'])
        except (FileNotFoundError,ProcessLookupError):current=None
        if current is not None and current['start_ticks']==saved['start_ticks']:raise ValueError('known candidate worker/descendant remains present')
        process_checks.append({'pid':saved['pid'],'saved_start_ticks':saved['start_ticks'],'saved_identity_present':False,'reason':'absent_or_PID_reused'})
    receipt_path=confined(root,Path(target['output'])/'receipt.json');receiptstat,raw=opaque_file(root,receipt_path,metadata=True);receipt=parse_json(raw)
    expected={'kind':'biological_candidate_fit_v1','status':'completed_three_conditions_one_seed','seed':target['seed'],
              'config_sha256':c['config_sha256'],'observed_manifest_sha256':c['observed_manifest_sha256'],
              'prepared_manifest_sha256':c['prepared_manifest_sha256'],'source_sha256':c['candidate_source_sha256'],'TEST_parents':0}
    if any(not exact(receipt.get(k),v) for k,v in expected.items()) or set(receipt['models'])!=set(c['models']):raise ValueError('genuine exact three-model completed candidate receipt required')
    for name,model in receipt['models'].items():
        delta=model.get('selected_maximum_parameter_delta_from_initial')
        if (model.get('status')!='completed' or model.get('optimizer_fit_completed') is not True
            or type(model.get('optimizer_steps')) is not int or model['optimizer_steps']<=0
            or type(delta) not in {int,float} or not math.isfinite(delta) or delta<=0):raise ValueError('each candidate model must have genuine completed optimizer fit')
    return {'terminal_job':job,'receipt':{'path':str(receipt_path.relative_to(root)),'sha256':receiptstat['sha256']},'saved_identity_checks':process_checks,'registry_path':str(registry_path.relative_to(root)),'registry_sha256':regstat['sha256']}

def full_inventory(root,target):
    output=confined(root,target['output']);log=confined(root,target['retained_pool_log'])
    if not stat.S_ISDIR(output.lstat().st_mode):raise ValueError('canonical output directory required')
    entries=[];directory_before={}
    def walk_error(error):raise error
    for base,dirs,files in os.walk(output,followlinks=False,onerror=walk_error):
        dirs.sort();files.sort();base=confined(root,base)
        for name in dirs+files:
            p=confined(root,base/name);info=p.lstat()
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):raise ValueError('symlink/nonregular output rejected')
        info=base.lstat()
        if not stat.S_ISDIR(info.st_mode):raise ValueError('directory changed type')
        directory_before[str(base)]=stat_fields(info)
        entries.append({'scope':'output','relative_path':str(base.relative_to(output)),'entry_type':'directory',**stat_fields(info)})
        for name in files:
            p=base/name;record,_=opaque_file(root,p)
            if record['link_count']!=1:raise ValueError('multiply linked retained file rejected')
            entries.append({'scope':'output','relative_path':str(p.relative_to(output)),'entry_type':'regular_file',**record})
    for p,record in directory_before.items():
        if stat_fields(Path(p).lstat())!=record:raise ValueError('directory membership/stat drift during inventory')
    record,_=opaque_file(root,log)
    if record['link_count']!=1:raise ValueError('multiply linked retained log rejected')
    entries.append({'scope':'retained_pool_log','relative_path':log.name,'entry_type':'regular_file',**record})
    entries.sort(key=lambda e:(e['scope'],e['relative_path']))
    charged=sum(x['retained_charge_bytes'] for x in entries)
    if charged>=target['new_bytes']:raise ValueError('retained output/log exceeds conservative15GB charge')
    return {'key':target['key'],'seed':target['seed'],'output':str(output),'retained_pool_log':str(log),'entries':entries,'conservative_retained_charge_bytes':charged,'opaque_byte_hashes_only':True}
def stat_membership(inv):
    values=[{k:v for k,v in x.items() if k!='sha256'} for x in inv['entries']]
    return sorted(values,key=lambda e:(e['scope'],e['relative_path']))
def proposal(c,state,module):
    if (state['baseline_bytes']!=c['baseline_bytes'] or state['external_reserve_bytes']!=c['external_reserve_bytes']
        or state['snapshot_sha256']!=c['snapshot']['sha256'] or state['limit_bytes']!=c['warning_bytes']):raise ValueError('historical/external/snapshot ledger bindings differ')
    proposed=copy.deepcopy(state)
    for t in c['targets']:
        expected={'output':str(confined(Path(state['root']),t['output'])),'reserved_bytes':t['old_bytes']}
        if not exact(state['jobs'].get(t['key']),expected):raise ValueError('exact original two slot charges required')
        proposed['jobs'][t['key']]['reserved_bytes']=t['new_bytes']
    inverse=copy.deepcopy(proposed)
    for t in c['targets']:inverse['jobs'][t['key']]['reserved_bytes']=t['old_bytes']
    if module.encoded(inverse)!=module.encoded(state):raise ValueError('unrelated ledger mutation')
    return proposed
@contextlib.contextmanager
def existing_mutex(root,c):
    p=confined(root,c['scheduler_lock']);before=p.lstat()
    if not stat.S_ISREG(before.st_mode):raise ValueError('existing canonical scheduler mutex required')
    fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW)
    with os.fdopen(fd,'r+') as locked:
        current=os.fstat(locked.fileno())
        if (current.st_dev,current.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('canonical mutex identity changed')
        fcntl.flock(locked,fcntl.LOCK_EX)
        if (p.lstat().st_dev,p.lstat().st_ino)!=(current.st_dev,current.st_ino):raise ValueError('canonical mutex replaced')
        yield locked

def run(root,c,mode,authority,prior,module,inspection=None,review=None,source_recheck=None):
    validate_authority(root,c,mode,authority,inspection,review)
    auditdir=confined(root,c['inspection_dir'] if mode=='inspect' else c['apply_dir'])
    if auditdir.exists():raise ValueError('fresh fixed root audit output required; retain previous evidence')
    auditdir.parent.mkdir(parents=True,exist_ok=True);auditdir.mkdir()
    attempted=False;atomic_completed=False
    try:
        with existing_mutex(root,c):
            snapshot=module.snapshot_record(root,str(confined(root,c['snapshot']['path'])),c['snapshot']['sha256'])
            def available():
                info=os.statvfs(root);return info.f_bavail*info.f_frsize
            ledger=module.StorageLedger(root,snapshot,c['external_reserve_bytes'],c['warning_bytes'],available)
            if ledger.path!=confined(root,c['ledger_path']):raise ValueError('exact persistent ledger path required')
            ledgerstat,before=opaque_file(root,ledger.path,metadata=True)
            if ledgerstat['sha256']!=c['before_ledger_sha256']:raise ValueError('ledger drift since sealed prior inventory; stop without settlement')
            state=ledger._load()
            if not exact(state,parse_json(before)):raise ValueError('ledger drift during locked read')
            proposed=proposal(c,state,module);before_account=ledger._check(state);after_account=ledger._check(proposed)
            terminal_passes=[];inventories=[];inventory_refs=[]
            for pass_number in [1,2]:
                terminal_rows={t['key']:terminal(root,c,t,prior[t['key']],module) for t in c['targets']}
                current={t['key']:full_inventory(root,t) for t in c['targets']}
                for t in c['targets']:
                    key=t['key']
                    if not exact(stat_membership(current[key]),sorted(prior[key]['prior_stat_entries'],key=lambda e:(e['scope'],e['relative_path']))):raise ValueError('output/stat membership differs from prior sealed inventory')
                    if pass_number==2 and not exact(current[key],inventories[0][key]):raise ValueError('full byte/stat inventory drift between locked passes')
                    if mode=='apply' and not exact(current[key],inspection['inventories'][key]):raise ValueError('full retained scope changed after root inspection')
                terminal_passes.append(terminal_rows);inventories.append(current)
                inventory_refs.append({t['key']:write_new(root,auditdir,f"seed{t['seed']}_full_inventory_pass{pass_number}.json",current[t['key']]) for t in c['targets']})
            final_terminal={t['key']:terminal(root,c,t,prior[t['key']],module) for t in c['targets']}
            for t in c['targets']:
                key=t['key']
                if any(not exact(final_terminal[key]['terminal_job'],x[key]['terminal_job']) or not exact(final_terminal[key]['receipt'],x[key]['receipt']) for x in terminal_passes):raise ValueError('terminal receipt or target identity drift during inventory')
                if mode=='apply' and not exact(final_terminal[key]['receipt'],inspection['terminal'][key]['receipt']):raise ValueError('terminal receipt differs from reviewed root inspection')
            if opaque_file(root,ledger.path,metadata=True)[1]!=before:raise ValueError('live ledger changed under scheduler mutex')
            if source_recheck is not None:source_recheck()
            audit={'kind':'biological_completed_candidates_storage_settlement_v1_'+mode+'_receipt','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_authorizer':'root_agent','applied':False,'procedure':reference(SCRIPT),'config':reference(CONFIG),'prior_inventory':c['prior_inventory'],'before_ledger_sha256':ledgerstat['sha256'],'scope_keys':[t['key'] for t in c['targets']],'operator_frozen_output_policy':c['operator_frozen_output_policy'],'known_saved_identity_absence_does_not_prove_unknown_writer_absence':True,'two_full_byte_stat_inventory_passes_under_scheduler_mutex':True,'inventory_refs':inventory_refs,'inventories':inventories[1],'terminal':final_terminal,'before_account':before_account,'after_account':after_account,'old_bytes_each':29281048576,'retained_bytes_each':15000000000,'reduction_bytes':28562097152,'no_product_files_deleted':True,'all_other_ledger_fields_charges_baseline_external_preserved':True,'historical_projection_not_global_remeasurement':True}
            if mode=='inspect':
                if opaque_file(root,ledger.path,metadata=True)[1]!=before:raise ValueError('inspect changed ledger')
                write_new(root,auditdir,'inspection.json',audit)
            else:
                if inspection['applied'] is not False or inspection['kind']!='biological_completed_candidates_storage_settlement_v1_inspect_receipt' or inspection['before_ledger_sha256']!=c['before_ledger_sha256'] or not exact(inspection['scope_keys'],audit['scope_keys']) or not exact(inspection['procedure'],audit['procedure']) or not exact(inspection['config'],audit['config']):raise ValueError('exact fresh unexecuted root inspection required')
                audit.update(root_admission=authority,independent_review=authority['independent_review'],inspection=authority['inspection'])
                write_new(root,auditdir,'ledger_before.json',parse_json(before));write_new(root,auditdir,'preapply_audit.json',audit)
                attempted=True;module.atomic_json(ledger.path,proposed);atomic_completed=True
                afterinfo,afterbytes=opaque_file(root,ledger.path,metadata=True)
                if not exact(parse_json(afterbytes),proposed):raise ValueError('post-write ledger differs from exact two-delta proposal')
                audit.update(applied=True,after_ledger_sha256=afterinfo['sha256'])
                write_new(root,auditdir,'settlement_receipt.json',audit)
            return audit
    except BaseException as error:
        write_new(root,auditdir,'failure_receipt.json',{'kind':'biological_completed_candidates_storage_settlement_v1_preserved_failure','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error_type':type(error).__name__,'error':str(error),'mode':mode,'ledger_mutation_attempted':attempted,'atomic_json_completed':atomic_completed,'no_automatic_rollback':True})
        raise

def load_pool(c):
    p=confined(ROOT,c['pool_source']['path'])
    if sha(p)!=c['pool_source']['sha256']:raise ValueError('frozen pool source changed')
    spec=importlib.util.spec_from_file_location('_candidate_settlement_frozen_pool',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--inspect',action='store_true');modes.add_argument('--apply',action='store_true')
    parser.add_argument('--expected-source-sha256');parser.add_argument('--expected-config-sha256');parser.add_argument('--admission');parser.add_argument('--expected-admission-sha256')
    args=parser.parse_args(argv);c=policy();verify_sources(c)
    if not args.inspect and not args.apply:
        print(json.dumps({'kind':'biological_completed_candidates_storage_settlement_v1_source_report','procedure':reference(SCRIPT),'config':reference(CONFIG),'targets':c['targets'],'prior_inventory':c['prior_inventory'],'actual_inventory_inspect_apply_performed':False,'live_ledger_registry_PID_or_GPU_access':False},indent=2));return 0
    if not all([args.expected_source_sha256,args.expected_config_sha256,args.admission,args.expected_admission_sha256]):raise ValueError('root exact source/config/admission opt-in required before live access')
    if sha(SCRIPT)!=args.expected_source_sha256 or sha(CONFIG)!=args.expected_config_sha256:raise ValueError('reviewed source/config changed')
    authority=bound_json(ROOT,{'path':args.admission,'sha256':args.expected_admission_sha256});mode='apply' if args.apply else 'inspect'
    inspection=bound_json(ROOT,authority['inspection'],ROOT/c['inspection_dir']/'inspection.json') if mode=='apply' else None
    review=bound_json(ROOT,authority['independent_review']) if mode=='apply' else None
    validate_authority(ROOT,c,mode,authority,inspection,review)
    prior=prior_evidence(ROOT,c);module=load_pool(c)
    def recheck():
        verify_sources(c)
        if sha(SCRIPT)!=args.expected_source_sha256 or sha(CONFIG)!=args.expected_config_sha256:raise ValueError('source/config drift during audit')
        if bound_json(ROOT,{'path':args.admission,'sha256':args.expected_admission_sha256})!=authority:raise ValueError('root authority changed')
        prior_evidence(ROOT,c)
        if mode=='apply':
            if not exact(bound_json(ROOT,authority['inspection']),inspection) or not exact(bound_json(ROOT,authority['independent_review']),review):raise ValueError('inspection/review changed during audit')
    result=run(ROOT,c,mode,authority,prior,module,inspection,review,recheck)
    print(json.dumps({'applied':result['applied'],'reduction_bytes':result['reduction_bytes'],'before_projected_bytes':result['before_account']['projected_bytes'],'after_projected_bytes':result['after_account']['projected_bytes']},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
