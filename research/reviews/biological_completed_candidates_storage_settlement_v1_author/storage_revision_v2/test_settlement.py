"""Handmade project fixtures only; no real product, registry, process or ledger reads."""
from pathlib import Path
import copy,importlib.util,json,os
from types import SimpleNamespace
import pytest
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
AUTHOR=ROOT/'research/reviews/biological_completed_candidates_storage_settlement_v1_author/storage_revision_v2'
spec=importlib.util.spec_from_file_location('candidate_settlement_author_fixture',AUTHOR/'audit_and_settle.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
def dump(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')
def ref(root,p):return {'path':str(p.relative_to(root)),'sha256':h.opaque_file(root,p)[0]['sha256']}
def admission_record(target):
    return {'observed':{'index':target['physical_gpu'],'uuid':f"GPU-00000000-0000-0000-0000-{target['physical_gpu']:012d}",'used_mib':0,'total_mib':24576,'utilization':0,'compute':[]},'active_peak_mib':0,'incoming_peak_mib':12288,'unattributed_mib':0,'projected_mib':13312,'foreign_compute_pids':[],'overrun_jobs':[],'blocked_by':[]}
def fixture(tmp_path,monkeypatch):
    c=copy.deepcopy(h.FIXED_CONFIG);root=tmp_path
    module=h.load_pool(c);monkeypatch.setattr(module,'proc_identity',lambda pid:None)
    monkeypatch.setattr(h.os,'statvfs',lambda p:SimpleNamespace(f_bavail=10**15,f_frsize=1))
    lock=root/c['scheduler_lock'];lock.parent.mkdir(parents=True);lock.touch()
    snapshot=root/'snapshot.json';dump(snapshot,{'schema':'biological_candidate_storage_snapshot_v1','status':'complete','returncode':0,'project_root':str(root),'command':['du','-s','-B1',str(root)],'used_bytes':167336640512,'completed_utc':'2026-10-10T20:00:00+00:00'})
    c['snapshot']=ref(root,snapshot);c['inspection_dir']='audit/inspect';c['apply_dir']='audit/apply'
    prior={};terminal_rows=[]
    for t in c['targets']:
        out=root/t['output'];out.mkdir(parents=True);(out/'nested').mkdir();(out/'nested/opaque.npz').write_bytes(b'handmade opaque content, never decoded');(out/'checkpoint.pt').write_bytes(b'handmade checkpoint bytes')
        receipt={'kind':'biological_candidate_fit_v1','status':'completed_three_conditions_one_seed','seed':t['seed'],'config_sha256':c['config_sha256'],'observed_manifest_sha256':c['observed_manifest_sha256'],'prepared_manifest_sha256':c['prepared_manifest_sha256'],'source_sha256':c['candidate_source_sha256'],'TEST_parents':0,'models':{name:{'status':'completed','optimizer_fit_completed':True,'optimizer_steps':17,'selected_maximum_parameter_delta_from_initial':0.125} for name in c['models']}}
        dump(out/'receipt.json',receipt)
        log=root/t['retained_pool_log'];log.parent.mkdir(parents=True,exist_ok=True);log.write_text('handmade completed log\n')
        job={'logical_job':t['key'].split('/',1)[1],'state':'succeeded','exit_code':0,'pid':t['seed']+1000,'leader_start_ticks':1234,'process_identities':[{'pid':t['seed']+1000,'start_ticks':1234},{'pid':t['seed']+2000,'start_ticks':2345}],'module_source_sha256':c['candidate_source']['sha256'],'log_path':str(log),'requested':{'module':'arggnn.biological_candidate','seed':t['seed'],'output':str(out),'storage_bytes':t['old_bytes'],'peak_mib':12288},'argv':['handmade-python','-m','arggnn.biological_candidate','--train','--seed',str(t['seed'])],'log_bytes':len(log.read_bytes()),'log_dropped_bytes':0,'started_unix':1700000001.0,'finished_unix':1700001001.0}
        current=copy.deepcopy(job);current.update(admission=admission_record(t),submitted_unix=1700000000.0)
        dump(root/t['registry'],{'safety_error':None,'jobs':[current]})
        inv=h.full_inventory(root,t);entries=h.stat_membership(inv)
        prior[t['key']]={'terminal_job':job,'prior_stat_entries':entries}
        terminal_rows.append({'key':t['key'],'terminal_job':job})
        t['prior_stat_passes']=[]
        for index in [1,2]:
            p=root/f'prior/seed{t["seed"]}_pass{index}.json';dump(p,{'key':t['key'],'seed':t['seed'],'output':str(out),'retained_pool_log':str(log),'entries':entries});t['prior_stat_passes'].append(ref(root,p))
    state={'schema':'gpu_pool_storage_ledger_v1','root':str(root),'snapshot_sha256':c['snapshot']['sha256'],'baseline_bytes':167336640512,'external_reserve_bytes':130000000000,'limit_bytes':450000000000,'controllers':{'active_controller':{'pid':678,'reserved_bytes':67108864}},'jobs':{t['key']:{'output':str(root/t['output']),'reserved_bytes':t['old_bytes']} for t in c['targets']}}
    state['jobs']['active_job']={'output':str(root/'other_active'),'reserved_bytes':52321048576}
    ledger=root/c['ledger_path'];dump(ledger,state);c['before_ledger_sha256']=ref(root,ledger)['sha256']
    c['prior_metadata']=[]
    for name in ['before','after']:
        p=root/f'prior/metadata_{name}.json';dump(p,{'candidates':terminal_rows,'ledger_sha256':c['before_ledger_sha256']});c['prior_metadata'].append(ref(root,p))
    p=root/'prior/inventory_receipt.json';dump(p,{'schema':'completed_candidate_storage_inventory_receipt_v1','status':'preliminary_stat_inventory_complete','settlement_applied':False,'settlement_authorized':False,'current_ledger_sha256':c['before_ledger_sha256'],'before_ledger_sha256':c['before_ledger_sha256'],'artifact_refs':c['prior_metadata']+[x for t in c['targets'] for x in t['prior_stat_passes']]});c['prior_inventory']=ref(root,p)
    authority={'kind':'biological_completed_candidates_storage_settlement_v1_root_inspect_admission',**h.common_bindings(c)}
    return root,c,prior,module,authority,ledger,state

def inspect_fixture(f):
    root,c,prior,module,a,ledger,state=f
    return h.run(root,c,'inspect',a,prior,module)
def apply_parts(f,inspection):
    root,c,prior,module,a,ledger,state=f
    p=root/c['inspection_dir']/'inspection.json';ir=ref(root,p)
    review={'kind':'biological_completed_candidates_storage_settlement_v1_independent_review','review_role':'independent_reviewer','verdict':'PASS','procedure':h.reference(h.SCRIPT),'config':h.reference(h.CONFIG),'inspection':ir,'prior_inventory':c['prior_inventory'],'expected_before_ledger_sha256':c['before_ledger_sha256'],'scope_keys':[t['key'] for t in c['targets']],'operator_frozen_output_contract_accepted':True,'actual_apply_performed':False}
    rp=root/'audit/review.json';dump(rp,review)
    admission={'kind':'biological_completed_candidates_storage_settlement_v1_root_apply_admission',**h.common_bindings(c),'inspection':ir,'independent_review':ref(root,rp)}
    return admission,review

def test_source_only_default_never_enters_runtime_or_live_paths(monkeypatch,capsys):
    monkeypatch.setattr(h,'load_pool',lambda *args:(_ for _ in ()).throw(AssertionError('pool imported')))
    monkeypatch.setattr(h,'prior_evidence',lambda *args:(_ for _ in ()).throw(AssertionError('actual prior runtime')))
    assert h.main([])==0
    report=json.loads(capsys.readouterr().out);assert not report['actual_inventory_inspect_apply_performed'] and not report['live_ledger_registry_PID_or_GPU_access']
@pytest.mark.parametrize('mode',['--inspect','--apply'])
def test_runtime_modes_require_exact_optin_before_live_paths(mode,monkeypatch):
    monkeypatch.setattr(h,'load_pool',lambda *args:(_ for _ in ()).throw(AssertionError('live pool')))
    with pytest.raises(ValueError,match='opt-in'):h.main([mode])

def test_inspect_is_readonly_two_pass_complete_and_prior_bound(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f;before=ledger.read_bytes()
    assert h.prior_evidence(root,c)==prior
    result=inspect_fixture(f)
    assert not result['applied'] and ledger.read_bytes()==before
    assert result['two_full_byte_stat_inventory_passes_under_scheduler_mutex']
    assert len(result['inventory_refs'])==2 and len(result['inventories'])==2
    assert result['before_account']['projected_bytes']-result['after_account']['projected_bytes']==28562097152
    for t in c['targets']:
        inv=result['inventories'][t['key']];assert any(x.get('sha256') for x in inv['entries'])
        assert sum(x['retained_charge_bytes'] for x in inv['entries'])==inv['conservative_retained_charge_bytes']

def test_handmade_apply_only_two_reserved_bytes_changes_and_retains_all_products(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f
    inspection=inspect_fixture(f);admission,review=apply_parts(f,inspection)
    outputs={t['key']:h.full_inventory(root,t) for t in c['targets']}
    result=h.run(root,c,'apply',admission,prior,module,inspection,review)
    assert result['applied'] and (root/c['apply_dir']/'preapply_audit.json').is_file()
    after=json.loads(ledger.read_text())
    for t in c['targets']:
        assert after['jobs'][t['key']]['reserved_bytes']==15000000000
        assert h.full_inventory(root,t)==outputs[t['key']]
        after['jobs'][t['key']]['reserved_bytes']=29281048576
    assert after==state
@pytest.mark.parametrize('change',['not_terminal','wrong_seed','module','reservation','unsafe_registry','duplicate_job','receipt_status','receipt_seed','source_pin','missing_model','fit_false','zero_steps','zero_delta','query_TEST'])
def test_terminal_provenance_and_per_model_fit_corruptions_fail_without_reduction(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f;t=c['targets'][0];reg=root/t['registry'];r=json.loads(reg.read_text());job=r['jobs'][0];receipt=root/t['output']/'receipt.json';data=json.loads(receipt.read_text());name=c['models'][0]
    if change=='not_terminal':job['state']='running'
    elif change=='wrong_seed':job['requested']['seed']=101
    elif change=='module':job['requested']['module']='other'
    elif change=='reservation':job['requested']['storage_bytes']=15000000000
    elif change=='unsafe_registry':r['safety_error']='unknown descendant'
    elif change=='duplicate_job':r['jobs'].append(copy.deepcopy(job))
    elif change=='receipt_status':data['status']='running'
    elif change=='receipt_seed':data['seed']=101
    elif change=='source_pin':data['source_sha256']={}
    elif change=='missing_model':data['models'].pop(name)
    elif change=='fit_false':data['models'][name]['optimizer_fit_completed']=False
    elif change=='zero_steps':data['models'][name]['optimizer_steps']=0
    elif change=='zero_delta':data['models'][name]['selected_maximum_parameter_delta_from_initial']=0
    else:data['TEST_parents']=1
    dump(reg,r);dump(receipt,data);before=ledger.read_bytes()
    with pytest.raises(ValueError):inspect_fixture(f)
    assert ledger.read_bytes()==before and (root/c['inspection_dir']/'failure_receipt.json').is_file()

def test_known_saved_worker_blocks_but_PID_reuse_is_distinguished(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f;t=c['targets'][0]
    monkeypatch.setattr(module,'proc_identity',lambda pid:{'start_ticks':1234})
    with pytest.raises(ValueError,match='remains present'):h.terminal(root,c,t,prior[t['key']],module)
    monkeypatch.setattr(module,'proc_identity',lambda pid:{'start_ticks':9999})
    assert all(not row['saved_identity_present'] for row in h.terminal(root,c,t,prior[t['key']],module)['saved_identity_checks'])
@pytest.mark.parametrize('change',['symlink','hardlink','extra_file','missing_file','modified_file','unsafe_log'])
def test_retained_scope_symlink_membership_stat_drift_rejected(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f;t=c['targets'][0];p=root/t['output']/'nested/opaque.npz'
    if change=='symlink':(p.parent/'link').symlink_to(p)
    elif change=='hardlink':os.link(p,p.parent/'alias')
    elif change=='extra_file':(p.parent/'extra').write_bytes(b'new')
    elif change=='missing_file':p.unlink()
    elif change=='modified_file':p.write_bytes(b'changed bytes')
    else:
        log=root/t['retained_pool_log'];log.unlink();log.symlink_to(p)
    before=ledger.read_bytes()
    with pytest.raises(ValueError):inspect_fixture(f)
    assert ledger.read_bytes()==before

def test_walk_errors_fail_closed(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f
    def broken(*args,**kwargs):
        kwargs['onerror'](OSError('synthetic walk failure'))
        yield
    monkeypatch.setattr(h.os,'walk',broken)
    with pytest.raises(OSError,match='synthetic walk'):inspect_fixture(f)

def test_second_full_inventory_drift_is_rejected(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f;original=h.full_inventory;calls=0
    def inventory(r,t):
        nonlocal calls
        calls+=1;value=original(r,t)
        if calls==3:value['entries'][1]['sha256']='0'*64
        return value
    monkeypatch.setattr(h,'full_inventory',inventory)
    with pytest.raises(ValueError,match='between locked passes'):inspect_fixture(f)
@pytest.mark.parametrize('change',['ledger_hash','missing_slot','wrong_output','already_reduced','other_charge','baseline','external','snapshot'])
def test_old_ledger_or_slot_drift_blocks_without_fallback(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f;t=c['targets'][0]
    if change=='ledger_hash':c['before_ledger_sha256']='0'*64;a=h.common_bindings(c);a['kind']='biological_completed_candidates_storage_settlement_v1_root_inspect_admission'
    else:
        if change=='missing_slot':state['jobs'].pop(t['key'])
        elif change=='wrong_output':state['jobs'][t['key']]['output']=str(root/'other')
        elif change=='already_reduced':state['jobs'][t['key']]['reserved_bytes']=15000000000
        elif change=='other_charge':state['jobs']['active_job']['reserved_bytes']+=1
        elif change=='baseline':state['baseline_bytes']+=1
        elif change=='external':state['external_reserve_bytes']+=1
        else:state['snapshot_sha256']='0'*64
        dump(ledger,state)
    before=ledger.read_bytes()
    with pytest.raises(ValueError):h.run(root,c,'inspect',a,prior,module)
    assert ledger.read_bytes()==before
@pytest.mark.parametrize('change',['missing_freeze','active_false','keys','review_verdict','review_inspection_hash','review_role','changed_output_after_inspect','inspection_applied'])
def test_apply_requires_exact_frozen_scope_independent_review_and_inspection(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f;inspection=inspect_fixture(f);admission,review=apply_parts(f,inspection)
    if change=='missing_freeze':admission['operator_freeze_confirmed']=False
    elif change=='active_false':admission['active']=False
    elif change=='keys':admission['scope_keys']=admission['scope_keys'][:1]
    elif change=='review_verdict':review['verdict']='BLOCKED'
    elif change=='review_inspection_hash':review['inspection']['sha256']='0'*64
    elif change=='review_role':review['review_role']='author'
    elif change=='changed_output_after_inspect':(root/c['targets'][0]['output']/'new').write_bytes(b'drift')
    else:inspection['applied']=True
    before=ledger.read_bytes()
    with pytest.raises(ValueError):h.run(root,c,'apply',admission,prior,module,inspection,review)
    assert ledger.read_bytes()==before

def test_atomic_failure_preserves_preapply_and_failure_evidence(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f;inspection=inspect_fixture(f);admission,review=apply_parts(f,inspection);before=ledger.read_bytes()
    monkeypatch.setattr(module,'atomic_json',lambda *args:(_ for _ in ()).throw(OSError('synthetic atomic failure')))
    with pytest.raises(OSError,match='synthetic atomic'):h.run(root,c,'apply',admission,prior,module,inspection,review)
    assert ledger.read_bytes()==before
    failure=json.loads((root/c['apply_dir']/'failure_receipt.json').read_text());assert failure['ledger_mutation_attempted'] and not failure['atomic_json_completed']
    assert (root/c['apply_dir']/'ledger_before.json').is_file() and (root/c['apply_dir']/'preapply_audit.json').is_file()

def test_canonical_mutex_is_same_as_frozen_library_descriptor(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,*_=f
    with module.lock_file(root,'scheduler.lock') as library:
        with h.existing_mutex(root,c) as procedure:
            a=os.fstat(library.fileno());b=os.fstat(procedure.fileno());assert (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)

def test_path_traversal_and_duplicate_or_nonfinite_JSON_rejected(tmp_path):
    for p in [tmp_path/'..'/'escape',Path('/tmp/external')]:
        with pytest.raises(ValueError):h.confined(tmp_path,p)
    for raw in ['{"x":1,"x":2}','{"x":NaN}']:
        with pytest.raises(ValueError):h.parse_json(raw)


def current_job(f,target_index=0):
    root,c,*_=f;t=c['targets'][target_index];p=root/t['registry'];r=json.loads(p.read_text());return t,p,r,r['jobs'][0]

def test_actual_schema_projection_preserves_and_records_full_live_job_extras(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,*_=f;t,p,r,current=current_job(f)
    result=h.terminal(root,c,t,prior[t['key']],module)
    assert result['terminal_job']==current and result['sealed_terminal_projection']==prior[t['key']]['terminal_job']
    assert result['terminal_metadata_extras']=={k:current[k] for k in ['admission','submitted_unix']}
    assert result['exact_prior_projection_fields']==c['terminal_job_schema']['required_sealed_projection_fields']
    inspection=inspect_fixture(f)
    assert inspection['terminal'][t['key']]['terminal_metadata_extras']==result['terminal_metadata_extras']

@pytest.mark.parametrize('field',['argv','exit_code','finished_unix','leader_start_ticks','log_bytes','log_dropped_bytes','log_path','logical_job','module_source_sha256','pid','process_identities','requested','started_unix','state'])
def test_every_sealed_projection_field_change_or_missing_is_rejected(tmp_path,monkeypatch,field):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,*_=f;t,p,r,current=current_job(f);sealed=prior[t['key']]['terminal_job']
    missing=copy.deepcopy(current);missing.pop(field)
    with pytest.raises(ValueError,match='missing expected'):h.terminal_projection(c,t,missing,sealed)
    changed=copy.deepcopy(current);v=changed[field]
    if isinstance(v,dict):v['synthetic_change']=True
    elif isinstance(v,list):v.append('synthetic_change')
    elif type(v) in {int,float}:changed[field]=v+1
    else:changed[field]=str(v)+'synthetic_change'
    with pytest.raises(ValueError,match='projected fields differ'):h.terminal_projection(c,t,changed,sealed)

@pytest.mark.parametrize('change',['unknown_extra','missing_admission','missing_submitted','prior_missing_argv','prior_extra'])
def test_unknown_extra_missing_actual_extra_or_incomplete_prior_coverage_rejected(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,*_=f;t,p,r,current=current_job(f);sealed=copy.deepcopy(prior[t['key']]['terminal_job'])
    if change=='unknown_extra':current['future_unreviewed_field']={'foo':1}
    elif change=='missing_admission':current.pop('admission')
    elif change=='missing_submitted':current.pop('submitted_unix')
    elif change=='prior_missing_argv':sealed.pop('argv')
    else:sealed['unreviewed_prior_field']=1
    with pytest.raises(ValueError):h.terminal_projection(c,t,current,sealed)

@pytest.mark.parametrize('value',[True,None,'1700000000',float('nan'),float('inf'),-1,0,1700000002.0])
def test_submitted_timestamp_is_typed_finite_positive_and_bound_to_exact_start_finish(tmp_path,monkeypatch,value):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,*_=f;t,p,r,current=current_job(f);current['submitted_unix']=value
    with pytest.raises(ValueError):h.terminal_projection(c,t,current,prior[t['key']]['terminal_job'])

@pytest.mark.parametrize('change',['not_dict','unknown_field','missing_field','wrong_device','index_bool','bad_uuid','used_over_total','total_zero','util_over100','compute_not_list','duplicate_compute_pid','bad_compute_name','unknown_compute_field','negative_active','bool_active','wrong_incoming','unattributed_bool','projected_over_total','zero_headroom','foreign','overrun','blocker'])
def test_launch_admission_extras_are_exact_typed_and_bound_to_requested_peak_and_target(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,*_=f;t,p,r,current=current_job(f);a=current['admission'];o=a['observed']
    if change=='not_dict':current['admission']=[]
    elif change=='unknown_field':a['future']=1
    elif change=='missing_field':a.pop('projected_mib')
    elif change=='wrong_device':o['index']=5
    elif change=='index_bool':o['index']=True
    elif change=='bad_uuid':o['uuid']='GPU-unbound'
    elif change=='used_over_total':o['used_mib']=o['total_mib']+1
    elif change=='total_zero':o['total_mib']=0
    elif change=='util_over100':o['utilization']=101
    elif change=='compute_not_list':o['compute']={}
    elif change=='duplicate_compute_pid':o['compute']=[{'pid':111,'name':'fixture','used_mib':1},{'pid':111,'name':'fixture','used_mib':1}]
    elif change=='bad_compute_name':o['compute']=[{'pid':111,'name':'bad\0name','used_mib':1}]
    elif change=='unknown_compute_field':o['compute']=[{'pid':111,'name':'fixture','used_mib':1,'future':0}]
    elif change=='negative_active':a['active_peak_mib']=-1
    elif change=='bool_active':a['active_peak_mib']=False
    elif change=='wrong_incoming':a['incoming_peak_mib']=1
    elif change=='unattributed_bool':a['unattributed_mib']=False
    elif change=='projected_over_total':a['projected_mib']=o['total_mib']+1
    elif change=='zero_headroom':a['projected_mib']=a['incoming_peak_mib']
    elif change=='foreign':a['foreign_compute_pids']=[111]
    elif change=='overrun':a['overrun_jobs']=['another-job']
    else:a['blocked_by']=['memory_headroom']
    with pytest.raises(ValueError):h.terminal_projection(c,t,current,prior[t['key']]['terminal_job'])

def test_typed_extra_metadata_drift_between_passes_is_rejected(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f;original=h.terminal;calls=0
    def terminal(r,conf,t,pr,m):
        nonlocal calls
        calls+=1
        if calls==3:
            p=root/t['registry'];reg=json.loads(p.read_text());reg['jobs'][0]['submitted_unix']=1700000000.5;dump(p,reg)
        return original(r,conf,t,pr,m)
    monkeypatch.setattr(h,'terminal',terminal);before=ledger.read_bytes()
    with pytest.raises(ValueError,match='target identity drift'):inspect_fixture(f)
    assert ledger.read_bytes()==before

@pytest.mark.parametrize('extra',['admission','submitted_unix'])
def test_apply_binds_full_extra_metadata_to_reviewed_inspection(tmp_path,monkeypatch,extra):
    f=fixture(tmp_path,monkeypatch);root,c,prior,module,a,ledger,state=f;inspection=inspect_fixture(f);admission,review=apply_parts(f,inspection)
    t,p,r,current=current_job(f)
    if extra=='submitted_unix':current['submitted_unix']=1700000000.5
    else:current['admission']['observed']['utilization']=1
    dump(p,r);before=ledger.read_bytes()
    with pytest.raises(ValueError,match='full terminal job/extras differ'):h.run(root,c,'apply',admission,prior,module,inspection,review)
    assert ledger.read_bytes()==before
