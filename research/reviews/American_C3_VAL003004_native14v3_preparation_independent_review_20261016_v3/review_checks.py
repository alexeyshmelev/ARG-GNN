"""Independent source-only checks. All writes are inside this review directory."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ast, contextlib, copy, hashlib, importlib.util, io, json, os, stat, subprocess, sys, types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=Path(__file__).parent
SRC=ROOT/'research/reviews/American_C3_VAL003004_native14v3_preparation_source_20261016_v3'
OLD=SRC.with_name(SRC.name[:-1]+'2')
EXPIRY='2026-10-16T20:04:16.988073+00:00'
CHECKS=[]
def check(name, value=True, **detail):
    if not value: raise AssertionError(name)
    CHECKS.append(dict(name=name,status='PASS',**detail))
def read(p): return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def auth(r):
    p=ROOT/r['path'];s=p.lstat()
    assert p.resolve()==p and p.is_relative_to(ROOT) and stat.S_ISREG(s.st_mode) and s.st_nlink==1
    assert p.suffix in ('.py','.sh','.json','.lock','.md') and s.st_size<=32*2**20
    assert ref(p)=={k:r[k] for k in ('path','sha256')},str(p)
    if 'bytes' in r: assert s.st_size==r['bytes']
def save(name,v):
    p=HERE/name;p.parent.mkdir(exist_ok=True,parents=True)
    assert p.is_relative_to(HERE)
    with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    return ref(p)
def reject(name,fn,message=None):
    try:fn()
    except ValueError as e:
        check(name,message is None or message in str(e),actual_exception=str(e));return
    raise AssertionError(name+' did not reject')
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def funcs(p):return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def extract(node,ns):exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),'<independent-source-extract>','exec'),ns)
def dump(n):return ast.dump(n,include_attributes=False)
expected_handoff='d635a24b1893ff9d509d60749cd46f74c9f752c55938c97a89971f9917fbe4dd'
expected_manifest='21900676c0cccbeec1cc6b2359ec39455d7a27c1aee933083d239293b19b35ed'
check('candidate_handoff_exact_sha',ref(SRC/'source_handoff.v1.json')['sha256']==expected_handoff)
handoff=read(SRC/'source_handoff.v1.json');manifest=read(SRC/'source_manifest.v1.json')
check('candidate_manifest_exact_sha',ref(SRC/'source_manifest.v1.json')['sha256']==expected_manifest)
for r in handoff['files']:auth(r)
check('every_handoff_file_authenticated',count=len(handoff['files']))
for r in manifest['refs']:auth(r)
check('all308_manifest_refs_authenticated',len(manifest['refs'])==308)
sealed={r['path']:ref(r['path']) for r in [*handoff['files'],*manifest['refs']]}
oldhand=read(OLD/'source_handoff.v1.json');auth(handoff['prior_v2_handoff_ref'])
for r in oldhand['files']:auth(r)
check('all37_original_v2_sealed_files_unchanged',len(oldhand['files'])==37)
audit_ref={'path':'research/reviews/VAL003_admission_partial_audit_20261016_v1/conclusion.v1.json','sha256':'69936ff6f13ae820453c21e026f3fa6e5adeca749845ae805a14cc7c468bb140'}
auth(audit_ref);audit=read(ROOT/audit_ref['path'])
for r in audit['actual_evidence_refs']:
    p=ROOT/r['path'];assert ref(p)=={k:r[k] for k in ('path','sha256')} and p.stat().st_size==r['bytes']
check('original_partial_audit_and_actual_evidence_authenticated')
check('historical_failure_preserves_unknown_cause',audit['original_tool90737_failure_cause']=='UNKNOWN')
oldmanifest=read(OLD/'source_manifest.v1.json');currentmap={r['path']:r for r in manifest['refs']}
unchanged_external=[r for r in oldmanifest['refs'] if not r['path'].startswith(str(OLD.relative_to(ROOT))+'/')]
check('every_original_external_closure_ref_retained_exactly',all(currentmap.get(r['path'])==r for r in unchanged_external),count=len(unchanged_external))
parsed=0
for r in manifest['refs']:
    if Path(r['path']).suffix=='.py':ast.parse((ROOT/r['path']).read_text());parsed+=1
check('all_manifest_python_sources_parse',count=parsed)
ct=load(SRC/'root_contract_v1.py','_independent_VAL003004_v3_contract')
v4=load(ROOT/'src/arggnn/project_budget_v4.py','_independent_real_unchanged_v4')
auth(ct.V4_REF)
check('unchanged_actual_v4_validator_authenticated',ref(ROOT/'src/arggnn/project_budget_v4.py')==ct.V4_REF)
ct.POLICY_REF=read(OLD/'request_val003_infer.root_active_v1.json')['policy_ref'];auth(ct.POLICY_REF)
check('actual_native14_policy_binding_exact',ct.POLICY_REF['sha256']=='72ed727feb1a824fa8f9c2b31ce374dca6728ca4e655e9562b4dedc84122f417')
check('production_authenticator_visits_complete_closure',len(ct.source_paths())==308)
slots=read(SRC/'slot_resource_identity.v3.json')
check('exact_original_pool_two_slots',slots['maximum_CPU_producers']==2 and manifest['original_shared_slot_refs']==[r['ref'] for r in slots['slots']])
for row in slots['slots']:
    p=ROOT/row['ref']['path'];s=p.lstat()
    check('slot_actual_identity_'+p.name,(s.st_dev,s.st_ino,s.st_size,s.st_nlink)==(row['st_dev'],row['st_ino'],0,1) and stat.S_ISREG(s.st_mode) and ref(p)==row['ref'])
check('no_new_slot_files',not list(SRC.glob('cpu_slot_*.lock')))
reject('wrong_actual_slot_hash_rejected',lambda:ct.raw(dict(slots['slots'][0]['ref'],sha256='0'*64)))
reject('wrong_identity_receipt_hash_rejected',lambda:ct.raw(dict(ct.SLOT_IDENTITY_REF,sha256='0'*64)))
reject('unbound_identity_receipt_rejected',lambda:ct.validate_slot_resources([r for r in manifest['refs'] if r!=ct.SLOT_IDENTITY_REF]))
slot_fn=funcs(SRC/'root_contract_v1.py')['validate_slot_resources']
for field,value in [('st_ino',slots['slots'][0]['st_ino']+1),('st_dev',slots['slots'][0]['st_dev']+1),('st_size',1),('st_nlink',2)]:
    modified=copy.deepcopy(slots);modified['slots'][0][field]=value
    ns=dict(ct.__dict__,read=lambda r,v=modified:v);extract(slot_fn,ns)
    reject('fake_receipt_wrong_'+field+'_rejected',lambda:ns['validate_slot_resources'](manifest['refs']))
# Compare all production function bodies after only mechanically normalizing namespace/version path changes.
permitted={'root_contract_v1.py':{'source_paths','validate_request','bind','direct_roles'},'root_driver_v1.py':{'admit'},'worker_entry_v1.py':set(),'plan_worker_v1.py':set()}
changes={}
for filename,allowed in permitted.items():
    oldf=funcs(OLD/filename);newf=funcs(SRC/filename);delta=[]
    for name,node in oldf.items():
        text=ast.unparse(node).replace(OLD.name,SRC.name).replace('_native14v3_v2','_native14v3_v3').replace('-native14v3-v2','-native14v3-v3').replace("'_v2'","'_v3'")
        if dump(ast.parse(text).body[0])!=dump(newf[name]):delta.append(name)
    check('only_allowed_operational_delta_'+filename,set(delta)<=allowed,changed_bodies=delta)
    additions=set(newf)-set(oldf)
    check('only_new_slot_identity_validator_'+filename,additions==({'validate_slot_resources'} if filename=='root_contract_v1.py' else set()))
    changes[filename]=delta
for filename,name in [('root_contract_v1.py','cpu_slot'),('root_driver_v1.py','execute'),('plan_worker_v1.py','execute')]:
    check('raw_AST_identical_'+filename+'_'+name,dump(funcs(OLD/filename)[name])==dump(funcs(SRC/filename)[name]))
for filename in ['readiness.v1.json','scientific_basis.v1.json','caps.template.inactive.v1.json','reuse_provenance.v1.json','infer.scientific_config.template.inactive.v1.json','native_paths.scientific_config.template.inactive.v1.json','plan.scientific_config.template.inactive.v1.json']:
    check('science_caps_byte_identical_'+filename,(OLD/filename).read_bytes()==(SRC/filename).read_bytes())
# Reproduce all six complete unions against the real unwrapped validator.
oldauthority=read(OLD/'admitted_val003_infer_v2/reserve_authority.json')
auth({'path':str((OLD/'admitted_val003_infer_v2/reserve_authority.json').relative_to(ROOT)),'sha256':'7d84a1ec18c64d7c9b522f430a45ea5c6da952ab09506722f45813779bfcdb8e'})
base=list(oldauthority['bindings'].values());check('exact_actual_historical_base37',len(base)==37)
reject('v2_original_exact_union_reproduces_lock_suffix_failure',lambda:v4.scope_spec(ROOT,oldauthority['request']),'consumer bindings must be source/runtime authority metadata')
phase_results={}
for parent,row in read(SRC/'readiness.v1.json')['parents'].items():
    for stage in ct.STAGES:
        ct.select(parent,stage);tag=parent+'/'+stage
        fixture=lambda role:save('fixtures/'+parent+'_'+stage+'/'+role+'.json',dict(schema='INDEPENDENT_REVIEW_SOURCE_TEST_ONLY',role=role,parent=parent,stage=stage,actual_authority=False,actual_scientific_evidence=False))
        role_refs={role:fixture(role) for role in ['request','caps','config','closure','review']}
        if stage=='infer':
            prior={k:row[v] for k,v in [('completion_ref','simulation_completion_ref'),('wait_ref','simulation_wait_ref'),('outer_ref','simulation_outer_terminal_ref'),('close_ref','simulation_close_ref'),('observation_ref','simulation_observation_ref')]}
        else:prior={k:fixture('future_'+k) for k in ['completion_ref','wait_ref','outer_ref','close_ref','observation_ref']}
        q=dict(caps_ref=role_refs['caps'],source_review_ref=role_refs['review'],prior=prior)
        roles=ct.direct_roles(role_refs['request'],q,role_refs['config'],role_refs['closure'],ct.RUNTIME_REF)
        union=ct.unique([*base,*roles])
        check(tag+'_complete_actual_base_union62',len(union)==62)
        check(tag+'_only_allowed_direct_suffixes',all(Path(r['path']).suffix in ('.py','.sh','.json') for r in union))
        check(tag+'_locks_only_in_authenticated_transitive_closure',all(r in manifest['refs'] and r not in union for r in manifest['original_shared_slot_refs']))
        request=dict(key='INDEPENDENT_SOURCE_TEST_ONLY/'+parent+'_'+stage,output=str(HERE/'uncreated_products'/parent/stage),auxiliary_scopes=[str(HERE/'uncreated_aux'/parent/stage)],reserved_bytes=ct.CAPS['reservation_bytes'],consumer_bindings=union)
        paths=v4.scope_spec(ROOT,request);check(tag+'_real_unchanged_scope_spec_accepts',len(paths)==2)
        bad=copy.deepcopy(request);bad['consumer_bindings'].append(slots['slots'][0]['ref'])
        reject(tag+'_real_validator_rejects_lock',lambda:v4.scope_spec(ROOT,bad),'consumer bindings must be source/runtime authority metadata')
        bad=copy.deepcopy(request);bad['consumer_bindings']+=[fixture('extra'+str(i)) for i in range(3)]
        reject(tag+'_real_validator_rejects65',lambda:v4.scope_spec(ROOT,bad),'source/runtime consumer metadata bindings required')
        bad=copy.deepcopy(request);bad['consumer_bindings'][0]['sha256']='0'*64
        reject(tag+'_real_validator_rejects_wrong_hash',lambda:v4.scope_spec(ROOT,bad),'bound source or authority bytes changed')
        phase_results[tag]=dict(complete_count=len(union),consumer_bindings=union,accepted_by_real_unchanged_scope_spec=True,future_roles_are_explicit_review_fixtures_only=True)
check('no_product_or_aux_scope_created',not (HERE/'uncreated_products').exists() and not (HERE/'uncreated_aux').exists())
# Explicit expiry validation, using the production function with wholly fake metadata.
ct.select('american_c3_val_003','infer');fixed=datetime.fromisoformat(EXPIRY)
class FrozenDateTime:
    now=staticmethod(lambda tz:fixed-timedelta(seconds=1800))
    fromisoformat=staticmethod(datetime.fromisoformat)
q=dict(schema='American_C3_VAL003004_native14v3_request_v2',active=True,root_authorizer='root_agent',parent_id=ct.PARENT,stage=ct.STAGE,phase=ct.PHASE,created_at_utc=(fixed-timedelta(seconds=2000)).isoformat(),expires_at_utc=EXPIRY,execution_authorized=True,source_review_passed=True,adapter_manifest_ref=ref(SRC/'source_manifest.v1.json'),guard_ref=ct.GUARD_REF,policy_ref=ct.POLICY_REF,maximum_CPU_producers=2,query_truth_access_authorized=False,GNN_optimization_authorized=False,TEST_access_authorized=False,all30_C3_prediction_seals_before_VAL_truth=True,source_review_ref='review',prior={'fixture':True},scientific_config={'fixture':True},bindings=dict(parent_id=ct.PARENT,scientific_config={'fixture':True}),caps_ref='caps',runtime_ref=ct.RUNTIME_REF)
ns=dict(ct.__dict__,datetime=FrozenDateTime,read=lambda r:q if r=='q' else {'fixture':True},bind_policy=lambda r:None,select=lambda *a:None,reviewed=lambda r:None,source_paths=lambda:(),actual_prior=lambda *a:{'fixture':True},scientific_config=lambda *a:{'fixture':True},worker_caps=lambda s:{'fixture':True})
extract(funcs(SRC/'root_contract_v1.py')['active'],ns);extract(funcs(SRC/'root_contract_v1.py')['validate_request'],ns)
check('exact_expiry_request_accepted_in_fake_metadata_context',ns['validate_request']('q')[0] is q)
q['expires_at_utc']=(fixed+timedelta(seconds=1)).isoformat()
reject('validate_request_rejects_extended_expiry',lambda:ns['validate_request']('q'),'Exact original failed-admission expiry')
q['expires_at_utc']=(fixed-timedelta(seconds=1)).isoformat()
reject('validate_request_rejects_changed_earlier_expiry',lambda:ns['validate_request']('q'),'Exact original failed-admission expiry')
q['expires_at_utc']=EXPIRY
class ExpiredDateTime(FrozenDateTime):now=staticmethod(lambda tz:fixed)
ns['datetime']=ExpiredDateTime
reject('validate_request_rejects_expired_original_authority',lambda:ns['validate_request']('q'),'finite evidence margin')
saved={}
def fake_save(folder,name,v):saved[name]=copy.deepcopy(v);return {'fixture_ref':name}
ns=dict(ct.__dict__,datetime=FrozenDateTime,bind_policy=lambda r:None,original_guard=lambda:None,select=lambda *a:None,reviewed=lambda r:None,source_paths=lambda:(),actual_prior=lambda *a:{'fixture':True},scientific_config=lambda *a:{'fixture':True},read=lambda r:{'renewed_expires_at_utc':'2026-10-20T00:00:00+00:00'},save=fake_save,local=lambda p:p,ref=lambda p:{'fixture':str(p)},worker_caps=lambda s:{'fixture':True},validate_request=lambda r:None)
extract(funcs(SRC/'root_contract_v1.py')['bind'],ns)
ns['bind']('american_c3_val_003','infer',{},{});check('bind_emits_only_original_fixed_expiry',saved['request_val003_infer.root_active_v1.json']['expires_at_utc']==EXPIRY)
ns['datetime']=ExpiredDateTime;saved.clear()
reject('bind_after_original_expiry_rejected',lambda:ns['bind']('american_c3_val_003','infer',{},{}),'Original failed-admission deadline')
check('expired_bind_writes_nothing',not saved)
# The actual admit body must reject an old v2 key before its first fake packet write.
class ReachedPacket(Exception):pass
mkdir_calls=[]
class FakePacket:
    def exists(self):return False
    def mkdir(self):mkdir_calls.append(True);raise ReachedPacket()
packet=FakePacket();old_key='H041/American-C3-val003-full30mb-infer-native14v3-v2'
state=[{}, {'predecessor':{}}, {}, {}]
budget=types.SimpleNamespace(SOURCE_REL='fake_source',LAUNCHER_REL='fake_launcher',v4=types.SimpleNamespace(mutex=lambda root:contextlib.nullcontext()))
g=types.SimpleNamespace(budget=budget,_state_locked=lambda *a,**k:state)
fakec=types.SimpleNamespace(validate_request=lambda r:({'prior':{'closed_key':'predecessor'}},{},{}),phase_paths=lambda phase:(packet,'fake_output'),PHASE='fake_phase',original_guard=lambda:g)
ns=dict(c=fakec,ROOT=ROOT,need=ct.require,local=lambda p:packet,ref=lambda p:{'sha256':'0'*64})
extract(funcs(SRC/'root_driver_v1.py')['admit'],ns)
try:ns['admit']({})
except ReachedPacket:check('admit_without_v2_key_reaches_first_fake_packet_write',len(mkdir_calls)==1)
else:raise AssertionError('admit baseline failed')
state[0][old_key]={};mkdir_calls.clear()
reject('admit_with_v2_key_rejected_before_write',lambda:ns['admit']({}),'Original partial VAL003 admission must remain unreserved')
check('v2_key_rejection_writes_nothing',not mkdir_calls)
# Replay the inherited pure/fake suite, removing only its two final report writes.
fakepath=SRC/'fake_contracts_v3.py';tree=ast.parse(fakepath.read_text());removed=[];newbody=[]
for n in tree.body:
    if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='write_text':removed.append(n.lineno)
    else:newbody.append(n)
check('inherited_fake_replay_removes_only_report_writes',len(removed)==2,removed_line_numbers=removed)
tree.body=newbody;namespace={'__file__':str(fakepath),'__name__':'independent_inherited_replay'}
with contextlib.redirect_stdout(io.StringIO()):exec(compile(ast.fix_missing_locations(tree),str(fakepath),'exec'),namespace)
check('all139_inherited_fake_checks_replayed_without_source_writes',namespace['result']['passed']==139)
# Real default CLI only: no flags, no actual admission/execution.
env=dict(os.environ,PYTHONPATH='.:src:scripts',PYTHONDONTWRITEBYTECODE='1')
cli=subprocess.run([str(ROOT/'.runtime/env/bin/python'),'-B','-m',str(SRC.relative_to(ROOT)).replace('/','.')+'.root_driver_v1'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
check('real_default_CLI_inert',cli.returncode==0 and cli.stderr=='' and json.loads(cli.stdout)==dict(status='SOURCE_ONLY_INACTIVE',parents=list(ct.PARENTS),stages=list(ct.STAGES),maximum_CPU_producers=2,execution=False),stdout=cli.stdout.strip())
check('no_actual_stage_packet_or_request_created',not list(SRC.glob('admitted*')) and not list(SRC.glob('request*.root_active*.json')) and not list(SRC.glob('caps*.root_active*.json')))
check('source_fixture_futures_labelled_not_actual',all(read(ROOT/r['path']).get('actual_scientific_or_admission_evidence') is False for p in read(SRC/'real_scope_contract_results.v3.json')['phases'].values() for r in p['fixture_refs']))
check('candidate_and_transitive_seal_unchanged_after_review',all(ref(path)==r for path,r in sealed.items()))
results=dict(schema='VAL003004_v3_independent_source_checks_v1',status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),passed=len(CHECKS),checks=CHECKS,inherited_pure_fake_checks_replayed=namespace['result']['passed'],candidate_handoff_ref=ref(SRC/'source_handoff.v1.json'),source_manifest_ref=ref(SRC/'source_manifest.v1.json'),phase_unions=phase_results,changed_production_bodies=changes,original_expiry_utc=EXPIRY,all_writes_within=str(HERE.relative_to(ROOT)),actual_accounting_or_admission_calls=0,actual_scientific_jobs=0,actual_resource_probes=0,scientific_truth_or_payload_reads=0,actual_scope_spec_calls='six complete valid unions plus rejection cases; unchanged real function',inherited_replay_transform='Removed only two final write_text expressions so sealed source results were not rewritten')
save('review_checks.results.v1.json',results)
print(json.dumps(dict(status='PASS',independent_checks=len(CHECKS),inherited_fake_checks=namespace['result']['passed'],all_six_real_unions=62)))
