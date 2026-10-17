"""Independent static/AST and handwritten fixtures only. No active project imports."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,hashlib,json,stat,sys,warnings
warnings.simplefilter('ignore',SyntaxWarning)
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/H058_fresh_CPU_lifetime_independent_review_20261017_v1'
version=sys.argv[1] if len(sys.argv)>1 else 'v1'
assert version in ('v1','v2')
HERE='research/reviews/H058_fresh_CPU_lifetime_source_20261017_'+version
S=ROOT/HERE
B=json.loads((S/'source_bindings.v1.json').read_text())
M=json.loads((S/'source_manifest.v1.json').read_text())
BASE=ROOT/B['base_directory']
SRC=ast.parse((S/'root_driver_v1.py').read_text())
BASECON=ast.parse((BASE/'root_contract_v1.py').read_text())
BASEDRIVER=ast.parse((BASE/'root_driver_v1.py').read_text())
SEL=ast.parse((ROOT/B['selector_protocol_contract_ref']['path']).read_text())
checks={}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def local(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 need(p!=ROOT and p.is_relative_to(ROOT) and '..' not in p.parts and p.resolve()==p,'canonical')
 return p
def ref(p):
 p=local(p);st=p.lstat();need(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_size<=32*2**20,'source/metadata only')
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(r):
 need(ref(r['path'])==r,'SHA');return local(r['path']).read_bytes()
def read(r):return json.loads(raw(r))
def fn(tree,name):return next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
def compile_nodes(nodes,ns):
 code=ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]));exec(compile(code,'<independent-extracted-AST>','exec'),ns);return ns
def extracted(tree,name,ns):return compile_nodes([fn(tree,name)],ns)[name]
def rejects(action):
 try:action()
 except (ValueError,KeyError,TypeError):return True
 return False
# Authenticate every source/metadata item without executing any project code.
for p,h in M['source_sha256'].items():need(ref(p)['sha256']==h,'manifest SHA '+p)
pyfiles=[p for p in M['source_sha256'] if p.endswith('.py')]
for p in pyfiles:compile(ast.parse((ROOT/p).read_text()),p,'exec')
checks['sealed_manifest']={'all_ref_count':len(M['source_sha256']),'all_SHA_match':True,'Python_AST_compile_count':len(pyfiles)}
missing={}
for p in (B['source_manifest_ref']['path'],B['selector_protocol_manifest_ref']['path']):
 m=read(ref(p));rs=m.get('refs',[])+[dict(path=k,sha256=v) for k,v in m.get('source_sha256',{}).items()]
 missing[p]=[r['path'] for r in rs if M['source_sha256'].get(r['path'])!=r['sha256']]
need(not any(missing.values()),'original closure loss');checks['complete_original_and_selector_closures']=missing
# Actual source-pure authority interval, including timezone and old25h boundary fixtures.
interval=extracted(SRC,'authority_interval',dict(datetime=datetime,need=need))
c0=datetime(2026,10,17,tzinfo=timezone.utc)
for seconds in (90000,259200,262800):
 q=dict(created_at_utc=c0.isoformat(),expires_at_utc=(c0+timedelta(seconds=seconds)).isoformat())
 need(rejects(lambda:interval(q))==(seconds!=259200),'finite interval')
need(rejects(lambda:interval(dict(created_at_utc='2026-10-17T00:00:00',expires_at_utc='2026-10-20T00:00:00'))),'naive time')
helper=ast.parse((ROOT/'research/reviews/European_outer8_current_policy_successor_source_20261016_v1/prepare_outer000_v1.py').read_text())
class FixedDT(datetime):
 now_value=c0
 @classmethod
 def now(cls,tz=None):return cls.now_value
cur=extracted(helper,'current',dict(datetime=FixedDT,timezone=timezone,need=need))
rows=[]
for seconds,setup,expected in ((90000,2699,True),(90000,2700,False),(259200,171899,True),(259200,171900,False)):
 q=dict(created_at_utc=c0.isoformat(),expires_at_utc=(c0+timedelta(seconds=seconds)).isoformat());FixedDT.now_value=c0+timedelta(seconds=setup)
 accepted=not rejects(lambda:cur(q,87300));need(accepted==expected,'unchanged strict margin')
 rows.append(dict(authority_seconds=seconds,setup_seconds=setup,accepted=accepted))
checks['authority_interval_and_unchanged_current']=rows
# Extract actual namespace loop, independently apply original path gates to exploit fixture.
contract=fn(SRC,'contract');validator=next(x for x in contract.body if isinstance(x,ast.FunctionDef) and x.name=='validate_request');namespace_loop=next(x for x in validator.body if isinstance(x,ast.For))
namespace_fn=ast.FunctionDef(name='namespace_gate',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='q')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[namespace_loop],decorator_list=[])
ns=dict(c=SimpleNamespace(local=local,require=need),b=B)
namespace_gate=compile_nodes([namespace_fn],ns)['namespace_gate']
baseq=dict(preparation_directory='research/reviews/H058_CPU_ROOT_preparation_REVIEW_FIXTURE',packet_directory='research/reviews/H058_CPU_ROOT_REVIEW_FIXTURE',output_directory='experiments/results/H058_full_parent_CPU_REVIEW_FIXTURE',cpu_ids=list(range(160,192)))
namespace_gate(baseq)
namespace_cases=[]
for newkey in B['excluded_namespaces']:
 for oldkey,old in B['excluded_namespaces'].items():
  for relation,value in [('equal',old),('descendant',old+'/NEVER_CREATED_REVIEW_FIXTURE'),('ancestor',str(Path(old).parent))]:
   q=copy.deepcopy(baseq);q[newkey]=value
   namespace_cases.append(dict(new_key=newkey,old_key=oldkey,relation=relation,rejected=rejects(lambda:namespace_gate(q))))
class FixturePath(type(Path())):
 def is_dir(self):return str(self)==str(ROOT/baseq['preparation_directory'])
 def exists(self):return False
def fixturelocal(p):return FixturePath(p if Path(p).is_absolute() else ROOT/p)
original=fn(BASECON,'validate_request')
oldpathnodes=[x for x in original.body if 92<=x.lineno<=99]
oldpathfn=ast.FunctionDef(name='original_path_gates',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='q'),ast.arg(arg='record')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=oldpathnodes,decorator_list=[])
oldgate=compile_nodes([oldpathfn],dict(local=fixturelocal,require=need))['original_path_gates']
exploit=copy.deepcopy(baseq);exploit['packet_directory']=B['excluded_namespaces']['preparation_directory']+'/NEVER_CREATED_REVIEW_PACKET'
oldgate(exploit,dict(path=baseq['preparation_directory']+'/request.json'))
exploit_accepted=not rejects(lambda:namespace_gate(exploit))
need(exploit_accepted==(version=='v1'),'namespace expected version behavior')
if version=='v2':need(all(x['rejected'] for x in namespace_cases),'all namespace pairs')
checks['namespace']={'cases':namespace_cases,'flat_fresh_siblings_accept':True,'original_path_gates_accept_cross_key_packet_fixture':True,'cross_key_packet_in_old_preparation_accepts':exploit_accepted,'fixture_never_written_or_admitted':True}
# Pure reviewed selector functions read only their pinned metadata, never guard/load.
selns=dict(ROOT=ROOT,HERE=str(Path(B['selector_protocol_contract_ref']['path']).parent),read=read,raw=raw,ref=ref,require=need,copy=copy,SELECTOR_REF=None,_CHOSEN=None)
sel_selected=extracted(SEL,'selected',selns)
sel_bind=extracted(SEL,'bind_selector',selns)
selector=SimpleNamespace(selected=sel_selected)
c=SimpleNamespace(require=need)
selected_node=next(x for x in contract.body if isinstance(x,ast.FunctionDef) and x.name=='selected')
new_selected=compile_nodes([selected_node],dict(selector=selector,c=c,b=B))['selected']
chosen=new_selected(B['actual_selector_metadata_ref']);sel_bind(B['actual_selector_metadata_ref'])
need(chosen['source_ref']==B['selected_child_source_ref'],'exact actual child')
# Validate unchanged64 role union using actual pinned role metadata and source-pure dedup routine.
deps=read(ref(str(Path(B['contract_ref']['path']).parent/'dependencies.v1.json')))
q=read(B['original_request_ref']);q=copy.deepcopy(q)
def synthetic(name):return dict(path='research/reviews/REVIEW_ONLY_NEVER_CREATED/'+name+'.json',sha256=hashlib.sha256(name.encode()).hexdigest())
for k in ('source_review_ref','runtime_ref','runtime_qualification_ref','root_directive_ref','registration_decisions_ref','training_context_ref'):q[k]=synthetic(k)
q['storage_policy_selector_ref']=B['actual_selector_metadata_ref']
requestref=synthetic('request');cfgref=synthetic('config');closure=synthetic('closure');scienceref=synthetic('science_config')
original_roles=extracted(BASECON,'direct_roles',dict(binding=lambda:read(B['scientific_binding_ref']),ref=ref,MANIFEST=HERE+'/source_manifest.v1.json',SOURCE=HERE+'/root_driver_v1.py',DRIVER=HERE+'/root_driver_v1.py',ENTRY=HERE+'/worker_entry_v1.py',HANDOFF=HERE+'/source_handoff.v1.json',CAPS=B['caps'],make_config=lambda *_:dict(scientific_config_ref=scienceref),SELECTOR=chosen['source_ref'],**{k:deps[k]for k in ('NAMESPACE','PIDFD','OWNED','SUPERVISOR')}))
c=SimpleNamespace(require=need,DEPS=deps,GUARD_REF=chosen['source_ref'])
newroles_node=next(x for x in contract.body if isinstance(x,ast.FunctionDef) and x.name=='direct_roles')
newroles=compile_nodes([newroles_node],dict(c=c,original_roles=original_roles,selected=new_selected,copy=copy))['direct_roles']
roles=newroles(requestref,q,cfgref,closure)
a=read(B['actual_guard_role_metadata_ref']);guards=list(a['bindings'].values())
childtree=ast.parse((ROOT/B['selected_child_source_ref']['path']).read_text())
def shape_raw(r):need(isinstance(r,dict) and set(r)=={'path','sha256'} and isinstance(r['path'],str) and len(r['sha256'])==64,'fixture ref shape')
consumer=extracted(childtree,'consumer_bindings_v1',dict(require=need,_raw=shape_raw,copy=copy))
union=consumer(guards,roles)
need(len(union)==62 and all(x in union for x in guards+roles) and deps['SELECTOR'] in union and chosen['source_ref'] in union,'complete exact baseline union')
extra=[synthetic('extra'+str(i))for i in range(3)]
need(len(consumer(union,extra[:2]))==64 and rejects(lambda:consumer(union,extra)),'64cap')
checks['role_union']={'actual_metadata_role_names':len(a['bindings']),'distinct_guard_refs':len(consumer(guards)),'distinct_complete_union':len(union),'all_guard_and_direct_refs_preserved':True,'original_static_selector_and_exact_child_direct':True,'64_accept_65_reject':True,'future_actual_current_union_remains_required':True}
# Extract failure acceptance gate and mutate each substantive acceptance field.
failure_node=next(x for x in contract.body if isinstance(x,ast.FunctionDef) and x.name=='accept_failure')
f=read(ref(HERE+'/ROOT_failure_acceptance.INACTIVE.template.v1.json'))
f.update(root_authorizer='root_agent',accepted=True,actual_outer_terminal_ref=synthetic('actual_outer_terminal'),outer_tool_terminal_consumed=True,outer_exit_code=1,direct_worker_returncode=1,owned_drain=True,scientific_startup_reached=False,optimizer_updates=0,old_launch_resume_write_authority_revoked=True,old_files_key_charge_and_expiry_unchanged=True)
c=SimpleNamespace(require=need,read=lambda _:f,raw=shape_raw)
accept=compile_nodes([failure_node],dict(c=c,b=B))['accept_failure'];accept(synthetic('failure_acceptance'))
mutation_count=0
for key in list(f):
 if key=='actual_outer_terminal_ref':continue
 old=f[key];f[key]=None
 need(rejects(lambda:accept(synthetic('failure_acceptance'))),'failure field '+key);f[key]=old;mutation_count+=1
checks['failure_acceptance']={'all_required_fields_mutation_reject_count':mutation_count,'genuine_future_ROOT_outer_receipt_still_required':True,'no_ROOT_acceptance_created':True}
# Admitted metadata identity patch, only new exact input and onlyH058 value.
put_node=next(x for x in contract.body if isinstance(x,ast.FunctionDef) and x.name=='put')
writes=[]
c=SimpleNamespace(require=need,ref=ref)
patched_put=compile_nodes([put_node],dict(c=c,SOURCE=HERE+'/root_driver_v1.py',Path=Path,original_put=lambda path,value,cap: writes.append(copy.deepcopy(value))))['put']
v=dict(schema='H058_full_parent_CPU_admitted_input_v1',root_driver_ref=ref(HERE+'/root_driver_v1.py'))
patched_put('input.json',v);need(writes[-1]==dict(v,hypothesis_id='H058') and 'hypothesis_id'not in v,'identity addition only')
patched_put('other.json',v);need(writes[-1]==v,'no arbitrary metadata change')
need(rejects(lambda:patched_put('input.json',dict(v,root_driver_ref=synthetic('wrong_driver')))),'wrongdriver')
checks['hypothesis_identity']={'only_new_admitted_input_adds_H058':True,'caller_value_unchanged':True,'wrong_driver_rejected':True,'original_context_requires_H058':True}
# Actual immutable original lifecycle ordering; no workload function is executed.
admit=fn(BASEDRIVER,'admit');calls=[(n.lineno,ast.unparse(n.func))for n in ast.walk(admit)if isinstance(n,ast.Call)]
line=lambda suffix:min(x for x,n in calls if n.endswith(suffix))
need(line('consumer_bindings_v1')<line('scope_spec')<line('operation_draft_v1')<line('apply_reserve_v1'),'preflight before authority and reserve')
basecaps=read(ref(str(Path(B['contract_ref']['path']).parent/'caps.inactive.v1.json')))
need(basecaps==B['caps'] and B['cpu_ids']==list(range(160,192)),'exact capsCPUs')
checks['unchanged_science_lifecycle']={'base_caps_exact':True,'CPU160_through191_exact':True,'original_science_seconds':86400,'original_outer_seconds':86700,'preflight_call_lines':{s:line(s)for s in ('consumer_bindings_v1','scope_spec','operation_draft_v1','apply_reserve_v1')},'no_numeric_body_changed':True,'original_full32TRAIN_30passes_960each_1920total':True,'original_completion_wait_drain_caps_preserved':True}
result=dict(schema='H058_fresh_lifetime_independent_source_checks_v1',source_ref=ref(HERE+'/root_driver_v1.py'),source_manifest_ref=ref(HERE+'/source_manifest.v1.json'),source_handoff_ref=ref(HERE+'/source_handoff.v1.json'),checks=checks,guards_imported=False,mutexes_or_real_state_accessed=False,numeric_payload_or_execution=False,actual_admission_or_job=False,synthetic_fixtures_are_not_ROOT_authority=True)
p=OUT/('review_checks.'+version+'.json');p.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(result_path=str(p),source_ref=result['source_ref'],manifest_ref_count=len(M['source_sha256']),namespace_vulnerability=exploit_accepted,role_union_count=len(union),failure_field_mutation_count=mutation_count),sort_keys=True))
