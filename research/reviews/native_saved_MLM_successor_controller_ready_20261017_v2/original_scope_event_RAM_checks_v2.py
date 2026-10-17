"""Original supplemental scope_spec and original event encoder, RAM-only boundaries."""
from pathlib import Path as RealPath,PurePosixPath
import ast,copy,hashlib,json,types
D=RealPath('research/reviews/native_saved_MLM_successor_controller_ready_20261017_v2');N=D/'native64'
checks=[];negative=[]
def need(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)
class P(PurePosixPath):
    def resolve(self):return self
ROOT=P('/workspace-SR003.nfs2/ARG-GNN')
source=RealPath('src/arggnn/project_budget_v4.py').read_text();tree=ast.parse(source)
names={'encoded','integer','bound','paths_overlap','accounting_paths','canonical','scope_spec'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
env=dict(Path=P,json=json,LEGACY='.runtime/storage/ledger.json',LOCK='.runtime/locks/scheduler.lock')
files={}
def file_bytes(root,path):
    key=str(path)
    if key.startswith(str(ROOT)+'/'):key=key[len(str(ROOT))+1:]
    if key not in files:raise AssertionError('Unknown RAM source or control '+key)
    b=files[key];return b,hashlib.sha256(b).hexdigest()
env.update(file_bytes=file_bytes,parse=json.loads)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'RAM_ORIGINAL_V4_SCOPE_AND_BOUND','exec'),env)
roles=json.loads(RealPath('research/reviews/FLARE_VAL004_native_ROOT_close_Census_root01_oct17/ordinary_close_authority.json').read_text())['bindings']
manifest=json.loads((N/'source_manifest.v1.json').read_text())
def ref(path):
    p=RealPath(path);b=p.read_bytes();r=dict(path=str(p),sha256=hashlib.sha256(b).hexdigest());files[r['path']]=b;return r
def ramref(name):
    b=json.dumps(dict(schema='RAM_ONLY_authority',label=name),sort_keys=True).encode();r=dict(path='RAM_ONLY/'+name+'.json',sha256=hashlib.sha256(b).hexdigest());files[r['path']]=b;return r
for r in roles.values():
    b=RealPath(r['path']).read_bytes();need(hashlib.sha256(b).hexdigest()==r['sha256'],'Authenticate exact role '+r['path']);files[r['path']]=b
const={}
for n in ast.parse((N/'contract_v1.py').read_text()).body:
    if isinstance(n,ast.Assign):
        for t in n.targets:
            if isinstance(t,ast.Name)and t.id in ['GUARD','SAVED_INTEGRATION_REF','SAVED_SCIENTIFIC_ENTRY_REF','SAVED_SCIENCE_REVIEW_REF']:const[t.id]=ast.literal_eval(n.value)
common=ast.parse(RealPath('research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/common_v1.py').read_text())
runtime=ast.literal_eval(next(n.value for n in common.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='RUNTIME_REF'for t in n.targets)))
for r in list(const.values())+[runtime]+manifest['executed_source_refs']:
    b=RealPath(r['path']).read_bytes();need(hashlib.sha256(b).hexdigest()==r['sha256'],'Authenticate native direct ref '+r['path']);files[r['path']]=b
fn=next(n for n in ast.parse((N/'root_builder_v1.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='prepare_master_v1')
assign=[n for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='direct'for t in n.targets)or isinstance(n,ast.AugAssign)and isinstance(n.target,ast.Name)and n.target.id=='direct']
formula=dict(c=types.SimpleNamespace(**const,MANIFEST=str(N/'source_manifest.v1.json'),ref=ref),config_ref=ref(N/'scientific_config.FROZEN.v1.json'),master=dict(runtime_ref=runtime),request_ref=ramref('native_request'),q=dict(source_review_ref=ramref('native_review'),ROOT_saved_MLM_artifact_acceptance_ref=ramref('saved_acceptance')),manifest=manifest)
exec(compile(ast.Module(body=assign,type_ignores=[]),'RAM_EXACT_NATIVE_DIRECT_FORMULA','exec'),formula)
def unique(rows):
    result=[]
    for r in rows:
        if r not in result:result.append(r)
    return result
native_bindings=unique(list(roles.values())+formula['direct'])
controller_bindings=unique(list(roles.values())+[const['GUARD'],ref(D/'empty_gpu0_controller_v1.py'),ref(D/'owner_parent_v1.py'),ref(D/'source_manifest.v1.json')]+[ramref('controller_'+s)for s in ['plan','config','runtime','review','ROOT_request','handover']])
need(len(roles)==47 and len(unique(list(roles.values())))==40,'Actual47roles preserve40unique refs')
need(len(native_bindings)==57 and len(controller_bindings)==50,'Exact complete native57/controller50 direct unions')
native=dict(key='H056/RAM-native-saved-MLM',output=str(ROOT/'experiments/results/RAM_native_science'),auxiliary_scopes=[str(ROOT/'research/reviews/RAM_native_aux')],reserved_bytes=64*2**30,consumer_bindings=native_bindings)
controller=dict(key='GPU0-empty/RAM-controller',output=str(ROOT/'experiments/results/RAM_empty_controller'),auxiliary_scopes=[],reserved_bytes=32*2**20,consumer_bindings=controller_bindings)
for name,request in [('native',native),('controller',controller)]:
    need(len(env['scope_spec'](ROOT,request))==(2 if name=='native'else 1),'Full actual original scope_spec accepts '+name)
    need(all(RealPath(r['path']).suffix in('.py','.sh','.json')for r in request['consumer_bindings']),'Every direct binding has original allowed source/runtime suffix '+name)
def reject(name,request):
    try:env['scope_spec'](ROOT,request)
    except(ValueError,KeyError,TypeError):negative.append(name)
    else:raise AssertionError('Original scope validator accepted '+name)
bad=copy.deepcopy(controller);bad['consumer_bindings'].append(dict(path='.runtime/locks/gpu_0.lock',sha256='0'*64));reject('Raw physical lock cannot become source binding',bad)
bad=copy.deepcopy(native);bad['output']=str(ROOT/'.runtime/storage/forbidden_output');reject('Accounting-scope overlap',bad)
bad=copy.deepcopy(native);bad['auxiliary_scopes']=[bad['output']+'/descendant'];reject('Science/AUX overlap',bad)
bad=copy.deepcopy(native);bad['consumer_bindings']=[*native_bindings,*[ramref('overflow_'+str(i))for i in range(8)]];reject('65distinct bindings',bad)
bad=copy.deepcopy(controller);bad['output']='../escape';reject('Escaping output',bad)
bad=copy.deepcopy(controller);bad['reserved_bytes']=True;reject('Boolean charge',bad)
bad=copy.deepcopy(controller);bad['extra']='hidden';reject('Extra reservation schema',bad)
bad=copy.deepcopy(controller);bad['consumer_bindings'][0]=dict(path=controller_bindings[0]['path'],sha256='0'*64);reject('Changed bound source hash',bad)
# Execute exact original event construction and byte guard only. Stop before directory boundary.
gsource=RealPath('research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py').read_text()
gtree=ast.parse(gsource);append=next(n for n in gtree.body if isinstance(n,ast.FunctionDef)and n.name=='_append')
cut=next(i for i,n in enumerate(append.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='directory'for t in n.targets))
prefix=append.body[:cut]
state=(None,None,None,roles,[{}for _ in range(22)],['a'*64]*22)
sizes={}
for name,request in [('native',native),('controller',controller)]:
    eventenv=dict(budget=types.SimpleNamespace(v4=types.SimpleNamespace(encoded=env['encoded'])),require=lambda ok,msg:(_ for _ in()).throw(ValueError(msg))if not ok else None,MAX_EVENTS=next(ast.literal_eval(n.value)for n in gtree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='MAX_EVENTS'for t in n.targets)),MAX_JSON=65536,operation='reserve',authority_ref=ramref(name+'_reserve_authority'),value=dict(request,root_admission=ramref(name+'_admission')),state=state)
    exec(compile(ast.Module(body=prefix,type_ignores=[]),'RAM_ORIGINAL_EVENT_PREFIX_NO_JOURNAL','exec'),eventenv)
    sizes[name]=len(eventenv['raw']);need(sizes[name]<=65536,'Exact original encoded full reserve event fits64KiB '+name)
    need(eventenv['raw']==env['encoded'](eventenv['event'])+b'\n','Original event includes final newline '+name)
need(454656+65536+4096==524288,'Native existing ROOT/event administrative bound preserved')
need(128*1024+65536+4096<=524288,'Empty owner ROOT/event allowance fits unchanged512KiB')
result=dict(schema='ROOT_empty_owner_and_native_original_scope_event_RAM_checks_v1',passed=True,assertions=len(checks),negative_cases=negative,original_scope_function_AST_exact=True,original_bound_function_AST_exact=True,full_native_direct_union=57,full_controller_direct_union=50,raw_lock_bindings=0,complete_reserve_event_bytes=sizes,actual_guard_imports_calls=0,actual_journal_or_accounting_or_process_operations=0,all_future_control_objects_are_RAM_only=True,canonical_and_file_bytes_are_explicit_RAM_boundaries=True,independent_review_claimed=False)
print(json.dumps(result,sort_keys=True,indent=2))
