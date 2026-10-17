"""Independent full helper-v2 metadata→request→input→argv joins, all future IO in RAM.
No observe/capacity/guard/admit/execute/Popen/scientific method is invoked.
"""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import ast,contextlib,copy,hashlib,importlib,importlib.util,io,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');HERE='research/reviews/H058_GPU0_RSS_identity_independent_20261017_v7';P=ROOT/HERE
PKG='research/reviews/H058_GPU0_full_training_successor_source_20261017_v7'
PREP='research/reviews/H058_GPU0_v7_ROOT_preparation_source_20261017_v2'
OLD='research/reviews/H058_GPU0_full_training_successor_source_20261017_v5'
CHECKS=[];REAL_REFS={};REAL_JSON={};RAM={};LATEST={};DIRS=set();EVENTS=[]
def need(ok,msg):
 if not ok:raise AssertionError(msg)
def good(n,fn):fn();CHECKS.append(dict(name=n,status='PASS'))
def reject(n,fn,kind=(AssertionError,ValueError,KeyError),part=None):
 try:fn()
 except kind as e:
  need(part is None or part in str(e),'Wrong rejection '+n+': '+str(e));CHECKS.append(dict(name=n,status='PASS_REJECTED',message=str(e)));return
 raise AssertionError('Not rejected '+n)
def stable_ref(path):
 p=ROOT/path;need(p.suffix in('.py','.json','.sh','.md','.patch','.txt','.diff')and p.is_file()and not p.is_symlink()and p.stat().st_size<=32*2**20,'Source/small JSON only '+path)
 return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def ref(path):
 path=str(path)
 if Path(path).is_absolute():path=str(Path(path).relative_to(ROOT))
 if path in LATEST:return copy.deepcopy(LATEST[path])
 if path not in REAL_REFS:REAL_REFS[path]=stable_ref(path)
 return copy.deepcopy(REAL_REFS[path])
def read(r):
 key=(r['path'],r['sha256'])
 if key in RAM:return copy.deepcopy(RAM[key])
 need(ref(r['path'])==r and Path(r['path']).suffix=='.json','Exact JSON metadata')
 if key not in REAL_JSON:REAL_JSON[key]=json.loads((ROOT/r['path']).read_bytes())
 return copy.deepcopy(REAL_JSON[key])
def put(path,value):
 raw=(json.dumps(value,sort_keys=True,indent=2)+chr(10)).encode();r=dict(path=path,sha256=hashlib.sha256(raw).hexdigest())
 RAM[(path,r['sha256'])]=copy.deepcopy(value);LATEST[path]=r;return r
class RAMPath:
 def __init__(self,path):self.p=Path(path)
 def __truediv__(self,x):return RAMPath(self.p/str(x))
 def __str__(self):return str(self.p)
 def __eq__(self,x):return self.p==(x.p if isinstance(x,RAMPath)else x)
 def __hash__(self):return hash(self.p)
 @property
 def parent(self):return RAMPath(self.p.parent)
 def is_relative_to(self,x):return self.p.is_relative_to(x.p if isinstance(x,RAMPath)else x)
 def relative_to(self,x):return self.p.relative_to(x.p if isinstance(x,RAMPath)else x)
 def exists(self):return str(self.p)in DIRS or str(self.p.relative_to(ROOT))in LATEST
 def is_dir(self):return str(self.p)in DIRS
 def mkdir(self):
  need(not self.exists(),'RAM exclusive directory');DIRS.add(str(self.p));EVENTS.append('RAM_mkdir')
 def resolve(self):return self
 def is_symlink(self):return False
 def is_file(self):return str(self.p.relative_to(ROOT))in LATEST
# Actual inert imports: no module body invokes guard/resource/science functions.
sys.path.insert(0,str(ROOT));ns=PKG.replace('/','.')
c=importlib.import_module(ns+'.common_v1');d=importlib.import_module(ns+'.direct_gpu0_v1');driver=importlib.import_module(ns+'.root_driver_v1')
context=importlib.import_module(ns+'.context_factory_v1');producer=importlib.import_module(ns+'.producer_v1')
sp=importlib.util.spec_from_file_location('_independent_H058_v7_actual_preparer_v2',ROOT/PREP/'prepare_ROOT_v1.py');helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper)
need(ref(PREP+'/prepare_ROOT_v1.py')['sha256']=='b907567a25a2a02f588ad62de9e4bbd201e44c5f78b7f5d1920ec5fd8aae5a42','Exact helperv2')
need(ref(PREP+'/source_manifest.v1.json')['sha256']=='0ac78001845458fddc8278ea2d89617730606ebed42aeacd9c8d0b70a3578b8b','Exact helper manifestv2')
for path in(PREP+'/source_manifest.v1.json',PREP+'/MANIFEST.v1.json',PKG+'/MANIFEST.v7.json'):
 v=read(ref(path))
 if 'source_sha256'in v:
  for p,h in v['source_sha256'].items():need(ref(p)['sha256']==h,'Sealed source '+p)
 for r in v.get('refs',[]):need(ref(r['path'])==r,'Sealed ref '+r['path'])
good('Actual inert common/direct/context/driver/producer all bind explicit v7',lambda:need(c.HERE==d.HERE==driver.HERE==context.HERE==producer.HERE==PKG and d.validate_native_resource_disposition_v1 is c.validate_native_resource_disposition_v1 and driver.validate_request_v1 is c.validate_request_v1,'One coherent package'))
original_local=c.local
new_prefixes=('research/reviews/H058_GPU0_v7_ROOT_preparation_','research/reviews/H058_GPU0_v7_ROOT_actual_','experiments/results/H058_GPU0_v7_full_parent_')
def local(path):
 if any(str(path).startswith(p)for p in new_prefixes):return RAMPath(ROOT/str(path))
 return original_local(path)
def raw(r):
 value=read(r)if Path(r['path']).suffix=='.json'else None
 if value is not None:return (json.dumps(value,sort_keys=True)+chr(10)).encode()
 need(ref(r['path'])==r,'Exact source bytes');return(ROOT/r['path']).read_bytes()
def helper_save(name,value):return put(str((helper.P/name).relative_to(ROOT)),value)
helper.ROOT=RAMPath(ROOT);helper.ref=ref;helper.read=read;helper.save=helper_save
c.ref=ref;c.read=read;c.raw=raw;c.local=local
# Only clock/UID/FS/JSON IO is adapted. All actual metadata/science/source/prior,
# native-disposition, review, caps, clock and request/input business predicates run.
now=datetime(2026,10,17,9,0,0,tzinfo=timezone.utc)
class FixtureDatetime(datetime):
 @classmethod
 def now(cls,tz=None):return now
c.datetime=FixtureDatetime;c.time=SimpleNamespace(time=lambda:now.timestamp())
old=read(c.CLOCK_ROOT_REQUEST_V4);oldcfg=read(old['config_ref'])
native=read(old['native_resource_disposition_ref']);oldabsence=read(native['fresh_targeted_native_absence_ref'])
wait=read(c.NATIVE_RESOURCE_WAIT_REF);fixture_uid=wait['last_capacity_sample']['actual_worker_start_identity']['uid']
c.os=SimpleNamespace(getuid=lambda:fixture_uid)
review=dict(schema='H058_GPU0_independent_source_review_v1',independent=True,verdict='SOURCE_PASS',blockers=[],source_only=True,no_execution_authority=True,SYNTHETIC_RAM_ONLY=True,source_ref=ref(PKG+'/root_driver_v1.py'),source_manifest_ref=ref(PKG+'/source_manifest.v1.json'),ROOT_preparation_source_ref=ref(PREP+'/prepare_ROOT_v1.py'),ROOT_preparation_source_manifest_ref=ref(PREP+'/source_manifest.v1.json'))
review_ref=put('RAM_FIXTURE_ONLY/independent_review.json',review)
ns,c2,d2,q,old2,cfg2=helper.context_v1(review_ref)
good('Actual helperv2 context consumes complete review schema and actual coherent v7 modules',lambda:need(c2 is c and d2 is d and ns==PKG.replace('/','.') and old2==old and cfg2==oldcfg,'Actual context and predecessor joins'))
for key,value in [('source_ref',ref(OLD+'/root_driver_v1.py')),('source_manifest_ref',ref(OLD+'/source_manifest.v1.json')),('ROOT_preparation_source_ref',ref('research/reviews/H058_GPU0_v7_ROOT_preparation_source_20261017_v1/prepare_ROOT_v1.py')),('ROOT_preparation_source_manifest_ref',ref('research/reviews/H058_GPU0_v7_ROOT_preparation_source_20261017_v1/source_manifest.v1.json'))]:
 v=copy.deepcopy(review);v[key]=value;rr=put('RAM_FIXTURE_ONLY/wrong_review.json',v)
 reject('Helper context rejects stale '+key,lambda rr=rr:helper.context_v1(rr))
terminal=read(old['controller_terminal_qualification_ref']);registry=read(terminal['registry_snapshot_ref'])
observation=dict(native=native,oldabsence=oldabsence,rows=[dict(expected=e,observed=None,exact_old_identity_live=False)for e in native['native_process_identities']],groups=[dict(pgid=g,absent=True,method='RAM_ONLY_targeted_absence_fixture',checked_utc=now.isoformat())for g in native['native_process_groups']],GPU0=dict(index=0,uuid=d.GPU0_UUID,total_mib=81920,used_mib=0,compute=[]),observed_at_utc=now.isoformat(),registry=registry)
# Complete helper construction. P.mkdir and all generated active-shaped records
# exist only in RAM; observe_v1 and the wrapper's actual capacity phase are never called.
request_ref,request,absence_ref=helper.metadata_v1(review_ref,c,d,q,old,oldcfg,observation)
good('Complete actual helperv2 metadata_v1 reaches actual v7 validate_request',lambda:need(read(request_ref)==request and helper.P.is_dir(),'Full metadata construction'))
returned=c.validate_request_v1(request_ref);req,cfg,prod,protocol,manifest=returned
good('Whole constructed request independently passes actual v7 validator again',lambda:need(req==request and prod==ref(PKG+'/producer_v1.py'),'Full actual request'))
good('Original genuine clocks, prior failure/UNKNOWN, full1920 and v7 repeat reason preserved',lambda:need((req['created_at_utc'],req['expires_at_utc'])==(old['created_at_utc'],old['expires_at_utc']) and read(req['prior_H058_CPU_disposition_ref'])['new_GPU_budget']==1920 and 'Separately reviewed v7'in read(req['prior_H058_CPU_disposition_ref'])['repeat_reason'],'Clock/science/failure lineage'))
good('Generated scientific config/handoff/producer/context and clock acceptance point to v7',lambda:need(cfg['producer_ref']==prod and cfg['source_manifest_ref']==read(req['science_handoff_ref'])['source_manifest_ref'] and req['fit_clock_compatibility_v5']==c.fit_clock_acceptance_v5(),'All exact prospective source joins'))
# Execute only exact original metadata construction/return AST expressions from
# the driver. No reserve/memo/guard/action call is selected or invoked.
def function(path,name):return next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
admit=function(PKG+'/root_driver_v1.py','admit_v1')
def assigned(fn,name):return next(n for n in ast.walk(fn) if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets))
def run_node(node,scope):exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),'<exact sealed metadata expression only>','exec'),scope)
def ram_save(directory,name,value):return put(str((directory/name).relative_to(ROOT)),value)
scope=dict(vars(driver),q=req,cfg=cfg,producer=prod,protocol=protocol,manifest=manifest,request_ref=request_ref,staging=local(req['preparation_directory']),packet=local(req['packet_directory']),output=local(req['output_directory']),ref=ref,read=read,save=ram_save,direct=d)
run_node(assigned(admit,'runtime'),scope)
run_node(assigned(admit,'original'),scope)
run_node(assigned(admit,'closure'),scope)
def needed_nodes(fn):
 result=[]
 for n in fn.body:
  if isinstance(n,(ast.Assign,ast.AugAssign))and any(isinstance(t,ast.Name)and t.id=='needed'for t in (n.targets if isinstance(n,ast.Assign)else[n.target])):result.append(n)
  if isinstance(n,ast.If):result.extend(x for x in n.body if isinstance(x,ast.AugAssign)and isinstance(x.target,ast.Name)and x.target.id=='needed')
 return result
for n in needed_nodes(admit):run_node(n,scope)
# A RAM-only reserved-role fixture preserves all recorded older roles except
# those exact old operational refs replaced by the new construction. No real
# current returned-role union or eligibility is claimed.
old_input=read(c.PRIOR_V5_INPUT_REF);oldq=read(old_input['root_stage_request_ref']);oldcfg5=read(oldq['config_ref']);oldstorage=read(old_input['refs']['storage'])
old_driver=importlib.import_module(OLD.replace('/','.')+'.root_driver_v1');old_direct=importlib.import_module(OLD.replace('/','.')+'.direct_gpu0_v1')
oldscope=dict(vars(old_driver),q=oldq,cfg=oldcfg5,producer=old_input['producer_ref'],request_ref=old_input['root_stage_request_ref'],runtime=old_input['refs']['runtime'],closure=old_input['consumer_manifest_ref'],ref=ref,direct=old_direct)
for n in needed_nodes(function(OLD+'/root_driver_v1.py','admit_v1')):run_node(n,oldscope)
old_operational=oldscope['needed'];remaining=[r for r in oldstorage['reservation']['consumer_bindings']if r not in old_operational]
consumers=c.unique(remaining+scope['needed']);scope.update(consumers=consumers,auxiliary=req['packet_directory']+'/worker_aux')
run_node(assigned(admit,'reserve'),scope)
authority_ref=put('RAM_FIXTURE_ONLY/reserve_authority.json',dict(SYNTHETIC_RAM_ONLY=True))
reservation=dict(scope['reserve'],root_admission=authority_ref)
scope.update(g=SimpleNamespace(MODE=oldstorage['accounting_mode']),receipt=dict(value=reservation,initial_ledger_pins=oldstorage['initial_ledger_pins']),implementation=oldstorage['budget_components']['implementation'],launcher=oldstorage['budget_components']['launcher'])
run_node(assigned(admit,'storage'),scope)
scope['storage_ref']=ram_save(scope['packet'],'storage.json',scope['storage'])
run_node(assigned(admit,'s'),scope)
input_ref=ram_save(scope['packet'],'input.json',scope['s'])
validated=c.validate_input_v1(input_ref)
good('Exact driver input/storage/runtime/closure construction passes full actual v7 validate_input',lambda:need(validated[0]==scope['s']and validated[1]==req,'Full actual input source and metadata joins'))
good('RAM role fixture keeps original64 ceiling without pruning real returned roles',lambda:need(len(consumers)<=64 and all(r in consumers for r in remaining+scope['needed']),'Full fixture union cap'))
# Exact emitted arguments, without their process calls.
ret=admit.body[-1];need(isinstance(ret,ast.Return),'Original driver return')
return_scope=dict(scope,input_ref=input_ref,admitted_ref=put('RAM_FIXTURE_ONLY/admitted.json',dict(SYNTHETIC_RAM_ONLY=True)))
emitted=eval(compile(ast.Expression(ret.value),'<exact admitted argv expression>','eval'),return_scope)
child_fn=function(PKG+'/root_driver_v1.py','execute_v1')
child_scope=dict(vars(driver),input_ref=input_ref,gate_read=901,attachment_ref=put('RAM_FIXTURE_ONLY/attachment.json',dict(SYNTHETIC_RAM_ONLY=True)))
run_node(assigned(child_fn,'command'),child_scope)
child_if=next(n for n in child_fn.body[0].body if False)if False else None
# The conditional command extension lives in execute's try body; select only its
# exact original argument append, never Popen/attachment or any other statement.
append=next(n for n in ast.walk(child_fn)if isinstance(n,ast.If)and ast.unparse(n.test)=='attachment_ref is not None'and any(isinstance(x,ast.AugAssign)and isinstance(x.target,ast.Name)and x.target.id=='command'for x in n.body))
run_node(append,child_scope)
prepare=function(PREP+'/prepare_ROOT_v1.py','prepare_ROOT_v1');pr=prepare.body[-1]
helper_emitted=eval(compile(ast.Expression(pr.value),'<exact preparation argv expression>','eval'),dict(vars(helper),request_ref=request_ref,summary_ref=put('RAM_FIXTURE_ONLY/summary.json',dict(SYNTHETIC_RAM_ONLY=True)),ns=ns))
good('Exact helper admission argv dispatches v7 -m driver',lambda:need(helper_emitted['admit_argv'][helper_emitted['admit_argv'].index('-m')+1]==ns+'.root_driver_v1'and '--admit'in helper_emitted['admit_argv'],'Preparation→admit argv'))
good('Exact admitted execute argv dispatches v7 -m driver',lambda:need(emitted['one_shot_exec_argv'][emitted['one_shot_exec_argv'].index('-m')+1]==ns+'.root_driver_v1'and '--execute'in emitted['one_shot_exec_argv'],'Admit→execute argv'))
good('Exact owned child argv dispatches v7 worker with the same input and attachment',lambda:need(child_scope['command'][child_scope['command'].index('-m')+1]==ns+'.root_driver_v1'and '--worker-input-ref'in child_scope['command']and '--attachment-ref'in child_scope['command'],'Execute→owned child argv'))
def bad_request(change):
 v=copy.deepcopy(req);change(v);return c.validate_request_v1(put(req['preparation_directory']+'/RAM_negative_request.json',v))
reject('Whole request rejects stale v5 manifest',lambda:bad_request(lambda v:v.update(adapter_manifest_ref=ref(OLD+'/source_manifest.v1.json'))),part='Explicit fresh reviewed')
reject('Whole request rejects stale fit-clock source acceptance',lambda:bad_request(lambda v:v['fit_clock_compatibility_v5'].update(compatibility_source_ref=ref(OLD+'/common_v1.py'))),part='Explicit separately reviewed')
reject('Whole request rejects false source review',lambda:bad_request(lambda v:v.update(source_review_passed=False)),part='Explicit fresh reviewed')
reject('Whole request rejects rewritten clocks',lambda:bad_request(lambda v:v.update(created_at_utc='2026-10-17T08:00:00+00:00')),part='Explicit separately reviewed')
for module in(c,d,driver,context,producer):need(module.HERE==PKG,'No package alias drift')
need(not any(k=='torch'or k.startswith(('arggnn','numpy'))for k in sys.modules),'No scientific import')
need(c._SELECTOR_API is None,'No guard selected/imported')
for r in list(REAL_REFS.values()):need(stable_ref(r['path'])==r,'Original source/metadata drift '+r['path'])
result=dict(schema='independent_H058_v7_helper2_complete_RAM_metadata_join_v1',status='PASS_SOURCE_ONLY',source_only=True,active=False,execution_authorized=False,check_count=len(CHECKS),checks=CHECKS,source_ref=ref(PKG+'/root_driver_v1.py'),source_manifest_ref=ref(PKG+'/source_manifest.v1.json'),ROOT_preparation_source_ref=ref(PREP+'/prepare_ROOT_v1.py'),ROOT_preparation_source_manifest_ref=ref(PREP+'/source_manifest.v1.json'),RAM_records=len(RAM),RAM_dirs=len(DIRS),RAM_consumer_binding_count=len(consumers),real_source_metadata_refs=sorted(REAL_REFS.values(),key=lambda r:r['path']),actual_resource_observe_capacity_guard_admit_execute_Popen_or_science_calls=False,business_validator_functions_stubbed=[],IO_boundary_adaptations=['RAM FS creation/existence/read/write/ref for future metadata','bounded source and recorded JSON metadata cache with final pin revalidation','fixed RAM clock and UID matching genuine recorded native metadata'],active_fixture_metadata_persisted=False,emitted_module=ns+'.root_driver_v1',review_fields_consumable=True,original_v6_SOURCE_BLOCKED_preserved=True)
with(P/'helper2_full_join_checks.result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write(chr(10))
print(json.dumps(dict(status=result['status'],checks=len(CHECKS),RAM_records=len(RAM),RAM_consumer_binding_count=len(consumers),emitted_module=result['emitted_module']),sort_keys=True),flush=True)
