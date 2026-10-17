"""Independent narrow v6 source/RAM checks. No process/resource/guard/scientific action."""
from pathlib import Path
from types import SimpleNamespace,FunctionType,CodeType
import ast,copy,hashlib,importlib.util,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');HERE='research/reviews/H058_GPU0_RSS_identity_independent_20261017_v6';P=ROOT/HERE
V6='research/reviews/H058_GPU0_full_training_successor_source_20261017_v6';V5=V6[:-1]+'5'
CHECKS=[];PINS={}
BLOCKERS=[]
def diagnose(name,fn):
 try:fn();CHECKS.append(dict(name=name,status='PASS'))
 except AssertionError as e:CHECKS.append(dict(name=name,status='FAIL_BLOCKER',message=str(e)));BLOCKERS.append(dict(name=name,message=str(e)))
def need(ok,msg):
 if not ok:raise AssertionError(msg)
def good(name,fn):fn();CHECKS.append(dict(name=name,status='PASS'))
def reject(name,fn,kind=ValueError,part=None):
 try:fn()
 except kind as e:
  need(part is None or part in str(e),'Unexpected rejection '+name+': '+str(e));CHECKS.append(dict(name=name,status='PASS_REJECTED',error_type=type(e).__name__));return
 raise AssertionError('Accepted forbidden case '+name)
def ref(path):
 p=ROOT/path;need(p.is_file()and not p.is_symlink()and p.stat().st_size<=32*2**20 and p.suffix in('.py','.json','.sh','.md','.patch','.txt','.diff'),'Source/small JSON only '+path)
 r=dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest());PINS[path]=r;return r
def read(r):
 need(ref(r['path'])==r,'Authentic metadata '+r['path']);need(Path(r['path']).suffix=='.json','JSON only');return json.loads((ROOT/r['path']).read_bytes())
def norm(s):return s.replace(V6,V5).replace('H058_GPU0_v6_','H058_GPU0_v5_').replace('H058-GPU0-v6-','H058-GPU0-v5-')
def code(path,name,scope):
 compiled=compile((ROOT/path).read_text(),str(ROOT/path),'exec',dont_inherit=True,optimize=0)
 matches=[n for n in compiled.co_consts if isinstance(n,CodeType)and n.co_name==name]
 need(len(matches)==1,'Exact source function '+name);return FunctionType(matches[0],scope,name)
def astsame(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
source_manifest=read(ref(V6+'/source_manifest.v1.json'));old_manifest=read(ref(V5+'/source_manifest.v1.json'))
for path,sha in source_manifest['source_sha256'].items():need(ref(path)['sha256']==sha,'Full source map '+path)
need({r['path']:r['sha256']for r in source_manifest['refs']}==source_manifest['source_sha256'],'Aggregate refs/source map exact')
good('Complete v6 aggregate authenticates every source/metadata ref',lambda:need(len(source_manifest['refs'])==339,'All339 refs'))
good('All320 predecessor refs preserved with exact digests',lambda:need(all(source_manifest['source_sha256'].get(p)==h for p,h in old_manifest['source_sha256'].items()),'Frozen v5 source closure'))
for name in('__init__.py','context_factory_v1.py','direct_gpu0_v1.py','observations_gpu_v1.py','producer_v1.py'):
 good(name+' unchanged except explicit fresh namespace',lambda name=name:need(norm((ROOT/V6/name).read_text())==(ROOT/V5/name).read_text(),'Namespace-only file '+name))
old=ast.parse((ROOT/V5/'root_driver_v1.py').read_text());new=ast.parse(norm((ROOT/V6/'root_driver_v1.py').read_text()))
oldrss=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='rss_identity_v1')
new.body=[copy.deepcopy(oldrss)if isinstance(n,ast.FunctionDef)and n.name=='rss_identity_v1'else n for n in new.body]
good('Driver entire AST unchanged beyond exact RSS helper and namespace',lambda:need(astsame(old,new),'Driver narrowness'))
old=ast.parse((ROOT/V5/'common_v1.py').read_text());new=ast.parse(norm((ROOT/V6/'common_v1.py').read_text()))
new.body=[n for n in new.body if not(isinstance(n,ast.FunctionDef)and n.name=='validate_prior_v5_terminal_v6')and not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id.startswith('PRIOR_V5_')for t in n.targets))]
q=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='validate_request_v1')
q.body=[n for n in q.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value)=='validate_prior_v5_terminal_v6(prior)')]
good('Common entire AST unchanged beyond new genuine-v5 prior gate and namespace',lambda:need(astsame(old,new),'Common narrowness'))
# Construct exact source-derived RSS function with RAM-only path/owned/error boundaries.
identity=dict(pid=101,start=777);matching=copy.deepcopy(identity)
owned_path='research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py';same=code(owned_path,'_same',{})
ref(owned_path)
def rss_case(before=matching,after=matching,body='100 8 0',read_error=None,sysconf_error=None,same_error=None):
 stats=[before,after];trace=[]
 def stat(pid):
  trace.append('stat');need(pid==identity['pid'],'Exact owned PID');value=stats.pop(0)
  if isinstance(value,BaseException):raise value
  return value
 def issame(actual,expected):
  if same_error:raise same_error
  return same(actual,expected)
 class RAMPath:
  def __init__(self,path):need(path=='/proc/101/statm','Exact source path construction')
  def read_text(self):
   trace.append('read_statm')
   if read_error:raise read_error
   return body
 def sysconf(name):
  trace.append('sysconf');need(name=='SC_PAGE_SIZE','Exact source page size name')
  if sysconf_error:raise sysconf_error
  return 4096
 def require(v,msg):
  if not v:raise ValueError(msg)
 fn=code(V6+'/root_driver_v1.py','rss_identity_v1',dict(Path=RAMPath,os=SimpleNamespace(sysconf=sysconf),require=require))
 result=fn(SimpleNamespace(_stat=stat,_same=issame),identity);return result,trace
for name,kwargs,expected_trace,result in [
 ('live matching RSS',{},['stat','read_statm','stat','sysconf'],32768),
 ('before disappeared',dict(before=None),['stat'],0),
 ('after disappeared',dict(after=None),['stat','read_statm','stat'],0),
 ('statm ENOENT disappeared',dict(read_error=FileNotFoundError()),['stat','read_statm'],0),
 ('statm ESRCH disappeared',dict(read_error=ProcessLookupError()),['stat','read_statm'],0)]:
 good(name,lambda kwargs=kwargs,expected_trace=expected_trace,result=result:need(rss_case(**kwargs)==(result,expected_trace),'RSS exact trace/result'))
for name,kwargs in [('before positive start mismatch',dict(before=dict(pid=101,start=778))),('before positive PID mismatch',dict(before=dict(pid=102,start=777))),('after positive start mismatch',dict(after=dict(pid=101,start=778))),('after positive PID mismatch',dict(after=dict(pid=102,start=777)))]:
 reject(name,lambda kwargs=kwargs:rss_case(**kwargs),ValueError,'RSS observation identity changed')
for name,kwargs,kind in [
 ('before permission error',dict(before=PermissionError('RAM')),PermissionError),
 ('after permission error',dict(after=PermissionError('RAM')),PermissionError),
 ('statm permission error',dict(read_error=PermissionError('RAM')),PermissionError),
 ('statm other OS error',dict(read_error=OSError('RAM')),OSError),
 ('statm decode error',dict(read_error=UnicodeError('RAM')),UnicodeError),
 ('statm malformed pages',dict(body='100 invalid'),ValueError),
 ('statm missing pages',dict(body='100'),IndexError),
 ('page-size error',dict(sysconf_error=OSError('RAM')),OSError),
 ('owned identity helper error',dict(same_error=KeyError('RAM')),KeyError)]:
 reject(name,lambda kwargs=kwargs:rss_case(**kwargs),kind)
# The actual common module has only stdlib imports/constants/functions; import it
# for real pinned prior metadata checks, never calling any guard or resource API.
sp=importlib.util.spec_from_file_location('_independent_H058_v6_actual_common',ROOT/V6/'common_v1.py');c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
prior=dict(prior_GPU_v5_actual_wait_ref=c.PRIOR_V5_ACTUAL_WAIT_REF,prior_GPU_v5_actual_outer_terminal_ref=c.PRIOR_V5_ACTUAL_OUTER_REF,prior_GPU_v5_files_charges_clocks_sources_and_failure_preserved=True,prior_GPU_v5_future_launch_resume_write_routes_permanently_revoked=True,prior_GPU_v5_actual_optimizer_steps={'same_marker_aligned':None,'same_marker_alignment_null':None},prior_GPU_v5_final_optimizer_totals_status='UNKNOWN_no_new_update_verified',no_CPU_or_GPU_optimizer_state_transfer=True,new_GPU_budget=1920,new_GPU_budget_per_arm=960)
good('Actual v6 prior gate accepts genuine pinned outer241/worker−15/drain/no-completion/UNKNOWN lineage with RAM full-budget disposition',lambda:c.validate_prior_v5_terminal_v6(prior))
for key,value in [('prior_GPU_v5_future_launch_resume_write_routes_permanently_revoked',False),('prior_GPU_v5_actual_optimizer_steps',{'same_marker_aligned':0,'same_marker_alignment_null':0}),('prior_GPU_v5_final_optimizer_totals_status','zero_actual'),('no_CPU_or_GPU_optimizer_state_transfer',False),('new_GPU_budget',1919),('new_GPU_budget_per_arm',959)]:
 p=copy.deepcopy(prior);p[key]=value
 reject('Prior gate rejects '+key,lambda p=p:c.validate_prior_v5_terminal_v6(p),ValueError,'Separate ROOT full-budget disposition')
for name,r in [('wait',c.PRIOR_V5_ACTUAL_WAIT_REF),('minimal',c.PRIOR_V5_MINIMAL_WAIT_REF),('outer',c.PRIOR_V5_ACTUAL_OUTER_REF),('input',c.PRIOR_V5_INPUT_REF)]:read(r)
ref(c.PRIOR_V5_LOG_REF['path'])
# Authenticity is independently checked above; altered body fixtures below are
# explicit IO-boundary negatives and are never written under genuine ROOT refs.
original_read=c.read
for label,target,key,value in [('wrong outer',c.PRIOR_V5_ACTUAL_OUTER_REF,'actual_outer_exit_code',0),('unconsumed outer',c.PRIOR_V5_ACTUAL_OUTER_REF,'actual_terminal_consumed',False),('invented zero totals',c.PRIOR_V5_ACTUAL_OUTER_REF,'actual_final_optimizer_totals','0'),('missing drain',c.PRIOR_V5_ACTUAL_WAIT_REF,'all_owned_processes_and_groups_drained',False),('false success',c.PRIOR_V5_ACTUAL_WAIT_REF,'returncode',0)]:
 def changed(record,target=target,key=key,value=value):
  v=original_read(record)
  if record==target:v[key]=value
  return v
 c.read=changed
 reject('Actual prior gate rejects '+label,lambda:c.validate_prior_v5_terminal_v6(prior))
 c.read=original_read
# Source/metadata joins required before either ROOT execute or child construction.
driver=(ROOT/V6/'root_driver_v1.py').read_text();common=(ROOT/V6/'common_v1.py').read_text()
good('ROOT emitted argv uses current -m package entry',lambda:need('HERE.replace("/","." )+".root_driver_v1"'in driver and '"--execute","--input-ref"'in driver,'Root argv namespace'))
diagnose('Child exact module/env/owned gate paths use current package',lambda:need(V6.replace('/','.')+'.root_driver_v1'in driver and 'PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")])'in driver and '"--worker-input-ref"'in driver and 'No science before owned parent launch receipt'in driver,'Child construction joins'))
# Actual inert module import proves the operational constant binding; no main,
# execute/admit/worker/guard/resource method is called.
sys.path.insert(0,str(ROOT))
sp=importlib.util.spec_from_file_location('_independent_actual_v6_driver_inert',ROOT/V6/'root_driver_v1.py')
entry=importlib.util.module_from_spec(sp);sp.loader.exec_module(entry)
diagnose('Actual v6 driver inert import binds v6 HERE/DRIVER/MANIFEST',lambda:need(entry.HERE==V6 and entry.DRIVER==V6+'/root_driver_v1.py' and entry.MANIFEST==V6+'/source_manifest.v1.json','Actual imported constants are '+str((entry.HERE,entry.DRIVER,entry.MANIFEST))))
for name in('context_factory_v1.py','direct_gpu0_v1.py'):
 diagnose(name+' operational common import names v6',lambda name=name:need('from '+V6.replace('/','.')+'.common_v1 import *'in(ROOT/V6/name).read_text(),'Stale v5 common import in '+name))
good('Producer/config/handoff/source map checks remain exact',lambda:need('seal.get("producer_ref")==producer'in common and 'seal.get("source_manifest_ref")==cfg["source_manifest_ref"]'in common and 'producer["path"].startswith(HERE+"/")'in common,'Science metadata join'))
good('Review consumption requires actual exact source/manifest SOURCE_PASS',lambda:need('review.get("independent") is True'in common and 'review.get("verdict")=="SOURCE_PASS"'in common and 'review.get("source_ref")==ref(DRIVER)'in common and 'review.get("source_manifest_ref")==ref(MANIFEST)'in common,'Consumable exact review joins'))
good('No performance source draft integrated',lambda:need('immutable_parse_reuse'not in driver+common and 'guard_latency_minimal_reuse'not in driver+common,'No perf integration'))
for r in list(PINS.values()):need(ref(r['path'])==r,'Original source/metadata changed')
need(not any(k=='torch'or k.startswith(('arggnn','numpy'))for k in sys.modules),'No science import')
result=dict(schema='independent_H058_v6_narrow_core_source_RAM_checks_v1',status='SOURCE_BLOCKED' if BLOCKERS else 'PASS_SOURCE_ONLY',source_only=True,execution_authorized=False,blockers=BLOCKERS,actual_inert_driver_namespace=dict(HERE=entry.HERE,DRIVER=entry.DRIVER,MANIFEST=entry.MANIFEST),check_count=len(CHECKS),checks=CHECKS,source_ref=ref(V6+'/root_driver_v1.py'),source_manifest_ref=ref(V6+'/source_manifest.v1.json'),original_source_and_JSON_pins=sorted(PINS.values(),key=lambda r:r['path']),source_manifest_refs=339,whole_ROOT_preparer_review_pending=True,actual_process_proc_resource_guard_or_science_calls=False,active_fixture_records_persisted=False)
with(P/'narrow_core_checks.result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write(chr(10))
print(json.dumps(dict(status=result['status'],checks=len(CHECKS),source_manifest_refs=339,preparer_review_pending=True),sort_keys=True),flush=True)
