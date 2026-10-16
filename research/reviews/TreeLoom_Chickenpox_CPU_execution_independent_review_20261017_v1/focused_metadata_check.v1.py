"""Independent source + focused in-memory metadata checks; never real execution."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,contextlib,copy,hashlib,importlib,io,json,os,re,stat,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/TreeLoom_Chickenpox_CPU_execution_source_20261017_v1'
REVIEW='research/reviews/TreeLoom_Chickenpox_CPU_execution_independent_review_20261017_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
MANIFEST_SHA='3fae92998113d6b5b534aa0e8966920297f214701642056c5a87fd9ed096f7ee'
HANDOFF_SHA='1162b6228a80925c81b690837337dce1642a206dadaf8d167d2f24aa186f20e6'
def need(ok,msg='fixture assertion'):
 if not ok:raise ValueError(msg)
def pin(path):
 return dict(path=str(path),sha256=hashlib.sha256(str(path).encode()).hexdigest())
def extract(path,name,namespace):
 node=next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
 return namespace[name]
def rejected(call):
 try:call()
 except (ValueError,TypeError):return True
 raise AssertionError('expected fixture rejection')
def main():
 need(sys.dont_write_bytecode and sys.flags.no_user_site)
 for name,expected in [('source_manifest.v1.json',MANIFEST_SHA),('source_handoff.v1.json',HANDOFF_SHA)]:
  need(hashlib.sha256((ROOT/HERE/name).read_bytes()).hexdigest()==expected,'candidate seal')
 m=json.loads((ROOT/HERE/'source_manifest.v1.json').read_text())
 for path,sha in m['source_sha256'].items():
  need(path!='data/raw/chickenpox/hungarian-chickenpox-cases.zip','no payload in pre-gate source closure')
  need(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,'source hash '+path)
 d,w,r=[importlib.import_module(HERE.replace('/','.')+'.'+n)for n in ('root_driver_v1','worker_entry_v1','scientific_runner_v1')]
 c=d.c;b=c.binding();cfg=c.read(c.ref(HERE+'/scientific_config.INACTIVE.template.v1.json'));cfg['execution_enabled']=True
 need(w.c is c and r.c is c and c._h._GUARD is None)
 c.validate_config(cfg);c.context(b['data_source_manifest_ref']);c.scientific_review(b['scientific_review_ref'])
 reg=c.read(b['registration_ref'])
 c.registration(dict(registration_ref=b['registration_ref'],registration_decisions_ref=b['registration_decisions_ref'],created_at_utc=reg['registered_utc']))
 for module in (d,w):
  with contextlib.redirect_stdout(io.StringIO()):need(module.main([])==0)
 # Exact old union helper and v4 schema, extracted without importing their modules.
 ns=dict(require=need,_raw=lambda x:need(set(x)=={'path','sha256'} and re.fullmatch('[a-f0-9]{64}',x['sha256']) is not None),copy=copy)
 union=extract(c.GUARD_REF['path'],'consumer_bindings_v1',ns)
 roles=[pin('synthetic/role%d.py'%i)for i in range(65)]
 need(len(union(roles[:64],roles[:1]))==64);rejected(lambda:union(roles))
 def overlap(a,b):return a==b or a in b.parents or b in a.parents
 scope_ns=dict(Path=Path,integer=lambda n,*args:need(type(n)is int and n>=1),canonical=lambda root,x:Path(x),
  paths_overlap=overlap,accounting_paths=lambda root:[root/'synthetic_accounting'],bound=lambda root,x,*a:ns['_raw'](x))
 scope=extract(c.V4['path'],'scope_spec',scope_ns)
 # Actual admit code, with every I/O/guard/mutex operation replaced by an in-memory seam.
 def admission_fixture(fail_reserve=False,invalid_role=False):
  events=[];written={};made=set();reserved=[False]
  prep=ROOT/REVIEW/'SYNTHETIC_PREPARATION';packet=ROOT/'research/reviews/TreeLoom_Chickenpox_CPU_ROOT_actual_SYNTHETIC';out=ROOT/'experiments/results/TreeLoom_Chickenpox_CPU_SYNTHETIC'
  class P:
   def __init__(self,p):self.p=Path(p)
   @property
   def name(self):return self.p.name
   def __truediv__(self,n):return P(self.p/n)
   def __str__(self):return str(self.p)
   def exists(self):return self.p in made
   def mkdir(self):need(reserved[0],'charged packet before reserve');made.add(self.p);events.append('packet_mkdir')
  def local(x):return P(x if Path(x).is_absolute()else ROOT/x)
  def put(p,v):
   path=p.p;need(path not in written)
   if packet==path or packet in path.parents:need(reserved[0],'charged metadata before reserve')
   else:need(prep in path.parents,'staging outside preparation')
   written[path]=copy.deepcopy(v);events.append('put:'+path.name);return pin(str(path.relative_to(ROOT)))
  q=dict(packet_directory=str(packet.relative_to(ROOT)),output_directory=str(out.relative_to(ROOT)),preparation_directory=str(prep.relative_to(ROOT)),cpu_ids=[223],
   source_review_ref=pin('synthetic/review.json'),scientific_review_ref=b['scientific_review_ref'],runtime_ref=pin('synthetic/runtime.json'),runtime_qualification_ref=pin('synthetic/qualification.json'),
   root_directive_ref=pin('synthetic/directive.json'),registration_ref=b['registration_ref'],registration_decisions_ref=b['registration_decisions_ref'],data_context_ref=b['data_source_manifest_ref'],
   storage_policy_selector_ref=pin('synthetic/selector.json'),scientific_config=cfg,created_at_utc='2026-10-16T00:00:00+00:00',expires_at_utc='2026-10-18T00:00:00+00:00')
  returned={str(i):pin('synthetic/current_guard%d.py'%i)for i in range(30)}
  if invalid_role:returned['bad']=pin('synthetic/forbidden.md')
  state=({},None,{},returned)
  def schema(root,value):
   events.append('original_v4_scope_spec');paths=scope(root,value)
   return [local(x)for x in paths]
  def draft(*a,**kw):events.append('authority_draft');return {}
  def apply(*a,**kw):
   events.append('apply_reserve');need(not made and all(prep in p.parents for p in written),'all charged scopes absent at reserve')
   if fail_reserve:raise ValueError('synthetic failed reserve')
   reserved[0]=True
   reserve=copy.deepcopy(last_reserve[0]);reserve['root_admission']=pin('synthetic/reserve.json')
   return dict(value=reserve,initial_ledger_pins={})
  last_reserve=[]
  def wrapped_scope(root,value):last_reserve.append(value);return schema(root,value)
  g=NS(budget=NS(SOURCE_REL='scripts/project_budget_v9.py',LAUNCHER_REL='scripts/launch_reserved_gpu_v6.py',v4=NS(mutex=lambda root:contextlib.nullcontext(),scope_spec=wrapped_scope)),
   _state_locked=lambda *a,**kw:state,consumer_bindings_v1=union,_policy=lambda:dict(never_credit_keys=[]),_pins=lambda v:{},operation_draft_v1=draft,apply_reserve_v1=apply,MODE='synthetic')
  cc=NS(**{name:getattr(c,name)for name in ('CAPS','PARENT','PHASE','POLICY_REF','GUARD_REF','ENTRY','make_config')})
  cc.SELECTOR_REF=q['storage_policy_selector_ref'];cc.validate_request=lambda rr:(q,c.CAPS,{})
  cc.sources=lambda:m;cc.direct_roles=c.direct_roles;cc.input_spec=lambda rr:need(rr['path'].endswith('/input.json'))
  gl=dict(c=cc,ROOT=ROOT,SOURCE=c.DRIVER,HERE=HERE,need=need,local=local,ref=lambda p:pin(str(p)),put=put,guard=lambda x:g,
   os=NS(sched_getaffinity=lambda x:{223}),json=json)
  admit=extract(c.DRIVER,'admit',gl)
  if fail_reserve or invalid_role:
   rejected(lambda:admit(pin('synthetic/request.json')));need(not made and not any(packet in p.parents for p in written))
   need('packet_mkdir'not in events)
   if invalid_role:need('authority_draft'not in events and 'apply_reserve'not in events)
  else:
   result=admit(pin('synthetic/request.json'));need(events.index('original_v4_scope_spec')<events.index('authority_draft')<events.index('apply_reserve')<events.index('packet_mkdir'))
   need(all(v in last_reserve[0]['consumer_bindings']for v in returned.values()));need(result['execution_argv'][0]=='setsid')
  return dict(events=events,charged_paths_created=bool(made),guard_roles_simulated=len(returned),real_operations=False)
 admission=[admission_fixture(),admission_fixture(fail_reserve=True),admission_fixture(invalid_role=True)]
 # Actual writer save method using only in-memory filesystem and Torch-save surrogate.
 files={};fds={};written_names=[]
 class F(io.BytesIO):
  def __init__(self,name):super().__init__();self.name=name
  def fileno(self):return 42
  def __exit__(self,*args):files[self.name]=self.getvalue();return super().__exit__(*args)
 class OP:
  def __init__(self,name=''):self.name=name
  def __truediv__(self,name):return OP(name)
  def lstat(self):return NS(st_mode=stat.S_IFREG,st_nlink=1,st_dev=1,st_ino=2)
  def relative_to(self,root):return Path('SYNTHETIC_OUTPUT')/self.name
 def opened(path,*args):need(path.name not in files);written_names.append(path.name);fds[42]=path.name;return 42
 fake_os=NS(O_WRONLY=os.O_WRONLY,O_CREAT=os.O_CREAT,O_EXCL=os.O_EXCL,O_NOFOLLOW=os.O_NOFOLLOW,
  statvfs=lambda p:NS(f_bavail=10**9,f_frsize=4096),open=opened,fdopen=lambda fd,mode:F(fds[fd]),fsync=lambda fd:None,
  fstat=lambda fd:NS(st_dev=1,st_ino=2))
 writer=object.__new__(w.BoundedWriter);writer.fast=lambda *a:None;writer.s={'storage':{}};writer.g=NS(budget_check_v9=lambda *a,**k:None)
 writer.torch=NS(save=lambda value,buf:buf.write(b'synthetic state serialization, no model/tensor'))
 writer.jsonable=lambda value:value;writer.path=OP();writer.used=8192;writer.json_used=0;writer.torch_files=0;writer.names=set()
 old_os=w.os;w.os=fake_os
 old_raw,old_local,old_importlib=c.raw,c.local,r.importlib
 archive=NS(stat=lambda:NS(st_size=14765))
 c.local=lambda path:archive if path==b['archive_ref']['path']else old_local(path)
 c.raw=lambda record:b'SYNTHETIC_ARCHIVE_NOT_OPENED'if record==b['archive_ref']else old_raw(record)
 development=NS(phase='development',counties=b['counties'],counts=NS(shape=(522,20)),mean=NS(tolist=lambda:[0.0]*20),scale=NS(tolist=lambda:[1.0]*20))
 interface_node=next(n for n in ast.parse((ROOT/b['generic_core_ref']['path']).read_text()).body if isinstance(n,ast.ClassDef)and n.name=='TreeLoomGenericForecastV1')
 interface_fn=next(n for n in interface_node.body if isinstance(n,ast.FunctionDef)and n.name=='interface_contract')
 contract_ns={};exec(compile(ast.Module(body=[interface_fn],type_ignores=[]),'synthetic_interface','exec'),contract_ns)
 simulated=[0,0]
 def fit(prepared,*,seed,boundary,checkpoint_sink):
  result={}
  for epoch in range(1,41):
   for arm in c.ARMS:
    for k in range(1,21):
     update=(epoch-1)*20+k;targets=list(range(52,365))[(k-1)*16:k*16]
     boundary(dict(event='train_batch',seed=seed,arm=arm,epoch=epoch,update=update,targets=targets));simulated[0]+=1
    for k in range(5):boundary(dict(event='predict_batch',targets=list(range(365,443))[k*16:(k+1)*16],shuffled=False));simulated[1]+=1
  for arm in c.ARMS:
   hist=[dict(epoch=e,updates=e*20,forecast_weeks=78,mae_cases=float(e),rmse_cases=float(e),per_county_mae=[float(e)]*20,per_week_mae=[float(e)]*78,prediction_clipping='none')for e in range(1,41)]
   record=dict(seed=seed,arm=arm,selected_epoch=1,selected_validation_mae_cases=1.,epochs_completed=40,optimizer_updates=800,training_targets=list(range(52,365)),validation_targets=list(range(365,443)),
    parameters={'full':25378,'no_memory':24865}[arm],test_accessed=False,history=hist,family_contract=contract_ns['interface_contract'](NS(memory=object()if arm=='full'else None)))
   boundary(dict(event='save_selected_checkpoint',seed=seed,arm=arm))
   cp=checkpoint_sink(seed,arm,{'synthetic':NS(device=NS(type='cpu'))},record);result[arm]=dict(selected_checkpoint_ref=cp)
  return result
 r.importlib=NS(import_module=lambda name:NS(load_development_v1=lambda archive:development,fit_development_pair_v1=fit))
 try:
  writer.save('started.json',dict(input_ref=pin('synthetic/input.json'),config=cfg,seeds=list(c.SEEDS),arms=list(c.ARMS)))
  result=r.run_train_v1(cfg,writer=writer,runtime_guard=lambda *a:None)
  writer.save('completed.json',dict(result_ref=result,status=c.STATUSES[c.PHASE],parent_id=c.PARENT))
  decoded={name:json.loads(data)for name,data in files.items()if name.endswith('.json')}
  need(simulated==[4800,1200]and writer.torch_files==6 and len(decoded)==10,'full metadata bridge route')
  need(len(decoded['completed_TRAIN.json']['fits'])==6 and decoded['completed_TRAIN.json']['total_optimizer_updates']==4800)
  # Exact actual CappedBuffer and JSON rejection are pre-write failures.
  buf=w.CappedBuffer(8);buf.write(b'12345678');rejected(lambda:buf.write(b'9'))
  before=len(written_names);rejected(lambda:writer.save('worker_failure.json',{'opaque':object()}));need(before==len(written_names))
  rejected(lambda:writer.save('worker_failure.json',{'message':'x'*(c.CAPS['total_JSON_bytes']+1)}));need(before==len(written_names))
  rejected(lambda:writer.save('selected_seed101_full.pt',{},kind='torch'));need(before==len(written_names))
 finally:w.os=old_os;c.raw=old_raw;c.local=old_local;r.importlib=old_importlib
 need(not any(x in sys.modules for x in ('torch','numpy','torch_geometric','scripts.project_budget_v9','arggnn.project_budget_v4')))
 need(c._h._GUARD is None and c.SELECTOR_REF is None)
 return dict(schema='H061_independent_focused_metadata_check_v1',status='PASS_SOURCE_AND_SYNTHETIC_METADATA_ONLY',candidate_source_count=len(m['source_sha256']),
  admission_fixtures=admission,original_union_64_accepted_65_rejected=True,original_v4_invalid_direct_metadata_rejected_before_authority=True,
  actual_writer_JSON_identity_serializer_native_metadata_passed=True,actual_bridge_simulated_callbacks=simulated,
  six_synthetic_state_products=True,real_optimizer_updates=0,Torch_Numpy_model_imports=False,real_archive_or_checkpoint_reads=False,
  real_guards_mutex_admission_process_probes_resources_jobs_install_Git=False,actual_JSON_bytes=writer.json_used,synthetic_total_output_bytes=sum(len(v)for v in files.values()),
  cap_duplicate_and_nonJSON_failures_precede_file_write=True,limits='All filesystem/guard/authority/process seams in exercised operational methods were in-memory substitutes; no real execution, capacity, lifecycle wait, numerical behavior or fit is established.')
if __name__=='__main__':
 result=main();out=ROOT/REVIEW/'focused_metadata_check.result.v1.json'
 out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result,sort_keys=True,indent=2))
