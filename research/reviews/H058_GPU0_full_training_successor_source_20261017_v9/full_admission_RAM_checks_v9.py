"""Actual v9 request/admit/memo-v2/reserve/input path, source-authenticated AST and RAM IO only."""
from pathlib import Path,PurePosixPath
from types import ModuleType,SimpleNamespace
from contextlib import contextmanager
from datetime import datetime,timezone,timedelta
import ast,collections,contextlib,copy,hashlib,importlib,io,json,os,stat,sys,time
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D='research/reviews/H058_GPU0_full_training_successor_source_20261017_v9';sys.path.insert(0,str(ROOT))
c=importlib.import_module(D.replace('/','.')+'.common_v1');op=importlib.import_module(D.replace('/','.')+'.operational_v1');driver=op.driver_role_v1()
checks=[]
def ck(n,v=True):
 if not v:raise AssertionError(n)
 checks.append(dict(name=n,status='PASS'))
def need(v,n):
 if not v:raise ValueError(n)
def realref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def readreal(p):return json.loads((ROOT/p).read_bytes())
def emit_source_module(path,names,scope):
 text=(ROOT/path).read_text();tree=ast.parse(text);mod=ModuleType('_H058_RAM_'+Path(path).stem);mod.__file__=str(ROOT/path);mod.__dict__.update(scope)
 nodes=[n for n in tree.body if (isinstance(n,ast.ImportFrom)and n.module=='__future__')or(isinstance(n,ast.FunctionDef)and n.name in names)]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/path),'exec',dont_inherit=True,optimize=0),mod.__dict__);return mod
memory={};dirs={str(ROOT)};fds={};seq=[0];locked=[False];lock_events=[];reads=collections.Counter();writes=[];append_count=[0]
allowed={r['path']:r['sha256']for r in readreal(D+'/source_manifest.v1.json')['refs']};allowed[D+'/source_manifest.v1.json']=realref(D+'/source_manifest.v1.json')['sha256']
def relative(path):
 p=PurePosixPath(path);return str(p.relative_to(str(ROOT)))if p.is_absolute()else str(p)
def rawpath(path):
 key=relative(path);reads[key]+=1
 if key in memory:return memory[key]
 need(key in allowed,'Only explicit sealed source/metadata or RAM record may be read: '+key)
 raw=(ROOT/key).read_bytes();need(hashlib.sha256(raw).hexdigest()==allowed[key],'Exact immutable allowed fixture source');return raw
def exists(path):
 key=relative(path);return key in memory or str(path)in dirs or key in allowed
class MemoryPath(PurePosixPath):
 def resolve(self):return self
 def exists(self):return exists(self)
 def is_file(self):return self.exists()and str(self)not in dirs
 def is_dir(self):return str(self)in dirs
 def is_symlink(self):return False
 def mkdir(self,parents=False,exist_ok=False,mode=0o777):
  need(exist_ok or not self.exists(),'RAM exclusive mkdir');dirs.add(str(self))
 def lstat(self):
  if str(self)in dirs:data=b'';mode=stat.S_IFDIR|0o700
  else:data=rawpath(self);mode=stat.S_IFREG|0o600
  return SimpleNamespace(st_dev=1,st_ino=int(hashlib.sha256(str(self).encode()).hexdigest()[:12],16),st_mode=mode,st_nlink=1,st_size=len(data),st_mtime_ns=0,st_ctime_ns=0)
 def iterdir(self):
  candidates=set(dirs)|{str(MemoryPath(str(ROOT))/k)for k in memory}
  return iter(sorted((MemoryPath(x)for x in candidates if MemoryPath(x).parent==self),key=str))
 def read_bytes(self):return rawpath(self)
ramroot=MemoryPath(str(ROOT));LOCK='.runtime/scheduler.lock';memory[LOCK]=b'RAM lock'
class RAMFile(io.BytesIO):
 def __init__(self,fd,mode):
  self.fd,self.mode=fd,mode;path=fds[fd];super().__init__(b''if 'w'in mode else rawpath(path))
 def fileno(self):return self.fd
 def close(self):
  if self.closed:return
  path=fds.get(self.fd)
  if 'w'in self.mode:memory[relative(path)]=self.getvalue()
  if path==ramroot/LOCK and locked[0]:
   need(all(getattr(v4,k)is v for k,v in primitive_originals.items())and budget.successor_paths is successor_original,'Every memo patch restored before unlock')
   locked[0]=False;lock_events.append('unlock')
  fds.pop(self.fd,None);super().close()
def ropen(path,flags,mode=0o600):
 path=MemoryPath(path);key=relative(path)
 if flags&os.O_CREAT:
  need(not(flags&os.O_EXCL)or key not in memory,'RAM O_EXCL');memory.setdefault(key,b'')
 seq[0]+=1;fds[seq[0]]=path;return seq[0]
def rclose(fd):fds.pop(fd,None)
def flock(stream,flags):
 need(not locked[0],'No nested original mutex');locked[0]=True;lock_events.append('lock')
ros=SimpleNamespace(**{n:getattr(os,n)for n in ('O_RDONLY','O_RDWR','O_WRONLY','O_NOFOLLOW','O_CREAT','O_EXCL','O_DIRECTORY')},open=ropen,fdopen=lambda fd,mode:RAMFile(fd,mode),fstat=lambda fd:fds[fd].lstat(),fsync=lambda fd:None,close=rclose)
v4path=c.V4['path'];v4=emit_source_module(v4path,('canonical','file_bytes','bound','reference','scope_spec','read_json','mutex','parse','encoded','integer','paths_overlap','accounting_paths'),dict(Path=MemoryPath,os=ros,stat=stat,hashlib=hashlib,json=json,contextmanager=contextmanager,contextlib=contextlib,fcntl=SimpleNamespace(flock=flock,LOCK_EX=2),MAX_JSON=2**20,LEGACY='RAM_ONLY/legacy.json',SUPPLEMENTAL='RAM_ONLY/supplemental.json',LOCK=LOCK,SOURCE=v4path,CONFIG='configs/project_budget_v4.json',FIXED_POLICY={}))
budgetpath='scripts/project_budget_v9.py';budget=emit_source_module(budgetpath,('successor_paths','digest'),dict(v4=v4,hashlib=hashlib,JOURNAL_REL='.runtime/RAM_successor',SOURCE_REL=budgetpath,LAUNCHER_REL='scripts/launch_reserved_gpu_v6.py'))
primitive_originals={k:getattr(v4,k)for k in ('bound','reference','scope_spec','canonical','file_bytes','mutex')};successor_original=budget.successor_paths
memory['RAM_ONLY/legacy.json']=b'{"jobs":{}}'
binding=readreal('research/reviews/H059_outer000_Relate64_execution_source_20261017_v1/source_bindings.v1.json');guards=copy.deepcopy(readreal(binding['guard_role_snapshot_ref']['path'])['bindings']);guards['Census_observation_explicit_selector_capsule']=c.SELECTOR_REF
allowed.update({r['path']:r['sha256']for r in guards.values()})
ck('all47_original_current_guard_roles_preserved',len(guards)==47)
api_source=(ROOT/c.GUARD['path']).read_text();at=ast.parse(api_source)
constants={}
for n in at.body:
 if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
  try:constants[n.targets[0].id]=ast.literal_eval(n.value)
  except Exception:pass
names=('_state_locked','apply_reserve_v1','_append','_paths','_pins','_events','_authority','operation_draft_v1','_json','_raw','_ref','_expire','consumer_bindings_v1','charge','warning_fields','check_prefix','assert_open','require')
g=emit_source_module(c.GUARD['path'],names,dict(constants,ROOT=ramroot,budget=budget,Path=MemoryPath,os=ros,copy=copy,json=json,hashlib=hashlib,datetime=datetime,timezone=timezone))
g._POLICY_REF=c.POLICY;g._SELECTOR_REF=c.SELECTOR_REF
basepins=dict(frozen_fixture_parent=True)
baseview=dict(accounting_mode=g.BASE_MODE,projected_bytes=1,limit_bytes=g.CEILING_BYTES,hard_limit_bytes=g.HUMAN_LIMIT_BYTES,new_reserved_bytes=0,completed_credit_bytes=0)
def parentstate(*args,**kwargs):need(locked[0]and kwargs==dict(fresh=True),'Fresh state under original owned mutex');return {},set(),dict(baseview),dict(guards),[],[],{}
g._BASE=SimpleNamespace(_state_locked=parentstate,_pins=lambda v:basepins,_paths=lambda own:budget.successor_paths(ramroot,own))
g._policy=lambda:dict(implementation_ref=realref(budget.SOURCE_REL),launcher_ref=realref(budget.LAUNCHER_REL),base_pins=basepins,base_projected_bytes=1,never_credit_keys=[])
g._bindings=lambda inherited,policy:copy.deepcopy(guards)
# Source-authenticated original _state/_reserve/_append and v4 six primitives remain actual AST callables.
# Only lower policy/ancestor/accounting IO is handwritten RAM; no current guard entrypoint is imported.
now=datetime.now(timezone.utc);created=now.isoformat();expires=(now+timedelta(hours=27)).isoformat()
def putram(path,value):
 raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(path not in memory,'Exclusive RAM metadata');memory[path]=raw;return dict(path=path,sha256=hashlib.sha256(raw).hexdigest())
def refr(path):
 if path in memory:return dict(path=path,sha256=hashlib.sha256(memory[path]).hexdigest())
 return c.ref(path)
def rawr(r):
 raw=memory[r['path']]if r['path']in memory else c.raw(r);need(hashlib.sha256(raw).hexdigest()==r['sha256'],'Exact RAM bound ref');return raw
def readr(r):return json.loads(rawr(r))
q=readreal(D+'/ROOT_request.INACTIVE.template.v1.json');q.update(instance_id='ram_only',job_id='H058-GPU0-v9-fulltrain-ram_only',preparation_directory='research/reviews/H058_GPU0_v9_ROOT_preparation_ram_only',packet_directory='research/reviews/H058_GPU0_v9_ROOT_actual_ram_only',output_directory='experiments/results/H058_GPU0_v9_full_parent_ram_only',status='active',execution_authorized=True,source_review_passed=True,created_at_utc=created,expires_at_utc=expires,historical_ROOT_clocks_v8=c.historical_clocks_v8(),fit_clock_compatibility_v5=c.fit_clock_acceptance_v5())
dirs.add(str(ramroot/q['preparation_directory']))
cfg=readreal(D+'/scientific_config.INACTIVE.template.v1.json');cfg.update(execution_enabled=True,root_resource_caps_reviewed=True,producer_ref=realref(c.DRIVER),source_manifest_ref=realref(D+'/producer_manifest.v1.json'))
layout=readreal(D+'/layout_resource_proof.INACTIVE.template.v1.json');layout.update(root_authorizer='root_agent',status='admitted_first_scheduled_full_parent_step',source_count_ref=cfg['source_count_ref'],physical_gpu=0,reduced_data=False,counted_in_960_updates=True);cfg['layout_resource_proof_ref']=putram('RAM_ONLY/layout.json',layout)
directive=readreal(D+'/full_training_directive.INACTIVE.template.v1.json');directive.update(root_authorizer='root_agent',status='active',created_at_utc=created,expires_at_utc=expires,full30pass_training_authorized=True,extra_probe_updates_authorized=False,GPU_entry_ref=realref(c.DRIVER),new_finite_v8_interval_authorized=True,historical_ROOT_clocks_v8=c.historical_clocks_v8(),historical_files_scopes_charges_clocks_failures_and_unknowns_preserved=True);cfg['full_training_directive_ref']=putram('RAM_ONLY/directive.json',directive)
q['config_ref']=putram('RAM_ONLY/config.json',cfg);q['source_review_ref']=putram('RAM_ONLY/review.json',dict(independent=True,verdict='SOURCE_PASS',blockers=[],source_ref=realref(c.DRIVER),source_manifest_ref=realref(c.MANIFEST)))
prior=readreal(D+'/prior_CPU_disposition.INACTIVE.template.v1.json')
for key in ('actual_terminal_consumed','all_owned_processes_and_groups_drained','old_launch_resume_write_routes_revoked','old_files_charges_clocks_and_counters_preserved','new_full_GPU_budget_explicitly_authorized','prior_GPU_v4_files_charges_clocks_and_failure_preserved','prior_GPU_v5_files_charges_clocks_sources_and_failure_preserved','prior_GPU_v5_future_launch_resume_write_routes_permanently_revoked','no_CPU_or_GPU_optimizer_state_transfer','prior_GPU_v7_files_scopes_full_charges_clocks_sources_failures_and_unknowns_preserved','prior_GPU_v7_future_launch_resume_write_routes_permanently_revoked','actual_prior_clocks_match_immutable_source_lineage','new_full_1920_update_unfinished_cell_registration'):prior[key]=True
prior.update(root_authorizer='root_agent',new_gpu_request_job_id=q['job_id'],historical_ROOT_clocks_v8=c.historical_clocks_v8(),actual_wait_ref=putram('RAM_ONLY/CPU_wait.json',dict(RAM_NOT_AUTHORITY=True)),actual_outer_terminal_ref=putram('RAM_ONLY/CPU_outer.json',dict(RAM_NOT_AUTHORITY=True)))
absence=readreal(D+'/v7_targeted_absence.INACTIVE.template.v1.json')
for key in ('actual_supervisor_terminal_consumed','current_known_v7_identities_groups_and_aux_writers_absent','historical_roster_was_truncated','remaining_historical_ownership_uncertainty_ROOT_accepted_for_fresh_scope','all_v7_launch_resume_write_routes_permanently_revoked'):absence[key]=True
absence.update(observed_at_utc=created,actual_outer_terminal_ref=c.PRIOR_V7_ACTUAL_OUTER_REF,actual_wait_ref=c.PRIOR_V7_ACTUAL_WAIT_REF,historical_complete_roster_claimed=False);prior['fresh_targeted_v7_absence_ref']=putram('RAM_ONLY/v7_absence.json',absence);q['prior_H058_CPU_disposition_ref']=putram('RAM_ONLY/prior.json',prior)
native=readreal(D+'/native_resource_disposition.INACTIVE.template.v1.json');native.update(status='resources_released',H058_job_id=q['job_id'],created_at_utc=created,expires_at_utc=expires,observed_at_utc=created,all_old_native_launch_resume_write_routes_permanently_revoked=True,no_native_GPU0_attachment_alive=True)
na=readreal(D+'/native_targeted_absence.INACTIVE.template.v1.json')
for key in ('native_known_roots_and_contexts_absent','no_native_GPU0_attachment_alive','all_exact_native_identities_and_groups_absent','all_old_native_launch_resume_write_routes_permanently_revoked','historical_unknowns_preserved'):na[key]=True
na.update(root_authorizer='root_agent',native_actual_wait_ref=c.NATIVE_RESOURCE_WAIT_REF,native_actual_outer_terminal_ref=c.NATIVE_RESOURCE_OUTER_REF,native_process_identities=native['native_process_identities'],native_process_groups=native['native_process_groups'],observed_at_utc=created);native['fresh_targeted_native_absence_ref']=putram('RAM_ONLY/native_absence.json',na);q['native_resource_disposition_ref']=putram('RAM_ONLY/native.json',native)
q['direct_attachment_authority_ref']=putram('RAM_ONLY/direct_authority.json',dict(RAM_NOT_AUTHORITY=True));q['controller_terminal_qualification_ref']=putram('RAM_ONLY/controller.json',dict(RAM_NOT_AUTHORITY=True))
request_ref=putram(q['preparation_directory']+'/ROOT_request.json',q)
common_scope=dict(vars(c),ref=refr,raw=rawr,read=readr,local=lambda path:MemoryPath(str(ROOT))/path,os=SimpleNamespace(getuid=lambda:1000))
ct=ast.parse((ROOT/D/'common_v1.py').read_text());nodes=[n for n in ct.body if isinstance(n,ast.FunctionDef)and n.name in ('unique','validate_request_v1','validate_input_v1','science_v1','validate_gate_v1','validate_native_resource_disposition_v1','validate_prior_v4_terminal_v5','validate_prior_v5_terminal_v6','validate_prior_v7_terminal_v8','validate_root_fit_clock_v5','source_refs_v1')];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual_H058_complete_request_input_RAM_metadata>','exec'),common_scope)
# All original H058 request gates execute; refs outside RAM are sealed small source/metadata only.
validated=common_scope['validate_request_v1'](request_ref);ck('complete_actual_H058_request_predicate_accepts_RAM_authority',validated[0]==q)
loaded=[]
def loadmemo(record,name):
 module=c.load(record,name)
 if record==c.MEMO:
  original_source_bytes=module.source_bytes;module.source_bytes=module.clone(original_source_bytes,dict(ROOT=ROOT));module.ROOT=ramroot;loaded.append(module)
 return module
def save(directory,name,value):
 path=str((MemoryPath(directory)/name).relative_to(str(ROOT)));writes.append(path);return putram(path,value)
scope=dict(vars(driver),unique=common_scope['unique'],ref=refr,raw=rawr,read=readr,local=lambda path:MemoryPath(str(ROOT))/path,load=loadmemo,save=save,guard_v1=lambda **kwargs:g,validate_request_v1=common_scope['validate_request_v1'],validate_input_v1=common_scope['validate_input_v1'],source_refs_v1=common_scope['source_refs_v1'])
dt=ast.parse(op._DRIVER_SOURCE);nodes=[n for n in dt.body if isinstance(n,ast.FunctionDef)and n.name in ('current_uncached_paths_v9','mutation_memo_v1','admit_v1')];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual_driver_memo_admit_RAM>','exec'),scope)
answer=scope['admit_v1'](request_ref);spec,q2,cfg2,storage,protocol=common_scope['validate_input_v1'](answer['input_ref'])
ck('actual_admit_constructs_accepted_v2_current_Census_memo',len(loaded)==1 and loaded[0].SOURCE==c.MEMO['path'])
ck('actual_source_authenticator_accepts_current_Census_and_original_primitives',g.__file__==str(ROOT/c.GUARD['path'])and g._state_locked.__globals__ is vars(g))
ck('actual_read_state_then_reserve_take_two_original_nonnested_locks',lock_events==['lock','unlock','lock','unlock'])
ck('actual_original_append_creates_one_RAM_journal_event',len([p for p in memory if p.startswith(g.JOURNAL+'/')])==1)
ck('actual_reserve_receipt_and_input_join_full256MiB',storage['reserved_new_bytes']==c.CAPS['reservation_bytes'] and storage['reservation']['root_admission']['path'].endswith('.reserve_authority.json'))
ck('actual_current47_roles_and_full64_direct_union',len(storage['reservation']['consumer_bindings'])==64 and all(r in storage['reservation']['consumer_bindings']for r in guards.values()))
old_memo='research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py'
ck('real_v2_direct_role_replaces_obsolete_unused_memo_role',c.MEMO in storage['reservation']['consumer_bindings'] and not any(r['path']==old_memo for r in storage['reservation']['consumer_bindings']))
ck('all_original_primitives_restored_and_cache_generation_destroyed',all(getattr(v4,k)is v for k,v in primitive_originals.items())and budget.successor_paths is successor_original and loaded[0]._ACTIVE is False and not locked[0])
ck('actual_admission_and_future_child_route_stay_shared_v9',answer['one_shot_exec_argv'][4]==D.replace('/','.')+'.operational_v1'and spec['producer_ref']==spec['main_source_ref']==realref(c.DRIVER))
ck('fresh27h_and_original24h_budget_preserved',spec['created_at_utc']==created and spec['expires_at_utc']==expires and q2['caps']['max_seconds']==86400)
ck('current_operation_authority_and_manifest_have_original_final_reads',reads[q['preparation_directory']+'/'+q['instance_id']+'.reserve_authority.json']>=1 and reads[c.MANIFEST]>=1)
ck('no_science_framework_imports',not any(n=='torch'or n.startswith(('torch.','torch_geometric','numpy'))for n in sys.modules))
# Material negatives: current bridge cannot authenticate an old-guard pathname, and no spare direct role exists.
before_events=len([p for p in memory if p.startswith(g.JOURNAL+'/')]);original_guard_file=g.__file__;g.__file__=str(ROOT/'research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py')
try:scope['mutation_memo_v1'](g,q,cfg,request_ref)
except ValueError as exc:ck('wrong_old_guard_path_rejected_by_actual_source_authenticator',str(exc)=='Exact supported module pathname')
else:raise AssertionError('Wrong old guard accepted')
finally:g.__file__=original_guard_file
ck('rejected_bridge_never_reaches_another_state_reserve_or_append',len([p for p in memory if p.startswith(g.JOURNAL+'/')])==before_events and lock_events==['lock','unlock','lock','unlock'])
extra=putram('RAM_ONLY/extra_role.json',dict(RAM_NOT_AUTHORITY=True))
try:common_scope['unique'](storage['reservation']['consumer_bindings']+[extra])
except ValueError as exc:ck('actual_original_direct_role_ceiling_rejects_65',str(exc)=='Original64 direct-consumer ceiling')
else:raise AssertionError('65 direct roles accepted')
result=dict(schema='H058_v9_complete_actual_request_admit_memo2_reserve_input_RAM_join_v1',source_only=True,independent=False,status='PASS',assertions=len(checks),checks=checks,source_ref=realref(c.DRIVER),source_manifest_ref=realref(c.MANIFEST),memo_source_ref=c.MEMO,actual_guard_roles=47,actual_direct_union=64,actual_guard_ledger_resource_process_or_numerical_operations=False,all_operational_IO_handwritten_RAM=True,actual_source_authenticator_and_original_predicate_bodies_executed=True,original_lock_events=lock_events)
if '--emit-result'in sys.argv:
 with(ROOT/D/'full_admission_RAM.result.v9.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k!='checks'},sort_keys=True))
