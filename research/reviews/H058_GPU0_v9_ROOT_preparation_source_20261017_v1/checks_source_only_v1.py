"""Bounded actual preparer/native/v7/request joins; all operational IO is RAM.
No actual clock, process, GPU, lock, registry, guard, accounting or science calls.
"""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
from datetime import datetime,timezone
import ast,contextlib,copy,hashlib,importlib,io,json,sys
ROOT_REAL=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/H058_GPU0_v9_ROOT_preparation_source_20261017_v1'
CANDIDATE='research/reviews/H058_GPU0_full_training_successor_source_20261017_v9'
sys.path.insert(0,str(ROOT_REAL))
checks=[];memory={};dirs={str(ROOT_REAL)};reads=set();operational=[]
def ck(name,v=True):
 if not v:raise AssertionError(name)
 checks.append(dict(name=name,status='PASS'))
def safe_bytes(path):
 rel=str(PurePosixPath(path).relative_to(str(ROOT_REAL))) if PurePosixPath(path).is_absolute() else str(path)
 if rel in memory:return memory[rel]
 p=ROOT_REAL/rel
 assert '..' not in PurePosixPath(rel).parts and p.suffix in ('.py','.json','.txt','.md','.sh','.patch','.yaml','.yml','.toml'),rel
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<=2**20,rel
 reads.add(rel);return p.read_bytes()
class RAMText(io.StringIO):
 def __init__(self,path):super().__init__();self.path=path
 def close(self):
  if not self.closed:memory[self.path]=self.getvalue().encode()
  super().close()
class MemoryPath(PurePosixPath):
 def exists(self):return str(self) in dirs or str(self.relative_to(str(ROOT_REAL))) in memory
 def is_dir(self):return str(self) in dirs
 def resolve(self):return self
 def is_symlink(self):return False
 def read_bytes(self):return safe_bytes(self)
 def mkdir(self):
  assert not self.exists();dirs.add(str(self));operational.append('RAM mkdir')
 def open(self,mode):
  assert mode=='x' and not self.exists();return RAMText(str(self.relative_to(str(ROOT_REAL))))
ramroot=MemoryPath(str(ROOT_REAL))
FIXED=datetime.fromisoformat('2026-10-17T12:00:00+00:00')
class FrozenDateTime(datetime):
 @classmethod
 def now(cls,tz=None):return FIXED if tz is not None else FIXED.replace(tzinfo=None)
def refreal(path):return dict(path=path,sha256=hashlib.sha256(safe_bytes(path)).hexdigest())
def readreal(path):return json.loads(safe_bytes(path))
def putram(path,value):
 assert path not in memory
 memory[path]=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();return refreal(path)
text=safe_bytes(HERE+'/prepare_ROOT_v1.py').decode();tree=ast.parse(text)
scope=dict(__name__='_inactive_preparer_RAM',__file__=str(ROOT_REAL/HERE/'prepare_ROOT_v1.py'))
exec(compile(text,str(ROOT_REAL/HERE/'prepare_ROOT_v1.py'),'exec'),scope)
scope.update(ROOT=ramroot,datetime=FrozenDateTime)
# Actual default main does not call context, observe, metadata, capacity or clock.
original_prepare=scope['prepare_ROOT_v1']
scope['prepare_ROOT_v1']=lambda *a:(_ for _ in ()).throw(AssertionError('Default path performed actual work'))
with contextlib.redirect_stdout(io.StringIO()) as captured:status=scope['main']([])
ck('actual_default_main_inert',status==0 and json.loads(captured.getvalue())==dict(status='SOURCE_ONLY_INACTIVE',ROOT_metadata_created=False,resource_probes=False,admission=False,execution=False))
scope['prepare_ROOT_v1']=original_prepare
c=importlib.import_module(CANDIDATE.replace('/','.')+'.common_v1')
common=dict(vars(c),ROOT=ramroot,datetime=FrozenDateTime,time=SimpleNamespace(time=lambda:FIXED.timestamp()),
 local=lambda p:ramroot/p,os=SimpleNamespace(getuid=lambda:1000),ref=scope['ref'],read=scope['read'],raw=lambda r:rawr(r))
def rawr(r):
 b=safe_bytes(r['path']);assert hashlib.sha256(b).hexdigest()==r['sha256'];return b
# Exact common predicate bodies, including clocks, lineage, native50 and science.
ct=ast.parse(safe_bytes(CANDIDATE+'/common_v1.py').decode())
nodes=[n for n in ct.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual_v9_common_RAM>','exec'),common)
# Keep the already authenticated pure source loader; it reads pinned code only.
common.update(load=c.load,ref=scope['ref'],read=scope['read'],raw=rawr,local=lambda p:ramroot/p)
cr=SimpleNamespace(**common)
old=scope['read'](c.CLOCK_ROOT_REQUEST_V4);oldcfg=scope['read'](old['config_ref'])
review=dict(independent=True,verdict='SOURCE_PASS',blockers=[],source_ref=scope['ref'](scope['SOURCE']),
 source_manifest_ref=scope['ref'](scope['MANIFEST']),candidate_source_ref=scope['CANDIDATE_DRIVER_REF'],
 candidate_source_manifest_ref=scope['CANDIDATE_MANIFEST_REF'],candidate_whole_source_review_ref=scope['CANDIDATE_REVIEW_REF'])
review_ref=putram('RAM_ONLY/preparer_review.json',review)
directive=readreal(HERE+'/ROOT_preparation_directive.INACTIVE.template.v1.json')
for key in ('all_prior_H058_launch_resume_write_routes_permanently_revoked','all_v7_launch_resume_write_routes_permanently_revoked','remaining_historical_ownership_uncertainty_ROOT_accepted_for_fresh_scope','historical_files_scopes_full_charges_clocks_sources_failures_and_unknowns_preserved','new_full_1920_update_unfinished_cell_authorized','new27h_interval_for_fresh_v9_only','all_old_native_launch_resume_write_routes_permanently_revoked','additional_old_controller_submissions_revoked_while_H058_attached','old_controller_preserved_until_H058_drained'):directive[key]=True
directive.update(status='active',prepare_once=True,created_at_utc=FIXED.isoformat(),expires_at_utc='2026-10-18T15:00:00+00:00',
 preparation_source_manifest_ref=scope['ref'](scope['MANIFEST']),preparation_review_ref=review_ref,historical_ROOT_clocks_v8=cr.historical_clocks_v8())
registration=readreal(HERE+'/unfinished_cell_registration.INACTIVE.template.v1.json')
registration.update(status='registered',new_full_unfinished_cell=True,fresh_AdamW_from_original_H045_own_state=True,no_old_CPU_or_GPU_optimizer_state_transfer=True,
 repeat_reason='RAM fixture only, never an actual registration',created_at_utc=directive['created_at_utc'],expires_at_utc=directive['expires_at_utc'])
directive['unfinished_cell_registration_ref']=putram('RAM_ONLY/registration.json',registration)
directive_ref=putram('RAM_ONLY/directive.json',directive)
ck('actual_authority_join_accepts_RAM_ROOT_review_directive_and_registration',scope['authority_v1'](review_ref,directive_ref,cr)==directive)
def reject_authority(name,change):
 value=copy.deepcopy(directive);change(value);r=putram('RAM_ONLY/'+name+'.json',value)
 try:scope['authority_v1'](review_ref,r,cr)
 except (AssertionError,ValueError):ck(name)
 else:raise AssertionError('Accepted '+name)
reject_authority('missing_residual_history_acceptance_rejected',lambda v:v.update(remaining_historical_ownership_uncertainty_ROOT_accepted_for_fresh_scope=False))
reject_authority('old_route_reuse_rejected',lambda v:v.update(all_v7_launch_resume_write_routes_permanently_revoked=False))
reject_authority('old_clocks_rejected',lambda v:v.update(created_at_utc=old['created_at_utc'],expires_at_utc=old['expires_at_utc']))
reject_authority('wrong_outer_terminal_ref_rejected',lambda v:v.update(prior_v7_actual_outer_terminal_ref=dict(path='RAM_ONLY/fake_outer.json',sha256='0'*64)))
reject_authority('controller_renewal_rejected',lambda v:v.update(controller_capability_renewal_authorized=True))
badreg=dict(registration,updates_all_new_arms=960);badregref=putram('RAM_ONLY/bad_registration.json',badreg)
reject_authority('partial_update_registration_rejected',lambda v:v.update(unfinished_cell_registration_ref=badregref))
identities,groups,supervisor=scope['known_v7_targets_v1'](cr)
ck('exact_genuine_v7_wait_outer_and_target_origin_join',len(identities)==64 and groups==[4127469] and supervisor==4122785)
# Actual targeted v7 function, fake process identities and fake ESRCH only.
rows={}
def proc(pid):
 operational.append('RAM targeted proc')
 if pid in rows:return copy.deepcopy(rows[pid])
 raise ProcessLookupError(pid)
def killpg(pgid,sig):operational.append('RAM targeted group ESRCH');raise ProcessLookupError(pgid)
scope['os']=SimpleNamespace(killpg=killpg)
base=SimpleNamespace(proc_identity=proc,real_inventory=lambda:[dict(index=0,uuid='GPU-ff06ca1f-403a-f2d0-b584-71e177452667',compute=[])])
v7=scope['observe_v7_v1'](cr,base)
ck('actual_v7_observer_checks_all64_group_supervisor_and_limited_AUX',len(v7['identities'])==64 and v7['known_AUX_writer_contexts_absent'] and not v7['historical_complete_roster_claimed'])
rows[identities[0]['pid']]=dict(start_ticks=identities[0]['start'],uid=1000,state='Z')
try:scope['observe_v7_v1'](cr,base)
except AssertionError:ck('exact_known_v7_zombie_identity_blocks')
else:raise AssertionError('Old exact zombie accepted')
rows[identities[0]['pid']]['start_ticks']+=1
ck('reused_PID_different_start_is_not_old_identity',scope['observe_v7_v1'](cr,base)['identities'][0]['exact_old_identity_present'] is False)
rows.clear();rows[supervisor]=dict(start_ticks=1,uid=1000,state='Z')
try:scope['observe_v7_v1'](cr,base)
except AssertionError:ck('numeric_supervisor_presence_blocks_without_saved_start')
else:raise AssertionError('Supervisor numeric presence accepted')
rows.clear();scope['os']=SimpleNamespace(killpg=lambda *a:None)
try:scope['observe_v7_v1'](cr,base)
except ValueError:ck('known_v7_group_presence_blocks')
else:raise AssertionError('Live group accepted')
scope['os']=SimpleNamespace(killpg=killpg)
# Actual baseline native50 observation body and added v7/probe, all IO RAM.
registry=dict(RAM_ONLY=True)
registry_dir='RAM_ONLY/controller';memory[registry_dir+'/registry.json']=json.dumps(registry).encode()
storage_ref=putram('RAM_ONLY/storage.json',dict(output_directory=registry_dir))
cap_ref=putram('RAM_ONLY/cap.json',dict(controller_storage_ref=storage_ref))
probes=[]
d=SimpleNamespace(base_v1=lambda:base,CURRENT_CAP=cap_ref,GPU0_UUID='GPU-ff06ca1f-403a-f2d0-b584-71e177452667',held_writable_probe_v1=lambda p:probes.append(str(p)) or True)
observation,observed_base=scope['observe_v1'](cr,d,old)
ck('actual_native50_and_v7_observation_plus_writable_probe_RAM',len(observation['rows'])==50 and len(observation['v7']['identities'])==64 and probes==[str(ramroot/'.runtime/locks/gpu_0.lock')] and observed_base is base)
# Original native observation statements remain byte-for-byte inside the port.
baseline=safe_bytes('research/reviews/H058_GPU0_v7_ROOT_preparation_source_20261017_v2/prepare_ROOT_v1.py').decode()
def function_source(source,name):return next(ast.get_source_segment(source,n)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name==name)
old_observe=function_source(baseline,'observe_v1');new_observe=function_source(text,'observe_v1')
old_prefix=old_observe[:old_observe.index(' return dict(native=native')]
ck('all_original_native_identity_group_GPU0_registry_observation_statements_unchanged',new_observe.startswith(old_prefix))
q=readreal(CANDIDATE+'/ROOT_request.INACTIVE.template.v1.json');q.update(scope['SCOPES'])
q.update(created_at_utc=directive['created_at_utc'],expires_at_utc=directive['expires_at_utc'],historical_ROOT_clocks_v8=cr.historical_clocks_v8(),fit_clock_compatibility_v5=cr.fit_clock_acceptance_v5(),fresh_scope_lineage_ref=directive_ref)
request_ref,q_out,na_ref,v7_ref=scope['metadata_v1'](review_ref,directive_ref,cr,d,q,old,oldcfg,observation,directive)
ck('actual_metadata_builder_returns_real_complete_v9_request_schema_PASS',cr.validate_request_v1(request_ref)[0]==q_out)
cfg=scope['read'](q_out['config_ref']);prior=scope['read'](q_out['prior_H058_CPU_disposition_ref']);native=scope['read'](q_out['native_resource_disposition_ref'])
changed={k for k in cfg if cfg[k]!=oldcfg.get(k)}
ck('original_scientific_config_changes_only_required_authenticated_refs',changed=={'producer_ref','source_manifest_ref','runtime_equivalence_ref','layout_resource_proof_ref','full_training_directive_ref'})
ck('original_runtime_and_layout_proofs_byte_values_preserved',all(scope['read'](cfg[k])==scope['read'](oldcfg[k])for k in ('runtime_equivalence_ref','layout_resource_proof_ref')))
ck('full1920_original32_TRAIN_science_and_all_caps_unchanged',cfg['scientific_protocol']==oldcfg['scientific_protocol'] and q_out['caps']==old['caps']==cr.CAPS)
ck('new27h_preserves_all_historical_clocks_and_UNKNOWN_v7_totals',q_out['historical_ROOT_clocks_v8']==cr.historical_clocks_v8() and q_out['created_at_utc']!=old['created_at_utc'] and prior['prior_GPU_v7_actual_optimizer_steps']=={'same_marker_aligned':None,'same_marker_alignment_null':None})
ck('candidate_review_stays_exact_sealed_and_helper_review_separate',q_out['source_review_ref']==scope['CANDIDATE_REVIEW_REF'] and q_out['ROOT_preparation_source_review_ref']==review_ref)
ck('native50_roster_and_groups_unchanged',native['native_process_identities']==observation['native']['native_process_identities'] and native['native_process_groups']==observation['native']['native_process_groups'])
ck('new_controller_metadata_clocks_change_without_capability_renewal',all(scope['read'](q_out[k])['expires_at_utc']==q_out['expires_at_utc'] and scope['read'](q_out[k])['controller_capability_ref']==scope['read'](old[k])['controller_capability_ref']for k in ('controller_terminal_qualification_ref','direct_attachment_authority_ref')))
# The actual wrapper calls unchanged capacity once after complete metadata, returns
# the shared v9 admission argv, and never invokes reserve/admit/execute.
capacity_calls=[];original_context=scope['context_v1'];original_metadata=scope['metadata_v1'];original_observe=scope['observe_v1']
d.capacity_v1=lambda q,b:capacity_calls.append((q,b)) or dict(RAM_ONLY=True)
scope['context_v1']=lambda *args:(CANDIDATE.replace('/','.'),cr,d,q_out,old,oldcfg,directive)
scope['observe_v1']=lambda *args:(observation,base)
scope['metadata_v1']=lambda *args:(request_ref,q_out,na_ref,v7_ref)
answer=scope['prepare_ROOT_v1'](review_ref,directive_ref)
ck('actual_wrapper_calls_preserved_capacity_once_and_emits_separate_shared_v9_admit',len(capacity_calls)==1 and answer['admit_argv'][4]==CANDIDATE.replace('/','.')+'.operational_v1' and answer['admit_argv'][5]=='--admit' and answer['no_reserve_admission_or_science'] is True)
# No stateful entrypoint or broad process walk is reachable from preparer source.
called={n.func.attr if isinstance(n.func,ast.Attribute)else n.func.id for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,(ast.Name,ast.Attribute))}
ck('source_has_no_reserve_admit_launch_delete_or_process_scan_call',not called.intersection({'reserve_v1','apply_reserve_v1','admit_v1','execute_v1','Popen','run','unlink','rmtree','iterdir','glob','walk'}))
ck('only_RAM_operational_events_and_no_scientific_framework_imports',all(x.startswith('RAM ')for x in operational) and not any(n=='torch'or n.startswith(('torch.','torch_geometric','numpy'))for n in sys.modules))
result=dict(schema='H058_v9_ROOT_preparation_bounded_source_only_RAM_checks_v1',source_only=True,independent=False,status='PASS',assertions=len(checks),checks=checks,
 source_ref=refreal(HERE+'/prepare_ROOT_v1.py'),source_manifest_ref=refreal(HERE+'/source_manifest.v1.json'),candidate_source_ref=scope['CANDIDATE_DRIVER_REF'],candidate_whole_source_review_ref=scope['CANDIDATE_REVIEW_REF'],
 actual_resource_process_lock_registry_guard_accounting_clock_or_science_operations=False,actual_native50_v7_64_metadata_and_complete_request_predicate_bodies_executed=True,
 operational_events_all_RAM=True,read_source_or_bounded_metadata_files=len(reads),RAM_metadata_files=len(memory))
if '--emit-result' in sys.argv:
 with(ROOT_REAL/HERE/'checks.source_only.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k!='checks'},sort_keys=True))
