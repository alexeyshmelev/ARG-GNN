"""Independent exact-source/immutable-metadata review. No operational imports."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from types import SimpleNamespace
import ast,argparse,copy,hashlib,io,json,stat,sys,contextlib,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
D='research/reviews/FLARE_VAL002_native_current_Census_close_source_20261017_v1/'
OUT=ROOT/'research/reviews/FLARE_VAL002_native_current_Census_close_independent_20261017_v1'
verified={};checks=[];negative=[]
def need(ok,message):
 if not ok:raise ValueError(message)
def check(name,ok):need(ok,name);checks.append(name)
def enc(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def pack(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def bounded(path,expected=None):
 need(not path.startswith('research/reviews/FLARE_VAL002_native_ROOT_close_Census_'),'No actual operation namespace reads')
 p=ROOT/path;s=p.lstat();need(p.resolve()==p and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=2**20,'Bounded regular source/metadata')
 b=p.read_bytes();t=p.lstat();signature=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(signature(s)==signature(t)and len(b)==s.st_size,'Stable source/metadata read');need(expected is None or sha(b)==expected,'Exact immutable pin '+path);verified[path]=sha(b);return b
def functions(b,names):return [n for n in ast.parse(b).body if isinstance(n,ast.FunctionDef)and n.name in names]
def compile_nodes(nodes,env,name):exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),name,'exec',dont_inherit=True,optimize=0),env)
class RAMPath(PurePosixPath):
 @classmethod
 def cwd(cls):return cls(str(ROOT))
 def resolve(self):return self
class RAMDate(datetime):
 @classmethod
 def now(cls,tz=None):return cls(2100,1,1,12,0,0,tzinfo=timezone.utc)
def main():
 h=json.loads(bounded(D+'source_handoff.v1.json','a3f898fd2fda9376d38973e2b445996a60968fe0d042dd40b8644f4825df33e0'))
 mb=bounded(D+'source_manifest.v1.json','2b5cebb482ec5eb3b2edb457b21dc0ca93dfcd0a8affe5b3537c48dab2ac3f96');m=json.loads(mb)
 blobs={D+'source_manifest.v1.json':mb}
 for r in m['source_refs']:blobs[r['path']]=bounded(r['path'],r['sha256'])
 for v in h.values():
  if type(v)is dict and set(v)=={'path','sha256'}:blobs[v['path']]=bounded(v['path'],v['sha256'])
 audit=json.loads(blobs[h['audit_manifest_ref']['path']])
 for r in audit['refs']:blobs[r['path']]=bounded(r['path'],r['sha256'])
 source=blobs[D+'close_ROOT_v1.py'];tree=ast.parse(source);p=json.loads(blobs[D+'pins.v1.json'])
 check('exact_handoff_manifest_source_and_all23refs_authenticated',len(m['source_refs'])==23 and sha(source)=='ecd132e3731f5c487a5c8cd76ba6edf65ff11ec8d75ff62de43f5b1af88aa4e8')
 def raw(r):need(type(r)is dict and set(r)=={'path','sha256'}and sha(blobs[r['path']])==r['sha256'],'Exact authenticated RAM ref');return blobs[r['path']]
 def read(r):return json.loads(raw(r))
 records={}
 for name,r in p.items():
  if type(r)is dict and set(r)=={'path','sha256'}:
   try:records[name]=read(r)
   except json.JSONDecodeError:pass
 env={'ROOT':RAMPath(str(ROOT)),'Path':RAMPath,'need':need,'datetime':RAMDate,'timezone':timezone,'json':json,'sys':sys,'argparse':argparse,'__doc__':'Independent inert pure functions','HERE':D.rstrip('/'),'SOURCE':D+'close_ROOT_v1.py','MANIFEST':D+'source_manifest.v1.json','raw':raw,'read':read,'ref':lambda path:dict(path=path,sha256=sha(blobs[path]))}
 compile_nodes(functions(source,('local','metadata_join','request_value','current','pins','main')),env,'exact_native_close_pure_functions_RAM')
 joined=env['metadata_join'](p);inp,wait,done,launch,storage,own,pids,groups=joined
 check('genuine_native_worker96960_supervisor92246_and_recorded_children',pids==groups==[92246,96960,100887,105434]and launch['ownership']['owned_identities']==[dict(pid=96960,start=491757276)])
 check('genuine_direct_supervisor_start491728067_and_wait0',records['direct_supervisor_wait']==dict(actual_direct_wait=True,actual_returncode=0,pid=92246)and '491728067' in records['direct_spawn']['proc_stat'])
 check('same_authenticated_embedded_worker_driver_source',p['producer_driver']==p['producer_worker']==inp['root_driver_ref']==inp['main_source_ref'])
 check('actual_admission_and_issued_native_argv_exact63bindings',records['admission_receipt']['consumer_binding_count']==63 and records['issued_once']['actual_once_execution_argv']==records['direct_spawn']['argv']==records['admission_receipt']['execution_argv'])
 check('genuine_completed_native_wait0_drain_consumed6358outer0',done['status']=='completed_C3_native_FLARE'and wait['returncode']==wait['actual_OS_wait_status']==0 and records['outer']['tool_session_id']==6358 and records['outer']['actual_terminal_consumed']is True)
 check('entire_original_reservation63bindings2021479363bytes',sha(enc(own))==p['reservation_sha256']=='bf41e79f497a4f69631634989bc53bea1d4b33a2a628beb6ae6506c2a1f58544'and len(own['consumer_bindings'])==63 and own['reserved_bytes']==2021479363 and own['output']==str(ROOT/inp['output_directory'])and own['auxiliary_scopes']==[str(ROOT/inp['packet_directory'])])
 check('current_Census_selector_and_policy_pinned',records['selector']['source_ref']==records['policy']['source_ref']==p['guard']and records['selector']['policy_ref']==p['policy']and records['policy']['active']is True)
 def reject_metadata(name,mutate):
  rr=copy.deepcopy(records);mutate(rr);by_path={p[k]['path']:v for k,v in rr.items()}
  env['read']=lambda r:copy.deepcopy(by_path[r['path']])
  try:env['metadata_join'](p)
  except(ValueError,TypeError,KeyError):negative.append(name);return
  raise AssertionError('Accepted metadata counterfactual '+name)
 reject_metadata('worker_returncode1',lambda r:r['wait'].update(returncode=1))
 reject_metadata('raw_OSwait256',lambda r:r['wait'].update(actual_OS_wait_status=256))
 reject_metadata('unknown_worker_wait',lambda r:r['wait'].update(actual_direct_worker_wait=False))
 reject_metadata('unknown_launcher_wait',lambda r:r['wait'].update(actual_direct_launcher_wait=False))
 reject_metadata('incomplete_owned_drain',lambda r:r['wait'].update(all_owned_processes_and_groups_drained=False))
 reject_metadata('adopted_nonzero_wait',lambda r:r['wait'].update(adopted_waits=[dict(returncode=1)]))
 reject_metadata('unconsumed_outer',lambda r:r['outer'].update(actual_terminal_consumed=False))
 reject_metadata('wrong_outer_session',lambda r:r['outer'].update(tool_session_id=6359))
 reject_metadata('conversion_completion',lambda r:r['completed'].update(status='completed_C3_conversion_FLARE'))
 reject_metadata('wrong_worker_entry_pid',lambda r:r['wait']['actual_worker_scientific_entry'].update(pid=96961))
 reject_metadata('nonzero_supervisor_direct_wait',lambda r:r['direct_supervisor_wait'].update(actual_returncode=1))
 reject_metadata('wrong_supervisor_pid',lambda r:r['direct_supervisor_wait'].update(pid=92247))
 reject_metadata('wrong_supervisor_start',lambda r:r['direct_spawn'].update(proc_stat=r['direct_spawn']['proc_stat'].replace('491728067','491728068')))
 reject_metadata('changed_issued_native_argv',lambda r:r['issued_once']['actual_once_execution_argv'].pop())
 reject_metadata('dropped_native_admission_binding_count',lambda r:r['admission_receipt'].update(consumer_binding_count=62))
 reject_metadata('launch_start_not_integer',lambda r:r['launch']['ownership']['owned_identities'][0].update(start='491757276'))
 reject_metadata('recorded_launch_pid_missing',lambda r:r['wait'].update(known_owned_pids=[100887,105434]))
 reject_metadata('recorded_group_not_drained',lambda r:r['wait']['group_observations'][0].update(absent=False))
 reject_metadata('recorded_group_roster_dropped',lambda r:r['wait']['group_observations'].pop())
 reject_metadata('original_reservation_changed',lambda r:r['storage']['reservation'].update(reserved_bytes=2021479362))
 reject_metadata('original_binding_dropped',lambda r:r['storage']['reservation']['consumer_bindings'].pop())
 env['read']=read
 at=json.loads(blobs[D+'ROOT_once_authorization.INACTIVE.template.v1.json']);rt=json.loads(blobs[D+'ROOT_permanent_revocation.INACTIVE.template.v1.json'])
 check('templates_are_inactive_without_clocks_review_or_revocation',at['active']is at['once_authorized']is rt['active']is False and at['created_at_utc']is at['expires_at_utc']is rt['revoked_at_utc']is None and at['source_review_ref']is at['permanent_route_revocation_ref']is None)
 close=next(n for n in functions(source,('close_once',)));prefix=copy.deepcopy(close);prefix.name='preoperation_controls'
 stop=next(n.lineno for n in close.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='mkdir')
 prefix.body=[n for n in prefix.body if n.lineno<stop]+[ast.Return(value=ast.Constant(value=True))]
 compile_nodes([prefix],env,'exact_native_close_preoperation_prefix_RAM')
 sr=dict(path=D+'close_ROOT_v1.py',sha256=sha(source));mr=dict(path=D+'source_manifest.v1.json',sha256=sha(mb))
 def control_case(mutate=None):
  a=copy.deepcopy(at);r=copy.deepcopy(rt);v=dict(independent=True,verdict='SOURCE_PASS',blockers=[],source_ref=sr,source_manifest_ref=mr)
  a.update(active=True,once_authorized=True,root_authorizer='root_agent',created_at_utc='2100-01-01T11:30:00+00:00',expires_at_utc='2100-01-01T12:30:00+00:00',ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions=True)
  r.update(active=True,root_authorizer='root_agent',revoked_at_utc='2100-01-01T11:00:00+00:00')
  if mutate:mutate(a,r,v)
  ram=dict(blobs)
  def add(name,value):b=pack(value);ram[name]=b;return dict(path=name,sha256=sha(b))
  a['source_review_ref']=add('RAM_ONLY_review.json',v);a['permanent_route_revocation_ref']=add('RAM_ONLY_revocation.json',r);ar=add('RAM_ONLY_authorization.json',a)
  def rr(ref):need(type(ref)is dict and set(ref)=={'path','sha256'}and sha(ram[ref['path']])==ref['sha256'],'Exact RAM-only control DAG');return ram[ref['path']]
  env['raw']=rr;env['read']=lambda r:json.loads(rr(r));env['ref']=lambda path:dict(path=path,sha256=sha(ram[path]))
  return env['preoperation_controls'](ar)
 check('exact_whole_control_prefix_passes_without_mkdir_import_or_probe',control_case())
 def reject_control(name,mutate):
  try:control_case(mutate)
  except(ValueError,TypeError,KeyError):negative.append(name);return
  raise AssertionError('Accepted invalid control '+name)
 reject_control('inactive_ROOT_authorization',lambda a,r,v:a.update(active=False))
 reject_control('once_false',lambda a,r,v:a.update(once_authorized=False))
 reject_control('unresolved_prior_inflight',lambda a,r,v:a.update(ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions=False))
 reject_control('wrong_outer_control_ref',lambda a,r,v:a['actual_outer_terminal_ref'].update(sha256='f'*64))
 reject_control('wrong_reservation_digest_control',lambda a,r,v:a.update(reservation_sha256='f'*64))
 reject_control('wrong_current_selector_control',lambda a,r,v:a['storage_policy_selector_ref'].update(sha256='f'*64))
 reject_control('nonindependent_review',lambda a,r,v:v.update(independent=False))
 reject_control('review_blocker',lambda a,r,v:v.update(blockers=['RAM blocker']))
 reject_control('wrong_source_review_digest',lambda a,r,v:v.update(source_ref=dict(path=sr['path'],sha256='f'*64)))
 reject_control('revocation_inactive',lambda a,r,v:r.update(active=False))
 reject_control('routes_not_permanently_revoked',lambda a,r,v:r.update(all_original_launch_resume_retry_write_routes_permanently_revoked=False))
 reject_control('revocation_wrong_scope',lambda a,r,v:r.update(packet_directory='RAM_ONLY_different_scope'))
 reject_control('future_revocation',lambda a,r,v:r.update(revoked_at_utc='2100-01-01T12:00:01+00:00'))
 reject_control('expired_authority',lambda a,r,v:a.update(expires_at_utc='2100-01-01T12:00:00+00:00'))
 reject_control('authority_over_one_hour',lambda a,r,v:a.update(created_at_utc='2100-01-01T11:29:59+00:00'))
 guard=blobs[p['guard']['path']];g={'ROOT':RAMPath(str(ROOT)),'Path':RAMPath,'require':need,'copy':copy,'datetime':RAMDate,'timezone':timezone}
 def integer(v,label,minimum):need(type(v)is int and v>=minimum,label);return v
 g['budget']=SimpleNamespace(digest=lambda v:sha(enc(v)),v4=SimpleNamespace(encoded=enc,integer=integer,canonical=lambda root,path:RAMPath(path)if RAMPath(path).is_absolute()else root/path))
 compile_nodes(functions(guard,('_pointer','_identity','_completed_identity','_proof','consumer_bindings_v1')),g,'exact_current_Census_original_functions_RAM')
 obsref=dict(path='RAM_ONLY_observation.json',sha256='a'*64)
 obs=dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',key=own['key'],reservation_sha256=p['reservation_sha256'],storage_ref=p['storage'],launch_ref=p['launch'],direct_wait_ref=p['wait'],completed_ref=p['completed'],launcher_pid=launch['pid'],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=[dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc='1999-01-01T00:00:00+00:00')for pid in pids])
 absents=[]
 def absent(pid):absents.append(pid);return dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc='2100-01-01T12:00:00+00:00')
 g['_absent']=absent;g['_json']=lambda r:copy.deepcopy(obs)if r==obsref else read(r);g['_completed_json']=read
 request=env['request_value'](p,own,launch,obsref)
 a=g['_proof'](request,own,observe=False);b=g['_proof'](request,own,observe=True)
 check('exact_original_proof_both_RAM_observe_branches_accept_native_metadata',a['completion_verified']is b['completion_verified']is True and absents==pids)
 check('ordinary_proof_nonfuture_only_has_no120s_age_bound',all(x['checked_utc']=='1999-01-01T00:00:00+00:00'for x in obs['observations']))
 def reject_proof(name,mutate):
  rr=copy.deepcopy(request);oo=copy.deepcopy(obs);mutate(rr,oo);g['_json']=lambda r:copy.deepcopy(oo)if r==obsref else read(r)
  try:g['_proof'](rr,own,observe=False)
  except(ValueError,TypeError,KeyError):negative.append(name);return
  raise AssertionError('Accepted invalid original proof '+name)
 reject_proof('proof_wrong_reservation_digest',lambda r,o:r.update(reservation_sha256='f'*64))
 reject_proof('proof_wrong_wait_identity',lambda r,o:r['wait_identity'].update(returncode=1))
 reject_proof('proof_wrong_native_completed_identity',lambda r,o:r['completed_identity'].update(status='completed_C3_conversion_FLARE'))
 reject_proof('proof_nonabsent_record',lambda r,o:o['observations'][0].update(absent=False))
 reject_proof('proof_future_record',lambda r,o:o['observations'][0].update(checked_utc='2100-01-01T12:00:01+00:00'))
 reject_proof('proof_boolean_PID',lambda r,o:o['pids'].__setitem__(0,True))
 extra={}
 def union_raw(r):
  if r['path']in extra:return extra[r['path']]
  if r['path']in blobs:return raw(r)
  need(r in own['consumer_bindings'],'Exact original reservation ref in RAM union fixture')
  return b'RAM_ONLY_authenticated_ref_provider_for_union_logic'
 g['_raw']=union_raw
 check('exact_original_union63_plus47_RAM_subset_keeps63',g['consumer_bindings_v1'](own['consumer_bindings'],own['consumer_bindings'][:47])==own['consumer_bindings'])
 additions=[]
 for i in range(2):
  path='RAM_ONLY_extra_'+str(i)+'.py';extra[path]=str(i).encode();additions.append(dict(path=path,sha256=sha(extra[path])))
 check('original_union_allows64refs',len(g['consumer_bindings_v1'](own['consumer_bindings'],additions[:1]))==64)
 try:g['consumer_bindings_v1'](own['consumer_bindings'],additions)
 except ValueError:negative.append('original_union_rejects65refs')
 else:raise AssertionError('65-ref cap unexpectedly accepted')
 with contextlib.redirect_stdout(io.StringIO())as output:check('default_main_inert_return',env['main']([])==0)
 check('default_main_inert_output',json.loads(output.getvalue())==dict(status='SOURCE_ONLY_INACTIVE',guard_imported=False,operation=False))
 text=source.decode();ct=ast.get_source_segment(text,close);apply=functions(guard,('apply_close_completed_v1',))[0];applytext=ast.unparse(apply)
 check('exclusive_invocation_before_guard_import_probe_and_state',ct.index('directory.mkdir()')<ct.index('sp.loader.exec_module(module)')<ct.index('g._absent(pid)')<ct.index('g._state_locked'))
 check('all47roles_entire63_under_original64_cap_and_fresh_state','len(state[3])==47'in text and 'complete==own["consumer_bindings"]and len(complete)<=64'in text and 'fresh=True'in ct)
 check('only_ordinary_close_no_retirement_or_new_charge','g.operation_draft_v1("close_completed"'in ct and 'g.apply_close_completed_v1('in ct and 'apply_retire'not in ct and 'apply_reserve'not in ct)
 check('finite_expiry_passed_to_original_authority_unchanged','authority.update(active=True,expires_at_utc=auth["expires_at_utc"])'in ct)
 check('original_apply_checks_state_then_expiry_then_proof_measurement_postcheck_append',applytext.index('_state_locked(')<applytext.index('_authority(')<applytext.index('_proof(')<applytext.index('measure_scopes(')<applytext.index('for pid in proof')<applytext.index('_append('))
 check('no_added120s_300s_or_postmeasurement_expiry','monotonic'not in applytext and '_expire('not in applytext and '120'not in ct and '300'not in ct)
 check('guard_module_never_imported',not any(k.startswith('_ROOT_FLARE_VAL002_native_exact_current_Census_close')for k in sys.modules))
 return dict(schema='FLARE_VAL002_native_current_Census_independent_checks_v1',verdict='PASS',checks=checks,negative_cases=negative,positive_count=len(checks),negative_count=len(negative),verified_refs=verified,manifest_refs=23,actual_metadata_join=True,all_current_state_role_fit_absence_retained_bytes_credit_or_prefix_claims=False,original_proof_RAM_observe_both=True,actual_guard_imports=0,actual_clock_reads=0,actual_state_mutex_process_group_probe_measurement_journal_launch_or_Git_calls=0,active_controls_written=0,RAM_clock='2100-01-01T12:00:00+00:00',actual_recorded_PIDs_and_groups=pids,original_reserved_bytes=own['reserved_bytes'],original_consumer_binding_count=63,original_returned_role_requirement=47,original_binding_cap=64,timing='Ordinary proof nonfuture only. Finite ROOT<=1h controls may expire during mutex/replay. Original apply checks authority expiry after fresh state and reobserves PIDs before/after measure_scopes, without added postmeasurement expiry or retirement timer. No actual runtime fit or current state is inferred.')
try:result=main()
except BaseException:
 result={'verdict':'HARNESS_OR_SOURCE_FAILURE','diagnostic':traceback.format_exc()};print(result['diagnostic'])
with (OUT/'checks.result.v2.json').open('xb')as f:f.write(pack(result))
print(json.dumps({k:result.get(k)for k in ['verdict','positive_count','negative_count','actual_metadata_join','actual_guard_imports','actual_clock_reads']},indent=2))
