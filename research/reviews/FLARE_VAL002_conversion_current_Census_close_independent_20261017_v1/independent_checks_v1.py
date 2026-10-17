"""Narrow independent exact-source/metadata/control review. Operational modules never imported."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from types import SimpleNamespace
import argparse,ast,contextlib,copy,hashlib,io,json,stat,sys,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
D='research/reviews/FLARE_VAL002_conversion_current_Census_close_source_20261017_v1/'
OUT=ROOT/'research/reviews/FLARE_VAL002_conversion_current_Census_close_independent_20261017_v1'
verified={};checks=[];negative=[]
def need(ok,message):
 if not ok:raise ValueError(message)
def check(name,ok):need(ok,name);checks.append(name)
def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def packed(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(data):return hashlib.sha256(data).hexdigest()
def bounded(path,expected=None):
 need(not path.startswith('research/reviews/FLARE_VAL002_conversion_ROOT_close_Census_'),'No actual run namespace reads/probes')
 p=ROOT/path;s=p.lstat();need(p.resolve()==p and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=1048576,'Bounded pinned source/metadata')
 data=p.read_bytes();a=p.lstat();identity=lambda v:(v.st_dev,v.st_ino,v.st_mode,v.st_nlink,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
 need(identity(s)==identity(a)and len(data)==s.st_size,'Stable bounded bytes');digest=sha(data);need(expected is None or digest==expected,'Authenticated '+path);verified[path]=digest;return data
def functions(data,names):return [n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef)and n.name in names]
def compile_nodes(nodes,env,label):exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),label,'exec',dont_inherit=True,optimize=0),env)
class RAMPath(PurePosixPath):
 @classmethod
 def cwd(cls):return cls('/workspace-SR003.nfs2/ARG-GNN')
 def resolve(self):return self
class FixtureDate(datetime):
 @classmethod
 def now(cls,tz=None):return cls(2001,1,1,12,0,0,tzinfo=timezone.utc)
def main():
 handoff=json.loads(bounded(D+'source_handoff.v1.json','8a3f2a61e56435792b5a497cd70fa4df4942efcb2f64ee8891e6d78e4441073f'))
 manifest_bytes=bounded(D+'source_manifest.v1.json','4192dc056f51916574d9f4acad3e79171b429624086c1e22c84704d277c15a95');manifest=json.loads(manifest_bytes)
 blobs={D+'source_manifest.v1.json':manifest_bytes}
 for ref in manifest['source_refs']:blobs[ref['path']]=bounded(ref['path'],ref['sha256'])
 for key,value in handoff.items():
  if isinstance(value,dict)and set(value)=={'path','sha256'}:blobs[value['path']]=bounded(value['path'],value['sha256'])
 check('all19manifest_sources_and_handoff_direct_refs_authenticated',len(manifest['source_refs'])==19)
 source=blobs[D+'close_ROOT_v1.py'];tree=ast.parse(source);p=json.loads(blobs[D+'pins.v1.json'])
 check('exact_source_and_manifest_pins',sha(source)=='73b9ed1fded392502547be6811bcbc14c25b5201abb897b8231d7d097e3ece2d'and handoff['source_manifest_ref']['sha256']==sha(manifest_bytes))
 records={name:json.loads(blobs[ref['path']]) for name,ref in p.items()if isinstance(ref,dict)and set(ref)=={'path','sha256'}and ref['path'].endswith('.json')}
 real_own=records['storage']['reservation']
 check('exact_genuine_original_reservation_digest62bindings899404366',sha(encode(real_own))==p['reservation_sha256']=='5a632d6d3b6dc79f34d590c0d90d0ea725050a90fd1c7dadb18ea31b2a09b92b' and real_own['reserved_bytes']==899404366 and len(real_own['consumer_bindings'])==62)
 env={'Path':RAMPath,'ROOT':RAMPath(str(ROOT)),'need':need,'json':json,'datetime':FixtureDate,'timezone':timezone,'sys':sys,'argparse':argparse,'__doc__':'Independent RAM exact source functions'}
 env['HERE']=D.rstrip('/');env['SOURCE']=D+'close_ROOT_v1.py';env['MANIFEST']=D+'source_manifest.v1.json'
 compile_nodes(functions(source,('local','metadata_join','request_value','current','pins','main')),env,'exact_close_pure_metadata_RAM')
 def ram_raw(ref):need(ref==dict(path=ref['path'],sha256=sha(blobs[ref['path']])),'Exact RAM authenticated ref');return blobs[ref['path']]
 env['raw']=ram_raw;env['read']=lambda r:json.loads(ram_raw(r));env['ref']=lambda path:dict(path=path,sha256=sha(blobs[path]))
 joined=env['metadata_join'](p);inp,wait,done,launch,storage,own,pids,groups=joined
 check('actual_metadata_joins_worker_supervisor_groups',pids==groups==[4167377,4168865]and launch['ownership']['owned_identities']==[dict(pid=4168865,start=490547826)])
 check('genuine_consumed18397outer0_and_direct_wait0',records['outer']['tool_session_id']==18397 and records['outer']['actual_terminal_consumed']is True and records['outer']['actual_exit_code']==0 and wait['actual_direct_worker_wait']is True and wait['actual_OS_wait_status']==wait['returncode']==0)
 check('preserves_entire_data_and_packet_auxiliary_scope',own['output']==str(ROOT/inp['output_directory'])and own['auxiliary_scopes']==[str(ROOT/inp['packet_directory'])])
 selector=records['selector'];policy=records['policy']
 check('current_Census_selector_policy_source_exact',selector['source_ref']==p['guard']and selector['policy_ref']==p['policy']and policy['source_ref']==p['guard']and policy['active']is True)
 auth_template=json.loads(blobs[D+'ROOT_once_authorization.INACTIVE.template.v1.json']);rev_template=json.loads(blobs[D+'ROOT_permanent_revocation.INACTIVE.template.v1.json'])
 check('author_templates_remain_inactive_without_ROOT_clocks',auth_template['active']is auth_template['once_authorized']is rev_template['active']is False and auth_template['created_at_utc']is auth_template['expires_at_utc']is rev_template['revoked_at_utc']is None and auth_template['source_review_ref']is auth_template['permanent_route_revocation_ref']is None)
 close=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='close_once')
 # Exact prefix includes every source/review/auth/metadata/revocation check; it stops BEFORE mkdir/import/probes/state.
 prefix=copy.deepcopy(close);prefix.name='exact_preoperation_controls'
 prefix.body=[n for n in prefix.body if n.lineno<109]+[ast.Return(value=ast.Constant(value=True))]
 compile_nodes([prefix],env,'exact_preoperation_controls_RAM_only')
 source_ref=dict(path=D+'close_ROOT_v1.py',sha256=sha(source));manifest_ref=dict(path=D+'source_manifest.v1.json',sha256=sha(manifest_bytes))
 def control_case(mutate=None):
  auth=copy.deepcopy(auth_template);rev=copy.deepcopy(rev_template)
  review=dict(independent=True,verdict='SOURCE_PASS',blockers=[],source_ref=source_ref,source_manifest_ref=manifest_ref)
  auth.update(active=True,once_authorized=True,root_authorizer='root_agent',created_at_utc='2001-01-01T11:30:00+00:00',expires_at_utc='2001-01-01T12:30:00+00:00',ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions=True)
  rev.update(active=True,root_authorizer='root_agent',revoked_at_utc='2001-01-01T11:00:00+00:00')
  if mutate:mutate(auth,rev,review)
  reviewref=dict(path='RAM_ONLY_review.json',sha256=sha(packed(review)));revref=dict(path='RAM_ONLY_revocation.json',sha256=sha(packed(rev)))
  auth['source_review_ref']=reviewref;auth['permanent_route_revocation_ref']=revref
  authref=dict(path='RAM_ONLY_authorization.json',sha256=sha(packed(auth)))
  ram={**blobs,reviewref['path']:packed(review),revref['path']:packed(rev),authref['path']:packed(auth)}
  def raw(ref):need(type(ref)is dict and set(ref)=={'path','sha256'}and sha(ram[ref['path']])==ref['sha256'],'Exact ref in RAM control DAG');return ram[ref['path']]
  env['read']=lambda r:json.loads(raw(r));env['raw']=raw;env['ref']=lambda path:dict(path=path,sha256=sha(ram[path]))
  return env['exact_preoperation_controls'](authref)
 check('complete_RAM_ROOT_control_review_revocation_join_passes_before_any_operation',control_case())
 def reject(name,mutate):
  try:control_case(mutate)
  except(ValueError,KeyError,TypeError):negative.append(name);return
  raise AssertionError('Accepted invalid control '+name)
 reject('inactive_ROOT_authority',lambda a,r,v:a.update(active=False))
 reject('once_authorized_false',lambda a,r,v:a.update(once_authorized=False))
 reject('prior_inflight_mutations_not_resolved',lambda a,r,v:a.update(ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions=False))
 reject('wrong_actual_outer_ref',lambda a,r,v:a['actual_outer_terminal_ref'].update(sha256='f'*64))
 reject('wrong_reservation_digest',lambda a,r,v:a.update(reservation_sha256='f'*64))
 reject('wrong_current_selector',lambda a,r,v:a['storage_policy_selector_ref'].update(sha256='f'*64))
 reject('nonindependent_source_review',lambda a,r,v:v.update(independent=False))
 reject('source_review_with_blocker',lambda a,r,v:v.update(blockers=['fixture blocker']))
 reject('source_review_wrong_source_digest',lambda a,r,v:v.update(source_ref=dict(path=source_ref['path'],sha256='f'*64)))
 reject('permanent_route_revocation_inactive',lambda a,r,v:r.update(active=False))
 reject('permanent_routes_not_revoked',lambda a,r,v:r.update(all_original_launch_resume_retry_write_routes_permanently_revoked=False))
 reject('revocation_wrong_packet_scope',lambda a,r,v:r.update(packet_directory='RAM_ONLY_other_packet'))
 reject('revocation_in_future',lambda a,r,v:r.update(revoked_at_utc='2001-01-01T12:00:01+00:00'))
 reject('expired_authority_after_wait_or_scan',lambda a,r,v:a.update(expires_at_utc='2001-01-01T12:00:00+00:00'))
 reject('authority_lifetime_exceeds_one_hour',lambda a,r,v:a.update(created_at_utc='2001-01-01T11:29:59+00:00'))
 # Current ordinary proof, observe=False only. Any attempted absence/probe would fail this review immediately.
 guard_source=blobs[p['guard']['path']];g={'Path':RAMPath,'ROOT':RAMPath(str(ROOT)),'require':need,'datetime':FixtureDate,'timezone':timezone}
 def integer(value,label,minimum):need(type(value)is int and value>=minimum,label)
 g['budget']=SimpleNamespace(digest=lambda v:sha(encode(v)),v4=SimpleNamespace(encoded=encode,integer=integer,canonical=lambda root,path:RAMPath(path)if RAMPath(path).is_absolute()else root/path))
 compile_nodes(functions(guard_source,('_pointer','_identity','_completed_identity','_proof')),g,'exact_original_ordinary_proof_RAM')
 obsref=dict(path='RAM_ONLY_observation.json',sha256='1'*64)
 obs=dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',key=own['key'],reservation_sha256=p['reservation_sha256'],storage_ref=p['storage'],launch_ref=p['launch'],direct_wait_ref=p['wait'],completed_ref=p['completed'],launcher_pid=launch['pid'],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=pids,observations=[dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc='1999-01-01T00:00:00+00:00')for pid in pids])
 g['_json']=lambda ref:copy.deepcopy(obs)if ref==obsref else json.loads(ram_raw(ref));g['_completed_json']=lambda ref:json.loads(ram_raw(ref))
 g['_absent']=lambda pid:(_ for _ in ()).throw(AssertionError('No actual or RAM absence calls requested'))
 request=env['request_value'](p,own,launch,obsref);proof=g['_proof'](request,own,observe=False)
 check('original_current_ordinary_proof_accepts_genuine_metadata_without_operational_calls',proof['pids']==pids and proof['completion_verified']is True)
 check('ordinary_proof_has_nonfuture_only_no120second_max_age',all(r['checked_utc']=='1999-01-01T00:00:00+00:00'for r in obs['observations']))
 calls=lambda n:[ast.unparse(x.func)for x in ast.walk(n)if isinstance(x,ast.Call)]
 apply=next(n for n in functions(guard_source,('apply_close_completed_v1',)))
 apply_calls=calls(apply)
 check('original_apply_fresh_state_before_authority_expiry_and_pid_reobserve',apply_calls.index('_state_locked')<apply_calls.index('_authority')and "_proof(request, own, observe=True)"in ast.unparse(apply)and 'for pid in proof[\'pids\']:'in ast.unparse(apply)and 'budget.v4.measure_scopes' in apply_calls)
 text=source.decode();close_text=ast.get_source_segment(text,close)
 check('exclusive_consumed_invocation_precedes_guard_import_and_target_checks',close_text.index('directory.mkdir()')<close_text.index('save(directory,"invocation.json"')<close_text.index('sp.loader.exec_module(module)')<close_text.index('g._absent(pid)')<close_text.index('g._state_locked'))
 check('original47roles62refs64cap_ordinary_API_preserved','len(state[3])==47'in text and 'complete==own["consumer_bindings"]and len(complete)<=64'in text and 'g.operation_draft_v1("close_completed"'in text and 'g.apply_close_completed_v1('in text and 'apply_retire'not in text)
 check('authority_expiry_literal_copied_unchanged','authority.update(active=True,expires_at_utc=auth["expires_at_utc"])'in text)
 check('default_exact_main_is_inert',True)
 with contextlib.redirect_stdout(io.StringIO())as output:need(env['main']([])==0,'Inert default return')
 need(json.loads(output.getvalue())==dict(status='SOURCE_ONLY_INACTIVE',guard_imported=False,operation=False),'Inert default exact output')
 check('original_ordinary_apply_no300timer_or_postmeasurement_expiry','monotonic'not in ast.unparse(apply)and '_expire('not in ast.unparse(apply))
 # Scope limitation: historical role map subset is corroboration, never a current state capture.
 capture=json.loads(bounded('research/reviews/Census_observation_ROOT_actual_inspect_retention_20261017_v1/fresh_capture.root_v1.json','a203f211e5927c68d85026f424d26734d653dc966c7c2cca5785d38cbb15974c'))
 check('historical47role_refs_already_in_original62_no_new_binding',len(capture['guard_roles'])==47 and all(r in own['consumer_bindings']for r in capture['guard_roles'].values()))
 return dict(schema='FLARE_VAL002_current_Census_close_independent_targeted_checks_v1',verdict='PASS',checks=checks,positive_count=len(checks),negative_cases=negative,negative_count=len(negative),verified_selected_refs=verified,source_manifest_refs=19,actual_metadata_join=True,original_ordinary_proof_RAM_only=True,operational_modules_imported=False,actual_clock_read=False,actual_run_namespace_inspected_or_created=False,actual_guard_state_lock_process_group_probe_measurement_journal_launch_or_Git_calls=False,active_controls_saved=False,fixture_now='2001-01-01T12:00:00+00:00',timing_predicate_assessment='Ordinary _proof only rejects future timestamps, with no120second age bound. Finite ROOT<=1hour expiry is checked after capture state and again after apply fresh state. Mutex/replay can exhaust it before authority acceptance. Original apply reobserves PIDs before/after measurement; no300s timer and no postmeasurement/postappend expiry check are added. No runtime fit inferred.')
try:result=main()
except BaseException:
 diagnostic=traceback.format_exc();(OUT/'first_diagnostic.v1.txt').write_text(diagnostic);print(diagnostic,flush=True);raise
with (OUT/'checks.result.v1.json').open('xb')as stream:stream.write(packed(result))
print(json.dumps({k:result[k]for k in ('verdict','positive_count','negative_count','actual_metadata_join','operational_modules_imported','actual_clock_read','actual_run_namespace_inspected_or_created')},sort_keys=True))
