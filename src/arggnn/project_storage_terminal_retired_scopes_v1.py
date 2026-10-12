"""Retire only two obsolete reservations in the existing terminal journal.

Preserve historical readers/events and successful partial artifacts. Root must
finish every terminal-success writer before activation. No import-time action,
scientific completion substitution, artifact payload read or global scan.
"""
from __future__ import annotations
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import stat
from . import project_storage_terminal_success_scopes_v1 as previous

old=previous.old
budget=previous.budget
ROOT=previous.ROOT
SOURCE='src/arggnn/project_storage_terminal_retired_scopes_v1.py'
JOURNAL=previous.JOURNAL
MODE='journal_terminal_retired_scopes_credit_v1'
PREVIOUS_SHA256='2dfbcc2711fc8310fd8a8b9866ee84e9107e1d2c4356396e0ebc0e99fdb7c3b3'
METADATA_CHARGE=previous.METADATA_CHARGE
MECHANISM='budgeted_shared_pool_v3/gpu3_fits_v1/mechanisms303-cache-v1'
ANCHOR='anchor-parent-full-support-CPU-v2-r001'
POOL_ROOT='research/reviews/gpu3_parallel_fits_v1_root/'
POOL_OUTPUT='experiments/results/gpu3_parallel_fits_v1_root/'
ANCHOR_ROOT='research/reviews/anchor_parent_cpu_profile_v2_root/'
ANCHOR_OUTPUT='experiments/results/anchor_parent_full_support_CPU_profile_v2_r001/'
CELL='age_reset_float64_cp0'
CASES={
 MECHANISM:dict(kind='failed_before_output',allowance=52321048576,output='results/biological_mechanism_controls_v1_seed303_cached_gpu3_v1',
  pids=[1313616,1313627,1313663,1313689,1313732],evidence={
   'pool_admission':POOL_ROOT+'pool_admission.json','registry':'.runtime/gpup/gpu3_fits_v1/registry.json',
   'worker_log':'.runtime/gpup/gpu3_fits_v1/jobs/mechanisms303-cache-v1.log','cache_admission':POOL_ROOT+'cache_admission.json',
   'storage_snapshot':POOL_ROOT+'worker_storage_snapshot.json','supervisor_launch':POOL_ROOT+'launch.json',
   'supervisor_source':POOL_ROOT+'supervisor.py','supervisor_exit':POOL_OUTPUT+'exit.json',
   'controller_started':POOL_OUTPUT+'controller_started.json','paired_shutdown':POOL_ROOT+'paired_restart_reason.json',
   'controls_source':'src/arggnn/biological_mechanism_controls_v1.py','gpu_pool_source':'src/arggnn/gpu_pool_v1.py'}),
 ANCHOR:dict(kind='abandoned_partial_CPU_matrix',allowance=40000000000,output=ANCHOR_OUTPUT.rstrip('/'),pids=[1386393,1386398],evidence={
   'matrix_launch':ANCHOR_ROOT+'launch.json','matrix_source':ANCHOR_ROOT+'run_matrix.py','matrix_started':ANCHOR_ROOT+'matrix_started.json',
   'matrix_failure':ANCHOR_ROOT+'matrix_failure.json','matrix_plan':ANCHOR_ROOT+'execution_plan.json',
   'cell_started':ANCHOR_ROOT+'cells/'+CELL+'.started.json','cell_exit':ANCHOR_ROOT+'cells/'+CELL+'.exit_observed.json',
   'cell_admission':ANCHOR_ROOT+'cells/'+CELL+'.admission.json','recovery':ANCHOR_ROOT+'resume_v3/recovery.json',
   'recovered_terminal':ANCHOR_ROOT+'resume_v3/recovered_first_cell.terminal.json','resume_plan':ANCHOR_ROOT+'resume_v3/execution_plan.json',
   'resume_source':ANCHOR_ROOT+'resume_v3/run_matrix.py','evidence':ANCHOR_OUTPUT+CELL+'/evidence.json',
   'compact_evidence':ANCHOR_OUTPUT+CELL+'/compact_evidence.json'})}
SOURCE_PINS={
 'controls_source':'28f3e7d3f28a48eeee0944b2e4ad55e3ea6ebb314d8b0342bd785facb08dcbeb',
 'gpu_pool_source':'ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1',
 'supervisor_source':'6a3771fe56422102854decee46d483162433d87b89a04652d16524733eff660d',
 'matrix_source':'3ffee1fa5ac1dda7f70d5a192c4095d53bf83c73b3accc6233c24f72e1a23205',
 'resume_source':'d6507d6590e1fc78c93c7aa1dd3b2734a4c8df1eb14ebf32f9cfc32a8f0032ef'}
_CACHE=None


def _utc():return datetime.now(timezone.utc).isoformat()
def _ref(path):return budget.v4.reference(ROOT,path)
def _bound(ref):return budget.v4.bound(ROOT,ref,True)
def _pins(view):return previous._pins(view)
def _paths(own):return previous._paths(own)
def _ceiling(value,view):return previous._ceiling(value,view)
def _clone(method,**changes):return previous._clone(method,**changes)


def _frozen():
 budget.v4.bound(ROOT,{'path':previous.SOURCE,'sha256':PREVIOUS_SHA256})
 return previous._frozen()


def _authority(ref,operation,bindings,*,fresh):
 value=_bound(ref)
 required=dict(kind='project_budget_terminal_retired_root_admission_v1',root_authorizer='root_agent',active=True,
  operation=operation,accounting_mode=MODE,bindings=bindings,all_budget_writers_use_scheduler_mutex=True,
  all_consumers_and_launchers_use_terminal_retired_guard=True,
  all_terminal_success_writers_finished_before_retirement_activation=True,
  historical_records_mutation_authorized=False,global_rebase_authorized=False,scientific_execution_authorized=False,GPU_authorized=False)
 if any(budget.v4.encoded(value.get(k))!=budget.v4.encoded(v) for k,v in required.items()):
  raise ValueError('Exact retirement authority and completed old-writer barrier required')
 if operation=='retire_obsolete_scope' and any(value.get(k) is not True for k in (
  'scope_write_authority_revoked_for_all_cooperative_project_jobs','closed_scope_future_writes_and_relaunch_revoked',
  'all_original_and_resume_capabilities_for_retired_scope_revoked','external_NFS_writers_outside_observation')):
  raise ValueError('Explicit original/resume capability and write/relaunch revocation required')
 if operation=='reserve' and any(ref not in value['reservation']['consumer_bindings'] for ref in bindings.values()):
  raise ValueError('Every original and retirement source must be reserved')
 if fresh:
  expiry=datetime.fromisoformat(value['expires_at_utc'].replace('Z','+00:00'))
  if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc):raise ValueError('Expired retirement authority')
 return value


def _mixed_events(bindings):
 directory=budget.v4.canonical(ROOT,JOURNAL)
 if not directory.exists():return [],[],[],None
 if not directory.is_dir():raise ValueError('Canonical existing shared journal required')
 before=directory.lstat();names=sorted(p.name for p in directory.iterdir())
 if len(names)>4096:raise ValueError('Shared event count cap exceeded')
 success={k:v for k,v in bindings.items() if k!='terminal_retired_overlay'}
 terminal={k:v for k,v in success.items() if k!='terminal_success_overlay'}
 first,second,retired,tail,total=[],[],[],None,0
 for number,name in enumerate(names,1):
  if name!=f'{number:06d}.json':raise ValueError('Shared journal gap/unknown entry')
  raw,sha=budget.v4.file_bytes(ROOT,directory/name);total+=len(raw)
  if len(raw)>METADATA_CHARGE or total>32*1024*1024:raise ValueError('Shared metadata charge/cap exceeded')
  event=budget.v4.parse(raw)
  if (set(event)!={'schema','sequence','previous_sha256','bindings','operation','authority','value'}
   or event['schema']!='project_budget_terminal_overlay_event_v1' or type(event['sequence']) is not int
   or event['sequence']!=number or event['previous_sha256']!=tail):raise ValueError('Immutable shared chain drift')
  if event['bindings']==terminal:
   if second or retired or event['operation'] not in ('reserve','close_failed_cpu'):raise ValueError('Old terminal writer after successor activation')
   authority=old._authority(event['authority'],event['operation'],terminal,fresh=False);first.append(event)
  elif event['bindings']==success:
   if retired or event['operation'] not in ('reserve','close_completed_cpu_observed_absent'):raise ValueError('Old success writer after retirement activation')
   authority=previous._authority(event['authority'],event['operation'],success,fresh=False);second.append(event)
  else:
   if event['bindings']!=bindings or event['operation'] not in ('reserve','retire_obsolete_scope'):raise ValueError('Exact retirement event/bindings required')
   authority=_authority(event['authority'],event['operation'],bindings,fresh=False);retired.append(event)
  if authority.get('expected_before',{}).get('terminal_overlay_tail_sha256')!=tail:raise ValueError('Shared authority predecessor drift')
  tail=sha
 after=directory.lstat()
 if (before.st_dev,before.st_ino,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Shared journal changed during read')
 return first,second,retired,tail


def _mechanism(e,refs,own):
 registry=e['registry'];jobs=registry['jobs'];plan=e['pool_admission']['plan']
 rows=[row for row in jobs if row['logical_job']=='mechanisms303-cache-v1']
 admitted=[row for row in plan['jobs'] if row['spec']['job']=='mechanisms303-cache-v1']
 if len(rows)!=1 or len(admitted)!=1:raise ValueError('Exact original mechanism job required')
 job,item=rows[0],admitted[0]
 if (registry.get('pool')!='gpu3_fits_v1' or registry.get('lifecycle')!='closed' or registry.get('safety_error') is not None
  or registry.get('ambiguous_descendants_or_lease_holders')!=[] or any(row['state'] not in ('failed','succeeded','cancelled') for row in jobs)
  or job.get('state')!='failed' or type(job.get('exit_code')) is not int or job['exit_code']!=1 or job.get('pid')!=1313663
  or job.get('log_dropped_bytes')!=0 or job.get('log_bytes')!=len(e['worker_log'])
  or item['reservation']!=own or item['spec']!=job['requested'] or item['argv_sha256']!=budget.digest(job['argv'])
  or item['module_source']['sha256']!=job.get('module_source_sha256')
  or registry['controller']['pid']!=1313627 or e['controller_started']['identity']['pid']!=1313627
  or e['supervisor_launch'].get('supervisor_pid')!=1313616 or e['supervisor_exit'].get('status')!='failed_partial_preserved'
  or e['paired_shutdown']['shutdown_reply']['controller']['pid']!=1313627
  or not e['worker_log'].endswith(b'ValueError: Storage snapshot must prove a complete successful project-root scan\n')
  or b'storage = AccountedStorage.admit' not in e['worker_log'] or (ROOT/CASES[MECHANISM]['output']).exists()):
  raise ValueError('Bound failed-before-output worker and fully stopped original pool required')
 for ref in [item['module_source'],*item['worker_bindings']]:
  if ref not in own['consumer_bindings']:raise ValueError('Original pool/worker source bindings drift')
  budget.v4.bound(ROOT,ref)
 argv=job['argv']
 for name,flag,sha_flag in [('cache_admission','--cache-admission','--expected-cache-admission-sha256'),
  ('storage_snapshot','--storage-snapshot','--expected-storage-snapshot-sha256')]:
  if refs[name]!={'path':argv[argv.index(flag)+1],'sha256':argv[argv.index(sha_flag)+1]}:raise ValueError('Actual worker input/snapshot pins required')
 args=e['supervisor_launch']['argv']
 if (args[args.index('--authority-sha256')+1]!=refs['pool_admission']['sha256']
  or args[args.index('--supervisor-sha256')+1]!=refs['supervisor_source']['sha256']):raise ValueError('Original supervisor command pins required')
 pids={e['supervisor_launch']['supervisor_pid'],registry['controller']['pid']}
 for row in jobs:
  if row.get('pid'):pids.add(row['pid'])
  pids.update(record['pid'] for record in row.get('process_identities',[]))
 if sorted(pids)!=CASES[MECHANISM]['pids']:raise ValueError('Original recorded owned pool PID set drift')
 return dict(status='retired_failed_before_output',scientific_completion_claimed=False,known_worker_exit_code=1,
  supervisor_exit_status_known=False,preserved_successful_cell=None,preserved_artifact_identities={},output_absent=True)


def _anchor(e,refs,own):
 job=ANCHOR+'/'+CELL
 if (e['matrix_launch'].get('pid')!=1386393 or e['matrix_started']['supervisor']['pid']!=1386393
  or e['matrix_started']['plan']!=refs['matrix_plan'] or e['matrix_started']['source_sha256']!=refs['matrix_source']['sha256']
  or e['matrix_plan'].get('reservation')!=own or e['resume_plan'].get('reservation')!=own
  or e['matrix_failure'].get('status')!='failed_partial_no_fallback' or e['matrix_failure'].get('cell')!=CELL
  or e['cell_started']['child']['pid']!=1386398 or e['cell_exit']['child']!=e['cell_started']['child']
  or type(e['cell_exit'].get('returncode')) is not int or e['cell_exit']['returncode']!=0 or e['cell_exit'].get('watchdog_reason') is not None
  or e['cell_started']['admission']!=refs['cell_admission'] or e['cell_admission'].get('matrix_job_id')!=ANCHOR
  or e['cell_admission'].get('job_id')!=job or e['recovery'].get('kind')!='root_metadata_only_resume_of_completed_full_support_cell'
  or e['recovery']['actual_exit']!=refs['cell_exit'] or e['recovery']['original_matrix_failure']!=refs['matrix_failure']
  or e['recovery']['evidence']!=refs['evidence'] or e['recovery']['terminal']!=refs['recovered_terminal']
  or e['recovered_terminal'].get('status')!='complete' or e['recovered_terminal'].get('returncode')!=0
  or e['recovered_terminal'].get('job_id')!=job or e['recovered_terminal']['evidence_ref']!=refs['evidence']
  or e['recovered_terminal']['admission_ref']!=refs['cell_admission']
  or e['evidence'].get('status')!='completed_full_support_observed_only_CPU_synthetic_numerical_cell'
  or e['evidence'].get('job_id')!=job or e['evidence']['admission_ref']!=refs['cell_admission'] or e['evidence'].get('GPU_used') is not False
  or e['compact_evidence'].get('status')!=e['evidence']['status'] or e['compact_evidence'].get('job_id')!=job
  or set(e['resume_plan'].get('resume_completed',{}))!={CELL}):raise ValueError('Stopped partial matrix and actual recovered successful cell required')
 for name in ('matrix_started.json','matrix_failure.json','matrix_terminal.json','cells','launch.json'):
  if (ROOT/ANCHOR_ROOT/'resume_v3'/name).exists():raise ValueError('Resume has execution metadata; this narrow abandoned case is insufficient')
 artifacts=e['recovery']['artifacts'];identities={}
 expected={'manual_original':ANCHOR_OUTPUT+CELL+'/manual_original.pt','matched_parent_v3':ANCHOR_OUTPUT+CELL+'/matched_parent_v3.pt'}
 if set(artifacts)!=set(expected):raise ValueError('Exact recovered successful artifacts required')
 for name,path in expected.items():
  if artifacts[name]['path']!=path or e['resume_plan']['resume_completed'][CELL][name]!=artifacts[name]:raise ValueError('Recovered artifact metadata drift')
  p=budget.v4.canonical(ROOT,path);s=p.lstat()  # Targeted identity only; no artifact payload read.
  if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_size!=1072691832:raise ValueError('Preserve both actual successful-cell artifacts')
  identities[path]=[s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
 return dict(status='retired_abandoned_partial_CPU_matrix',scientific_completion_claimed=False,known_worker_exit_code=0,
  supervisor_exit_status_known=False,preserved_successful_cell=CELL,preserved_artifact_identities=identities,output_absent=False)


def _retirement(request,own,*,observe=False):
 if set(request)!={'key','retirement_kind','evidence_refs','observation_ref'} or request['key'] not in CASES:
  raise ValueError('Only two exact typed obsolete scopes may retire')
 case=CASES[request['key']];refs=request['evidence_refs']
 if (request['key']!=own['key'] or request['retirement_kind']!=case['kind'] or own['reserved_bytes']!=case['allowance']
  or own['output']!=str(ROOT/case['output']) or own['auxiliary_scopes']!=[] or set(refs)!=set(case['evidence'])
  or any(refs[name]['path']!=path for name,path in case['evidence'].items())):raise ValueError('Exact original reservation/evidence closure required')
 _paths(own)  # Unchanged validation of every original root/source/config pin.
 evidence={}
 for name,ref in refs.items():
  if name in SOURCE_PINS and ref['sha256']!=SOURCE_PINS[name]:raise ValueError('Exact original proof source pin required')
  raw=budget.v4.bound(ROOT,ref)
  evidence[name]=raw if name in SOURCE_PINS or name=='worker_log' else budget.v4.parse(raw)
 proof=(_mechanism if request['key']==MECHANISM else _anchor)(evidence,refs,own)
 terminal=_bound(request['observation_ref'])
 fields={'kind','root_authorizer','key','retirement_kind','evidence_refs','pids','observations','all_original_and_resume_capabilities_revoked'}
 if (set(terminal)!=fields or terminal['kind']!='project_budget_obsolete_scope_terminal_observation_v1'
  or terminal['root_authorizer']!='root_agent' or terminal['key']!=request['key'] or terminal['retirement_kind']!=case['kind']
  or terminal['evidence_refs']!=refs or terminal['pids']!=case['pids'] or len(terminal['observations'])!=len(case['pids'])
  or terminal['all_original_and_resume_capabilities_revoked'] is not True):raise ValueError('Exact root stopped-PID/revoked-capability receipt required')
 for pid,row in zip(case['pids'],terminal['observations']):old._observation(row,pid)
 proof['PID_observations']=[_frozen()['_absent'](pid) for pid in case['pids']] if observe else terminal['observations']
 return proof


def _state_locked(source_sha,launcher_sha,*,extra=(),fresh=False):
 global _CACHE
 cpu=_frozen()
 bindings=dict(implementation={'path':budget.SOURCE_REL,'sha256':source_sha},launcher={'path':budget.LAUNCHER_REL,'sha256':launcher_sha},
  cpu_overlay={'path':old.CPU_SOURCE,'sha256':old.CPU_SHA256},terminal_overlay={'path':old.SOURCE,'sha256':previous.OLD_SHA256},
  terminal_success_overlay={'path':previous.SOURCE,'sha256':PREVIOUS_SHA256},terminal_retired_overlay=_ref(SOURCE))
 deps=_CACHE[3] if _CACHE is not None else ();watched=sorted(set(extra)|set(deps)|{SOURCE,previous.SOURCE,old.SOURCE,JOURNAL})
 before=cpu['_fingerprint'](watched)
 if not fresh and _CACHE is not None and _CACHE[0]==bindings and _CACHE[1]==before:return copy.deepcopy(_CACHE[2])
 first,second,suffix,tail=_mixed_events(bindings);prefix=first+second
 prefix_tail=budget.v4.file_bytes(ROOT,ROOT/JOURNAL/f'{len(prefix):06d}.json')[1] if prefix else None
 base_reader=_clone(previous._state_locked,_CACHE=None,_mixed_events=lambda bindings:(first,second,prefix_tail))
 reservations,closed,base,_,_=base_reader(source_sha,launcher_sha,extra=extra,fresh=fresh)
 reservations=copy.deepcopy(reservations);closed=set(closed);legacy=budget.v4.read_json(ROOT,budget.v4.LEGACY)[0]
 occupied=[budget.v4.canonical(ROOT,row['output']) for row in legacy['jobs'].values()]
 for own in reservations.values():occupied.extend(_paths(own))
 discovered=set(base_reader.__globals__['_CACHE'][3])|set(extra)|{SOURCE,previous.SOURCE,old.SOURCE,JOURNAL}
 def collect(value):
  if isinstance(value,dict):
   if set(value)=={'path','sha256'}:discovered.add(value['path'])
   for child in value.values():collect(child)
  elif isinstance(value,(list,tuple)):
   for child in value:collect(child)
 credit=added=0
 for event in suffix:
  authority=_authority(event['authority'],event['operation'],bindings,fresh=False);value=event['value'];collect(event);collect(authority)
  if event['operation']=='reserve':
   request=authority['reservation'];paths=_paths(value);key=value['key']
   if (value!={**request,'root_admission':event['authority']} or len(reservations)>=4096 or key in reservations or key in legacy['jobs']
    or any(budget.v4.paths_overlap(a,b) for a in paths for b in occupied)):raise ValueError('Retirement-mode reservation authority/key/scope collision')
   reservations[key]=value;occupied.extend(paths);added+=value['reserved_bytes']
  else:
   request=authority['retirement'];key=request['key'];own=reservations.get(key)
   if (own is None or key in closed or set(value)!={'request','reservation_sha256','measurement','retirement_proof','retired_utc','future_writes_and_relaunch_revoked'}
    or value['request']!=request or value['reservation_sha256']!=budget.digest(own) or value['future_writes_and_relaunch_revoked'] is not True):raise ValueError('Unknown/duplicate retirement or certificate drift')
   proof=_retirement(request,own);recorded=value['retirement_proof']
   if set(recorded)!=set(proof) or any(recorded[k]!=proof[k] for k in proof if k!='PID_observations'):raise ValueError('Honest retirement/preserved artifact proof drift')
   if len(recorded['PID_observations'])!=len(proof['PID_observations']):raise ValueError('Retirement PID proof count drift')
   for current,root in zip(recorded['PID_observations'],proof['PID_observations']):
    if old._observation(current,root['pid'])<old._observation(root,root['pid']):raise ValueError('Applied PID proof predates root observation')
   measurement=value['measurement']
   if (set(measurement)!={'conservative_scope_bytes','entries','scope_paths','stat_only'} or measurement['stat_only'] is not True
    or measurement['scope_paths']!=[str(p) for p in _paths(own)]):raise ValueError('Exact one owned-scope stat certificate required')
   actual=budget.v4.integer(measurement['conservative_scope_bytes'],'retired retained bytes');budget.v4.integer(measurement['entries'],'retired retained entries')
   if actual>own['reserved_bytes'] or (proof['output_absent'] and (actual!=0 or measurement['entries']!=0)):raise ValueError('Retired retained allowance/absence drift')
   credit+=own['reserved_bytes']-actual;closed.add(key)
   for ref in request['evidence_refs'].values():
    collect(ref)
    if Path(ref['path']).suffix=='.json':collect(_bound(ref))
   collect(_bound(request['observation_ref']))
 projected=base['projected_bytes']+added-credit+len(suffix)*METADATA_CHARGE;_ceiling(projected,base)
 view=dict(base,original_terminal_success_projected_bytes=base['projected_bytes'],projected_bytes=projected,
  terminal_retired_allowance_credit_bytes=credit,terminal_retired_reserved_bytes=added,terminal_retired_metadata_charge_bytes=len(suffix)*METADATA_CHARGE,
  terminal_overlay_tail_sha256=tail,terminal_overlay_events=len(prefix)+len(suffix),terminal_retired_events=len(suffix),accounting_mode=MODE)
 result=reservations,closed,view,bindings,prefix+suffix
 if before!=cpu['_fingerprint'](watched):raise ValueError('Shared retirement namespace changed during replay')
 _CACHE=bindings,cpu['_fingerprint'](sorted(discovered)),copy.deepcopy(result),tuple(sorted(discovered));return result


def inspect_v1(*,expected_source_sha256,expected_launcher_sha256):
 with budget.v4.mutex(ROOT):return _state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)[2]


def _append(operation,authority_ref,value,state):return old._append(operation,authority_ref,value,state)


def apply_retire_obsolete_scope_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
 with budget.v4.mutex(ROOT):
  state=_state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True);reservations,closed,view,bindings,_=state
  authority=_authority(authority_ref,'retire_obsolete_scope',bindings,fresh=True)
  if authority['expected_before']!=_pins(view):raise ValueError('Fresh shared-journal retirement pins required')
  request=authority['retirement'];key=request['key']
  if key not in reservations or key in closed:raise ValueError('Unknown/already closed obsolete reservation')
  own=reservations[key];proof=_retirement(request,own,observe=True)
  measurement=budget.v4.measure_scopes(ROOT,_paths(own))  # Exactly one actual full owned-scope inventory.
  for row in proof['PID_observations']:_frozen()['_absent'](row['pid'])
  actual=measurement['conservative_scope_bytes']
  if actual>own['reserved_bytes'] or (proof['output_absent'] and (actual!=0 or measurement['entries']!=0)):raise ValueError('Retired output allowance/absence drift')
  projected=view['projected_bytes']-(own['reserved_bytes']-actual)+METADATA_CHARGE;_ceiling(projected,view)
  value=dict(request=copy.deepcopy(request),reservation_sha256=budget.digest(own),measurement=measurement,
   retirement_proof=proof,retired_utc=_utc(),future_writes_and_relaunch_revoked=True)
  tail=_append('retire_obsolete_scope',authority_ref,value,state)
  return dict(kind='project_budget_obsolete_scope_retirement_receipt_v1',key=key,value=value,retained_bytes=actual,
   allowance_bytes=own['reserved_bytes'],credit_bytes=own['reserved_bytes']-actual,projected_bytes=projected,
   terminal_overlay_tail_sha256=tail,scope_measurements=1,scientific_completion_claimed=False,
   original_records_mutated=False,global_process_scan_performed=False)


def _adapt(name):
 return _clone(_frozen()[name],SOURCE=SOURCE,MODE=MODE,JOURNAL=JOURNAL,_state_locked=_state_locked,_pins=_pins,
  _paths=_paths,_authority=_authority,_append=_append,budget_check_v9=budget_check_v9,check_storage_locked_v1=check_storage_locked_v1)


def apply_reserve_v1(authority_ref,*,expected_source_sha256,expected_launcher_sha256):
 value=_adapt('apply_reserve_v1')(authority_ref,expected_source_sha256=expected_source_sha256,expected_launcher_sha256=expected_launcher_sha256)
 value['kind']='project_budget_terminal_retired_reserve_receipt_v1';value['terminal_overlay_tail_sha256']=value.pop('cpu_overlay_tail_sha256');return value


def check_storage_locked_v1(storage,*,first=False,boundary=False,required_write_bytes=0):
 return _adapt('check_storage_locked_v1')(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)


def budget_check_v9(storage,*,first=False,boundary=False,required_write_bytes=0):
 with budget.v4.mutex(ROOT):return check_storage_locked_v1(storage,first=first,boundary=boundary,required_write_bytes=required_write_bytes)


def validate_storage_v9(spec,cfg,storage,*,source_ref,config_ref,runtime_ref):
 return _adapt('validate_storage_v9')(spec,cfg,storage,source_ref=source_ref,config_ref=config_ref,runtime_ref=runtime_ref)


class PilotOutputV9(previous.PilotOutputV9):
 __init__=_clone(previous.PilotOutputV9.__init__,budget_check_v9=budget_check_v9)
 save=_clone(previous.PilotOutputV9.save,budget_check_v9=budget_check_v9)
 discard=_clone(previous.PilotOutputV9.discard,budget_check_v9=budget_check_v9)


if __name__=='__main__':
 print(json.dumps(dict(status='source_only_retirement_not_applied',source=_ref(SOURCE),accounting_mode=MODE,journal=JOURNAL,exact_keys=list(CASES)),indent=2))
