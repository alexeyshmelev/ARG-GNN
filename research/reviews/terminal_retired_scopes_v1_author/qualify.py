"""Read-only real proof metadata; all root observations/events/transitions are MOCKS."""
import contextlib,copy,json
from types import SimpleNamespace
from arggnn import project_storage_terminal_retired_scopes_v1 as m


def main():
 fixtures={};checks=[]
 def bound(ref):return copy.deepcopy(fixtures[ref['path']]) if ref['path'] in fixtures else m._bound(ref)
 proof_fn=m._clone(m._retirement,_bound=bound);authority_fn=m._clone(m._authority,_bound=bound)
 core,launcher=m._ref(m.budget.SOURCE_REL),m._ref(m.budget.LAUNCHER_REL)
 with m.budget.v4.mutex(m.ROOT):state=m._state_locked(core['sha256'],launcher['sha256'],fresh=True)
 first,second,existing,tail=m._mixed_events(state[3]);assert not existing
 suffix=[];retained={m.MECHANISM:0,m.ANCHOR:2150000000}
 for index,(key,case) in enumerate(m.CASES.items()):
  req=dict(key=key,retirement_kind=case['kind'],evidence_refs={name:m._ref(path) for name,path in case['evidence'].items()},
   observation_ref={'path':'research/reviews/terminal_retired_scopes_v1_author/mock_'+case['kind']+'.json','sha256':'0'*64})
  obs=dict(kind='project_budget_obsolete_scope_terminal_observation_v1',root_authorizer='root_agent',key=key,
   retirement_kind=case['kind'],evidence_refs=req['evidence_refs'],pids=case['pids'],observations=[dict(pid=pid,absent=True,
   method='targeted os.kill(pid,0) ESRCH',checked_utc=m._utc()) for pid in case['pids']],all_original_and_resume_capabilities_revoked=True)
  fixtures[req['observation_ref']['path']]=obs;own=state[0][key];proof=proof_fn(req,own)
  assert proof['scientific_completion_claimed'] is False
  assert len(proof['preserved_artifact_identities'])==(2 if key==m.ANCHOR else 0)
  checks.append(case['kind']+'_real_metadata_proof_with_mock_root_absence')
  obs['all_original_and_resume_capabilities_revoked']=False
  try:proof_fn(req,own)
  except ValueError:checks.append(case['kind']+'_unrevoked_capabilities_rejected')
  else:raise AssertionError('Unrevoked capabilities accepted')
  obs['all_original_and_resume_capabilities_revoked']=True
  auth_ref={'path':f'research/reviews/terminal_retired_scopes_v1_author/mock_authority_{index}.json','sha256':'0'*64}
  authority=dict(kind='project_budget_terminal_retired_root_admission_v1',root_authorizer='root_agent',active=True,
   operation='retire_obsolete_scope',accounting_mode=m.MODE,bindings=state[3],retirement=req,expected_before=m._pins(state[2]),
   all_budget_writers_use_scheduler_mutex=True,all_consumers_and_launchers_use_terminal_retired_guard=True,
   all_terminal_success_writers_finished_before_retirement_activation=True,historical_records_mutation_authorized=False,
   global_rebase_authorized=False,scientific_execution_authorized=False,GPU_authorized=False,
   scope_write_authority_revoked_for_all_cooperative_project_jobs=True,closed_scope_future_writes_and_relaunch_revoked=True,
   all_original_and_resume_capabilities_for_retired_scope_revoked=True,external_NFS_writers_outside_observation=True)
  fixtures[auth_ref['path']]=authority
  authority['all_terminal_success_writers_finished_before_retirement_activation']=False
  try:authority_fn(auth_ref,'retire_obsolete_scope',state[3],fresh=False)
  except ValueError:checks.append('unfinished_old_writer_barrier_rejected')
  else:raise AssertionError('Premature barrier accepted')
  authority['all_terminal_success_writers_finished_before_retirement_activation']=True
  measure=dict(conservative_scope_bytes=retained[key],entries=6 if key==m.ANCHOR else 0,scope_paths=[str(p) for p in m._paths(own)],stat_only=True)
  certificate=dict(request=req,reservation_sha256=m.budget.digest(own),measurement=measure,retirement_proof=proof,
   retired_utc=m._utc(),future_writes_and_relaunch_revoked=True)
  suffix.append(dict(operation='retire_obsolete_scope',authority=auth_ref,value=certificate))
  calls=dict(measure=0,append=0)
  def measurement(root,paths):calls['measure']+=1;return measure
  def append(*args):calls['append']+=1;return 'f'*64
  applying=m._clone(m.apply_retire_obsolete_scope_v1,
   budget=SimpleNamespace(v4=SimpleNamespace(mutex=lambda root:contextlib.nullcontext(),measure_scopes=measurement),digest=m.budget.digest),
   _state_locked=lambda *args,**kwargs:state,_authority=lambda *args,**kwargs:authority,
   _retirement=lambda req,own,**kwargs:proof_fn(req,own),_append=append,_frozen=lambda:{'_absent':lambda pid:None})
  receipt=applying(auth_ref,expected_source_sha256=core['sha256'],expected_launcher_sha256=launcher['sha256'])
  assert calls==dict(measure=1,append=1) and receipt['credit_bytes']==case['allowance']-retained[key]
  checks.append(case['kind']+'_mock_transition_one_scope_inventory')
 replay=m._clone(m._state_locked,_CACHE=None,_mixed_events=lambda bindings:(first,second,suffix,'f'*64),_bound=bound,
  _authority=authority_fn,_retirement=proof_fn)
 result=replay(core['sha256'],launcher['sha256'],fresh=True)
 credit=sum(m.CASES[key]['allowance']-value for key,value in retained.items())
 assert result[2]['projected_bytes']==state[2]['projected_bytes']-credit+2*m.METADATA_CHARGE
 assert set(m.CASES)<=result[1]
 checks.append('two_mock_retirements_replay_exact_R_to_A_arithmetic')
 suffix[0]['value']['retirement_proof']['scientific_completion_claimed']=True
 try:replay(core['sha256'],launcher['sha256'],fresh=True)
 except ValueError:checks.append('fabricated_scientific_completion_rejected')
 else:raise AssertionError('Fabricated completion accepted')
 suffix[0]['value']['retirement_proof']['scientific_completion_claimed']=False;suffix.append(suffix[0])
 try:replay(core['sha256'],launcher['sha256'],fresh=True)
 except ValueError:checks.append('duplicate_retirement_rejected')
 else:raise AssertionError('Duplicate retirement accepted')
 print(json.dumps(dict(status='PASS_actual_metadata_and_mock_retirement_replay',source=m._ref(m.SOURCE),checks=checks,
  actual_projected_bytes=state[2]['projected_bytes'],shared_prefix_events=state[2]['terminal_overlay_events'],actual_retirement_events=0,
  actual_apply_performed=False,actual_PID_observations=0,actual_scope_measurements=0,artifact_payloads_read=False,
  targeted_artifact_identity_stats_only=True,mock_records_persisted=False),indent=2))


if __name__=='__main__':main()
