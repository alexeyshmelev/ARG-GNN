"""Inactive ROOT-only H058v9 preparer. Default invocation has no writes/probes.
Actual preparation creates metadata, observes exact known old contexts and checks
GPU0 capacity. It never reserves, admits, launches, resets history or renews C3.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,importlib,json,os,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/H058_GPU0_v9_ROOT_preparation_source_20261017_v1'
SOURCE=HERE+'/prepare_ROOT_v1.py'
MANIFEST=HERE+'/source_manifest.v1.json'
CANDIDATE='research/reviews/H058_GPU0_full_training_successor_source_20261017_v9'
CANDIDATE_DRIVER_REF={'path':CANDIDATE+'/operational_v1.py','sha256':'f173a4d07457afc3bb386c6d9261ac16431891a24bd5135c1c209b6e2395e138'}
CANDIDATE_MANIFEST_REF={'path':CANDIDATE+'/source_manifest.v1.json','sha256':'196b662fcfb7ea2d6de57595b830c2e9109be14cbd861ee503bb6789af170004'}
CANDIDATE_REVIEW_REF={'path':'research/reviews/H058_GPU0_v9_whole_source_independent_20261017_v1/review.v1.json','sha256':'327c21d288272978268429c9e4df6d931d9784110d567aafe041ca113d41b87d'}
V7_TARGET_ORIGIN_REF={'path':'research/reviews/research_execution_ROOT_20261017_v27/H058_v7_actual_worker_failure_targeted_observation.root_v1.json','sha256':'66bfb102956286530b70f23991e11fc4a25ce5d3c306a259d911f4bd4b14d3c1'}
INSTANCE='root01_oct17'
SCOPES=dict(instance_id=INSTANCE,job_id='H058-GPU0-v9-fulltrain-'+INSTANCE,
 preparation_directory='research/reviews/H058_GPU0_v9_ROOT_preparation_'+INSTANCE,
 packet_directory='research/reviews/H058_GPU0_v9_ROOT_actual_'+INSTANCE,
 output_directory='experiments/results/H058_GPU0_v9_full_parent_'+INSTANCE)
DIRECTIVE='ROOT authorizes one fresh H058v9 full1920 unfinished-cell metadata preparation with a new27h interval, permanent old-route revocation and explicit truncated-v7-history acceptance; no reserve, admission or science by this helper'
P=None
def ref(p):
 b=(ROOT/p).read_bytes();return {'path':p,'sha256':hashlib.sha256(b).hexdigest()}

def read(r):
 b=(ROOT/r['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==r['sha256'];return json.loads(b)

def save(name,v):
 p=P/name
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
 return ref(str(p.relative_to(ROOT)))

def authority_v1(review_ref,root_directive_ref,c):
 """Bounded metadata checks. Caller must separately establish current authority."""
 review=read(review_ref)
 assert review['independent'] is True and review['verdict']=='SOURCE_PASS' and review['blockers']==[]
 assert review['source_ref']==ref(SOURCE) and review['source_manifest_ref']==ref(MANIFEST)
 assert review['candidate_source_ref']==CANDIDATE_DRIVER_REF and review['candidate_source_manifest_ref']==CANDIDATE_MANIFEST_REF
 assert review['candidate_whole_source_review_ref']==CANDIDATE_REVIEW_REF
 candidate=read(CANDIDATE_REVIEW_REF)
 assert candidate['independent'] is True and candidate['verdict']=='SOURCE_PASS' and candidate['blockers']==[]
 assert candidate['source_ref']==CANDIDATE_DRIVER_REF and candidate['source_manifest_ref']==CANDIDATE_MANIFEST_REF
 directive=read(root_directive_ref)
 assert directive['schema']=='ROOT_H058_v9_preparation_directive_v1' and directive['root_authorizer']=='root_agent' and directive['status']=='active'
 assert directive['ROOT_instruction']==DIRECTIVE and directive['prepare_once'] is True
 assert directive['preparation_source_ref']==ref(SOURCE) and directive['preparation_source_manifest_ref']==ref(MANIFEST)
 assert directive['preparation_review_ref']==review_ref and directive['candidate_whole_source_review_ref']==CANDIDATE_REVIEW_REF
 assert directive['candidate_source_ref']==CANDIDATE_DRIVER_REF and directive['candidate_source_manifest_ref']==CANDIDATE_MANIFEST_REF
 assert all(directive[k]==v for k,v in SCOPES.items())
 for key in ('all_prior_H058_launch_resume_write_routes_permanently_revoked','all_v7_launch_resume_write_routes_permanently_revoked',
  'remaining_historical_ownership_uncertainty_ROOT_accepted_for_fresh_scope','historical_files_scopes_full_charges_clocks_sources_failures_and_unknowns_preserved',
  'new_full_1920_update_unfinished_cell_authorized','new27h_interval_for_fresh_v9_only','all_old_native_launch_resume_write_routes_permanently_revoked',
  'additional_old_controller_submissions_revoked_while_H058_attached','old_controller_preserved_until_H058_drained'):
  assert directive[key] is True,key
 assert directive['historical_complete_v7_roster_claimed'] is False
 assert directive['controller_capability_renewal_authorized'] is False
 assert directive['reserve_admission_or_science_by_preparer_authorized'] is False
 assert directive['prior_v7_actual_outer_terminal_ref']==c.PRIOR_V7_ACTUAL_OUTER_REF
 assert directive['historical_ROOT_clocks_v8']==c.historical_clocks_v8()
 a=datetime.fromisoformat(directive['created_at_utc']);b=datetime.fromisoformat(directive['expires_at_utc'])
 assert a.tzinfo and b.tzinfo and (b-a).total_seconds()==27*3600
 assert all(directive[k]!=c.historical_clocks_v8()['v4'][k] for k in ('created_at_utc','expires_at_utc'))
 registration=read(directive['unfinished_cell_registration_ref'])
 assert registration['schema']=='ROOT_H058_v9_full1920_unfinished_cell_registration_v1'
 assert registration['root_authorizer']=='root_agent' and registration['status']=='registered'
 assert all(registration[k]==v for k,v in SCOPES.items())
 assert registration['hypothesis_id']=='H058' and registration['new_full_unfinished_cell'] is True
 assert registration['updates_all_new_arms']==1920 and registration['updates_per_arm']==960
 assert registration['TRAIN_parent_count']==32 and registration['fresh_AdamW_from_original_H045_own_state'] is True
 assert registration['no_old_CPU_or_GPU_optimizer_state_transfer'] is True and registration['repeat_reason']
 assert registration['candidate_source_ref']==CANDIDATE_DRIVER_REF and registration['candidate_source_manifest_ref']==CANDIDATE_MANIFEST_REF
 assert all(registration[k]==directive[k] for k in ('created_at_utc','expires_at_utc'))
 return directive

def context_v1(review_ref,root_directive_ref):
 ns=CANDIDATE.replace('/','.')
 # Authentication precedes imports and every actual resource observation.
 assert ref(CANDIDATE_DRIVER_REF['path'])==CANDIDATE_DRIVER_REF
 assert ref(CANDIDATE_MANIFEST_REF['path'])==CANDIDATE_MANIFEST_REF
 c=importlib.import_module(ns+'.common_v1');d=importlib.import_module(ns+'.direct_gpu0_v1')
 assert ref(c.DRIVER)==CANDIDATE_DRIVER_REF and ref(c.MANIFEST)==CANDIDATE_MANIFEST_REF
 assert c.HERE==d.HERE==CANDIDATE and d.SOURCE==c.DIRECT
 assert d.validate_native_resource_disposition_v1 is c.validate_native_resource_disposition_v1
 directive=authority_v1(review_ref,root_directive_ref,c);c.active(directive)
 q=read(ref(c.HERE+'/ROOT_request.INACTIVE.template.v1.json'));q.update(SCOPES)
 q.update(created_at_utc=directive['created_at_utc'],expires_at_utc=directive['expires_at_utc'],
  historical_ROOT_clocks_v8=c.historical_clocks_v8(),fit_clock_compatibility_v5=c.fit_clock_acceptance_v5(),
  fresh_scope_lineage_ref=root_directive_ref)
 assert not (ROOT/q['preparation_directory']).exists()
 assert not (ROOT/q['packet_directory']).exists() and not (ROOT/q['output_directory']).exists()
 c.require_work_window_v2(q)
 old=read(c.CLOCK_ROOT_REQUEST_V4);oldcfg=read(old['config_ref'])
 return ns,c,d,q,old,oldcfg,directive

def known_v7_targets_v1(c):
 """Exact retained history only; never manufacture an exhaustive old roster."""
 wait=read(c.PRIOR_V7_ACTUAL_WAIT_REF);outer=read(c.PRIOR_V7_ACTUAL_OUTER_REF);origin=read(V7_TARGET_ORIGIN_REF)
 assert outer['tool_session_id']==41764 and outer['actual_tool_exit_code']==241 and outer['actual_tool_terminal_consumed'] is True
 assert outer['actual_worker_returncode']==-15 and outer['actual_wait_ref']==c.PRIOR_V7_ACTUAL_WAIT_REF
 assert outer['completed_ref'] is None and outer['first_optimizer_update_verified'] is False
 assert wait['returncode']==-15 and wait['actual_OS_wait_status']==15 and wait['completed_ref'] is None
 owned=wait['ownership'];assert owned['complete_owned_roster_known'] is False and owned['owned_roster_truncated'] is True
 identities=owned['retained_owned_identities'];assert len(identities)==len({(r['pid'],r['start']) for r in identities})==64
 assert all(set(r)=={'pid','start'} and type(r['pid']) is int and type(r['start']) is int and r['pid']>0 and r['start']>0 for r in identities)
 seed=owned['actual_direct_identity_seed'];assert seed['pid']==4127469 and seed['start']==490203517 and seed['ppid']==4122785
 assert dict(pid=seed['pid'],start=seed['start']) in identities
 groups=[r['pgid'] for r in wait['group_observations']]
 assert groups==[4127469] and seed['group']==seed['session']==groups[0] and all(r['absent'] is True for r in wait['group_observations'])
 assert origin['actual_wait_ref']==c.PRIOR_V7_ACTUAL_WAIT_REF and origin['original_outer_tool_session_id']==41764
 assert {r['pid'] for r in origin['targeted_current_root_observations']}=={4122785,4127469}
 return copy.deepcopy(identities),groups,seed['ppid']

def observe_v7_v1(c,base):
 identities,groups,supervisor=known_v7_targets_v1(c);rows=[]
 for expected in identities:
  try:observed=base.proc_identity(expected['pid'])
  except (FileNotFoundError,ProcessLookupError):observed=None
  present=bool(observed and observed['start_ticks']==expected['start'])
  assert not present,'Known exact v7 identity still present, including zombie'
  rows.append(dict(expected=expected,observed=observed,exact_old_identity_present=present))
 try:observed=base.proc_identity(supervisor)
 except (FileNotFoundError,ProcessLookupError):observed=None
 assert observed is None,'Supervisor has no historical start identity; numeric PID absence required'
 group_rows=[]
 for pgid in groups:
  try:os.killpg(pgid,0)
  except ProcessLookupError:pass
  else:raise ValueError('Known v7 process group remains present')
  group_rows.append(dict(pgid=pgid,absent=True,method='targeted os.killpg(pgid,0) ESRCH'))
 return dict(identities=rows,groups=group_rows,supervisor=dict(pid=supervisor,original_start_identity_known=False,observed=None,numeric_PID_absent=True),
  known_AUX_writer_contexts_absent=True,
  known_AUX_writer_absence_basis='Exact64 retained PID/start identities, known group and numeric supervisor absent. This is limited to authenticated known contexts; no complete historical or global open-FD writer census is claimed.',
  historical_complete_roster_claimed=False)


def observe_v1(c,d,old):
 """Actual ROOT only. Separate IO phase, never called by source/RAM metadata tests."""
 base=d.base_v1()
 native=read(old['native_resource_disposition_ref'])
 oldabsence=read(native['fresh_targeted_native_absence_ref'])
 rows=[]
 for expected in native['native_process_identities']:
  try:observed=base.proc_identity(expected['pid'])
  except (FileNotFoundError,ProcessLookupError):observed=None
  live=bool(observed and observed['uid']==expected['uid'] and observed['start_ticks']==expected['start_ticks'] and observed['state'] not in ('Z','X'))
  assert not live
  rows.append(dict(expected=expected,observed=observed,exact_old_identity_live=live))
 groups=[]
 for pgid in native['native_process_groups']:
  try:os.killpg(pgid,0)
  except ProcessLookupError:pass
  else:raise ValueError('Original native process group remains present')
  groups.append(dict(pgid=pgid,absent=True,method='targeted os.killpg(pgid,0) ESRCH',checked_utc=datetime.now(timezone.utc).isoformat()))
 gpu=[r for r in base.real_inventory() if r['index']==0]
 assert len(gpu)==1 and gpu[0]['uuid']==d.GPU0_UUID
 assert not any(x['pid'] in {r['pid'] for r in native['native_process_identities']} for x in gpu[0]['compute'])
 observed_at=datetime.now(timezone.utc).isoformat()
 cap=read(d.CURRENT_CAP);controller_storage=read(cap['controller_storage_ref'])
 registry=json.loads((ROOT/controller_storage['output_directory']/'registry.json').read_bytes())
 v7=observe_v7_v1(c,base)
 assert d.held_writable_probe_v1(c.local('.runtime/locks/gpu_0.lock')) is True
 writable_probe=dict(path='.runtime/locks/gpu_0.lock',held=True,method='exact accepted direct_gpu0_v1.held_writable_probe_v1 writable r+ independent exclusion probe',borrowed_ownership_claimed=False)
 return dict(v7=v7,writable_probe=writable_probe,native=native,oldabsence=oldabsence,rows=rows,groups=groups,GPU0=gpu[0],observed_at_utc=observed_at,registry=registry),base

def metadata_v1(review_ref,root_directive_ref,c,d,q,old,oldcfg,observation,directive):
 """Complete original metadata construction with supplied observation IO result.

The ROOT wrapper always obtains this observation through observe_v1.
Independent review can execute this exact construction with RAM IO and fixtures.
No guard/process/GPU/accounting call occurs in this function.
"""
 global P
 P=ROOT/q['preparation_directory'];assert not P.exists()
 assert not (ROOT/q['packet_directory']).exists() and not (ROOT/q['output_directory']).exists()
 created=q['created_at_utc'];expires=q['expires_at_utc']
 assert all(q[k]==directive[k] for k in ('created_at_utc','expires_at_utc'))
 assert q['fit_clock_compatibility_v5']==c.fit_clock_acceptance_v5()
 assert observation['writable_probe']['held'] is True
 identities,groups,supervisor=known_v7_targets_v1(c);v7=observation['v7']
 assert [r['expected'] for r in v7['identities']]==identities and all(r['exact_old_identity_present'] is False for r in v7['identities'])
 assert [r['pgid'] for r in v7['groups']]==groups and all(r['absent'] is True for r in v7['groups'])
 assert v7['supervisor']==dict(pid=supervisor,original_start_identity_known=False,observed=None,numeric_PID_absent=True)
 assert v7['known_AUX_writer_contexts_absent'] is True and v7['historical_complete_roster_claimed'] is False
 P.mkdir()
 handoff=read(q['science_handoff_ref'])
 cfg=copy.deepcopy(oldcfg)
 cfg['producer_ref']=handoff['producer_ref'];cfg['source_manifest_ref']=handoff['source_manifest_ref']
 for k,name in [('runtime_equivalence_ref','runtime_equivalence.root_active_v1.json'),('layout_resource_proof_ref','layout_resource_proof.root_active_v1.json')]:
  cfg[k]=save(name,read(oldcfg[k]))
 full_directive=read(oldcfg['full_training_directive_ref'])
 full_directive.update(GPU_entry_ref=cfg['producer_ref'],created_at_utc=created,expires_at_utc=expires,new_finite_v8_interval_authorized=True,
  historical_ROOT_clocks_v8=c.historical_clocks_v8(),historical_files_scopes_charges_clocks_failures_and_unknowns_preserved=True,
  ROOT_preparation_directive_ref=root_directive_ref,unfinished_cell_registration_ref=directive['unfinished_cell_registration_ref'])
 cfg['full_training_directive_ref']=save('full_training_directive.root_active_v1.json',full_directive)
 q['config_ref']=save('scientific_config.root_active_v1.json',cfg)
 absence_v7=read(ref(c.HERE+'/v7_targeted_absence.INACTIVE.template.v1.json'))
 absence_v7.update(status='active',actual_outer_terminal_ref=c.PRIOR_V7_ACTUAL_OUTER_REF,actual_wait_ref=c.PRIOR_V7_ACTUAL_WAIT_REF,
  actual_supervisor_terminal_consumed=True,current_known_v7_identities_groups_and_aux_writers_absent=True,
  remaining_historical_ownership_uncertainty_ROOT_accepted_for_fresh_scope=directive['remaining_historical_ownership_uncertainty_ROOT_accepted_for_fresh_scope'],
  all_v7_launch_resume_write_routes_permanently_revoked=directive['all_v7_launch_resume_write_routes_permanently_revoked'],
  actual_targeted_observations=copy.deepcopy(v7),observed_at_utc=observation['observed_at_utc'],
  ROOT_preparation_directive_ref=root_directive_ref,known_target_origin_ref=V7_TARGET_ORIGIN_REF)
 absence_v7_ref=save('v7_targeted_absence.root_active_v1.json',absence_v7)
 prior=read(old['prior_H058_CPU_disposition_ref'])
 prior.update(new_gpu_request_job_id=q['job_id'],new_full_GPU_budget_explicitly_authorized=True,
  old_launch_resume_write_routes_revoked=True,old_files_charges_clocks_and_counters_preserved=True,
  prior_GPU_v4_actual_wait_ref=c.PRIOR_V4_ACTUAL_WAIT_REF,
  prior_GPU_v4_actual_outer_terminal_ref=c.PRIOR_V4_ACTUAL_OUTER_REF,
  prior_GPU_v4_files_charges_clocks_and_failure_preserved=True,
  prior_GPU_v5_actual_wait_ref=c.PRIOR_V5_ACTUAL_WAIT_REF,
  prior_GPU_v5_actual_outer_terminal_ref=c.PRIOR_V5_ACTUAL_OUTER_REF,
  prior_GPU_v5_files_charges_clocks_sources_and_failure_preserved=True,
  prior_GPU_v5_future_launch_resume_write_routes_permanently_revoked=True,
  prior_GPU_v5_actual_optimizer_steps={'same_marker_aligned':None,'same_marker_alignment_null':None},
  prior_GPU_v5_final_optimizer_totals_status='UNKNOWN_no_new_update_verified',
  no_CPU_or_GPU_optimizer_state_transfer=True,new_GPU_budget=1920,new_GPU_budget_per_arm=960,
  created_at_utc=created,expires_at_utc=expires,
  repeat_reason='Superseded below by exact preserved v7 failure and fresh v9 unfinished-cell lineage.')
 prior.update(prior_GPU_v7_actual_wait_ref=c.PRIOR_V7_ACTUAL_WAIT_REF,prior_GPU_v7_actual_wait_minimal_ref=c.PRIOR_V7_MINIMAL_WAIT_REF,
  prior_GPU_v7_log_ref=c.PRIOR_V7_LOG_REF,prior_GPU_v7_input_ref=c.PRIOR_V7_INPUT_REF,prior_GPU_v7_actual_outer_terminal_ref=c.PRIOR_V7_ACTUAL_OUTER_REF,
  prior_GPU_v7_files_scopes_full_charges_clocks_sources_failures_and_unknowns_preserved=True,
  prior_GPU_v7_future_launch_resume_write_routes_permanently_revoked=directive['all_v7_launch_resume_write_routes_permanently_revoked'],
  prior_GPU_v7_actual_optimizer_steps={'same_marker_aligned':None,'same_marker_alignment_null':None},
  prior_GPU_v7_final_optimizer_totals_status='UNKNOWN_no_new_update_verified',prior_GPU_v7_verified_optimizer_steps=0,
  fresh_targeted_v7_absence_ref=absence_v7_ref,historical_ROOT_clocks_v8=c.historical_clocks_v8(),actual_prior_clocks_match_immutable_source_lineage=True,
  new_full_1920_update_unfinished_cell_registration=True,unfinished_cell_registration_ref=directive['unfinished_cell_registration_ref'],
  repeat_reason='Genuine v7 tool41764 outer241, worker-15/wait15, no completion; truncated historical roster and UNKNOWN actual optimizer totals preserved. ROOT separately registers a fresh v9 full1920 unfinished cell with a new27h interval, all32 original TRAIN parents, original H045 warmstart and fresh separate AdamW. No old state transfer, replay, retry, old clock reset or charge mutation.')
 q['prior_H058_CPU_disposition_ref']=save('prior_CPU_and_GPU_disposition.root_active_v1.json',prior)
 native=copy.deepcopy(observation['native']);absence=copy.deepcopy(observation['oldabsence'])
 absence.update(actual_identity_observations=copy.deepcopy(observation['rows']),
  actual_process_group_observations=copy.deepcopy(observation['groups']),
  actual_GPU0_inventory=copy.deepcopy(observation['GPU0']),observed_at_utc=observation['observed_at_utc'])
 absence_ref=save('native_targeted_absence.root_active_v1.json',absence)
 native.update(adapter_manifest_ref=ref(c.MANIFEST),H058_job_id=q['job_id'],
  fresh_targeted_native_absence_ref=absence_ref,observed_at_utc=observation['observed_at_utc'],
  created_at_utc=created,expires_at_utc=expires)
 q['native_resource_disposition_ref']=save('native_resource_disposition.root_active_v1.json',native)
 registry_ref=save('controller_registry.actual_snapshot.v1.json',observation['registry'])
 terminal=read(old['controller_terminal_qualification_ref']);terminal.update(registry_snapshot_ref=registry_ref,created_at_utc=created,expires_at_utc=expires)
 q['controller_terminal_qualification_ref']=save('controller_terminal.root_active_v1.json',terminal)
 authority=read(old['direct_attachment_authority_ref'])
 authority.update(H058_job_id=q['job_id'],native_resource_disposition_ref=q['native_resource_disposition_ref'],created_at_utc=created,expires_at_utc=expires)
 q['direct_attachment_authority_ref']=save('direct_attachment_authority.root_active_v1.json',authority)
 q.update(status='active',execution_authorized=True,source_review_passed=True,source_review_ref=CANDIDATE_REVIEW_REF,
  ROOT_preparation_source_review_ref=review_ref,ROOT_preparation_directive_ref=root_directive_ref)
 q['fit_clock_compatibility_v5']['accepted']=True
 request_ref=save('ROOT_request.root_active_v1.json',q)
 c.validate_request_v1(request_ref)
 return request_ref,q,absence_ref,absence_v7_ref

def prepare_ROOT_v1(review_ref,root_directive_ref):
 ns,c,d,q,old,oldcfg,directive=context_v1(review_ref,root_directive_ref)
 observation,base=observe_v1(c,d,old)
 request_ref,q,absence_ref,absence_v7_ref=metadata_v1(review_ref,root_directive_ref,c,d,q,old,oldcfg,observation,directive)
 capacity=d.capacity_v1(q,base)
 summary_ref=save('ROOT_preparation_summary.root_v1.json',dict(
  schema='ROOT_H058_GPU0_v9_actual_preparation_v1',root_authorizer='root_agent',
  actual_prepared_utc=datetime.now(timezone.utc).isoformat(),request_ref=request_ref,
  candidate_source_review_ref=CANDIDATE_REVIEW_REF,preparation_source_review_ref=review_ref,
  ROOT_preparation_directive_ref=root_directive_ref,unfinished_cell_registration_ref=directive['unfinished_cell_registration_ref'],
  ROOT_preparation_source_ref=ref(SOURCE),ROOT_preparation_source_manifest_ref=ref(MANIFEST),
  actual_native_absence_ref=absence_ref,actual_known_v7_absence_ref=absence_v7_ref,
  actual_writable_exclusion_probe=observation['writable_probe'],actual_GPU0_capacity=capacity,
  metadata_preflight='PASS',no_reserve_admission_or_science=True,
  created_at_utc=q['created_at_utc'],expires_at_utc=q['expires_at_utc'],historical_ROOT_clocks_v8=c.historical_clocks_v8(),
  original_files_scopes_full_charges_clocks_sources_failures_and_unknowns_preserved=True,
  new_full1920_unfinished_cell_authorized=True,new27h_interval_for_exact_fresh_v9_only=True,
  prior_v7_actual_final_optimizer_totals='UNKNOWN_no_new_update_verified'))
 return dict(request_ref=request_ref,summary_ref=summary_ref,no_reserve_admission_or_science=True,
  admit_argv=[str(ROOT/'.runtime/env/bin/python'),'-B','-s','-m',ns+'.operational_v1',
   '--admit','--request-ref',json.dumps(request_ref,separators=(',',':'))])
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--prepare-root',action='store_true');p.add_argument('--review-ref');p.add_argument('--root-directive');p.add_argument('--root-directive-ref')
 a=p.parse_args(argv)
 if not a.prepare_root:
  print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE',ROOT_metadata_created=False,resource_probes=False,admission=False,execution=False)));return 0
 assert a.root_directive==DIRECTIVE and a.review_ref and a.root_directive_ref,'Separate explicit ROOT instruction, directive/ref and complete helper/source-closure review required'
 print(json.dumps(prepare_ROOT_v1(json.loads(a.review_ref),json.loads(a.root_directive_ref)),sort_keys=True),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())

