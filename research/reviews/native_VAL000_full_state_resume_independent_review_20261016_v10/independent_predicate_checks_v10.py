"""Independent actual v10 predicate tests with synthetic resource endpoints."""
from pathlib import Path
import copy,hashlib,importlib,json,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v10'
PKG='research.reviews.native_VAL000_full_state_resume_source_20261016_v10'
c=importlib.import_module(PKG+'.amendment_contract_v1');d=importlib.import_module(PKG+'.direct_gpu0_v1')
audit=c.read(c.V9_CONTROLLER_AUDIT_REF);snap=c.read(audit['registry_observation']['immutable_exact_snapshot_ref'])
spec=c.read(c.ORIGINAL_INPUT);q0=c.read(spec['root_stage_request_ref'])
cap0=c.read(d.CURRENT_CAP);authority0=c.read(q0['direct_attachment_authority_ref'])
checks=[]
def setup():
 snapshot=copy.deepcopy(snap);cap=copy.deepcopy(cap0);authority=copy.deepcopy(authority0);q=copy.deepcopy(q0)
 row=next(r for r in snapshot['jobs'] if r['logical_job']==d.C3_JOB)
 identity=dict(uid=1234,start_ticks=d.CONTROLLER_START,state='S')
 def read(record):
  if record==d.CURRENT_CAP:return cap
  if record==q['direct_attachment_authority_ref']:return authority
  return c.read(record)
 ns=dict(d.controller_v1.__globals__,read=read,active_amendment_v1=lambda:None,active=lambda a:None,
  os=types.SimpleNamespace(getuid=lambda:1234),time=types.SimpleNamespace(time=lambda:snapshot['updated_unix']+1),
  _registry_v1=lambda *args:snapshot)
 actual=types.FunctionType(d.controller_v1.__code__,ns)
 base=types.SimpleNamespace(proc_identity=lambda pid:identity)
 return actual,q,base,snapshot,row,cap,authority,identity
actual,q,base,snapshot,row,cap,authority,identity=setup()
before=copy.deepcopy(snapshot);result=actual(q,base)
assert result[2] is row and snapshot==before and result[2]['state']=='failed'
assert result[2]['exit_code']==1 and result[2]['leader_wait_status_known'] is True and result[2]['owned_drain_observed'] is True
checks.append(dict(name='exact_sealed_failed_record_accepted_without_mutation',status='PASS'))
cases=[
 ('borrowed_attachment_authority_false',lambda q,s,r,cap,a,p:a.update(borrowed_physical_inode_attachment_authorized=False)),
 ('controller_preservation_authority_false',lambda q,s,r,cap,a,p:a.update(old_controller_preserved_until_native32_drained=False)),
 ('additional_submission_revocation_false',lambda q,s,r,cap,a,p:a.update(additional_old_controller_submissions_revoked_while_native32_attached=False)),
 ('C3_envelope_weakened',lambda q,s,r,cap,a,p:a.update(C3_fixed_envelope_bytes=0)),
 ('native_envelope_changed',lambda q,s,r,cap,a,p:a.update(native32_max_cuda_reserved_bytes=1)),
 ('headroom_changed',lambda q,s,r,cap,a,p:a.update(headroom_bytes=0)),
 ('authority_wrong_controller_start',lambda q,s,r,cap,a,p:a.update(controller_start_ticks=1)),
 ('controller_UID_changed',lambda q,s,r,cap,a,p:p.update(uid=9999)),
 ('controller_start_changed',lambda q,s,r,cap,a,p:p.update(start_ticks=1)),
 ('controller_zombie',lambda q,s,r,cap,a,p:p.update(state='Z')),
 ('GPU_UUID_changed',lambda q,s,r,cap,a,p:cap.update(gpu_uuid='GPU-SYNTHETIC_WRONG')),
 ('nonzero_physical_GPU',lambda q,s,r,cap,a,p:cap.update(physical_gpu=1)),
 ('controller_expired',lambda q,s,r,cap,a,p:cap.update(expires_unix=s['updated_unix'])),
 ('C3_declared_cap_changed',lambda q,s,r,cap,a,p:cap['jobs'][d.C3_JOB].update(cuda_cap_bytes=1)),
 ('registry_wrong_capability',lambda q,s,r,cap,a,p:s.update(capability_ref=dict(path='SYNTHETIC',sha256='0'*64))),
 ('registry_stale',lambda q,s,r,cap,a,p:s.update(updated_unix=s['updated_unix']-61)),
 ('failed_exit_boolean',lambda q,s,r,cap,a,p:r.update(exit_code=True)),
 ('failed_exit_float',lambda q,s,r,cap,a,p:r.update(exit_code=1.0)),
 ('failed_drain_missing',lambda q,s,r,cap,a,p:r.pop('owned_drain_observed')),
 ('failed_wait_unknown',lambda q,s,r,cap,a,p:r.update(leader_wait_status_known=False)),
 ('failed_other_leader_PID',lambda q,s,r,cap,a,p:r.update(pid=r['pid']+1)),
 ('failed_other_start',lambda q,s,r,cap,a,p:r.update(leader_start_ticks=r['leader_start_ticks']+1)),
 ('failed_other_finish',lambda q,s,r,cap,a,p:r.update(finished_unix=r['finished_unix']+1)),
 ('failed_other_descendant_wait',lambda q,s,r,cap,a,p:r['descendant_waits'][0].update(raw_wait_status=256)),
 ('failed_changed_requested',lambda q,s,r,cap,a,p:r['requested'].update(cuda_cap_bytes=1)),
 ('C3_missing',lambda q,s,r,cap,a,p:s['jobs'].remove(r)),
 ('C3_duplicate',lambda q,s,r,cap,a,p:s['jobs'].append(copy.deepcopy(r))),
]
# A stable fake clock separate from registry mutations exercises freshness.
for name,mutate in cases:
 actual,q,base,snapshot,row,cap,authority,identity=setup()
 original_time=snapshot['updated_unix']+1
 actual.__globals__['time']=types.SimpleNamespace(time=lambda:original_time)
 mutate(q,snapshot,row,cap,authority,identity)
 before=copy.deepcopy(snapshot)
 try:actual(q,base)
 except (ValueError,KeyError):pass
 else:raise AssertionError('Unexpected acceptance '+name)
 assert snapshot==before
 checks.append(dict(name=name,status='PASS_REJECTED'))
for state in ('queued','launching','running','draining'):
 actual,q,base,snapshot,row,cap,authority,identity=setup()
 other=next(r for r in snapshot['jobs'] if r['logical_job']!=d.C3_JOB);other['state']=state
 try:actual(q,base)
 except ValueError:pass
 else:raise AssertionError('Other active '+state)
 checks.append(dict(name='other_'+state,status='PASS_REJECTED'))
# Positive lineage consumes actual public immutable JSON; only source metadata.
a=dict(preserved_actual_v9_amendment_ref=c.V9_ACTUAL_AMENDMENT_REF,
 genuine_v9_controller_entry_failed_terminal_ref=c.V9_ACTUAL_TERMINAL_REF,sealed_v9_controller_failure_audit_ref=c.V9_CONTROLLER_AUDIT_REF,
 v9_future_launch_write_capabilities_permanently_revoked=True,exact_recorded_failed_and_drained_C3_exit1_attachment_authorized=True,
 failed_C3_relabel_or_outer_success_claim_authorized=False)
terminal,sealed=c.validate_v9_failure_lineage_v10(a)
assert terminal['actual_returncode']==1 and sealed['C3_recorded_terminal_lineage']['outer_ROOT_handle_74136_terminal_consumed'] is None
assert sealed['v9_actual_failed_attempt']['direct_worker_PID'] is None and sealed['v9_actual_failed_attempt']['actual_wait_ref'] is None
checks.append(dict(name='actual_failed_v9_lineage_preserves_NULL_wait_and_unknown_C3_outer',status='PASS_METADATA_ONLY'))
for key in a:
 bad=copy.deepcopy(a);bad[key]=None
 try:c.validate_v9_failure_lineage_v10(bad)
 except ValueError:pass
 else:raise AssertionError('Missing ROOT authority '+key)
 checks.append(dict(name='missing_ROOT_field_'+key,status='PASS_REJECTED'))
assert not any(n in sys.modules for n in ('torch','numpy','composer','torch_geometric','tskit','tsinfer'))
v=dict(schema='native_v10_independent_actual_predicate_boundary_tests_v1',status='PASS_SOFTWARE_ONLY',checks=checks,
 actual_imported_controller_code_object=True,actual_public_source_metadata_authenticated=True,
 real_resource_process_registry_RPC_guard_capacity_attachment_or_authority_endpoints_never_called=True,
 failed_row_not_relabelled_or_mutated=True)
(OUT/'independent_predicate_checks.v10.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=v['status'],checks=len(checks))))
