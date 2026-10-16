from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,importlib.util,json,os,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/completed_map_scope_root_close_20261016_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2')]
def ref(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 p=ROOT/r['path'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'];return json.loads(raw)
def save(name,value):
 p=HERE/name;assert not p.exists();p.write_text(json.dumps(value,indent=2)+'\n');return ref(p)
renewal_source=dict(path='research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py',sha256='491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468')
assert ref(renewal_source['path'])==renewal_source
sp=importlib.util.spec_from_file_location('_map_close_exact_renewal',ROOT/renewal_source['path']);renewal=importlib.util.module_from_spec(sp);sp.loader.exec_module(renewal)
renewal.install_renewal_v1(dict(path='research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json',sha256='66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'))
import root_contract_v1 as c
guard=c.original_guard()
storage_ref=ref('research/reviews/ARGMix_prospective1611_map_root_20261016_v2/admitted/storage.json');storage=read(storage_ref);own=storage['reservation']
assert own['key']=='DATA/ARGMix-prospective1611-chr22-map-v2'
completed_ref=dict(path='data/ARGMix_prospective1611_HapMapII_GRCh37_chr22_v1/completed.json',sha256='cadf4cc17f0734dc0b741790de48ef0ed169a5fd8d27815bef85435289e5465e')
completed=read(completed_ref)
terminal_ref=ref('research/reviews/ARGMix_prospective1611_map_root_20261016_v2/actual_terminal_and_map_contract_root_v2.json');terminal=read(terminal_ref)
assert completed['root_pid']==3078227 and completed['no_workers_spawned'] is True
assert terminal['actual_exec_session_id']==40685 and terminal['actual_exit_code']==0 and terminal['actual_tool_chunk_id']=='42b4b7' and terminal['completed_ref']==completed_ref and terminal['no_child_workers_spawned'] is True
pid=completed['root_pid'];absence=guard._absent(pid)
# Explicit linkage of a previously consumed genuine tool terminal to the producer PID.
# This is not a new wait, a model terminal, or a contemporaneous before-start launch receipt.
wait_ref=save('existing_actual_tool_terminal_PID_link.root_v1.json',dict(schema='root_existing_actual_tool_terminal_producer_PID_link_v1',root_authorizer='root_agent',producer_pid=pid,actual_exit_code=terminal['actual_exit_code'],actual_terminal_consumed=True,original_terminal_ref=terminal_ref,original_exec_session_id=40685,original_chunk_id='42b4b7',completed_ref=completed_ref,no_child_workers_spawned=True,linkage_only_no_new_wait_or_replay=True))
reservation_sha=guard.budget.digest(own)
obs_ref=save('targeted_observation.root_v1.json',dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',key=own['key'],reservation_sha256=reservation_sha,storage_ref=storage_ref,launch_ref=completed_ref,direct_wait_ref=wait_ref,completed_ref=completed_ref,launcher_pid=pid,all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=[pid],observations=[absence],original_producer_is_single_process=True,launch_PID_evidence_is_actual_producer_completed_record=True,actual_root_terminal_ref=terminal_ref))
request=dict(key=own['key'],reservation_sha256=reservation_sha,storage_ref=storage_ref,storage_locator=[],launch_ref=completed_ref,launch_pid_locator=['root_pid'],launch_identity=dict(status='completed_prospective_chr22_map_acquisition',root_pid=pid,no_workers_spawned=True),direct_wait_ref=wait_ref,wait_pid_locator=['producer_pid'],wait_exit_locator=['actual_exit_code'],wait_identity=dict(schema='root_existing_actual_tool_terminal_producer_PID_link_v1',actual_terminal_consumed=True,original_terminal_ref=terminal_ref,linkage_only_no_new_wait_or_replay=True),completed_ref=completed_ref,completed_identity=dict(status='completed_prospective_chr22_map_acquisition',root_pid=pid,no_workers_spawned=True),observation_ref=obs_ref)
with guard.budget.v4.mutex(ROOT):state=guard._state_locked(c.CORE,c.BASE,fresh=True)
assert own['key'] in state[0] and own['key'] not in state[1] and state[0][own['key']]==own
proof=guard._proof(request,own,observe=True)
auth=guard.operation_draft_v1('close_completed',request,expected_before=guard._pins(state[2]),bindings=state[3]);auth.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
auth_ref=save('close_authority.root_v1.json',auth)
result=guard.apply_close_completed_v1(auth_ref,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
result_ref=save('close_receipt.root_v1.json',result)
print(json.dumps(dict(result_ref=result_ref,retained_bytes=result['retained_bytes'],credit_bytes=result['credit_bytes'],projected_bytes=result['projected_bytes'],remaining_to_980GB=result['remaining_to_admission_ceiling_bytes'],artifacts_deleted=result['artifacts_deleted'],model_or_GPU_execution=False)),flush=True)
