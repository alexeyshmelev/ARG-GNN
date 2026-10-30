"""Prepare immutable ARGformer handoff contracts; never launch or signal."""
import copy
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue_audit_20261030.backfill import support_v1 as support
from research.joint_evaluation_20261029.queue_audit_20261030.backfill import boundary_v1 as boundary

HERE=Path(__file__).resolve().parent
BASE=r.ROOT/'research/joint_evaluation_20261029'

def main():
    original_ref=r.ref(BASE/'queue/sidecar_plan.json')
    original=r.read(original_ref)
    state=json.loads((BASE/'queue/records_v1/state.json').read_text())
    boundary.need(state['controller_pid']==1981120 and state['plan_ref']==original_ref,'Exact live sidecar owner required')
    raw=Path('/proc/1981120')
    ticks=(raw/'stat').read_text().rsplit(')',1)[1].split()[19]
    boundary.need(ticks=='591807927','Reviewed original controller start identity')
    command=(raw/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    finish=r.read(r.ref(BASE/'root/reporting_20261030_v1/finish_plan_candidate_v1.json'))
    config=r.read(finish['successor_config_ref'])
    baseline_ref=support.immutable_json(HERE/'baseline_state_v1.json',state)
    names=['boundary_v1.py','support_v1.py','gpu_v1.py','scheduler_v1.py','watch_v1.py']
    sources=[r.ref(HERE/name) for name in names]
    sources += [r.ref(BASE/'queue_audit_20261030/common/probe_scheduler_v1.py'),
                r.ref(BASE/'queue_audit_20261030/resources/probe_guard_v1.py'),
                r.ref(BASE/'queue/successor_v2_gpu.py'),r.ref(BASE/'queue/successor_v2_support.py'),
                r.ref(BASE/'queue/future_merge.py'),r.ref(BASE/'queue/storage_credit_v1.py')]
    relative=str(HERE.relative_to(r.ROOT))
    plan_path=relative+'/candidate_plan_v1.json'
    spec=dict(schema='argformer_idle_boundary_continuation_v1', original_plan_ref=original_ref,
        baseline_state_ref=baseline_ref,original_state_path=original['records_directory']+'/state.json',
        predecessor_controller=dict(pid=1981120,process_start_ticks=ticks,command=command),
        source_refs=sources, owner_lock_path=original['records_directory']+'/sidecar.lock',
        metadata_directory=relative,metadata_max_bytes=64*2**20,
        handoff_receipt_path=relative+'/handoff_v1.json',adoption_audit_path=relative+'/adoption_v1.json',
        replacement_log_path=relative+'/replacement_v1.log',replacement_started_path=relative+'/started_v1.json',
        launch_argv=[str(r.path('.runtime/env/bin/python')),'-B','-s','-u','-m',
                    'research.joint_evaluation_20261029.queue_audit_20261030.backfill.scheduler_v1','--plan',plan_path],
        accounting_template_ref=config['successor_template_ref'],accounting_config_ref=finish['successor_config_ref'],
        storage_credit_ref=finish['storage_credit_ref'],
        native_active_owner_registry_path='research/joint_evaluation_20261029/queue/native_argmix_retry_active_owner_v1.json',
        treeloom_manifest_ref=r.ref(BASE/'argmix/recovery_20261030_queue/controller_manifest_v1.json'),
        operational_delta='Only remove TEST009.prepare operational dependency on TEST007.retire; science and full resource caps unchanged.',
        gpu3_policy='Only TEST009 may use GPU3 after genuine root release, every TreeLoom GPU-stage wait and terminal native GPU rows; shared lifetime mutex is mandatory.',
        handoff_policy='Freeze only exact childless predecessor; every historical worker requires real wait and absent session; no worker signals or adoption.',
        production_launches=0,signals_sent=0)
    spec_ref=support.immutable_json(HERE/'continuation_v1.json',spec)
    plan=boundary.expected_plan(spec_ref)
    plan_ref=support.immutable_json(plan_path,plan)
    boundary.validate_plan(plan)
    accounting=support.admission_accounting(plan)
    print(json.dumps(dict(candidate_plan_ref=plan_ref,continuation_ref=spec_ref,
        accounting_total_bytes=accounting['total_bytes'],storage_headroom_bytes=accounting['ceiling_bytes']-accounting['total_bytes'],
        production_launches=0,signals_sent=0),sort_keys=True))

if __name__=='__main__':main()
