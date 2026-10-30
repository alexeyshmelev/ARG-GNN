"""Authenticate an additive sidecar continuation while retaining public lineage."""
import copy
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue import claims, storage_credit_v1 as credit
from research.joint_evaluation_20261029.queue import successor_v2_support as shared
from research.joint_evaluation_20261029.queue_audit_20261030.backfill import boundary_v1 as boundary

need = boundary.need
external_result = shared.external_result
external_evidence = shared.external_evidence
task_fits_scope = shared.task_fits_scope


def immutable_json(path, value):
    path = r.path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return r.ref(path)


def initial_rows(plan):
    spec = boundary.validate_plan(plan)
    handoff = r.read(r.ref(spec['handoff_receipt_path']))
    need(handoff['schema'] == 'argformer_verified_idle_handoff_v1' and
         handoff['operational_continuation_ref'] == plan['operational_continuation_ref'], 'Exact sealed handoff required')
    need(handoff['old_exit_observed'] is True and handoff['worker_signals_sent'] == [] and
         handoff['old_actual_wait_consumed'] is False, 'Only a childless controller may be replaced')
    old = spec['predecessor_controller']
    need(handoff['predecessor_controller'] == old, 'Handoff owner identity changed')
    path = Path('/proc') / str(old['pid']) / 'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        fields = None
    need(fields is None or fields[19] != old['process_start_ticks'] or fields[0] == 'Z', 'Old owner remains live')
    need(not processes.session_members(old['pid']), 'Old owner session remains live')
    sealed = r.read(handoff['sealed_state_ref'])
    need(json.loads(r.path(spec['original_state_path']).read_text()) == sealed, 'Original state changed after frozen sealing')
    merged = boundary.reconcile(spec, sealed)
    need(merged == r.read(handoff['reconciliation_ref']), 'Frozen genuine waits changed after sealing')
    immutable_json(spec['adoption_audit_path'], dict(schema='argformer_actual_wait_adoption_v1',
        handoff_ref=r.ref(spec['handoff_receipt_path']), original_plan_ref=spec['original_plan_ref'],
        actual_wait_refs=merged['actual_wait_refs'], reconciled_stale_running_ids=merged['reconciled_stale_running_ids'],
        original_waits_modified=False, running_workers_adopted=False, scientific_settings_changed=False))
    return copy.deepcopy(merged['rows'])


def state_metadata(plan, plan_path):
    spec = r.read(plan['operational_continuation_ref'])
    return dict(plan_ref=spec['original_plan_ref'], operational_plan_ref=r.ref(plan_path),
                operational_continuation_ref=plan['operational_continuation_ref'],
                gpu3_wait_reason='TEST009 waits for all priority root GPU chains to complete; its five-epoch MLM is nonpreemptible')


def permit_existing_output(plan, task, row):
    return claims.permit_claimed_output(r.read(r.read(plan['operational_continuation_ref'])['original_plan_ref']), task, row)


def accounting_template(plan):
    spec = r.read(plan['operational_continuation_ref'])
    return dict(r.read(spec['accounting_template_ref']), handoff_config_ref=spec['accounting_config_ref'])


def admission_accounting(plan):
    spec = r.read(plan['operational_continuation_ref'])
    from research.joint_evaluation_20261029.queue_audit_20261030.common import probe_scheduler_v1 as engine
    previous = r.scope_bytes
    try:
        r.scope_bytes = engine.strict_scope_bytes
        value = credit.commitment(accounting_template(plan), spec['storage_credit_ref'], argmix_override_bytes=25*2**30)
    finally:
        r.scope_bytes = previous
    value['items']['ARGformer boundary continuation metadata'] = spec['metadata_max_bytes']
    value['items']['Native retry metadata conservative reserve'] = 16*2**20
    value['items']['Parallel Relate producer metadata reserve'] = 16*2**20
    value['items']['Queue audit and launcher metadata reserve'] = 16*2**20
    value['total_bytes'] = sum(value['items'].values())
    value['admitted'] = value['total_bytes'] <= value['ceiling_bytes']
    return value


def metadata_within_bound(plan):
    spec = r.read(plan['operational_continuation_ref'])
    return shared.metadata_within_bound(plan) and r.scope_bytes(r.path(spec['metadata_directory'])) <= spec['metadata_max_bytes']


def foreign_owner_growth(plan):
    # Retain the complete original 960GiB reservation. It already bounds the
    # original campaign, the parallel ARGMix branch and other reviewed writers.
    return 0
