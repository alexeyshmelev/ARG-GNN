"""Exact prospective European reservation revision with retained-file evidence.

All original writer accounting is evaluated first. A root-authorized immutable
receipt may replace only the 160 GiB European allowance by 128 GiB, retaining
all failed scopes plus a full future attempt, fit and metadata allowances.
"""
import copy
import os
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue import successor_v2_support as original
from research.joint_evaluation_20261029.queue.successor_v2_handoff import identity

ALLOWANCE = 'Separate full European validation, entire cap'
RUN_BASE = 'research/efficiency_20261027/root/runs'
EXPECTED_SCOPES = {
    RUN_BASE + '/european_independent_validation_200mb_v1',
    RUN_BASE + '/european_independent_validation_200mb_v2_no_timeouts',
    RUN_BASE + '/european_short_development_val_10mb_v1',
    RUN_BASE + '/european_short_development_val_10mb_v2',
}


def need(value, message):
    original.need(value, message)


def scope_usage(scope):
    """Count only the exact declared historical scope, including filesystem blocks."""
    raw = Path(scope) if Path(scope).is_absolute() else r.ROOT / scope
    need(raw.resolve() == raw and '..' not in raw.parts, 'Ordinary exact retained European scope required')
    path = r.path(scope)
    need(path.is_dir() and not path.is_symlink(), 'Declared retained European scope is absent or linked')
    logical = allocated = 0
    for folder, directories, files in os.walk(path, followlinks=False):
        directory_stat = Path(folder).stat()
        logical += directory_stat.st_size
        allocated += directory_stat.st_blocks * 512
        for name in directories:
            need(not (Path(folder)/name).is_symlink(), 'No linked retained European directory')
        for name in files:
            item = Path(folder)/name
            need(not item.is_symlink() and item.is_file(), 'Ordinary retained European file required')
            stat = item.stat()
            logical += stat.st_size
            allocated += stat.st_blocks * 512
    return dict(logical_bytes=logical, allocated_bytes=allocated)


def ended(identity_record, label):
    pid = identity_record['pid']
    need(not processes.session_members(pid), label + ' session is still live')
    current = identity(pid)
    need(current is None or current['start_ticks'] != str(identity_record['start_ticks']),
         label + ' original process is still live')


def validate_credit(plan, credit_ref):
    evidence = r.read(credit_ref)
    need(evidence['schema'] == 'european_reservation_release_evidence_v1' and
         evidence['root_authorized'] is True, 'Root-authorized immutable European credit required')
    config = r.read(plan['handoff_config_ref'])
    need(evidence['original_successor_config_ref'] == plan['handoff_config_ref'] and
         evidence['original_storage_ledger_ref'] == config['storage_ledger_ref'],
         'Credit belongs to different original writer accounting')
    need(evidence['allowance_name'] == ALLOWANCE and evidence['prior_allowance_bytes'] == 160*2**30
         and evidence['revised_allowance_bytes'] == 128*2**30 and evidence['credit_bytes'] == 32*2**30,
         'Only exact160-to128GiB European allowance revision is authorized')
    need(evidence['full_future_attempt_bytes'] == 128_983_236_608
         and evidence['fit_allowance_bytes'] == 2*2**30 and evidence['small_work_allowance_bytes'] == 2**29
         and evidence['additional_other_work_allowance_bytes'] == 2**30,
         'Full future attempt, fit and small work reservations must remain')
    need(evidence['reservation_basis_refs'], 'Original full-attempt reservation basis required')
    for reference in evidence['reservation_basis_refs']:
        r.read(reference)
    runs = evidence['terminal_runs']
    need(len(runs) == 4 and {run['scope'] for run in runs} == EXPECTED_SCOPES,
         'Exact four retained failed European scopes required')
    retained = 0
    for run in runs:
        need(r.path(run['actual_wait_ref']['path']) == r.path(run['scope'])/'actual_wait.json',
             'Historical European wait must remain in its exact scope')
        wait = r.read(run['actual_wait_ref'])
        need(wait.get('actual_wait') is True and wait.get('actual_returncode') not in (None, 0)
             and wait.get('owned_family_absent') is True and wait.get('owned_group_absent') is True
             and wait.get('adopted_waits') == [],
             'Only genuine consumed failed European waits can release unused reservations')
        need(wait['pid'] == run['worker_identity']['pid'], 'Historical wait worker identity differs')
        ticks = wait.get('process_start_ticks', wait.get('start_ticks'))
        need(str(ticks) == str(run['worker_identity']['start_ticks']), 'Historical wait worker start identity differs')
        ended(run['worker_identity'], 'Historical European worker')
        family = wait['tracked_family']
        need(any(member['pid'] == wait['pid'] and str(member['start']) == str(ticks) for member in family),
             'Wait must retain the actual worker family identity')
        for member in family:
            current = identity(member['pid'])
            need(current is None or current['start_ticks'] != str(member['start']),
                 'Historical European family member is still live')
        for session in {member['session'] for member in family}:
            need(not processes.session_members(session), 'Historical European family session remains live')
        if run.get('supervisor_identity'):
            ended(run['supervisor_identity'], 'Historical European supervisor')
        usage = scope_usage(run['scope'])
        need(usage['logical_bytes'] <= run['logical_bytes'] and usage['allocated_bytes'] <= run['allocated_bytes'],
             'Retained European scope grew after credit evidence')
        retained += max(run['logical_bytes'], run['allocated_bytes'])
    need(retained + evidence['full_future_attempt_bytes'] + evidence['fit_allowance_bytes']
         + evidence['small_work_allowance_bytes'] + evidence['additional_other_work_allowance_bytes']
         <= evidence['revised_allowance_bytes'],
         'Revised allowance cannot cover retained files and the full prospective work')
    return dict(credit_ref=credit_ref, retained_scopes_charged_bytes=retained,
                full_future_attempt_bytes=evidence['full_future_attempt_bytes'],
                fit_allowance_bytes=evidence['fit_allowance_bytes'],
                small_work_allowance_bytes=evidence['small_work_allowance_bytes'],
                additional_other_work_allowance_bytes=evidence['additional_other_work_allowance_bytes'],
                revised_allowance_bytes=evidence['revised_allowance_bytes'],
                released_unused_reservation_bytes=evidence['credit_bytes'],
                files_deleted=False, historical_sources_modified=False)


def commitment(plan, credit_ref, argmix_override_bytes=None):
    before = original.commitment(plan, argmix_override_bytes=argmix_override_bytes)
    proof = validate_credit(plan, credit_ref)
    need(before['items'][ALLOWANCE] == 160*2**30, 'Original European allowance already changed')
    result = copy.deepcopy(before)
    result['items'][ALLOWANCE] = 128*2**30
    result['total_bytes'] = sum(result['items'].values())
    need(before['total_bytes'] - result['total_bytes'] == 32*2**30, 'Only exact credit may affect total')
    result['admitted'] = result['total_bytes'] <= result['ceiling_bytes'] == 1_200_000_000_000
    result['before_credit_total_bytes'] = before['total_bytes']
    result['prospective_european_reservation_revision'] = proof
    return result
