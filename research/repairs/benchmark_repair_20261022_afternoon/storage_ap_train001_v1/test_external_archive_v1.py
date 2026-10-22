"""CPU-only receipt validation; never invoke a scientific worker or replace raw files."""
from copy import deepcopy
from pathlib import Path
import json
from unittest.mock import patch
from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v2 as closed
from research.repairs.benchmark_repair_20261022_afternoon.storage_ap_train001_v1 import external_archive_v1 as ap

ROOT = closed.ROOT
DIRECTORY = ROOT / 'research/repairs/benchmark_repair_20261022_afternoon/storage_ap_train001_v1'
ADDITION = json.loads((DIRECTORY / 'plan_addition_v1.json').read_text())
SOURCE_REF = {'path': 'configs/benchmarks/executable_workflow_20261022_afternoon_v1.json',
              'sha256': 'e8a231523ef04dbf14d11a493bd73f88ec26ed952125e738ea6f788443c48492'}


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')
    return closed.reference(ROOT, path)


def rejected(action):
    try:
        action()
    except (ValueError, KeyError, FileNotFoundError):
        return
    raise AssertionError('Invalid receipt or ownership was accepted')


def main():
    tests = []
    base = closed.read(ROOT, SOURCE_REF)
    plan = deepcopy(base)
    plan['tasks'].append(deepcopy(ADDITION['task']))
    plan[ap.FIELD] = deepcopy(ADDITION['plan_field_value'])
    ap.validate_plan(plan)
    tests.append('real immutable ownership and supplemental inventories validate')
    state = {'tasks': {ap.OWNER: {'status': 'succeeded', 'external': True}, ap.TASK: {'status': 'pending'}}}
    for status in ('pending', 'running', 'failed', 'blocked'):
        state['tasks'][ap.TASK]['status'] = status
        assert ap.completed_storage_reduction(plan, state) == 0
    tests.append('pending running failed and blocked tasks grant zero credit')
    state['tasks'][ap.TASK] = {'status': 'succeeded', 'actual_controller_wait': 0,
                               'owned_process_group_absent': False}
    rejected(lambda: ap.completed_storage_reduction(plan, state))
    tests.append('successful label without actual owned-group drain is rejected')
    changed = deepcopy(plan)
    changed['tasks'][-1]['max_output_bytes'] -= 1
    rejected(lambda: ap.validate_plan(changed))
    tests.append('reduced output reservation is rejected')
    changed = deepcopy(plan)
    original_binding = next(row for row in changed['workflow_external_predecessors'] if row['source_task_id'] == ap.OWNER)
    original_binding['max_output_bytes'] -= 1
    rejected(lambda: ap.validate_plan(changed))
    tests.append('changed historical predecessor accounting is rejected')
    changed = deepcopy(plan)
    changed.pop(ap.FIELD)
    rejected(lambda: ap.validate_plan(changed))
    tests.append('unbound supplemental factory is rejected')
    fixture = DIRECTORY / 'validation_fixture_v1'
    fixture.mkdir()
    output = fixture / 'worker'
    native = output / 'native'
    native.mkdir(parents=True)
    plan['tasks'][-1]['output_directory'] = str(output.relative_to(ROOT))
    task = plan['tasks'][-1]
    plan_ref = save(fixture / 'plan.json', plan)
    inventory, target, _ = ap.validate_plan(plan)
    entries = [dict(original_path=str(closed.path(ROOT, row['path'])),
                    original_bytes=row['identity'][2], state='original_kept_no_storage_gain',
                    credited_bytes=0, decompressed_hash_and_size_verified=True)
               for row in target['original']['candidates']]
    manifest = dict(schema=closed.SCHEMA, inventory_ref=plan[ap.FIELD]['inventory_ref'], target_key=ap.TARGET,
        owner_scope=ap.SCOPE, archive_output_directory=str(native),
        original_wait_ref=target['original']['producer']['wait_ref'], original_status_preserved='succeeded',
        every_original_byte_restorable=True, protected_science_and_evidence_unchanged=True,
        raw_replacement_after_verified_durable_archive=True,
        archives_in_new_or_exact_separately_charged_failed_scopes=True,
        preserved_failed_attempts=[], old_failure_relabelled=False, entries=entries,
        before_scope_bytes=ap.RETAINED, repair_start_scope_bytes=ap.RETAINED,
        after_scope_bytes=ap.RETAINED, original_scope_reduction_bytes=0)
    manifest_ref = save(native / 'restore_manifest.json', manifest)
    spec = dict(schema=closed.SCHEMA, inventory_ref=plan[ap.FIELD]['inventory_ref'],
                target_key=ap.TARGET, output_directory=str(native))
    job = dict(plan_ref=plan_ref, task=task, call=dict(module=ap.MODULE, function='run', spec=spec))
    job_ref = save(output / 'job.json', job)
    save(output / 'actual_wait.json', dict(job_ref=job_ref, actual_direct_wait=True,
        actual_direct_returncode=0, owned_process_group_absent=True, supervisor_error=None))
    result_ref = save(output / 'result.json', dict(job_ref=job_ref, value=manifest_ref))
    state = dict(plan_ref=plan_ref, tasks={ap.OWNER: dict(status='succeeded', external=True),
        ap.TASK: dict(status='succeeded', actual_controller_wait=0, owned_process_group_absent=True,
                     job_ref=job_ref, result_ref=result_ref, retained_bytes=closed.old.scope_bytes(output))})
    assert ap.completed_storage_reduction(plan, state) == 0
    tests.append('complete new kept-original receipt grants exactly zero credit')
    changed_state = deepcopy(state)
    changed_state['plan_ref'] = SOURCE_REF
    rejected(lambda: ap.completed_storage_reduction(plan, changed_state))
    tests.append('new job from a different plan is rejected')
    changed_state = deepcopy(state)
    changed_state['tasks'][ap.TASK]['retained_bytes'] -= 1
    rejected(lambda: ap.completed_storage_reduction(plan, changed_state))
    tests.append('unaccounted archive output is rejected')
    changed_state = deepcopy(state)
    changed_state['tasks'][ap.OWNER]['retained_bytes'] = ap.RETAINED
    rejected(lambda: ap.completed_storage_reduction(plan, changed_state))
    tests.append('duplicate external-owner local charge is rejected')
    original_read = ap.read
    def changed_manifest(repo, ref, cap=32 * 2**20):
        value = original_read(repo, ref, cap)
        if ref == manifest_ref:
            value['original_scope_reduction_bytes'] = 1
        return value
    with patch.object(ap, 'read', side_effect=changed_manifest):
        rejected(lambda: ap.completed_storage_reduction(plan, state))
    tests.append('invented one-byte original-scope reduction is rejected')
    original_record = ap.record
    protected_path = closed.path(ROOT, target['original']['protected_files'][0]['path'])
    def changed_protected(path):
        value = original_record(path)
        if Path(path) == protected_path:
            value = deepcopy(value)
            value['identity'][2] += 1
        return value
    with patch.object(ap, 'record', side_effect=changed_protected):
        rejected(lambda: ap.completed_storage_reduction(plan, state))
    tests.append('changed protected original evidence is rejected')
    with patch.object(ap, '_authority', side_effect=ValueError('Missing original source drain')):
        rejected(lambda: ap.factory(inventory_ref=plan[ap.FIELD]['inventory_ref'],
            target_key=ap.TARGET, output_directory=str(native / 'not-created')))
    tests.append('factory cannot proceed without original source dispatch drain')
    report = dict(status='PASS', tests=tests, test_count=len(tests),
        no_scientific_worker_invoked=True, no_original_scope_writes=True,
        preserved_validation_fixture=str(fixture.relative_to(ROOT)))
    report_ref = save(DIRECTORY / 'validation_report_v1.json', report)
    print(json.dumps(dict(status='PASS', test_count=len(tests), report_ref=report_ref), sort_keys=True))


if __name__ == '__main__':
    main()
