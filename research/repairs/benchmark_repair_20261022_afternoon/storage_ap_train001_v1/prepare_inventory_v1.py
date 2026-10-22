"""Prepare only new metadata for the exact completed AP TRAIN001 scope."""
from pathlib import Path
import json
import os
from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v2 as closed
from research.repairs.benchmark_repair_20261022_afternoon.storage_ap_train001_v1 import external_archive_v1 as ap

ROOT = closed.ROOT
DIRECTORY = ROOT / 'research/repairs/benchmark_repair_20261022_afternoon/storage_ap_train001_v1'
SOURCE_PLAN = {'path': 'configs/benchmarks/executable_workflow_20261022_afternoon_v1.json',
               'sha256': 'e8a231523ef04dbf14d11a493bd73f88ec26ed952125e738ea6f788443c48492'}


def save(name, value):
    output = DIRECTORY / name
    with output.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')
    return closed.reference(ROOT, output)


def main():
    plan = closed.read(ROOT, SOURCE_PLAN)
    binding = next(row for row in plan['workflow_external_predecessors']
                   if row['source_task_id'] == ap.OWNER)
    source_plan = closed.read(ROOT, binding['source_plan_ref'])
    source_state = json.loads(closed.path(ROOT, source_plan['records_directory']).joinpath('state.json').read_text())
    entry = source_state['tasks'][ap.OWNER]
    proof_entry = {key: entry[key] for key in ('status', 'actual_controller_wait',
                   'owned_process_group_absent', 'job_ref', 'retained_bytes')}
    owner = closed.path(ROOT, ap.SCOPE)
    before = closed.old.scope_bytes(owner)
    closed.need(before == ap.RETAINED == proof_entry['retained_bytes'],
                'Frozen predecessor retained charge must equal the unchanged current scope')
    candidates, protected = [], []
    for base, dirs, files in os.walk(owner, followlinks=False):
        for name in files:
            item = Path(base) / name
            row = dict(path=str(item.relative_to(ROOT)), **closed.record(item))
            if closed.old.candidate(item.relative_to(owner), False):
                candidates.append(row)
            else:
                protected.append(row)
    candidates.sort(key=lambda row: row['path'])
    protected.sort(key=lambda row: row['path'])
    candidate_charge = sum(row['charge_bytes'] for row in candidates)
    largest = max(row['charge_bytes'] for row in candidates)
    original = dict(key=ap.TARGET, scope=ap.SCOPE, before_scope_bytes=before,
        accounting=dict(kind='workflow_predecessor', source_task_id=ap.OWNER),
        candidates=candidates, protected_files=protected,
        candidate_charge_bytes=candidate_charge, largest_candidate_charge_bytes=largest,
        max_output_bytes=candidate_charge + largest + 4 * closed.old.METADATA,
        future_old_scope_writes_require_root_revocation=True,
        producer=dict(task_id=ap.OWNER, source_plan_ref=binding['source_plan_ref'],
                      job_ref=binding['original_job_ref'],
                      wait_ref=closed.reference(ROOT, owner / 'actual_wait.json'), entry=proof_entry))
    first_ref = save('original_inventory_v1.json', dict(
        schema=closed.old.SCHEMA + '_inventory', initial_credit_bytes=0,
        historical_unknown_scopes_untouched=True,
        codec_ref=closed.reference(ROOT, closed.old.CODEC), targets=[original]))
    second_ref = save('supplemental_inventory_v1.json', dict(
        schema=closed.SCHEMA + '_inventory', initial_credit_bytes=0,
        original_inventory_ref=first_ref, preserved_failed_attempts=[],
        external_predecessor_binding=binding,
        targets=[dict(key=ap.TARGET, original=original, prior_archives=[], start_scope_bytes=before)]))
    task = dict(id=ap.TASK, dependencies=[ap.OWNER], resource='cpu',
        max_rss_bytes=2**30, max_output_bytes=original['max_output_bytes'], max_seconds=86400,
        output_directory='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_afternoon_v3/other/' + ap.TASK,
        factory_module=ap.MODULE, factory_function='factory',
        factory_arguments=dict(inventory_ref=second_ref, target_key=ap.TARGET))
    addition = dict(task=task, plan_field_name=ap.FIELD,
        plan_field_value=dict(schema=ap.SCHEMA, task_id=ap.TASK, inventory_ref=second_ref, initial_credit_bytes=0),
        additional_source_refs=[closed.reference(ROOT, value) for value in (ap.SOURCE, ap.SHIM,
            str(DIRECTORY.relative_to(ROOT) / 'prepare_inventory_v1.py'))] + [first_ref, second_ref],
        operational_order='Complete archive, admit VAL002 embeddings/retrieval/retirement, then dispatch C4 retry',
        original_owner_charge_bytes=before, candidate_charge_bytes=candidate_charge,
        maximum_original_scope_reduction_bytes=candidate_charge,
        initial_credit_bytes=0)
    addition_ref = save('plan_addition_v1.json', addition)
    proposed = dict(plan)
    proposed['tasks'] = list(plan['tasks']) + [task]
    proposed[ap.FIELD] = addition['plan_field_value']
    ap.validate_plan(proposed)
    call = ap.factory(inventory_ref=second_ref, target_key=ap.TARGET,
                      output_directory=str(closed.path(ROOT, task['output_directory']) / 'native'))
    closed.need(call['module'] == ap.MODULE and call['function'] == 'run', 'Exact new facade route required')
    print(json.dumps(dict(plan_addition_ref=addition_ref, candidate_count=len(candidates),
                          protected_count=len(protected), candidate_charge_bytes=candidate_charge,
                          task_cap_bytes=task['max_output_bytes'], initial_credit_bytes=0), sort_keys=True))


if __name__ == '__main__':
    main()
