"""Archive exact drained failed leaves and submit reviewed branch repairs.

No process signals, healthy task edits or scientific changes occur here.
The live controller validates the immutable request a second time before use.
"""
import argparse
import copy
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r, repairs, processes
from arggnn.benchmarking.paper_campaign.scheduler import validate

BASE = r.ROOT / 'research/benchmark_rebuild_20261024'
RECORDS = r.ROOT / '.runtime/benchmarking/paper_campaign_20261024_v2'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def current_plan(state):
    plan = r.read(state['plan_ref'])
    receipts = [r.read(ref) for ref in state.get('applied_repairs', {}).values()]
    for receipt in sorted(receipts, key=lambda value: value['resulting_plan_revision']):
        request = r.read(receipt['request_ref'])
        replacements = {task['id']: task for task in request['replacement_tasks']}
        plan['tasks'] = [replacements.get(task['id'], task) for task in plan['tasks']]
    return plan


def apply(branch):
    state = json.loads((RECORDS / 'state.json').read_text())
    plan = current_plan(state)
    if branch == 'gst':
        proposal = json.loads((BASE / 'gst_lai/runtime_v2_repair_proposal.json').read_text())
        old = proposal['archive_plan']
        source, destination = r.path(old['source_directory']), r.path(old['suggested_destination'])
        inventory = [dict(relative_path=str(r.path(item['path']).relative_to(source)),
                          bytes=item['bytes'], sha256=item['sha256']) for item in old['source_files']]
        request = dict(schema=repairs.SCHEMA, request_id='gst_four_worker_ram_v2',
                       reset_task_ids=proposal['reset_task_ids'],
                       replacement_tasks=r.read(proposal['replacement_tasks_ref']),
                       reason='Preserve the GST-LAI-8 model, all examples, seed and90 updates. Use four preparation workers with fit320GiB and prediction512GiB RAM reservations after the eight-worker copy peak exceeded128GiB.')
        archive_bytes = old['archived_bytes']
        failed_id = 'paper_gst_lai_8__all_train__fit'
    else:
        draft = json.loads((BASE / 'argmix/paper_argmix_native_ram256_repair_draft.json').read_text())
        request = copy.deepcopy(draft['request_template'])
        old = json.loads((BASE / 'argmix/paper_argmix_native_ram_archive_plan.json').read_text())
        source, destination = r.path(old['source_directory']), r.path(old['archive_directory'])
        inventory, archive_bytes = old['archive_files'], old['archived_bytes']
        failed_id = 'paper_argmix_published_recon_train_000.native_relate'
    name = request['request_id']
    need(name not in state.get('applied_repairs', {}), 'This repair is already applied')
    row = state['tasks'][failed_id]
    need(row['status'] == 'failed' and row['actual_wait'] and row['owned_process_group_absent'],
         'Exact failed task must have a consumed wait and drained session')
    need(not processes.session_members(row['pid']), 'Failed session has live members')
    targets = set(request['reset_task_ids']) | {t['id'] for t in request['replacement_tasks']}
    need(all(state['tasks'][name]['status'] in ('failed', 'blocked') for name in targets),
         'Only failed or blocked rows may change')
    request.update(expected_plan_ref=state['plan_ref'], expected_plan_revision=state.get('plan_revision', 0),
                   expected_attempts={name:state['tasks'][name].get('attempt', 1) for name in targets},
                   archived_bytes=archive_bytes)
    replacements = {t['id']: t for t in request['replacement_tasks']}
    candidate = copy.deepcopy(plan)
    candidate['tasks'] = [replacements.get(t['id'], t) for t in plan['tasks']]
    validate(candidate)
    for task in replacements.values():
        for ref in task.get('source_refs', []):
            need(r.ref(ref['path']) == ref, 'Changed replacement source: ' + ref['path'])
    need(source.is_dir() and not source.is_symlink() and not destination.exists(),
         'Expected failed source and fresh archive destination required')
    files = sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_file())
    need(files == sorted(item['relative_path'] for item in inventory), 'Failed output inventory changed')
    for item in inventory:
        p = source / item['relative_path']
        need(not p.is_symlink() and p.stat().st_size == item['bytes']
             and r.ref(p)['sha256'] == item['sha256'], 'Failed output bytes changed')
    need(sum(item['bytes'] for item in inventory) == archive_bytes, 'Archive accounting mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    manifest = dict(schema='paper_resource_repair_archive_v1', source_directory=str(source.relative_to(r.ROOT)),
                    destination_directory=str(destination.relative_to(r.ROOT)), files=inventory,
                    archived_bytes=archive_bytes, original_task_row=row, original_session_empty=True)
    request['archive_manifest_ref'] = r.save(BASE / 'integration' / (name + '.archive.json'), manifest)
    request_ref = r.save(RECORDS / 'repair_requests' / (name + '.request.json'), request)
    # This review reference is outside the dispatch directory and cannot launch work.
    review = BASE / 'integration' / (name + '.ref.json')
    r.save(review, request_ref)
    prepared, receipt = repairs._prepare(plan, state, RECORDS, review)
    need(receipt['status'] == 'applied', 'Preflight did not pass')
    live = json.loads((RECORDS / 'state.json').read_text())
    need(live.get('plan_revision', 0) == state.get('plan_revision', 0), 'Concurrent repair changed revision')
    submission = r.save(RECORDS / 'repair_requests' / review.name, request_ref)
    print(json.dumps(dict(status='submitted', request_ref=request_ref, submission_ref=submission,
                          archived_bytes=archive_bytes, expected_revision=request['expected_plan_revision'],
                          affected_tasks=len(targets))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('branch', choices=['gst', 'argmix'])
    apply(parser.parse_args().branch)
