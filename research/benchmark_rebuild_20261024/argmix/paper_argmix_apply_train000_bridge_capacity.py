"""Review/check, then explicitly apply the sealed TRAIN0 bridge-only repair.

Default mode reads evidence only. --apply archives exactly the drained failed
1843-byte inferred leaf, writes immutable metadata, preflights with the existing
repair API using a research-only submission, then publishes one atomic pointer.
No process signals, native conversion, reservation changes or source edits occur.
The live controller remains the admission and execution authority.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
from arggnn.benchmarking.paper_campaign import processes, repairs, runtime as r
from arggnn.benchmarking.paper_campaign.scheduler import validate

AREA = r.ROOT / 'research/benchmark_rebuild_20261024/argmix'
RECORDS = r.ROOT / '.runtime/benchmarking/paper_campaign_20261024_v2'
STATE = RECORDS / 'state.json'
TARGET = 'paper_argmix_published_recon_train_000.inferred'
REQUEST_ID = 'argmix_train000_bridge_capacity_v1'
DRAFT_REF = dict(path=str((AREA / 'paper_argmix_train000_bridge_capacity_repair_draft.json').relative_to(r.ROOT)),
                 sha256='ba29cabc1af2a04c2cee6edb8ec138d692a08620357b4f0c3b82ab83577f73f4')
REPAIRS_REF = dict(path='src/arggnn/benchmarking/paper_campaign/repairs.py',
                   sha256='e8e407b7145d7dae20f833c61e7c040961de4aa16b1bfcb7f3f895a7ca986d23')
PLAN_REF = dict(path='configs/benchmarks/paper_campaign_20261024_v2.json',
                sha256='9f0ee40f25e9def145a85318bc723903b4ef96bd7c85060685c7f959ce3ccb9b')
SCOPE = 'experiments/paper_argmix_1611_rebuild_20261024_v1'
METADATA_ALLOWANCE = 2**20


def need(ok, message):
    if not ok:
        raise ValueError(message)


def ordinary_path(filename):
    p = Path(filename)
    if not p.is_absolute():
        p = r.ROOT / p
    need(p.is_relative_to(r.ROOT), 'Project-contained lexical path required')
    for part in (p, *p.parents):
        if part == r.ROOT.parent:
            break
        need(not part.is_symlink(), 'No symbolic links allowed: ' + str(part))
    need(p.resolve() == p, 'Canonical path required: ' + str(p))
    return p


def sealed(reference):
    p = ordinary_path(reference['path'])
    need(p.is_file(), 'Ordinary sealed file required: ' + str(p))
    need(r.ref(p) == reference, 'Seal changed: ' + str(p))
    return p


def effective_state(revision):
    state = json.loads(STATE.read_text())
    need(state['plan_ref'] == PLAN_REF, 'Original plan reference changed')
    need(state.get('plan_revision', 0) == revision, 'Current revision differs from explicit --expected-plan-revision')
    plan = r.read(PLAN_REF)
    for reference in plan['controller_source_refs']:
        sealed(reference)
    receipts = [r.read(ref) for ref in state.get('applied_repairs', {}).values()]
    for receipt in sorted(receipts, key=lambda x: x['resulting_plan_revision']):
        request = r.read(receipt['request_ref'])
        changes = {task['id']: task for task in request['replacement_tasks']}
        plan['tasks'] = [changes.get(task['id'], task) for task in plan['tasks']]
    return plan, state


def immutable_json(filename, value):
    """Publish complete readable JSON atomically without overwriting evidence."""
    p = ordinary_path(filename)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if p.exists():
        need(p.is_file() and p.read_bytes() == encoded, 'Existing immutable metadata differs: ' + str(p))
        return r.ref(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.staging.' + str(os.getpid()))
    need(not temporary.exists(), 'Staging file already exists; inspect it before retrying')
    with temporary.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    # Hard-link publication is atomic and fails if another writer used this name.
    os.link(temporary, p)
    temporary.unlink()
    return r.ref(p)


def inventory(leaf, archive):
    need(leaf.is_dir() and not leaf.is_symlink(), 'Expected ordinary failed leaf')
    files = list(leaf.iterdir())
    need(len(files) == 1 and files[0].name == 'actual_wait.json', 'Failed leaf inventory changed')
    expected = archive['archive_files'][0]
    f = ordinary_path(files[0])
    need(f.is_file() and f.stat().st_size == expected['bytes'] == 1843, 'Failed receipt size changed')
    need(r.ref(f)['sha256'] == expected['sha256'], 'Failed receipt seal changed')
    wait = json.loads(f.read_text())
    need(wait['status'] == 'failed' and wait['attempt'] == 2 and wait['pid'] == 308608
         and wait['process_start_ticks'] == '549746915' and wait['actual_returncode'] == -15
         and wait['actual_wait'] is True and wait['owned_process_group_absent'] is True,
         'Failed receipt does not prove the expected drained attempt')
    return wait


def precheck(revision, draft, archive):
    plan, state = effective_state(revision)
    request = copy.deepcopy(draft['request_template'])
    request['expected_plan_revision'] = revision
    need(request['request_id'] == REQUEST_ID and len(request['replacement_tasks']) == 1
         and request['replacement_tasks'][0]['id'] == TARGET and len(request['reset_task_ids']) == 147,
         'Expected exact one-bridge/147-target repair')
    targets = set(request['reset_task_ids']) | {TARGET}
    by_id = {t['id']: t for t in plan['tasks']}
    replacement = request['replacement_tasks'][0]
    need(set(request['expected_attempts']) == targets, 'Attempt binding set changed')
    source = ordinary_path(archive['source_directory'])
    destination = ordinary_path(archive['archive_directory'])
    manifest_path = AREA / 'paper_argmix_train000_bridge_capacity_actual_archive.json'
    need(not (source.exists() and destination.exists()), 'Both source and archive exist; inspect manually')
    leaf = source if source.exists() else destination
    need(leaf.exists(), 'Failed evidence leaf is absent at both exact paths')
    wait = inventory(leaf, archive)
    for name in targets:
        old, row = by_id[name], state['tasks'][name]
        need(row['status'] in ('failed', 'blocked') and old['kind'] != 'external', 'Target is not failed/blocked: ' + name)
        need(row.get('attempt', 1) == request['expected_attempts'][name], 'Attempt changed: ' + name)
        output = ordinary_path(old['output_directory'])
        need(output.is_relative_to(r.ROOT / 'experiments'), 'Output escaped experiments')
        need(name == TARGET or not output.exists(), 'Unexpected downstream output requires separate review: ' + name)
    row = state['tasks'][TARGET]
    for key in ('pid', 'process_start_ticks', 'attempt', 'actual_returncode', 'actual_wait', 'owned_process_group_absent', 'status'):
        need(row[key] == wait[key], 'Failed state/receipt mismatch: ' + key)
    need(not processes.session_members(row['pid']), 'Old task session still has live members')
    for name in archive['keep_prior_successful_tasks']:
        row = state['tasks'][name]
        need(row['status'] == 'succeeded' and row['actual_returncode'] == 0
             and row['actual_wait'] is True and row['owned_process_group_absent'] is True,
             'Successful predecessor seal changed: ' + name)
        if row.get('result_ref'):
            sealed(row['result_ref'])
    terminal = r.read(archive['preserve_native_terminal_ref'])
    need(terminal['successful'] is True and terminal['actual_wait_consumed'] is True
         and len(terminal['native_stage_waits']) == 144
         and all(x['returncode'] == 0 and x['actual_wait_consumed'] is True for x in terminal['native_stage_waits']),
         'Native 144-stage completion seal changed')
    # This is a streaming SHA read of one known tree, never a tskit load.
    sealed(archive['preserve_native_tree_ref'])
    sealed(archive['old_outer_log_ref'])
    for key in ('kind', 'output_directory', 'additional_mutable_scopes'):
        need(replacement.get(key, []) == by_id[TARGET].get(key, []), 'Replacement scope/kind changed: ' + key)
    candidate = copy.deepcopy(plan)
    candidate['tasks'] = [replacement if t['id'] == TARGET else t for t in plan['tasks']]
    validate(candidate)
    for ref in replacement.get('source_refs', []):
        sealed(ref)
    need(replacement['max_rss_bytes'] == 128 * 2**30 and replacement['max_output_bytes'] == 16 * 2**30 + 8 * 2**20
         and replacement['max_seconds'] == 1800 and replacement['cpu_threads'] == 1, 'Capacity or execution parameters changed')
    now = time.time()
    need(-5 <= now - state['updated_unix'] <= 120, 'Controller storage snapshot is stale')
    reserved = state['project_reserved_bytes']
    peak = reserved + replacement['max_output_bytes'] + METADATA_ALLOWANCE
    need(peak <= plan['storage_ceiling_bytes'] == 990000000000, 'Conservative controller storage headroom insufficient')
    scope_bytes = r.scope_bytes(ordinary_path(SCOPE))
    limits = [x['max_bytes'] for x in plan['scope_limits'] if x['path'] == SCOPE]
    need(limits == [40 * 2**30] and scope_bytes + replacement['max_output_bytes'] <= limits[0], 'Unchanged ARGMix scope headroom insufficient')
    free = shutil.disk_usage(source.parent).free
    need(free >= replacement['max_output_bytes'] + METADATA_ALLOWANCE, 'Filesystem free space insufficient')
    # Only the exact scope is scanned; no project-wide storage traversal occurs.
    check = dict(checked_unix=now, plan_revision=revision, controller_updated_unix=state['updated_unix'],
                 controller_project_reserved_bytes=reserved, conservative_candidate_peak_bytes=peak,
                 storage_ceiling_bytes=plan['storage_ceiling_bytes'], argmix_scope_bytes=scope_bytes,
                 argmix_scope_limit_bytes=limits[0], filesystem_free_bytes=free,
                 reset_tasks=len(targets), replacement_tasks=1, native_stage_waits=144,
                 failed_leaf_bytes=1843, old_session_members=[], archive_already_moved=not source.exists())
    return plan, state, request, check, source, destination, manifest_path


def verify_existing_request(request, draft, archive):
    expected = copy.deepcopy(draft['request_template'])
    expected['expected_plan_revision'] = request['expected_plan_revision']
    manifest_path = AREA / 'paper_argmix_train000_bridge_capacity_actual_archive.json'
    expected['archive_manifest_ref'] = r.ref(sealed(request['archive_manifest_ref']))
    need(ordinary_path(request['archive_manifest_ref']['path']) == manifest_path, 'Unexpected archive manifest path')
    need(request == expected, 'Existing repair differs from the exact sealed one-bridge draft')
    manifest = r.read(request['archive_manifest_ref'])
    need(manifest['archived_bytes'] == 1843 and manifest['archive_plan_ref'] == draft['archive_plan_ref']
         and manifest['archive_directory'] == archive['archive_directory'], 'Existing archive receipt mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-plan-revision', type=int, required=True)
    parser.add_argument('--apply', action='store_true', help='Archive the sealed failed leaf and submit exactly one reviewed repair')
    args = parser.parse_args()
    sealed(REPAIRS_REF)
    draft = r.read(DRAFT_REF)
    for key in ('helper_ref', 'archive_plan_ref', 'column_audit_ref'):
        sealed(draft[key])
    archive = r.read(draft['archive_plan_ref'])
    state = json.loads(STATE.read_text())
    submission = RECORDS / 'repair_requests' / (REQUEST_ID + '.ref.json')
    if REQUEST_ID in state.get('applied_repairs', {}):
        receipt = r.read(state['applied_repairs'][REQUEST_ID])
        need(receipt['request_id'] == REQUEST_ID and receipt['status'] == 'applied', 'Applied receipt mismatch')
        verify_existing_request(r.read(receipt['request_ref']), draft, archive)
        print(json.dumps(dict(status='already_applied', receipt_ref=state['applied_repairs'][REQUEST_ID]), indent=2))
        return
    if submission.exists():
        reference = r.read(r.ref(submission))
        request = r.read(reference)
        need(request['request_id'] == REQUEST_ID, 'Queued request mismatch')
        verify_existing_request(request, draft, archive)
        marker = r.ref(submission)['sha256']
        processed = state.get('processed_repair_submissions', {}).get(marker)
        print(json.dumps(dict(status='already_processed' if processed else 'already_queued',
                              submission_ref=r.ref(submission), request_ref=reference,
                              processed_receipt_ref=processed), indent=2))
        return
    plan, state, request, check, source, destination, manifest_path = precheck(args.expected_plan_revision, draft, archive)
    if not args.apply:
        print(json.dumps(dict(status='read_only_checks_passed', checks=check,
                              mutation_performed=False, final_API_preflight_requires_archived_leaf=True), indent=2))
        return
    if source.exists():
        need(not manifest_path.exists(), 'Archive receipt exists while source remains; inspect manually')
        destination.parent.mkdir(parents=True, exist_ok=True)
        need(source.stat().st_dev == destination.parent.stat().st_dev, 'Archive must be a same-filesystem rename')
        need(not destination.exists(), 'Archive destination already exists')
        source.rename(destination)
    inventory(destination, archive)
    manifest = dict(schema='paper_argmix_bridge_failed_leaf_archive_v1', archived_bytes=1843,
                    source_directory=archive['source_directory'], archive_directory=archive['archive_directory'],
                    archive_files=archive['archive_files'], archive_plan_ref=draft['archive_plan_ref'],
                    preserve_native_tree_ref=archive['preserve_native_tree_ref'],
                    preserve_native_terminal_ref=archive['preserve_native_terminal_ref'],
                    files_moved_or_deleted_from_native_work=[], failed_attempt=2, failed_pid=308608,
                    failed_process_start_ticks='549746915', actual_wait=True, owned_process_group_absent=True)
    request['archive_manifest_ref'] = immutable_json(manifest_path, manifest)
    request_ref = immutable_json(RECORDS / 'repair_requests' / (REQUEST_ID + '.request.json'), request)
    inert_submission = AREA / 'bridge_capacity_preflight' / (REQUEST_ID + '.ref.json')
    immutable_json(inert_submission, request_ref)
    # The inert pointer has the exact basename and real request body reference,
    # but is outside the controller's scanned repair_requests directory.
    fresh_plan, fresh_state = effective_state(args.expected_plan_revision)
    need(time.time() - fresh_state['updated_unix'] <= 120, 'Controller snapshot became stale before final preflight')
    repairs._prepare(fresh_plan, fresh_state, RECORDS, inert_submission)
    checks_path = AREA / 'paper_argmix_train000_bridge_capacity_apply_checks.json'
    if checks_path.exists():
        prior_check = json.loads(checks_path.read_text())
        need(prior_check['plan_revision'] == args.expected_plan_revision, 'Prior apply check revision differs')
    else:
        immutable_json(checks_path, check)
    # Last operation is the dispatch boundary. No task is launched by this helper.
    published = immutable_json(submission, request_ref)
    print(json.dumps(dict(status='submitted_to_controller', submission_ref=published, request_ref=request_ref,
                          archive_manifest_ref=request['archive_manifest_ref'], checks=check), indent=2))


if __name__ == '__main__':
    main()
