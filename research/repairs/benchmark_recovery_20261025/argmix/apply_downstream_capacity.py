"""Check, then explicitly archive/submit the sealed ARGMix downstream repair.

Default is read-only. ROOT reviews and invokes --apply. Only the failed
1522-byte support leaf is archived; all four completed predecessors remain.
Only the final atomic repair-request pointer dispatches work to the controller.
No task launch, signals, scientific conversion, live source or reservation edit.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
from arggnn.benchmarking.paper_campaign import runtime as r, repairs, processes
from arggnn.benchmarking.paper_campaign.scheduler import validate

AREA = r.ROOT / 'research/repairs/benchmark_recovery_20261025/argmix'
RECORDS = r.ROOT / '.runtime/benchmarking/paper_campaign_20261024_v2'
DRAFT_REF = dict(path=str((AREA / 'downstream_capacity_repair_draft.json').relative_to(r.ROOT)),
                 sha256='603609abb2ced88421fc1ea09b3adc73ca79cf64da4fc7b6c634129eaa73f52b')
FAILED = 'paper_argmix_published_recon_train_000.support'
SCOPE = 'experiments/paper_argmix_1611_rebuild_20261024_v1'
IDENTIFIER = 'argmix_downstream_capacity_v1'
MANIFEST = AREA / 'support_capacity_actual_archive.json'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def ordinary(value):
    p = Path(value)
    p = p if p.is_absolute() else r.ROOT / p
    need(p.is_relative_to(r.ROOT) and p.resolve() == p, 'Canonical project path required')
    for part in (p, *p.parents):
        if part == r.ROOT.parent:
            break
        need(not part.is_symlink(), 'No symbolic links allowed: ' + str(part))
    return p


def sealed(reference):
    p = ordinary(reference['path'])
    expected = {k:reference[k] for k in ('path', 'sha256')}
    need(p.is_file() and r.ref(p) == expected, 'Seal changed: ' + str(p))
    if 'bytes' in reference:
        need(p.stat().st_size == reference['bytes'], 'Sealed size changed')
    return p


def immutable(filename, value):
    p = ordinary(filename)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if p.exists():
        need(p.is_file() and p.read_bytes() == data, 'Immutable metadata differs: ' + str(p))
        return r.ref(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    staging = p.with_name(p.name + '.staging.' + str(os.getpid()))
    with staging.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(staging, p)  # Atomic, no replacement of an existing immutable name.
    staging.unlink()
    return r.ref(p)


def effective(revision):
    state = json.loads((RECORDS / 'state.json').read_text())
    need(state.get('plan_revision', 0) == revision, 'Current revision differs from explicit argument')
    plan = r.read(state['plan_ref'])
    for ref in plan['controller_source_refs']:
        sealed(ref)
    receipts = [r.read(ref) for ref in state.get('applied_repairs', {}).values()]
    for receipt in sorted(receipts, key=lambda x:x['resulting_plan_revision']):
        request = r.read(receipt['request_ref'])
        changes = {x['id']:x for x in request['replacement_tasks']}
        plan['tasks'] = [changes.get(x['id'], x) for x in plan['tasks']]
    return plan, state


def inventory(leaf, archive):
    need(leaf.is_dir(), 'Exact failed evidence leaf required')
    items = list(leaf.iterdir())
    need(len(items) == 1 and items[0].name == 'actual_wait.json', 'Failed leaf inventory changed')
    item = ordinary(items[0])
    record = archive['archive_files'][0]
    need(item.is_file() and item.stat().st_size == record['bytes'] == 1522
         and r.ref(item)['sha256'] == record['sha256'], 'Failed receipt changed')
    wait = json.loads(item.read_text())
    need(wait['status'] == 'failed' and wait['attempt'] == 3
         and wait['pid'] == archive['failed_pid'] and wait['process_start_ticks'] == archive['failed_process_start_ticks']
         and wait['actual_returncode'] == -15 and wait['actual_wait'] is True
         and wait['owned_process_group_absent'] is True, 'Expected drained failed support attempt required')
    return wait


def precheck(revision, draft, archive):
    plan, state = effective(revision)
    request = copy.deepcopy(draft['request_template'])
    request['expected_plan_revision'] = revision
    need(state['plan_ref'] == request['expected_plan_ref'], 'Original plan changed')
    targets = set(request['reset_task_ids'])
    replacements = {x['id']:x for x in request['replacement_tasks']}
    by_id = {x['id']:x for x in plan['tasks']}
    need(len(targets) == 146 and len(replacements) == 55
         and set(request['expected_attempts']) == targets and set(replacements) <= targets,
         'Exact146-target/55-replacement batch required')
    source, destination = ordinary(archive['source_directory']), ordinary(archive['archive_directory'])
    need(not (source.exists() and destination.exists()), 'Source and archive both exist')
    wait = inventory(source if source.exists() else destination, archive)
    row = state['tasks'][FAILED]
    for key in ('status','attempt','pid','process_start_ticks','actual_returncode','actual_wait','owned_process_group_absent'):
        need(row[key] == wait[key], 'State/failed receipt mismatch: ' + key)
    need(not processes.session_members(row['pid']), 'Failed task still owns live session members')
    for name in targets:
        task, row = by_id[name], state['tasks'][name]
        need(row['status'] in ('failed','blocked') and task['kind'] != 'external', 'Target status/kind changed: ' + name)
        need(row.get('attempt', 1) == request['expected_attempts'][name], 'Attempt changed: ' + name)
        out = ordinary(task['output_directory'])
        need(out.is_relative_to(r.ROOT / 'experiments') and (name == FAILED or not out.exists()),
             'Unexpected old output scope: ' + name)
    unique_refs = {}
    for name, replacement in replacements.items():
        old = by_id[name]
        changed = {k for k in set(old) | set(replacement) if old.get(k) != replacement.get(k)}
        if name.endswith('.inferred'):
            need(changed == {'argv','source_refs','max_rss_bytes','max_output_bytes'}
                 and replacement['max_rss_bytes'] == 128*2**30
                 and replacement['max_output_bytes'] == 16*2**30+8*2**20,
                 'Unexpected bridge change: ' + name)
        else:
            need(name.endswith(('.support','.features','.treeloom_observation'))
                 and changed == {'max_rss_bytes'} and replacement['max_rss_bytes'] == 32*2**30,
                 'Unexpected reader change: ' + name)
        for ref in replacement.get('source_refs', []):
            unique_refs[json.dumps(ref,sort_keys=True)] = ref
    for ref in unique_refs.values():
        sealed(ref)
    candidate = copy.deepcopy(plan)
    candidate['tasks'] = [replacements.get(t['id'], t) for t in plan['tasks']]
    validate(candidate)
    for name, reference in archive['preserve_succeeded_tasks'].items():
        row = state['tasks'][name]
        need(name not in targets and row['status'] == 'succeeded' and row['actual_returncode'] == 0
             and row['actual_wait'] is True and row['owned_process_group_absent'] is True
             and row['result_ref'] == reference, 'Completed predecessor changed: ' + name)
        sealed(reference)
    terminal = r.read(archive['preserve_native_terminal_ref'])
    need(terminal['successful'] is True and len(terminal['native_stage_waits']) == 144
         and all(x['returncode'] == 0 and x['actual_wait_consumed'] for x in terminal['native_stage_waits']),
         'Native terminal completion changed')
    for ref in archive['preserve_large_files']:
        p = ordinary(ref['path'])
        need(p.is_file() and p.stat().st_size == ref['bytes'], 'Preserved large file size changed')
    sealed(archive['old_log_ref'])
    ready = [name for name in targets if all(state['tasks'][d]['status'] == 'succeeded' for d in by_id[name]['dependencies'])]
    need(ready == [FAILED], 'Only support may be immediately ready; review changed admission boundary')
    now = time.time()
    need(-5 <= now-state['updated_unix'] <= 120, 'Controller snapshot is stale')
    ready_growth = replacements[FAILED]['max_output_bytes']
    need(state['project_reserved_bytes']+ready_growth+2**20 <= plan['storage_ceiling_bytes'] == 990000000000,
         'Immediate support storage headroom insufficient')
    scope_bytes = r.scope_bytes(ordinary(SCOPE))
    limit = next(x['max_bytes'] for x in plan['scope_limits'] if x['path'] == SCOPE)
    need(scope_bytes+ready_growth <= limit == 40*2**30, 'Unchanged ARGMix scope headroom insufficient')
    need(shutil.disk_usage(source.parent).free >= ready_growth+2**20, 'Filesystem headroom insufficient')
    checks = dict(checked_unix=now,plan_revision=revision,reset_tasks=146,replacement_tasks=55,
                  ready_task_ids=ready,immediate_output_growth_bytes=ready_growth,
                  project_reserved_bytes=state['project_reserved_bytes'],scope_bytes=scope_bytes,
                  scope_limit_bytes=limit,failed_session_members=[],archive_bytes=1522,
                  preserved_succeeded_tasks=list(archive['preserve_succeeded_tasks']),
                  source_pins_verified=len(unique_refs),future_bridge_admission='Existing prerequisite chain and scheduler resources unchanged; profile unused growth is not an atomic controller reservation.')
    return plan,state,request,checks,source,destination


def existing(request, draft):
    expected = copy.deepcopy(draft['request_template'])
    expected['expected_plan_revision'] = request['expected_plan_revision']
    expected['archive_manifest_ref'] = request['archive_manifest_ref']
    need(request == expected and ordinary(request['archive_manifest_ref']['path']) == MANIFEST,
         'Existing submission differs from sealed draft')
    receipt = r.read(request['archive_manifest_ref'])
    need(receipt['archived_bytes'] == 1522 and receipt['archive_plan_ref'] == draft['archive_plan_ref'],
         'Existing archive receipt differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-plan-revision',type=int,required=True)
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    draft = r.read(DRAFT_REF)
    for name in ('archive_plan_ref','helper_ref','original_generator_ref','repair_API_ref'):
        sealed(draft[name])
    archive = r.read(draft['archive_plan_ref'])
    submission = RECORDS / 'repair_requests' / (IDENTIFIER+'.ref.json')
    state = json.loads((RECORDS/'state.json').read_text())
    if IDENTIFIER in state.get('applied_repairs', {}):
        reference = state['applied_repairs'][IDENTIFIER]
        receipt = r.read(reference)
        existing(r.read(receipt['request_ref']),draft)
        print(json.dumps(dict(status='already_applied',receipt_ref=reference),indent=2));return
    if submission.exists():
        reference = r.read(r.ref(submission))
        existing(r.read(reference),draft)
        processed = state.get('processed_repair_submissions',{}).get(r.ref(submission)['sha256'])
        print(json.dumps(dict(status='already_processed' if processed else 'already_queued',
            submission_ref=r.ref(submission),processed_receipt_ref=processed),indent=2));return
    _,_,request,checks,source,destination = precheck(args.expected_plan_revision,draft,archive)
    if not args.apply:
        print(json.dumps(dict(status='read_only_checks_passed',checks=checks,mutations_performed=False),indent=2));return
    # Streaming checksums of only two explicitly named completed files. Never load tskit.
    for ref in archive['preserve_large_files']:
        sealed(ref)
    # Recheck state/resource boundary after those streaming reads, before mutation.
    _,_,request,checks,source,destination = precheck(args.expected_plan_revision,draft,archive)
    if source.exists():
        need(not MANIFEST.exists(), 'Archive manifest exists while source remains')
        destination.parent.mkdir(parents=True,exist_ok=True)
        need(source.stat().st_dev == destination.parent.stat().st_dev, 'Same-filesystem archive required')
        need(not destination.exists(), 'Archive destination already exists')
        source.rename(destination)
    inventory(destination,archive)
    manifest = dict(schema='paper_argmix_support_failed_leaf_archive_v1',archived_bytes=1522,
                    source_directory=archive['source_directory'],archive_directory=archive['archive_directory'],
                    archive_files=archive['archive_files'],archive_plan_ref=draft['archive_plan_ref'],
                    preserve_succeeded_tasks=archive['preserve_succeeded_tasks'],
                    preserve_large_files=archive['preserve_large_files'],
                    files_moved_or_deleted_from_completed_or_native_scopes=[],
                    failed_task_id=FAILED,failed_attempt=3,failed_pid=archive['failed_pid'],
                    failed_process_start_ticks=archive['failed_process_start_ticks'],
                    actual_wait=True,owned_process_group_absent=True)
    request['archive_manifest_ref'] = immutable(MANIFEST,manifest)
    body_ref = immutable(RECORDS/'repair_requests'/(IDENTIFIER+'.request.json'),request)
    inert_pointer = AREA/'preflight'/(IDENTIFIER+'.ref.json')
    immutable(inert_pointer,body_ref)
    plan,state = effective(args.expected_plan_revision)
    repairs._prepare(plan,state,RECORDS,inert_pointer)
    checks_path = AREA/'actual_apply_checks.json'
    if not checks_path.exists():
        immutable(checks_path,checks)
    # Final atomic pointer is the sole dispatch boundary; controller owns admission.
    final_ref = immutable(submission,body_ref)
    print(json.dumps(dict(status='submitted_to_controller',submission_ref=final_ref,
        request_ref=body_ref,archive_manifest_ref=request['archive_manifest_ref'],checks=checks),indent=2))


if __name__ == '__main__':
    main()
