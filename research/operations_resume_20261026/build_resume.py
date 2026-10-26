"""Prepare a source-bound continuation without launching any benchmark process."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
from arggnn.benchmarking.paper_campaign import runtime as r, scheduler, processes, continuation

OUT = r.path('research/operations_resume_20261026')
PRIOR_RECORDS = r.path('.runtime/benchmarking/paper_campaign_20261024_v2')
NEW_RECORDS = '.runtime/benchmarking/paper_campaign_resume_20261026_v1'
PLAN_PATH = 'configs/benchmarks/paper_campaign_resume_20261026_v1.json'


def main():
    assert not (r.path(NEW_RECORDS) / 'state.json').exists(), 'Already launched'
    prior = r.read(r.ref(OUT / 'effective_prior_plan.json'))
    source_ref = r.ref(OUT / 'prior_state_snapshot.json')
    state = r.read(source_ref)
    assert r.read(r.ref(PRIOR_RECORDS / 'state.json')) == state
    archives = r.read(r.ref(OUT / 'european_gst/archive_instructions.json'))
    for item in archives['archives']:
        assert not processes.session_members(item['original_task_row']['pid'])
        source, target = r.path(item['source_directory']), r.path(item['destination_directory'])
        if source.exists():
            assert not target.exists()
            actual = {str(p.relative_to(source)): r.ref(p)['sha256'] for p in source.rglob('*') if p.is_file()}
            assert actual == {p['relative_path']: p['sha256'] for p in item['files']}
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        assert not source.exists() and target.exists()
        assert r.scope_bytes(target) == item['archived_bytes']
        for file in item['files']:
            assert r.ref(target / file['relative_path'])['sha256'] == file['sha256']
    r.save(OUT / 'archives_applied.json', dict(archives=archives['archives'], archived_bytes=archives['total_archived_bytes']))
    replacements = {task['id']: task for task in r.read(r.ref(OUT / 'european_gst/replacement_tasks.json'))}
    for p in sorted((OUT / 'american').glob('*.replacement_task.json')):
        task = r.read(r.ref(p))
        replacements[task['id']] = task
    assert {'paper_american_treeloom_fit', 'paper_american_argformer_american_validation_001_mlm'} <= set(replacements)
    preferred = {
        'paper_american_treeloom_fit': [3],
        'paper_american_argformer_american_validation_001_mlm': [4],
        'paper_ap_treeloom__all_test__predict': [5],
        'paper_gst_lai_8__all_train__fit': [5, 4, 3],
    }
    for name, order in preferred.items():
        replacements[name]['preferred_gpus'] = order
    plan = copy.deepcopy(prior)
    plan['tasks'] = [replacements.get(t['id'], t) for t in prior['tasks']]
    interim = r.read(r.ref(OUT / 'argmix/european_three_method_interim_task.json'))
    plan['tasks'].append(interim)
    # Keep the preserved interrupted native files in their original charged scope.
    receipts = {}
    by_id = {t['id']: t for t in plan['tasks']}
    for name, evidence_file, leaf in [
        ('paper_american_treeloom_fit', 'treeloom_evidence.json', 'resume_20261026_update238'),
        ('paper_american_argformer_american_validation_001_mlm', 'argformer_evidence.json', 'resume_20261026_batch119500'),
    ]:
        task = by_id[name]
        directory = r.path(task['output_directory'])
        preserved = OUT / 'preserved_root_metadata' / name
        preserved.mkdir(parents=True, exist_ok=True)
        files = []
        for p in sorted(directory.iterdir()):
            if not p.is_file():
                continue
            assert p.stat().st_size < 8 * 2**20, 'Unexpected large root metadata'
            dest = preserved / p.name
            if not dest.exists():
                shutil.copy2(p, dest)
            assert r.ref(p)['sha256'] == r.ref(dest)['sha256']
            files.append(dict(original_ref=r.ref(p), preserved_ref=r.ref(dest)))
        evidence = r.read(r.ref(OUT / 'american' / evidence_file))
        record = dict(task_id=name, output_directory=task['output_directory'],
                      old_wait_ref=r.ref(directory / 'actual_wait.json'),
                      preserved_wait_ref=r.ref(preserved / 'actual_wait.json'),
                      preserved_metadata=files, checkpoint_refs=[evidence['checkpoint_ref']],
                      fresh_native_directory=str((directory / 'native' / leaf).relative_to(r.ROOT)))
        receipts[name] = r.save(OUT / (name + '.existing_output.json'), record)
    # Old controller metadata/logs are no longer counted by the new records scope.
    # Charge them once in addition to the original fixed project reservation.
    old_records_bytes = r.scope_bytes(PRIOR_RECORDS)
    aux_growth = sum(max(0, r.scope_bytes(Path(scope)) - initial)
                     for scope, initial in state.get('auxiliary_initial_bytes', {}).items())
    plan['reserved_existing_project_bytes'] = prior['reserved_existing_project_bytes'] + old_records_bytes + aux_growth
    plan['storage_ceiling_bytes'] = 995_000_000_000
    plan['controller_metadata_allowance_bytes'] = 2**30
    plan['scope_limits'] = [dict(item, max_bytes=128 * 2**30) if item['path'] == 'experiments/paper_argmix_1611_rebuild_20261024_v1' else item for item in prior['scope_limits']]
    for old in prior.get('existing_workers', []):
        p = Path('/proc') / str(old['pid']) / 'stat'
        assert not p.exists() or p.read_text().rsplit(')',1)[1].split()[19] != old['start_ticks']
    plan['existing_workers'] = []
    plan['records_directory'] = NEW_RECORDS
    controller = r.read(r.ref('research/benchmark_rebuild_20261024/integration/master_v2_launch.json'))
    body = dict(schema='paper_campaign_continuation_v1',
                authorization='Owner: Continue our benchmarks, 26 October 2026',
                original_plan_ref=state['plan_ref'], source_state_ref=source_ref,
                effective_prior_plan_ref=r.ref(OUT / 'effective_prior_plan.json'),
                source_plan_revision=state['plan_revision'],
                prior_controller=dict(pid=controller['pid'], start_ticks=controller['start_ticks']),
                reset_task_ids=[name for name,row in state['tasks'].items() if row['status'] in ('failed','blocked')],
                replacement_task_ids=sorted(replacements), new_task_ids=[interim['id']],
                existing_output_receipts=receipts,
                archived_attempt_bytes=state.get('archived_attempt_bytes',0)+archives['total_archived_bytes'],
                retained_old_runtime_bytes=old_records_bytes, prior_auxiliary_growth_bytes=aux_growth,
                no_model_or_dataset_change=True)
    plan['continuation_ref'] = r.save(OUT / 'continuation.json', body)
    plan['controller_source_refs'] = [r.ref(p) for p in sorted(r.path('src/arggnn/benchmarking/paper_campaign').glob('*.py'))]
    plan['deliberate_retry_reason'] = 'Owner authorized continuation after stop; reuse successful stages and durable optimizer checkpoints'
    scheduler.validate(plan)
    unique = {}
    for task in plan['tasks']:
        for source in task.get('source_refs', []):
            assert source['path'] not in unique or unique[source['path']] == source
            unique[source['path']] = source
    for source in unique.values():
        assert r.ref(source['path']) == source, source['path']
    rows = continuation.initial_rows(plan)
    assert sum(row['status']=='succeeded' for row in rows.values()) == 32
    ref = r.save(PLAN_PATH, plan)
    r.save(OUT / 'preflight.json', dict(plan_ref=ref, tasks=len(plan['tasks']),
           preserved_successes=32, pending=len(rows)-32, replacements=sorted(replacements),
           distinct_task_source_pins=len(unique), retained_old_runtime_bytes=old_records_bytes,
           prior_auxiliary_growth_bytes=aux_growth, global_ceiling_bytes=plan['storage_ceiling_bytes'],
           archive_bytes=body['archived_attempt_bytes'], model_work_executed=False))
    print(json.dumps(r.read(r.ref(OUT / 'preflight.json')),indent=2))

if __name__ == '__main__':
    main()
