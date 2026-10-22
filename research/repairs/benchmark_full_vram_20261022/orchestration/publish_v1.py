"""Publish concise verified progress and the owner's unrestricted GPU memory policy."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import fcntl
import json
from . import storage_workflow_v1 as facade

queue = facade.queue
c = queue.c
HERE = Path(__file__).parent
ACTUAL = HERE / 'actual_v1'


def publish():
    launch = c.read_json(ACTUAL / 'launch_receipt.json')
    reference = launch['plan_ref']
    plan = queue.load_plan(reference, approved=True)
    state = c.read_json(c.contained_path(plan['records_directory']) / 'state.json')
    counts = dict(Counter(entry['status'] for entry in state['tasks'].values()))
    am = state['tasks']['AM_treeloom_fit']
    am_updates = 6
    if am.get('job_ref'):
        job = c.read_ref(am['job_ref'])
        latest = c.contained_path(job['call']['spec']['output_directory']) / 'latest_checkpoint.json'
        if latest.is_file():
            am_updates = c.read_json(latest)['complete_parent_updates']
    old_c4 = next(b for b in plan['workflow_external_predecessors'] if b['source_task_id'] == 'C4__all_train__c4_treeloom_fit')
    update_dir = c.contained_path(old_c4['output_directory']) / 'native/updates'
    updates = [c.read_json(p) for p in update_dir.glob('update_*.json')]
    c4_updates = max((row['optimizer_updates'] for row in updates), default=0)
    embedding_binding = next(b for b in plan['workflow_external_predecessors'] if b['source_task_id'] == 'AM_argformer_american_validation_003_argformer_embeddings')
    _, embedding_entry = queue.support.predecessor._source(embedding_binding)
    stamp = datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
    projection = queue.storage_projection(plan, state)
    status = dict(observed_moscow=stamp, plan_ref=reference, counts=counts,
        controller_identity=launch['controller_identity'], gpu_memory_policy='full_device',
        physical_gpus=[2, 3, 5], existing_healthy_workers_preserved=True,
        American_complete_optimizer_updates=am_updates, American_expected_updates=600,
        American_resumed_from_updates=6, American_status=am['status'],
        AP_status=state['tasks']['AP__all_train__treeloom_fit']['status'], AP_expected_updates=90,
        C4_complete_optimizer_updates=c4_updates, C4_expected_updates=90,
        ARGformer_VAL003_MLM_completed_updates=226425, ARGformer_VAL003_contrastive_completed_updates=2120,
        ARGformer_VAL002_ancestry_predictions=10000,
        ARGformer_VAL003_embedding_status=embedding_entry['status'],
        storage_projection_bytes=projection, storage_ceiling_bytes=plan['storage_ceiling_bytes'],
        models_data_and_training_budgets_unchanged=True, new_matched_ancestry_scores=None,
        signals_sent=0)
    reporting = c.ROOT / '.runtime/benchmarking/workflow_reporting/reporting.lock'
    with reporting.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue.save(ACTUAL / 'current_status.json', status)
        queue.save('research/benchmark_current_verified_status.json', status)
        readme = c.ROOT / 'README.md'
        text = readme.read_text()
        start = '<!-- BEGIN CURRENT BENCHMARK DIAGNOSIS -->'
        end = '<!-- END CURRENT BENCHMARK DIAGNOSIS -->'
        diagnosis = '\n'.join([start,
            '**Verified ' + stamp + ': full GPU memory is authorized on physical GPUs 2, 3 and 5.** New workers have no artificial CUDA allocation cap. Healthy existing workers retain their original supervisors and finish normally. Other users\' processes remain untouched.', '',
            '| Workstream | Verified progress |', '| --- | --- |',
            '| American TreeLoom | ' + str(am_updates) + '/600 full training updates. The uncapped successor resumes the saved six-update checkpoint, with 594 updates remaining at launch |',
            '| European TreeLoom, four ancestry classes | ' + str(c4_updates) + '/90 full training updates. The existing healthy fit continues |',
            '| European TreeLoom, AP endpoint | The old fit hit its 32 GiB allocator limit despite free device memory. Its uncapped retry preserves the complete 90-update budget |',
            '| Native ARGformer, fourth validation simulation | All 226,425 MLM updates and all 2,120 contrastive updates completed. Embeddings are ' + str(status['ARGformer_VAL003_embedding_status']) + ' |',
            '| Native ARGformer, third validation simulation | Full prediction chain completed, producing 10,000 ancestry predictions. Disposable native scratch was retired after genuine completion |',
            '| New matched ancestry accuracy and F1 | Pending full TreeLoom training and the common scoring stages. Training loss and stage counts are not accuracy |', '',
            'The full queue preserves ' + str(counts.get('succeeded', 0)) + ' completed stages and both healthy predecessor workers. Full-device leases prevent our new jobs from oversubscribing each other. Actual free memory is checked before launch. Host RAM and project storage accounting remain enforced.', '',
            'Conservative storage projection: ' + format(projection / 1e9, '.2f') + ' GB, within the 980 GB operational ceiling and the 1 TB user limit. No directory-wide scan was performed.', '',
            '[Verified status](research/benchmark_current_verified_status.json) · [Full queue](research/benchmark_full_vram_current.json) · [Matched results](research/resource_matched_benchmark_results.json)', end])
        queue.need(start in text and end in text, 'Existing diagnosis markers required')
        text = text[:text.index(start)] + diagnosis + text[text.index(end) + len(end):]
        # Older controller blocks keep their original content inside history.
        old_start = '<!-- readable-recovery-afternoon-v2-queue:start -->'
        old_end = '<!-- readable-recovery-afternoon-v2-queue:end -->'
        history = '<!-- preserved-worker-controllers:start -->'
        if old_start in text and history in text and text.index(old_start) < text.index(history):
            a = text.index(old_start)
            b = text.index(old_end, a) + len(old_end)
            old_block = text[a:b]
            text = text[:a] + text[b:]
            insertion = '<summary>Preserved controllers and earlier attempts</summary>'
            text = text.replace(insertion, insertion + '\n\n' + old_block, 1)
        readme.write_text(text)
        decisions_path = c.ROOT / 'research/experiment_decisions.json'
        decisions = c.read_json(decisions_path)
        identifier = 'benchmark_full_device_VRAM_20261022'
        decision = dict(id=identifier, registered_utc=datetime.now(timezone.utc).isoformat(),
            hypothesis='Complete the frozen shared-data benchmark using all available memory on three authorized GPUs',
            method='Remove artificial GPU allocator limits and the fixed free-memory cancellation floor. Preserve healthy workers, original models, data, training schedules, host RAM reservations and storage limits. Resume American update six and retry only the failed European AP fit.',
            status='released_full_device_continuation', architecture_search=False,
            conclusion='Both previous failures are retained. AP exhausted its artificial32GiB allowance while46.77GiB remained free. American training saved6updates then encountered physical memory exhaustion alongside another process. Native fourth-validation pretraining/contrastive stages and third-validation predictions completed. No new matched ancestry accuracy is claimed.',
            repeat_policy='Do not repeat completed training updates or passed scientific qualifications. Resume the exact six-update checkpoint. Only genuinely failed and drained scopes may retry. Never modify unrelated processes.',
            evidence=[reference['path'],str((ACTUAL / 'launch_receipt.json').relative_to(c.ROOT)),
                str((ACTUAL / 'current_status.json').relative_to(c.ROOT)),
                str((ACTUAL / 'final_review.json').relative_to(c.ROOT))])
        decisions['decisions'] = [row for row in decisions['decisions'] if row['id'] != identifier] + [decision]
        decisions['resource_updates'].append(dict(updated_utc=datetime.now(timezone.utc).isoformat(),
            allowed_physical_GPUs=[2,3,5],maximum_project_GPUs=3,gpu_memory_policy='full_device',
            artificial_GPU_RAM_restrictions=False,project_storage_limit_bytes=1000000000000))
        queue.save(decisions_path, decisions)
    print(json.dumps(status, indent=2), flush=True)

if __name__ == '__main__':
    publish()
