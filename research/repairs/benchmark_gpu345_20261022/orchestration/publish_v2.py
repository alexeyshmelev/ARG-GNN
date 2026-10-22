"""Record the user-directed GPU4 migration and verified checkpoint continuation."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import fcntl
import json
from . import storage_workflow_v2 as facade

queue = facade.queue
c = queue.c
HERE = Path(__file__).parent
ACTUAL = HERE / 'actual_v2'


def publish():
    launch = c.read_json(ACTUAL / 'launch_receipt.json')
    reference = launch['plan_ref']
    plan = queue.load_plan(reference, approved=True)
    state = c.read_json(c.contained_path(plan['records_directory']) / 'state.json')
    am = state['tasks']['AM_treeloom_fit']
    queue.need(am['status'] == 'running', 'Actual GPU4 worker must be running before reporting a completed move')
    job = c.read_ref(am['job_ref'])
    output = c.contained_path(job['call']['spec']['output_directory'])
    binding = c.read_json(output / 'gpu_binding.json')
    restored = c.read_json(output / 'resume_state.json')
    queue.need(binding['physical_gpu'] == 4 and binding['gpu_memory_policy'] == 'full_device'
               and restored['completed_updates_replayed'] == 0, 'Actual uncapped GPU4 startup and exact checkpoint continuation required')
    controls = c.read_json(HERE / 'GPU2_AM_control_receipt.json')
    old_job = c.read_ref(controls['job_ref'])
    old_wait = c.read_json(c.contained_path(old_job['task']['output_directory']) / 'actual_wait.json')
    queue.need(old_wait['owned_process_group_absent'] is True, 'Previous GPU2 task must actually drain')
    counts = dict(Counter(entry['status'] for entry in state['tasks'].values()))
    live = []
    for row in plan['workflow_external_predecessors']:
        _, entry = queue.support.predecessor._source(row)
        if entry['status'] == 'running':
            live.append(row['source_task_id'])
    stamp = datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
    status = dict(observed_moscow=stamp, plan_ref=reference, counts=counts,
        controller_identity=launch['controller_identity'], currently_authorized_physical_gpus=[3,4,5],
        gpu_memory_policy='full_device', American_physical_gpu=4, American_status=am['status'],
        American_completed_updates_restored=restored['completed_updates_restored'],
        American_next_optimizer_update=restored['next_optimizer_update'],
        American_expected_updates=600, American_completed_updates_replayed=0,
        American_GPU_binding_ref=c.file_ref(output / 'gpu_binding.json'),
        American_resume_receipt_ref=c.file_ref(output / 'resume_state.json'),
        previous_GPU2_owned_group_absent=True, healthy_predecessors=sorted(live),
        storage_projection_bytes=queue.storage_projection(plan,state),
        storage_ceiling_bytes=plan['storage_ceiling_bytes'], new_matched_ancestry_scores=None,
        model_inputs_and_training_budget_unchanged=True, unrelated_processes_touched=False,
        controlled_processes=[controls['supervisor_identity']])
    with (c.ROOT / '.runtime/benchmarking/workflow_reporting/reporting.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue.save(ACTUAL / 'current_status.json',status)
        queue.save('research/benchmark_current_verified_status.json',status)
        readme=c.ROOT/'README.md'
        text=readme.read_text()
        first='<!-- BEGIN CURRENT BENCHMARK DIAGNOSIS -->'
        last='<!-- END CURRENT BENCHMARK DIAGNOSIS -->'
        block='\n'.join([first,
            '**Verified '+stamp+': only physical GPUs 3, 4 and 5 are authorized.** American TreeLoom has moved from GPU2 to GPU4 and is running without an artificial GPU memory limit.', '',
            '| Workstream | Current status |', '| --- | --- |',
            '| American TreeLoom, GPU4 | Restored '+str(restored['completed_updates_restored'])+'/600 completed updates and all optimizer state. Next update: '+str(restored['next_optimizer_update'])+'. Completed updates replayed: 0 |',
            '| Native ARGformer, GPU3 | The ongoing fifth validation pretraining job is preserved. Completed fourth validation embeddings remain available for ancestry prediction |',
            '| European TreeLoom, GPU5 | Both the existing four-class fit and the AP preparation job are preserved under their original supervisors |',
            '| Fresh matched ancestry accuracy and F1 | Pending complete training and common scoring. This hardware move changes no scientific method or benchmark input |', '',
            'The old GPU2 worker exited under its own supervisor, and its exact cancellation wait is retained. No unrelated process was signalled. Existing GPU3/5 source files, approvals and jobs remain unchanged. Legacy GPU fields in the policy exist solely for those original workers. New execution is governed by the current explicit GPU3/4/5 authority.', '',
            'Conservative storage projection: '+format(status['storage_projection_bytes']/1e9,'.2f')+' GB, below the 980 GB operational ceiling and the 1 TB project limit. Full GPU memory remains enabled.', '',
            '[Verified migration](research/benchmark_current_verified_status.json) · [Current queue](research/benchmark_gpu345_current.json)',last])
        text=text[:text.index(first)]+block+text[text.index(last)+len(last):]
        text=text.replace('All computation and Git commands run on anogena, using physical GPUs 2, 3 and 5.',
                          'All computation and Git commands run on anogena, using physical GPUs 3, 4 and 5.')
        old_first='<!-- readable-full-vram-queue:start -->'
        old_last='<!-- readable-full-vram-queue:end -->'
        history='<!-- preserved-worker-controllers:start -->'
        if old_first in text and history in text and text.index(old_first)<text.index(history):
            a=text.index(old_first);b=text.index(old_last,a)+len(old_last)
            old_block=text[a:b];text=text[:a]+text[b:]
            insertion='<summary>Preserved controllers and earlier attempts</summary>'
            text=text.replace(insertion,insertion+'\n\n'+old_block,1)
        readme.write_text(text)
        decisions_path=c.ROOT/'research/experiment_decisions.json'
        decisions=c.read_json(decisions_path)
        identifier='benchmark_GPU2_to_GPU4_migration_20261022'
        if not any(row['id']==identifier for row in decisions['decisions']):
            decisions['decisions'].append(dict(id=identifier,registered_utc=datetime.now(timezone.utc).isoformat(),
                hypothesis='Continue the frozen benchmark on only the currently authorized GPUs3/4/5',
                method='Cancel only the owned GPU2 American worker for hardware migration. Preserve completed checkpoint, all optimizer state and original600-update schedule. Resume on GPU4 with full VRAM. Preserve healthy GPU3/5 jobs and historical evidence.',
                status='GPU4_running_after_verified_checkpoint_restore',architecture_search=False,
                conclusion='The GPU2 job drained with actual direct cancellation wait. GPU4 initialized successfully and restored'+str(restored['completed_updates_restored'])+'completed updates without replay. This is an operational move, with no new ancestry score.',
                repeat_policy='Never repeat completed updates or change biological/scientific inputs for a hardware migration. Future workers may use only GPUs3/4/5. Preserve unrelated processes.',
                evidence=[reference['path'],str((ACTUAL/'launch_receipt.json').relative_to(c.ROOT)),
                    str((ACTUAL/'current_status.json').relative_to(c.ROOT)),
                    str((HERE/'GPU2_AM_control_receipt.json').relative_to(c.ROOT))]))
            decisions['resource_updates'].append(dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                allowed_physical_GPUs=[3,4,5],maximum_project_GPUs=3,gpu_memory_policy='full_device',
                replaces_physical_GPU=2,replacement_physical_GPU=4,project_storage_limit_bytes=1000000000000))
        queue.save(decisions_path,decisions)
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':
    publish()
