"""Adapt the frozen fifteen-task four-class author baseline to CPU execution."""
import copy
import difflib
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r
from arggnn.benchmarking.paper_ancestralpaths_four.workflow import make_call
from research.joint_evaluation_20261029.queue.build_proposal import ROOT, HERE, ref, read, save


def main():
    source_ref = ref('configs/benchmarks/paper_ancestralpaths_four_20261027.json')
    source = read(source_ref)
    audit = read(ref(HERE/'cnn4_readiness.json'))
    assert audit['plan_sha256'] == source_ref['sha256'] and not audit['reference_mismatches']
    for pin in audit['source_refs']:
        assert ref(pin['path']) == pin
    tasks = []
    for original in source['tasks']:
        assert original['resource'] == 'cpu'
        args = copy.deepcopy(original['factory_arguments'])
        call = make_call(output_directory=original['output_directory'], **args)
        assert call['spec']['arguments'] == args['arguments']
        task = {k:copy.deepcopy(original[k]) for k in ('id','dependencies','output_directory','max_output_bytes','max_rss_bytes')}
        task.update(kind='cpu', python=call['python'], cpu_threads=4, max_seconds=9223372036854775807,
                    call={k:call[k] for k in ('module','function','spec')}, source_refs=call['source_refs'])
        for pin in task['source_refs']:
            assert ref(pin['path']) == pin
        assert not r.path(task['output_directory']).exists()
        tasks.append(task)
    assert len(tasks) == 15 and sum(t['max_output_bytes'] for t in tasks) == 384*2**20
    allocation_ref = save('cnn4_allocation.json', dict(schema='evaluation_suballocation_v1', root_confirmed=True,
            authorization='Root confirmed 640 MiB inside existing 1 GiB evaluation allowance; current evaluation about 14 MB.',
            cnn_reserved_bytes=640*2**20, payload_bytes=384*2**20, metadata_bytes=256*2**20,
            existing_evaluation_allowance_bytes=2**30))
    native_ref = ref(HERE/'argmix_plan.json');native = read(native_ref)
    original_scheduler = (ROOT/'src/arggnn/benchmarking/paper_campaign/scheduler.py').read_text()
    scheduler = original_scheduler
    replacements = {
        'from . import runtime as r':'from arggnn.benchmarking.paper_campaign import runtime as r\nfrom research.joint_evaluation_20261029.queue import cnn4_admission as admission',
        'from . import processes':'from arggnn.benchmarking.paper_campaign import processes',
        'from . import repairs':'from arggnn.benchmarking.paper_campaign import repairs',
        'from . import continuation':'from arggnn.benchmarking.paper_campaign import continuation',
        "if plan['storage_ceiling_bytes'] > 1_000_000_000_000:":"if plan['storage_ceiling_bytes'] > 1_200_000_000_000:",
        "lock = r.path('.runtime/benchmarking/paper_campaign_global.lock').open('a+')":"lock = (directory / 'cnn4.lock').open('a+')",
        'seeded = continuation.initial_rows(plan)':'admission.admit(plan)\n    seeded = None',
        '                remaining_growth = sum(':'                remaining_growth = plan[\'foreign_ram_reserve_bytes\'] + sum(',
        '                    scope_exceeded = any(':'                    scope_exceeded = not admission.within_shared_allowance(plan) or any(',
        "                active = [t for t in plan['tasks'] if t['id'] in running]":"                if not admission.within_shared_allowance(plan):\n                    row['waiting_reason'] = 'Combined evaluation scope allowance'\n                    continue\n                active = [t for t in plan['tasks'] if t['id'] in running]"}
    for old,new in replacements.items():
        assert scheduler.count(old) == 1, old
        scheduler = scheduler.replace(old,new)
    (HERE/'cnn4_scheduler.py').write_text(scheduler)
    (HERE/'cnn4_scheduler.diff').write_text(''.join(difflib.unified_diff(original_scheduler.splitlines(True), scheduler.splitlines(True), fromfile='paper_campaign/scheduler.py', tofile='queue/cnn4_scheduler.py')))
    plan = dict(schema='paper_campaign_v1', physical_gpus=[3,4,5],
            records_directory=str((HERE/'cnn4_records_v1').relative_to(ROOT)),
            storage_ceiling_bytes=1_200_000_000_000, reserved_existing_project_bytes=native['global_committed_bytes'],
            controller_metadata_allowance_bytes=256*2**20, storage_ledger_ref=native['storage_ledger_ref'],
            combined_native_queue_plan_ref=native_ref, allocation_ref=allocation_ref, original_factory_plan_ref=source_ref,
            combined_evaluation_allowance=dict(paths=['research/joint_evaluation_20261029','experiments/results/paper_ancestralpaths_four_20261027'], max_bytes=2**30),
            max_parallel_tasks=1, max_cpu_tasks=1, max_jobs_per_gpu=1, existing_workers=[],
            foreign_ram_reserve_bytes=1472*2**30, tasks=tasks,
            controller_source_refs=[ref(HERE/'cnn4_scheduler.py'),ref(HERE/'cnn4_admission.py'),ref('src/arggnn/benchmarking/paper_campaign/runtime.py'),ref('src/arggnn/benchmarking/paper_campaign/processes.py')])
    save('cnn4_plan.json', plan)
    save('cnn4_proposal_receipt.json', dict(plan_ref=ref(HERE/'cnn4_plan.json'),original_factory_plan_ref=source_ref,
            native_specs_unchanged=True, task_ids_and_output_paths_unchanged=True, byte_and_ram_caps_unchanged=True,
            only_wall_policy_changed_to_INT64_MAX=True, all_cpu=True, launched=False,
            project_committed_bytes=native['global_committed_bytes']+640*2**20))
    print(json.dumps(dict(prepared=True,tasks=15,all_cpu=True,project_committed_bytes=native['global_committed_bytes']+640*2**20,launched=False)))


if __name__ == '__main__':
    main()
