"""Build an additive reviewed successor; never edit a live plan or controller."""
import copy
import difflib
import json
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue import successor_v2_support as support

HERE = Path(__file__).resolve().parent


def main():
    effective_ref = r.ref(HERE/'original_effective_plan.json')
    original = r.read(effective_ref)
    state_path = '.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json'
    state = json.loads(r.path(state_path).read_text())
    native_ref = r.ref(HERE/'sidecar_plan.json')
    mix_ref = r.ref(HERE/'argmix_plan.json')
    sidecars = [dict(plan_ref=reference, state_path=r.read(reference)['records_directory']+'/state.json') for reference in (native_ref, mix_ref)]
    aliases = {}
    for sidecar in sidecars:
        for task in r.read(sidecar['plan_ref'])['tasks']:
            if task['kind'] != 'external':
                support.need(task['id'] not in aliases, 'Disjoint exact sidecar owners')
                aliases[task['id']] = sidecar
    tasks = []
    amendments = {}
    for old in original['tasks']:
        task = copy.deepcopy(old)
        if task['id'] in aliases:
            owner = aliases[task['id']]
            task = dict(id=task['id'], kind='external', dependencies=task['dependencies'],
                        state_path=owner['state_path'], source_task_id=task['id'], sidecar_plan_ref=owner['plan_ref'])
        elif state['tasks'][task['id']]['status'] == 'pending':
            task['max_seconds'] = support.MAX_SECONDS
            if 'argv' in task:
                args = task['argv']
                if '--timeout-seconds' in args:
                    args[args.index('--timeout-seconds')+1] = str(support.MAX_SECONDS)
                if task['id'].endswith('.native_relate'):
                    support.need(args[2:5] == ['-m','arggnn.benchmarking.paper_argmix.paper_argmix_cli','native-relate'], 'Exact original native CLI required')
                    args[3] = 'research.joint_evaluation_20261029.queue.successor_native'
                    task['source_refs'].extend([r.ref(HERE/'successor_native.py'), r.ref(HERE/'argmix_native_wrapper.py')])
            amendments[task['id']] = ['Remove arbitrary wall limit; unchanged native science, seeds, output paths, bytes and RAM']
        tasks.append(task)
    superseded = {}
    for task in tasks:
        name = task['id']
        if name == 'paper_argmix_treeloom.train':
            superseded[name] = 'joint_argmix.train'
        elif name == 'paper_argmix_treeloom.prediction_barrier':
            superseded[name] = 'joint_argmix.prediction_barrier'
        elif name == 'paper_argmix_treeloom.score':
            superseded[name] = 'joint_argmix.score'
        elif name.startswith('paper_argmix_') and name.endswith('.treeloom_prediction'):
            parent = name.removeprefix('paper_argmix_').removesuffix('.treeloom_prediction')
            superseded[name] = 'joint_argmix.' + parent + '.prediction'
    support.need(len(superseded) == 11, 'Exact eleven legacy ARGMix model stages')
    for name in superseded:
        row = state['tasks'][name]
        support.need(row['status'] in ('pending','blocked') and 'pid' not in row, 'Supersession may only replace unlaunched legacy model work')
    new_barrier = dict(id='joint_argmix.prediction_barrier',kind='external',dependencies=[],
        source_task_id='joint_argmix.prediction_barrier',producer_model_id='treeloom_joint_argmix_structure_v1',
        future_wait_path='experiments/results/joint_argmix_20261029/prediction_barrier/actual_wait.json',
        expected_result_path='experiments/results/joint_argmix_20261029/prediction_barrier/barrier.json',
        config_ref=r.ref('research/benchmark_rebuild_20261024/argmix/paper_argmix_recipe.json'))
    paired = next(t for t in tasks if t['id']=='paper_argmix_paired.prediction_barrier')
    paired['dependencies'] = [new_barrier['id'] if x=='paper_argmix_treeloom.prediction_barrier' else x for x in paired['dependencies']]
    paired['argv'][paired['argv'].index('--other-barrier')+1] = str(r.path(new_barrier['expected_result_path']))
    tasks.append(new_barrier)
    supersession_ref = r.save(HERE/'successor_v2_supersession.json',dict(
        schema='explicit_legacy_argmix_supersession_v1',root_authorized=True,
        authorization='Root explicitly requested the new joint core as replacement for never-launched legacy TreeLoom ARGMix fit/predict/score; preserve every old row and native method.',
        model_id='treeloom_joint_argmix_structure_v1',replacements=superseded,
        added_external_task=new_barrier,changed_comparison_task_id=paired['id'],
        comparison_change='Only second paired-barrier method and dependency become the new joint model; original target support and native metric functions unchanged.',
        scientific_completion_claimed=False))
    source_path = r.path('src/arggnn/benchmarking/paper_campaign/scheduler.py')
    source = source_path.read_text()
    scheduler = source
    replacements = {
        'from . import runtime as r': 'from arggnn.benchmarking.paper_campaign import runtime as r\nfrom research.joint_evaluation_20261029.queue import successor_v2_support as support\nfrom research.joint_evaluation_20261029.queue import successor_v2_gpu as gpu_ownership',
        'from . import processes': 'from arggnn.benchmarking.paper_campaign import processes',
        'from . import repairs\n': '',
        'from . import continuation': 'from arggnn.benchmarking.paper_campaign import continuation',
        "if plan['storage_ceiling_bytes'] > 1_000_000_000_000:": "if plan['storage_ceiling_bytes'] > 1_200_000_000_000:",
        '                    scope_exceeded = any(': '                    scope_exceeded = not support.metadata_within_bound(plan) or any(',
        "for key in ('pending', 'running', 'succeeded', 'failed', 'blocked')": "for key in ('pending', 'running', 'succeeded', 'failed', 'blocked', 'superseded')",
        "all(row['status'] in ('succeeded', 'failed', 'blocked') for row in state['tasks'].values())": "all(row['status'] in ('succeeded', 'failed', 'blocked', 'superseded') for row in state['tasks'].values())",
        "if not task.get('result_ref') and not task.get('state_path'):": "if not task.get('result_ref') and not task.get('state_path') and not task.get('future_wait_path'):",
        '    running = {}': '    running = {}\n    gpu_leases = {}',
        '                del running[task_id]': '                del running[task_id]\n                gpu_ownership.release(gpu_leases.pop(task_id, None))',
        "                    candidates = [i for i,g in snapshot.items() if g['free_bytes'] >= task['minimum_free_gpu_bytes']": "                    candidates = [i for i,g in snapshot.items() if i in gpu_ownership.allowed(plan) and g['free_bytes'] >= task['minimum_free_gpu_bytes']",
        '                output.parent.mkdir(parents=True, exist_ok=True)': "                lease = gpu_ownership.acquire(gpu) if gpu is not None else None\n                if lease is False:\n                    row['waiting_reason'] = 'Root or successor owns exclusive GPU3 lock'\n                    continue\n                gpu_leases[task['id']] = lease\n                output.parent.mkdir(parents=True, exist_ok=True)",
        'stdout=log, stderr=subprocess.STDOUT, start_new_session=True)': 'stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(() if lease is None else (lease,)))',
        '        r.save(state_path, state)\n\n\ndef main():': '        r.save(state_path, state)\n        for lease in gpu_leases.values():\n            gpu_ownership.release(lease)\n\n\ndef main():',
        'seeded = continuation.initial_rows(plan)': 'seeded = support.initial_rows(plan)',
        "        state['continuation_ref'] = plan['continuation_ref']\n        state['archived_attempt_bytes'] = r.read(plan['continuation_ref'])['archived_attempt_bytes']": "        state['handoff_ref'] = plan['handoff_ref']",
        '            repairs.apply_pending(plan, state, directory)\n': '',
        "                        row.update(status='succeeded', result_ref=result, reused=True)": "                        row.update(status='succeeded', result_ref=result, reused=True, **support.external_evidence(task))",
        "            state['gpu_observation'] = snapshot": "            state['gpu_observation'] = snapshot\n            accounting = support.commitment(plan)\n            state['closed_writer_accounting'] = accounting",
        "                if baseline + reserved + new_growth + pipeline_floor(plan, state, sizes, set(running), task) > plan['storage_ceiling_bytes']:": "                if not accounting['admitted'] or not support.metadata_within_bound(plan) or not support.task_fits_scope(plan, state, task, [t for t in plan['tasks'] if t['id'] in running]):",
        '                remaining_growth = sum(': "                remaining_growth = plan['foreign_ram_reserve_bytes'] + sum(",
        "            state['project_reserved_bytes'] = baseline + reserved + pipeline_floor(plan, state, sizes, set(running))": "            state['project_reserved_bytes'] = accounting['total_bytes']",
    }
    for old, new in replacements.items():
        support.need(scheduler.count(old) == 1, 'Unique scheduler delta: ' + old)
        scheduler = scheduler.replace(old, new)
    start = scheduler.index('def external_result(task):')
    end = scheduler.index('\n\ndef stop_owned(', start)
    scheduler = scheduler[:start] + 'def external_result(task):\n    return support.external_result(task)\n' + scheduler[end:]
    (HERE/'successor_v2_scheduler.py').write_text(scheduler)
    (HERE/'successor_v2_scheduler.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True), scheduler.splitlines(True), fromfile='paper_campaign/scheduler.py', tofile='queue/successor_v2_scheduler.py')))
    sources = [r.ref(HERE/name) for name in ('successor_v2_scheduler.py','successor_v2_support.py','successor_v2_handoff.py','successor_native.py','successor_v2_gpu.py','future_merge.py','argmix_native_wrapper.py','argmix_production_admission.py')]
    sources += [r.ref('src/arggnn/benchmarking/paper_campaign/'+name) for name in ('runtime.py','processes.py','continuation.py')]
    template = dict(schema='paper_campaign_v1', physical_gpus=[3,4,5], tasks=tasks,
        records_directory=str((HERE/'successor_records_v2').relative_to(r.ROOT)),
        storage_ceiling_bytes=1_200_000_000_000, reserved_existing_project_bytes=0,
        controller_metadata_allowance_bytes=256*2**20, controller_source_refs=sources,
        scope_limits=copy.deepcopy(original['scope_limits']), pipeline_storage_floors=[],
        max_parallel_tasks=2, max_cpu_tasks=2, max_jobs_per_gpu=1,
        foreign_ram_reserve_bytes=768*2**30, existing_workers=[], alias_task_ids=sorted(aliases),
        source_effective_plan_ref=effective_ref, operational_amendments=amendments,
        supersession_ref=supersession_ref,added_task_ids=[new_barrier['id']],
        root_gpu_priority_release_path=str((HERE/'successor_v2_gpu3_release.json').relative_to(r.ROOT)),
        required_root_model_chains=['american_joint','argmix_joint','european_four_joint'])
    template_ref = r.save(HERE/'successor_v2_template.json', template)
    fields = (Path('/proc')/str(state['controller_pid'])/'stat').read_text().rsplit(')',1)[1].split()
    config = dict(schema='idle_campaign_successor_v2_config_v1', source_refs=sources,
        original_state_path=state_path, effective_plan_ref=effective_ref,
        prior_controller=dict(pid=state['controller_pid'], start_ticks=fields[19]),
        original_identity=dict(plan_ref=state['plan_ref'], plan_revision=state.get('plan_revision',0), applied_repairs=state['applied_repairs']),
        sidecars=sidecars, successor_template_ref=template_ref,
        storage_ledger_ref=r.ref(HERE/'storage_ledger.json'),
        american_allocation_ref=r.ref('research/joint_evaluation_20261029/root/american_additive_allocation_v1.json'),
        argmix_joint_allocation_ref=r.ref('research/joint_evaluation_20261029/root/argmix_joint_additive_allocation_v1.json'),
        argmix_production_admission_path='research/joint_evaluation_20261029/root/argmix_joint_production_admission_v1.json',
        required_successful_boundary_tasks=['paper_american_argformer_american_test_002_predict','paper_american_argformer_american_test_002_retire'],
        required_claim_block_task='paper_american_argformer_american_test_003_prepare')
    config_ref = r.save(HERE/'successor_v2_config.json', config)
    print(json.dumps(dict(prepared=True, aliases=len(aliases), tasks=len(tasks), config_ref=config_ref, launched=False)))


if __name__ == '__main__':
    main()
