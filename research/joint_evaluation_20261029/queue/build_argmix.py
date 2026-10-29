"""Prepare native ARGMix CPU sidecar; no claim, launch, signal or live mutation."""
from pathlib import Path
import copy
import difflib
import json
import time
from research.joint_evaluation_20261029.queue.build_proposal import ROOT, HERE, ref, read, save


def main():
    side = read(ref(HERE / 'sidecar_plan.json'))
    original = read(ref(HERE / 'original_effective_plan.json'))
    state = json.loads((ROOT / side['original_state_path']).read_text())
    name = 'paper_argmix_published_recon_train_002.native_relate'
    old = next(t for t in original['tasks'] if t['id'] == name)
    task = copy.deepcopy(old)
    source = task['dependencies'][0]
    dependency = state['tasks'][source]
    assert dependency['status'] == 'succeeded' and dependency['actual_wait'] and dependency['actual_returncode'] == 0
    read(dependency['result_ref'])
    assert state['tasks'][name]['status'] == 'pending'
    assert not (ROOT / task['output_directory']).exists()
    argv = task['argv']
    def option(flag):
        return argv[argv.index(flag) + 1]
    assert argv[3] == 'arggnn.benchmarking.paper_argmix.paper_argmix_cli' and argv[4] == 'native-relate'
    for pin in old['source_refs']:
        assert ref(pin['path']) == pin, pin['path']
    wrapper_sources = [ref(HERE / 'argmix_native_wrapper.py'), ref(HERE / 'claims.py')]
    task['max_seconds'] = 9223372036854775807
    task['source_refs'] += wrapper_sources
    spec = dict(schema='native_claim_execution_spec_v1', source_refs=task['source_refs'],
                original_task_ref=save('argmix_original_task.json', old),
                original_argv=argv, config_ref=ref(option('--config')), commands_ref=dependency['result_ref'],
                output_directory=task['output_directory'], native_work=option('--native-work'),
                native_work_max_bytes=task['additional_mutable_scopes'][0]['max_bytes'],
                payload_max_output_bytes=int(option('--max-output-bytes')))
    identity = 'argmix_native_train002_exact_20261029_v1'
    spec['claim_marker'] = dict(schema='exact_scope_sidecar_claim_v1', owner=identity, task_id=name,
                               original_plan_ref=side['original_identity']['plan_ref'], output_directory=task['output_directory'])
    spec_ref = save('argmix_execution_spec.json', spec)
    task['argv'] = [task['python'], '-B', '-m', 'research.joint_evaluation_20261029.queue.argmix_native_wrapper', '--spec', str(ROOT / spec_ref['path'])]
    task['source_refs'].append(spec_ref)
    external = dict(id=source, kind='external', dependencies=[], result_ref=dependency['result_ref'])
    ledger = read(side['storage_ledger_ref'])
    argformer_reservation = sum(t['max_output_bytes'] for t in side['tasks']) + side['controller_metadata_allowance_bytes']
    base = sum(t['reserved_bytes'] for t in ledger['base_allowances']) + argformer_reservation
    total = base + task['max_output_bytes'] + sum(s['max_bytes'] for s in task['additional_mutable_scopes']) + 2**30
    assert total == read(ref(HERE / 'proposal_receipt.json'))['global_committed_bytes']
    scheduler_path = HERE / 'argmix_scheduler.py'
    upstream = (HERE / 'sidecar_scheduler.py').read_text()
    scheduler = upstream.replace('from research.joint_evaluation_20261029.queue import claims', 'from research.joint_evaluation_20261029.queue import argmix_claims as claims')
    assert scheduler != upstream
    scheduler_path.write_text(scheduler)
    (HERE / 'argmix_scheduler.diff').write_text(''.join(difflib.unified_diff(upstream.splitlines(True), scheduler.splitlines(True), fromfile='queue/sidecar_scheduler.py', tofile='queue/argmix_scheduler.py')))
    plan = dict(schema='paper_campaign_v1', physical_gpus=[3, 4, 5], allowed_sidecar_gpus=[],
                claim_id=identity, claim_task_ids=[name],
                original_state_path=side['original_state_path'], original_identity=side['original_identity'],
                guard_task_id=side['guard_task_id'], guard_pid=side['guard_pid'], guard_start_ticks=side['guard_start_ticks'],
                records_directory=str((HERE / 'argmix_records_v1').relative_to(ROOT)),
                storage_ceiling_bytes=1_200_000_000_000, storage_ledger_ref=side['storage_ledger_ref'],
                reserved_existing_project_bytes=base, other_queue_disk_reserved_bytes=argformer_reservation,
                global_committed_bytes=total, controller_metadata_allowance_bytes=2**30,
                foreign_ram_reserve_bytes=1216*2**30, existing_workers=[],
                max_parallel_tasks=1, max_cpu_tasks=1, max_jobs_per_gpu=1,
                tasks=[external, task], scope_limits=[limit for limit in original.get('scope_limits', []) if (ROOT / task['output_directory']).is_relative_to(ROOT / limit['path'])],
                controller_source_refs=[ref(scheduler_path), ref(HERE / 'argmix_claims.py'), ref(HERE / 'claims.py'), ref(HERE / 'coordination.py'), ref('src/arggnn/benchmarking/paper_campaign/runtime.py'), ref('src/arggnn/benchmarking/paper_campaign/processes.py')])
    save('argmix_plan.json', plan)
    save('argmix_proposal_receipt.json', dict(created_unix=time.time(), plan_ref=ref(HERE / 'argmix_plan.json'),
         original_task_ref=spec['original_task_ref'], source_refs_verified=len(old['source_refs']),
         native_algorithm_unchanged=True, disk_and_ram_caps_unchanged=True, global_committed_bytes=total,
         changes=['Exact owned output claim accepted once', 'Visible module wrapper replaces only CLI dispatch', 'Native subprocess timeout removed while real waits preserved', 'INT64_MAX controller and ScopeGuard wall policy'],
         claims_created=0, workers_launched=0, signals_sent=0))
    print(json.dumps(dict(prepared=True, global_committed_bytes=total, source_refs_verified=len(old['source_refs']), launched=False)))


if __name__ == '__main__':
    main()
