"""Continue thirteen unchanged stages after two genuine successful prefix waits.

The prior controller confused a minimum-free admission floor with a maximum
qualification allocation. Preserve its failure and use measured reservation
plus explicit headroom for GPU3 admission. Do not replay either completed stage.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from research.joint_evaluation_20261029.argmix.recovery_20261030_queue import controller_v1 as c

HERE = c.BASE / 'recovery_20261030_gpu_capacity_v1'
HEADROOM = 2 * 2**30
PREFIX_IDS = ['joint_argmix.train_cache', 'joint_argmix.gpu_qualification']


def authenticate_prefix(spec, manifest, plan, original_binding):
    c.require([t['id'] for t in plan['tasks'][:2]] == PREFIX_IDS and len(plan['tasks']) == 15,
              'Exact completed two-stage prefix and full original fifteen-stage plan')
    folder = Path(manifest['records_directory'])
    references = spec['prefix_actual_wait_refs']
    c.require(len(references) == 2, 'Both genuine prefix waits required')
    for reference, task in zip(references, plan['tasks'][:2]):
        expected_path = folder / task['id'] / 'actual_wait.json'
        c.require(reference == c.ref(expected_path), 'Exact original ordered prefix wait reference')
        row = c.read(reference)
        command = task['argv_by_physical_gpu']['3'] if task['kind'] == 'gpu' else task['argv']
        c.require(row['task_id'] == task['id'] and row['plan_ref'] == manifest['plan_ref'] and
                  row['freeze_ref'] == manifest['freeze_ref'] and row['command'] == command and
                  row['recovery_binding'] == original_binding and row['actual_wait'] is True and
                  row['actual_returncode'] == 0 and row['status'] == 'succeeded' and
                  row['owned_process_group_absent'] is True and row['products_complete'] is True and
                  c.historical_session_absent(row['pid'], row['process_start_ticks']),
                  'Successful exact prefix stage with genuinely waited and absent owned process family')
        product = c.completion_path(task)
        c.require(row['result_ref'] == c.ref(product), 'Exact prefix result content')
        c.require(c.read(c.ref(product.parent/'actual_wait.json')) == row, 'Unchanged colocated original actual wait')
        c.require_wait(product, task['id'])
    cache = c.read(c.read(references[0])['result_ref'])
    qualification = c.read(c.read(references[1])['result_ref'])
    c.require(cache['complete'] is True and cache['freeze_ref'] == manifest['freeze_ref'], 'Complete frozen TRAIN cache')
    c.require(qualification['passed'] is True and qualification['physical_gpu'] == 3 and
              qualification['freeze_ref'] == manifest['freeze_ref'] and qualification['full_minibatch_windows'] == 16 and
              qualification['largest_across_all_registered_TRAIN_windows'] is True and
              qualification['sampled_only'] is False and qualification['heldout_observations_or_truth_opened'] is False,
              'Successful full original sixteen-window qualification on physical GPU3')
    c.verify(qualification['source_ref'])
    c.require(qualification['complete_TRAIN_cache_ref']['path'] == c.read(references[0])['result_ref']['path'] and
              qualification['complete_TRAIN_cache_ref']['sha256'] == c.read(references[0])['result_ref']['sha256'],
              'Qualification used this complete cache')
    peak = qualification['maximum_peak_reserved_bytes']
    c.require(peak == max(group['peak_reserved_bytes'] for group in qualification['groups']), 'Exact measured peak reservation')
    c.require(spec['gpu_headroom_bytes'] == HEADROOM and spec['measured_peak_reserved_bytes'] == peak and
              spec['minimum_free_gpu_bytes'] == peak + HEADROOM, 'Measured reservation plus explicit2GiB headroom')
    return list(references)


def validate(continuation_ref):
    spec = c.read(continuation_ref)
    c.require(spec['schema'] == 'argmix_gpu_capacity_continuation_v1' and spec['root_authorized'] is True,
              'Explicit reviewed operational continuation authority')
    c.require(spec['physical_gpu'] == 3 and spec['automatic_retries_authorized'] is False,
              'Only physicalGPU3 and one explicit continuation attempt')
    c.verify(spec['previous_controller_log_ref'])
    for reference in spec['source_refs']:
        c.verify(reference)
    manifest_ref = spec['original_manifest_ref']
    manifest = c.read(manifest_ref)
    for reference in manifest['source_refs']:
        c.verify(reference)
    plan = c.read(manifest['plan_ref'])
    original_binding, host_minimum = c.validate_recovery(manifest, manifest_ref, plan)
    c.frozen(manifest['freeze_ref']['path'])
    previous = c.read(spec['previous_controller_launch_ref'])
    c.require(previous['manifest_ref'] == manifest_ref and
              c.historical_session_absent(previous['pid'], previous['process_start_ticks']),
              'Previous controller is absent; existing work cannot be adopted')
    failed = c.read(spec['preserved_failed_state_ref'])
    c.require(failed['status'] == 'failed' and failed['controller_pid'] == previous['pid'] and
              failed['manifest_ref'] == manifest_ref and failed['plan_ref'] == manifest['plan_ref'] and
              failed['reason'] == 'Global full16-window GPU qualification fits registered capacity',
              'Exact preserved admission-floor comparison failure')
    prefix = authenticate_prefix(spec, manifest, plan, original_binding)
    binding = dict(original_binding, operational_continuation_ref=continuation_ref)
    return spec, manifest, plan, prefix, binding, host_minimum


def gpu_requirement(spec, task):
    return max(task['minimum_free_gpu_bytes'], spec['minimum_free_gpu_bytes'])


def remaining_tasks(plan):
    c.require([task['id'] for task in plan['tasks'][:2]] == PREFIX_IDS, 'Never replay the completed prefix')
    return plan['tasks'][2:]


def chain_record(manifest, continuation_ref, binding, references, tasks, *, gpu=False):
    folder = Path(manifest['records_directory'])
    expected = [str((folder/task['id']/'actual_wait.json').relative_to(c.ROOT)) for task in tasks]
    c.require([reference['path'] for reference in references] == expected and len(set(expected)) == len(expected),
              'Complete unique ordered genuine stage references')
    c.require(len(references) == (11 if gpu else 15), 'Complete original chain stage count')
    last = c.read(references[-1])
    row = dict(schema='joint_argmix_gpu_chain_completed_v1' if gpu else 'joint_argmix_completed_chain_v1',
               model_id=c.MODEL_ID, plan_ref=manifest['plan_ref'], freeze_ref=manifest['freeze_ref'],
               recovery_binding=binding, operational_continuation_ref=continuation_ref,
               stage_actual_wait_refs=list(references), actual_wait=True, actual_returncode=0,
               owned_process_group_absent=True, products_complete=True, completed_utc=c.now())
    if gpu:
        row.update(chain_id='argmix_joint', status='succeeded', manifest_ref=binding['manifest_ref'],
                   gpu_mutex_released=True, pid=last['pid'], process_start_ticks=last['process_start_ticks'], physical_gpu=3)
    return row


def run(continuation_ref):
    spec, manifest, plan, prefix, binding, host_minimum = validate(continuation_ref)
    folder = Path(manifest['records_directory'])
    with (folder/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c.require(not (HERE/'started.json').exists(), 'No automatic replay of this continuation')
        c.require(c.ref(folder/'state.json')['sha256'] == spec['preserved_failed_state_ref']['sha256'],
                  'Preserve exact failed live state before any update')
        for task in remaining_tasks(plan):
            c.require(not (folder/task['id']).exists() and not c.completion_path(task).exists(),
                      'No unowned partial stage, existing product or automatic replay')
        c.require(not (folder/'completed.json').exists() and not (folder/'gpu_chain_completed.json').exists(),
                  'No prior completed chain may be overwritten')
        ticks = (Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()[19]
        c.atomic_record(HERE/'started.json',dict(pid=os.getpid(),process_start_ticks=ticks,
                        continuation_ref=continuation_ref,preserved_failed_state_ref=spec['preserved_failed_state_ref'],
                        started_utc=c.now()),HERE,immutable=True)
        previous = None
        def status(state, reason, missing=None):
            nonlocal previous
            key=(state,reason,tuple(missing or []))
            if key != previous:
                c.atomic_record(folder/'state.json',dict(status=state,reason=reason,
                    manifest_ref=binding['manifest_ref'],plan_ref=manifest['plan_ref'],model_id=c.MODEL_ID,
                    controller_pid=os.getpid(),operational_continuation_ref=continuation_ref,
                    missing_dependencies=missing or [],updated_utc=c.now()),folder)
                print(state+': '+reason,flush=True)
                previous=key
        _, protocol = c.frozen(manifest['freeze_ref']['path'])
        admission_path = c.ROOT/protocol['production_admission_path']
        admission = c.cache_admission(admission_path,protocol)
        admission_ref = c.ref(admission_path)
        external = {t['id']:t for t in c.read(manifest['original_effective_plan_ref'])['tasks']}
        internal = {t['id']:t for t in plan['tasks']}
        all_waits, gpu_waits = list(prefix), [prefix[1]]
        gpu_tasks = [task for task in plan['tasks'] if task['kind']=='gpu']
        gpu_lock = None
        try:
            for task in remaining_tasks(plan):
                while True:
                    missing = c.dependency_state(task,internal,external)
                    if not missing: break
                    status('waiting','Successful actual producer waits pending',missing);time.sleep(20)
                c.verify(admission_ref)
                while c.host_available_bytes() < host_minimum:
                    status('waiting','Amended family allowance plus64GiB host headroom pending');time.sleep(20)
                if task['kind']=='gpu':
                    gpu_lock=Path(plan['GPU_mutex_path']).open('a+b')
                    while True:
                        try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                        except BlockingIOError:
                            status('waiting','Another authorized root GPU chain holds the shared mutex');time.sleep(20)
                    while True:
                        uuid,available=c.gpu_capacity(3,gpu_requirement(spec,task))
                        if available and c.host_available_bytes()>=host_minimum:break
                        status('waiting','Empty GPU3 with measured reservation plus2GiB free headroom pending');time.sleep(20)
                    command=task['argv_by_physical_gpu']['3']
                    environment=dict(CUDA_VISIBLE_DEVICES=uuid,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                                     MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                else:command,environment=task['argv'],task['environment']
                status('running',task['id'])
                waited=c.stage(task,command,environment,folder,manifest['plan_ref'],manifest['freeze_ref'],
                               admission['cache_max_bytes'],gpu_lock,binding)
                all_waits.append(waited)
                if task['kind']=='gpu':
                    gpu_lock.close();gpu_lock=None
                    gpu_waits.append(waited)
                    if task['id']==gpu_tasks[-1]['id']:
                        c.atomic_record(folder/'gpu_chain_completed.json',
                            chain_record(manifest,continuation_ref,binding,gpu_waits,gpu_tasks,gpu=True),folder,immutable=True)
            c.atomic_record(folder/'completed.json',
                chain_record(manifest,continuation_ref,binding,all_waits,plan['tasks']),folder,immutable=True)
            status('completed','All original fifteen stages and genuine waits complete')
        except BaseException as error:
            status('failed',str(error));raise
        finally:
            if gpu_lock is not None:gpu_lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continuation',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    reference=c.ref(args.continuation)
    if args.check_only:
        spec,manifest,plan,prefix,binding,host_minimum=validate(reference)
        uuid,available=c.gpu_capacity(3,spec['minimum_free_gpu_bytes'])
        print(json.dumps(dict(valid=True,completed_prefix=len(prefix),remaining_stages=len(remaining_tasks(plan)),
            minimum_free_gpu_bytes=spec['minimum_free_gpu_bytes'],gpu3_available=available,
            host_available_bytes=c.host_available_bytes(),host_minimum_bytes=host_minimum,
            continuation_ref=reference,production_launched=False,signals_sent=0)),flush=True)
        return
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    run(reference)


if __name__=='__main__':main()
