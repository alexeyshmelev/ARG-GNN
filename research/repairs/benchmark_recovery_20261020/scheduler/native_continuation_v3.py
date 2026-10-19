"""One readable full-budget native contrastive continuation, plan-only by default."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import american_competitors as api
from arggnn.benchmarking import workflow as metadata
from arggnn.benchmarking import workflow_v4 as lifecycle
from research.reviews.benchmark_repairs_20261019.supervision.finite_pss_v1 import repaired_owned_group_class
from . import shared_gpu0_v1 as gpu

MODULE = 'research.repairs.benchmark_recovery_20261020.scheduler.native_continuation_v3'
WORKER = 'research.repairs.benchmark_recovery_20261020.scheduler.native_worker_v3'
DIRECTORY = 'research/repairs/benchmark_recovery_20261020/scheduler'
SOURCE_FILES = tuple(DIRECTORY+'/'+name for name in (
    'native_continuation_v3.py', 'shared_gpu0_v1.py', 'native_worker_v3.py',
    'native_seal_v1.py', 'native_telemetry_v1.py', 'activation_checkpoint_v1.py'))
OwnedGroup = repaired_owned_group_class(lifecycle)


def save(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_suffix('.pending')
        with temporary.open('w') as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, path)
    return c.file_ref(path)


def source_check(request):
    for reference in request['source_refs']:
        gpu.need(c.file_ref(reference['path']) == reference, 'Pinned continuation source changed: '+reference['path'])
    gpu.need(set(SOURCE_FILES) <= {row['path'] for row in request['source_refs']},
             'Every new readable wrapper must remain source-pinned')


def prepare(plan_ref, task_id, output_directory, failed_attempt_wait_ref, worker_pid=1270474):
    """Read metadata and return a reviewable request. No files or processes created."""
    plan = c.read_ref(plan_ref)
    state_path = c.contained_path(plan['records_directory'])/'state.json'
    state = c.read_json(state_path)
    gpu.need(state['plan_ref'] == plan_ref, 'Current workflow state differs from the frozen plan')
    tasks = [row for row in plan['tasks'] if row['id'] == task_id]
    gpu.need(len(tasks) == 1, 'One exact planned native task required')
    original_task=tasks[0]
    task=dict(original_task,output_directory=output_directory)
    gpu.need(task['factory_module'] == 'arggnn.benchmarking.american_competitors'
             and task['factory_function'] == 'argformer_factory'
             and task['factory_arguments'].get('stage') == 'contrastive'
             and task['resource'] == 'gpu0', 'Only the original native contrastive stage is eligible')
    gpu.need(state['tasks'][task_id]['status'] == 'pending', 'Never replay started or completed native work')
    dependencies = {}
    for identifier in task['dependencies']:
        entry = state['tasks'][identifier]
        gpu.need(entry['status'] == 'succeeded' and entry.get('owned_process_group_absent') is True
                 and entry.get('actual_controller_wait') == 0, 'Real predecessor success and cleanup required')
        dependencies[identifier] = entry['result_ref']
    output = c.contained_path(task['output_directory'])
    gpu.need(output != c.contained_path(original_task['output_directory']) and not output.exists(), 'Explicit fresh successor output required')
    failed=c.read_ref(failed_attempt_wait_ref)
    gpu.need(c.contained_path(failed_attempt_wait_ref['path']) == c.contained_path(original_task['output_directory'])/'actual_wait.json'
             and failed.get('actual_direct_wait') is True and failed.get('actual_direct_returncode') != 0
             and failed.get('owned_process_group_absent') is True, 'Preserved failed attempt must genuinely wait and drain')
    arguments = metadata.resolve(task['factory_arguments'], state)
    arguments['output_directory'] = str(output/'native')
    call = api.argformer_factory(**arguments)
    gpu.need(call['function'] == 'run_argformer' and call['spec']['stage'] == 'contrastive',
             'Exact frozen scientific contrastive descriptor required')
    refs = {row['path']: row for row in [*plan['sources'], *call['source_refs'],
             *[c.file_ref(path) for path in SOURCE_FILES],
             c.file_ref(lifecycle.__file__),
             c.file_ref('research/reviews/benchmark_repairs_20261019/supervision/finite_pss_v1.py')]}
    try:
        borrowed=gpu.borrow_snapshot(worker_pid)
    except FileNotFoundError:
        borrowed=None
    request = dict(schema='readable_shared_native_contrastive_checkpoint_v3', plan_ref=plan_ref,
        task=task, original_task=original_task, call=call, prerequisite_results=dependencies,
        failed_attempt_wait_ref=failed_attempt_wait_ref, activation_checkpoint_execution=True,
        borrowed_lease=borrowed,
        cuda_cap_bytes=gpu.CUDA_CAP, driver_margin_bytes=gpu.DRIVER_MARGIN,
        start_free_bytes=gpu.START_FREE, live_free_bytes=gpu.LIVE_FREE,
        source_refs=list(refs.values()), automatic_retry=False,
        science_and_batch_unchanged=True, measured_memory_fit=False,
        first_full_batch_telemetry_required=True, execution_authorized=False,
        concurrent_ROOT_preparation=dict(max_rss_bytes=128*2**30,max_output_bytes=64*2**20))
    source_check(request)
    return request


def validate_request(reference, *, approved=True):
    request = gpu.authority(reference) if approved else c.read_ref(reference)
    gpu.need(request.get('schema') == 'readable_shared_native_contrastive_checkpoint_v3'
             and request.get('automatic_retry') is False, 'Explicit one-shot contrastive request required')
    source_check(request)
    failed=c.read_ref(request['failed_attempt_wait_ref'])
    gpu.need(failed.get('actual_direct_wait') is True and failed.get('actual_direct_returncode') != 0 and failed.get('owned_process_group_absent') is True, 'Original failed attempt wait changed')
    task, call = request['task'], request['call']
    plan = c.read_ref(request['plan_ref'])
    original=request['original_task']
    gpu.need(original in plan['tasks'] and task==dict(original,output_directory=task['output_directory'])
             and c.contained_path(task['output_directory'])!=c.contained_path(original['output_directory'])
             and request.get('activation_checkpoint_execution') is True and task['factory_module'] == 'arggnn.benchmarking.american_competitors'
             and task['factory_arguments']['stage'] == 'contrastive'
             and call['module'] == 'arggnn.benchmarking.american_competitors'
             and call['function'] == 'run_argformer' and call['spec']['stage'] == 'contrastive',
             'Original planned native stage and frozen callable required')
    state = c.read_json(c.contained_path(plan['records_directory'])/'state.json')
    gpu.need(state['tasks'][task['id']]['status'] == 'pending', 'Original task is no longer pending')
    for identifier, result_ref in request['prerequisite_results'].items():
        entry = state['tasks'][identifier]
        gpu.need(entry['status'] == 'succeeded' and entry['result_ref'] == result_ref
                 and entry.get('owned_process_group_absent') is True,
                 'Completed predecessor evidence changed')
    arguments = metadata.resolve(task['factory_arguments'], state)
    arguments['output_directory'] = str(c.contained_path(task['output_directory'])/'native')
    gpu.need(api.argformer_factory(**arguments) == call, 'Frozen scientific descriptor changed')
    return request


def remaining_growth(cap, sample):
    resident = sample['enforcement_PSS_bytes'] if sample.get('complete') is True else 0
    return max(0,cap-resident)


def managed_RAM(plan,state):
    from arggnn.benchmarking import workflow_predecessor_v5 as predecessor
    rows=[]
    def add(identifier,cap,expected=None):
        sample=dict(complete=False,enforcement_PSS_bytes=0)
        if expected is not None:
            try:
                current=lifecycle.identity(expected['pid'])
                gpu.need(all(current[key]==expected[key] for key in ('pid','start_ticks','pgid','session')),
                         'Exact managed RAM group identity changed')
                tracker=OwnedGroup(current['pid'])
                tracker.measure()
                sample=tracker.latest
            except (OSError,ValueError,RuntimeError):
                sample=dict(complete=False,enforcement_PSS_bytes=0)
        rows.append(dict(id=identifier,full_cap_bytes=cap,
                         remaining_growth_bytes=remaining_growth(cap,sample),
                         PSS_complete=sample.get('complete') is True,
                         measured_PSS_bytes=sample.get('enforcement_PSS_bytes',0)))
    for task in plan['tasks']:
        entry=state['tasks'][task['id']]
        if entry['status']!='running' or task['resource']=='external':continue
        expected=None
        try:
            receipt=c.read_json(c.contained_path(task['output_directory'])/'supervision_identity.json')
            gpu.need(receipt['job_ref']==entry['job_ref'],'Managed job identity differs')
            expected=receipt['child_identity']
        except (OSError,ValueError,RuntimeError,KeyError):pass
        add(task['id'],task['max_rss_bytes'],expected)
    known={row['source_task_id']:row for row in predecessor.live_group_bindings(plan)}
    for binding in plan.get('workflow_external_predecessors',[]):
        _,entry=predecessor._source(binding)
        if predecessor._drained(binding,entry) is None:
            add(binding['source_task_id'],binding['max_rss_bytes'],known.get(binding['source_task_id']))
    for row in plan.get('external_resources',[]):
        drained=False
        try:
            queue=c.read_json(row['queue_path'])
            entry=next(item for item in queue['jobs'] if item['admission']['job_id']==row['job_id'])
            wait=c.read_json(row['wait_path'])
            drained=entry['status'] in ('succeeded','failed') and wait.get('actual_direct_wait') is True and wait.get('owned_process_group_absent') is True
        except (OSError,ValueError,KeyError,StopIteration):pass
        if not drained:add(row['job_id'],row['max_rss_bytes'])
    return rows


def capacity_admission(request):
    from arggnn.benchmarking import workflow_predecessor_v5 as predecessor
    from research.reviews.benchmark_repairs_20261019.resources_v9.resource_accounting_v1 import (
        validate_static_settlements, kernel_available_ram_bytes)
    plan=c.read_ref(request['plan_ref'])
    state=c.read_json(c.contained_path(plan['records_directory'])/'state.json')
    validate_static_settlements(plan,c.ROOT)
    additional=request['concurrent_ROOT_preparation']
    gpu.need(additional==dict(max_rss_bytes=128*2**30,max_output_bytes=64*2**20),
             'Preserve explicit forthcoming ROOT CPU preparation reservation')
    projection=metadata.storage_projection(plan,state,request['task']['max_output_bytes']+additional['max_output_bytes'])
    projection+=predecessor.storage_reservation_bytes(plan)
    old_entry=state['tasks'][request['original_task']['id']]
    failed_bytes=0 if 'reservation_bytes' in old_entry else lifecycle.scope_bytes(c.contained_path(request['original_task']['output_directory']))
    projection+=failed_bytes
    gpu.need(projection<min(plan['storage_ceiling_bytes'],980000000000),'Native full output cap exceeds storage admission')
    rows=managed_RAM(plan,state)
    full=sum(row['full_cap_bytes'] for row in rows)
    remaining=sum(row['remaining_growth_bytes'] for row in rows)
    new=request['task']['max_rss_bytes']+additional['max_rss_bytes']
    gpu.need(full+new<=plan['max_parallel_rss_bytes'],'Aggregate native and ROOT CPU RAM caps do not fit')
    available=kernel_available_ram_bytes()
    gpu.need(available-remaining>=new+plan['host_ram_margin_bytes'],
             'Kernel free RAM minus unallocated managed growth does not admit new caps')
    return dict(projected_storage_bytes=projection,live_managed_groups=rows,
        live_managed_full_caps_bytes=full,remaining_managed_growth_bytes=remaining,
        native_and_ROOT_CPU_new_caps_bytes=new,kernel_available_RAM_bytes=available,
        existing_resident_RAM_not_subtracted_twice=True,retained_failed_predecessor_bytes=failed_bytes,
        root_CPU_metadata_output_reservation_bytes=additional['max_output_bytes'])


def supervise(reference):
    request = validate_request(reference)
    task, spec = request['task'], request['call']['spec']
    output = c.contained_path(task['output_directory'])
    gpu.need(not output.exists(), 'No output reuse or automatic retry')
    capacity = capacity_admission(request)
    lifecycle.enable_subreaper()
    with gpu.reserve_native(reference, request) as (descriptor, first_sample):
        gpu.authority(reference)
        output.mkdir(parents=True, exist_ok=False)
        native = output/'native'
        native.mkdir()
        spec_ref = save(native/'native_stage_spec.json', spec, exclusive=True)
        job = dict(schema='arggnn_workflow_job_v1', plan_ref=request['plan_ref'], task=task,
            task_fingerprint=c.fingerprint(task), call=request['call'], source_refs=request['source_refs'],
            shared_native_request_ref=reference, original_task=request['original_task'],
            execution_amendment=dict(cuda_allocator_cap_bytes=gpu.CUDA_CAP,
                original_cap_bytes=task['max_cuda_reserved_bytes'], physical_gpu=0,
                borrowed_physical_lease=request['borrowed_lease'] is not None, science_unchanged=True,
                activation_checkpoint_execution=True,prior_failed_attempt_wait_ref=request['failed_attempt_wait_ref']))
        job_ref = save(output/'job.json', job, exclusive=True)
        save(output/'sharing_admission.json', dict(request_ref=reference,
             actual_first_gpu_sample=first_sample, borrowed_lease=request['borrowed_lease'],
             logical_reservation_bytes=gpu.CUDA_CAP+gpu.DRIVER_MARGIN, capacity_admission=capacity), exclusive=True)
        env = {key:value for key,value in os.environ.items()
               if not key.startswith('ARGGNN_GPU') and key != 'ARGGNN_PHYSICAL_GPU'}
        temporary = output/'tmp'
        temporary.mkdir()
        env.update(CUDA_VISIBLE_DEVICES=gpu.GPU0_UUID, ARGGNN_PHYSICAL_GPU='0',
            PYTORCH_NVML_BASED_CUDA_CHECK='1', ARGGNN_MAX_CUDA_RESERVED_BYTES=str(gpu.CUDA_CAP),
            ARGGNN_SHARED_NATIVE_FD=str(descriptor), ARGGNN_SHARED_NATIVE_REQUEST_REF=json.dumps(reference),
            PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            PYTHONPATH=str(c.ROOT/'src')+':'+str(c.ROOT),
            TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
            OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', NUMBA_NUM_THREADS='8')
        reservation=json.loads(os.pread(descriptor,65536,0))
        physical_descriptor=reservation.get('owned_physical_lease_fd')
        inherited=(descriptor,) if physical_descriptor is None else (descriptor,int(physical_descriptor))
        read_fd, write_fd = os.pipe()
        env['ARGGNN_SHARED_STARTUP_FD'] = str(read_fd)
        command = [str(c.ROOT/api.NATIVE_PYTHON), '-B', '-s', '-m', WORKER,
                   '--spec-ref', json.dumps(spec_ref)]
        save(output/'plaintext_command.json', dict(argv=command, cwd=str(c.ROOT),
             interpreter=api.NATIVE_PYTHON, source_file=SOURCE_FILES[2]), exclusive=True)
        monitor = gpu.HeadroomMonitor()
        deadline = time.monotonic()+task['max_seconds']
        process = tracker = None
        error = None
        cancelled = []
        handlers = {sig:signal.signal(sig, lambda number, frame:cancelled.append(number))
                    for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (output/'stage.log').open('xb') as log:
                try:
                    process = subprocess.Popen(command, cwd=c.ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True, pass_fds=inherited+(read_fd,))
                    tracker = OwnedGroup(process.pid)
                    save(output/'supervision_identity.json', dict(job_ref=job_ref,
                         child_identity=tracker.root, startup_barrier=True), exclusive=True)
                    os.close(read_fd)
                    read_fd = None
                    os.write(write_fd, b'1')
                    os.close(write_fd)
                    write_fd = None
                    while process.poll() is None:
                        gpu.authority(reference)
                        gpu.validate_reservation(request,descriptor,os.getpid())
                        gpu.need(not cancelled and time.monotonic() < deadline, 'Native worker cancelled or timed out')
                        gpu.need(tracker.measure() <= task['max_rss_bytes'], 'Native owned PSS cap exceeded')
                        gpu.need(lifecycle.scope_bytes(output) <= task['max_output_bytes'], 'Native output cap exceeded')
                        sample = gpu.gpu_sample()
                        owned_gpu_bytes = gpu.native_driver_bytes(set(tracker.capture()))
                        sample['owned_native_driver_bytes'] = owned_gpu_bytes
                        monitor.observe(sample)
                        save(output/'latest_gpu_memory.json', dict(sample, samples=monitor.samples,
                             consecutive_breaches=monitor.consecutive_breaches,
                             minimum_free_bytes=monitor.minimum_free_bytes))
                        time.sleep(1)
                except BaseException as exc:
                    error = type(exc).__name__+': '+str(exc)
                finally:
                    for fd in (read_fd, write_fd):
                        if fd is not None:os.close(fd)
                    if tracker is not None:
                        tracker.stop_and_wait(process)
                    elif process is not None:
                        # Only the actual owned Popen is eligible for this emergency cleanup.
                        process.terminate()
                        process.wait()
        finally:
            for sig, handler in handlers.items():signal.signal(sig, handler)
        direct = None if tracker is None else tracker.direct_wait
        drained = tracker is not None and not tracker.capture()
        result = dict(actual_direct_wait=bool(direct),
            actual_direct_returncode=None if direct is None else direct['actual_returncode'],
            supervisor_error=error, owned_process_group_absent=drained,
            output_bytes=lifecycle.scope_bytes(output), job_ref=job_ref, automatic_retry=False,
            shared_native_request_ref=reference,
            gpu_memory_monitor=dict(samples=monitor.samples, minimum_free_bytes=monitor.minimum_free_bytes),
            supervision=None if tracker is None else tracker.receipt())
        if result['actual_direct_returncode'] == 0 and drained and error is None:
            try:
                completed = api.read(api.ref(native/'scientific_completed.json'))
                gpu.need(completed['actual_optimizer_updates'] == spec['expected_updates']
                    and completed['optimizer_unique_step_values'] == [spec['expected_updates']]
                    and completed['completed_epochs'] == 1 and completed['parent_id'] == spec['parent_id'],
                    'Actual full unchanged contrastive budget must complete')
                activation=completed.get('initialization',{}).get('activation_checkpoint_execution',{})
                gpu.need(len(activation.get('layers',[]))==12 and activation.get('no_loss_microbatching') is True
                    and activation.get('parameter_identity_and_saved_schema_unchanged') is True,
                    'Actual full encoder checkpoint-execution receipt required')
                gpu.need((native/'first_full_batch_memory.json').is_file(), 'First actual full-batch memory receipt missing')
                native_wait = dict(actual_direct_wait=True, actual_direct_returncode=0,
                    native_process_absent=True, outer_owned_group_cleanup_pending=True, pid=process.pid)
                stage_ref = save(native/'stage_completed.json', dict(completed,
                    schema='american_fresh_ARGformer_stage_completed_v1', native_wait=native_wait,
                    outer_worker_wait_still_required=True), exclusive=True)
                save(output/'result.json', dict(job_ref=job_ref, value=stage_ref), exclusive=True)
                save(output/'successor_handoff.json', dict(task_id=task['id'], task_fingerprint=c.fingerprint(task),
                    original_plan_ref=request['plan_ref'], job_ref=job_ref, result_ref=c.file_ref(output/'result.json'),
                    actual_wait_path=str(output/'actual_wait.json'), original_task_must_be_adopted_not_replayed=True,
                    prerequisite_MLM_reused_without_updates=True), exclusive=True)
            except BaseException as exc:
                result['supervisor_error'] = type(exc).__name__+': '+str(exc)
        save(output/'actual_wait.json', result, exclusive=True)
        gpu.need(result['actual_direct_returncode'] == 0 and drained and result['supervisor_error'] is None,
                 'Native continuation failed: '+str(result['supervisor_error']))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-plan-ref')
    parser.add_argument('--task-id')
    parser.add_argument('--write-request')
    parser.add_argument('--output-directory')
    parser.add_argument('--failed-attempt-wait-ref')
    parser.add_argument('--request-ref')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.prepare_plan_ref:
        gpu.need(not args.execute and args.task_id and args.write_request, 'Preparation is metadata-only and needs a fresh request path')
        gpu.need(args.output_directory and args.failed_attempt_wait_ref,'Fresh output and actual failed wait are mandatory')
        request = prepare(json.loads(args.prepare_plan_ref), args.task_id,args.output_directory,json.loads(args.failed_attempt_wait_ref))
        reference = save(c.contained_path(args.write_request), request, exclusive=True)
        print(json.dumps(dict(request_ref=reference, execution_authorized=False, jobs_started=0)))
    else:
        gpu.need(args.request_ref, 'An exact reviewed request reference is required')
        reference = json.loads(args.request_ref)
        if args.execute:
            print(json.dumps(supervise(reference), sort_keys=True))
        else:
            validate_request(reference, approved=False)
            print(json.dumps(dict(execution_authorized=False, jobs_started=0)))


if __name__ == '__main__':
    main()
