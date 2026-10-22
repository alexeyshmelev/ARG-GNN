"""Readable scientific worker and direct, finite-PSS process supervision."""
from contextlib import ExitStack, nullcontext
import importlib
import inspect
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from . import contracts_parallel_v1 as c
from arggnn.benchmarking import workflow_v4 as lifecycle
from research.reviews.benchmark_repairs_20261019.supervision.finite_pss_v1 import repaired_owned_group_class
from . import queue_gpu_v1 as gpu

OwnedGroup = repaired_owned_group_class(lifecycle)
_DEFERRED_LEASES = []  # OS closes these only when the scientific process exits.


def child(ref, api):
    gate = int(os.environ.pop('ARGGNN_READABLE_STARTUP_FD'))
    try:
        api.need(os.read(gate, 1) == b'R', 'Supervisor startup receipt required')
    finally:
        os.close(gate)
    job = api.job_spec(ref)
    task, call = job['task'], job['call']
    deferred = task.get('gpu_after_preparation') is True
    output = c.contained_path(task['output_directory'])
    api.need(Path(sys.prefix).resolve() == c.contained_path(call['python']).parent.parent,
             'Exact declared scientific interpreter required')
    try:
        import setproctitle
        setproctitle.setproctitle('AI Auto Research Pipeline Test' if task['resource'] == 'gpu0' or deferred
                                 else 'AI Research Data Preparation')
    except ImportError:
        api.need(task['resource'] != 'gpu0' and not deferred, 'Training process title is required')
    if task['resource'] == 'gpu0':
        gpu.validate_child(job, ref,first_cuda=True)
    elif deferred:
        api.need(os.environ.get('CUDA_VISIBLE_DEVICES') == gpu.target(task)[1]
                 and 'ARGGNN_READABLE_GPU_FD' not in os.environ, 'Deferred preparation has no CUDA lease')
    else:
        api.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU worker cannot use CUDA')
    resource.setrlimit(resource.RLIMIT_FSIZE, (task['max_output_bytes'], task['max_output_bytes']))
    deadline = float(os.environ['ARGGNN_WORKFLOW_DEADLINE'])
    tracker = OwnedGroup(os.getpid())
    last_check = [0.0]

    def check(*args, **kwargs):
        now = time.monotonic()
        api.need(now < deadline, 'Scientific wall limit exceeded')
        if now - last_check[0] < 5:
            return
        last_check[0] = now
        api.need(api.current_policy()['approved_workflow_ref'] == job['plan_ref'], 'Release withdrawn')
        api.need(tracker.measure() <= task['max_rss_bytes'], 'Scientific process group PSS cap exceeded')
        api.need(lifecycle.scope_bytes(output) <= task['max_output_bytes'], 'Scientific output cap exceeded')
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            api.need('ARGGNN_READABLE_GPU_FD' in os.environ,'CUDA initialized before GPU admission')
            api.need(torch.cuda.memory_reserved(0) <= task['max_cuda_reserved_bytes'], 'CUDA allocator cap exceeded')

    def bounded_runner(argv, cwd, env, limits, log_path):
        argv=list(argv)
        if len(argv)>4 and argv[3]=='-m' and argv[4]=='arggnn.benchmarking.american_competitor_worker':
            argv[4]='research.repairs.benchmark_multigpu_20261022.runtime.queue_native_worker_v2'
        api.support.validate_native_command(argv)
        api.need(Path(cwd).resolve().is_relative_to(c.ROOT), 'Native working directory escaped project')
        log = c.contained_path(log_path)
        api.need(log.is_relative_to(output), 'Native log escaped output scope')
        log.parent.mkdir(parents=True, exist_ok=True)
        child_env = dict(os.environ, **env)
        protected = ('CUDA_VISIBLE_DEVICES', 'ARGGNN_PHYSICAL_GPU', 'ARGGNN_MAX_CUDA_RESERVED_BYTES',
                     'ARGGNN_WORKFLOW_DEADLINE', 'ARGGNN_READABLE_GPU_FD', 'ARGGNN_READABLE_JOB_REF')
        api.need(all(child_env.get(k) == os.environ.get(k) for k in protected), 'Native resource admission changed')
        for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME',
                    'MPLCONFIGDIR', 'NUMBA_CACHE_DIR', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR'):
            if child_env.get(key):
                api.need(c.contained_path(child_env[key]).is_relative_to(output), 'Native cache escaped scope')
        inherited = () if 'ARGGNN_READABLE_GPU_FD' not in os.environ else gpu.inherited(int(os.environ['ARGGNN_READABLE_GPU_FD']))
        with log.open('xb') as stream:
            process = subprocess.Popen(argv, cwd=cwd, env=child_env, stdout=stream,
                                       stderr=subprocess.STDOUT, pass_fds=inherited)
            try:
                end = min(deadline, time.monotonic() + limits.get('max_seconds', task['max_seconds']))
                while process.poll() is None:
                    check()
                    if time.monotonic() >= end:
                        raise TimeoutError('Native command wall limit exceeded')
                    time.sleep(1)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        return dict(actual_direct_wait=True, actual_direct_returncode=code,
                    native_process_absent=True, outer_owned_group_cleanup_pending=True, pid=process.pid)

    def acquire_gpu0():
        api.need(deferred,'Only an explicitly deferred task may request late GPU admission')
        api.need('ARGGNN_READABLE_GPU_FD' not in os.environ,'GPU admission callback is single use')
        torch=sys.modules.get('torch')
        api.need(torch is None or not torch.cuda.is_initialized(),'CUDA initialized during CPU observation preparation')
        while True:
            check()
            context=gpu.reserve(job,ref)
            try:descriptor=context.__enter__()
            except (BlockingIOError,gpu.CapacityUnavailable):
                time.sleep(2)
                continue
            break
        _DEFERRED_LEASES.append(context)
        gpu.set_root(descriptor,tracker.root)
        os.environ.update(ARGGNN_PHYSICAL_GPU=str(gpu.target(task)[0]),ARGGNN_MAX_CUDA_RESERVED_BYTES=str(task['max_cuda_reserved_bytes']),
            ARGGNN_READABLE_GPU_FD=str(descriptor),ARGGNN_READABLE_JOB_REF=json.dumps(ref))
        gpu.validate_child(job,ref,first_cuda=True)
        return dict(physical_gpu=gpu.target(task)[0],gpu_uuid=gpu.target(task)[1],max_cuda_reserved_bytes=task['max_cuda_reserved_bytes'])

    module, name = api.support.scientific_route(c.read_ref(job['plan_ref']), call)
    function = getattr(importlib.import_module(module), name)
    available = dict(resource_check=check, execution_check=lambda: api.job_spec(ref), bounded_runner=bounded_runner, acquire_gpu0=acquire_gpu0)
    callbacks = {key: value for key, value in available.items() if key in inspect.signature(function).parameters}
    result = function(call['spec'], **callbacks)
    check()
    return api.save(output/'result.json', dict(value=result, job_ref=ref), exclusive=True)


def execute(ref, api):
    job = api.job_spec(ref)
    task, call = job['task'], job['call']
    deferred = task.get('gpu_after_preparation') is True
    output = c.contained_path(task['output_directory'])
    api.need(not output.exists(), 'No output reuse or automatic retry')
    lifecycle.enable_subreaper()
    context = gpu.reserve(job, ref) if task['resource'] == 'gpu0' else nullcontext(None)
    with ExitStack() as stack:
        try:
            descriptor = stack.enter_context(context)
        except (BlockingIOError, gpu.CapacityUnavailable) as exc:
            return dict(deferred_before_scientific_start=True, reason=str(exc))
        output.mkdir(parents=True, exist_ok=False)
        temporary, cache = output/'tmp', output/'cache'
        temporary.mkdir()
        cache.mkdir()
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('ARGGNN_GPU', 'ARGGNN_READABLE_'))
               and key not in ('ARGGNN_PHYSICAL_GPU', 'ARGGNN_MAX_CUDA_RESERVED_BYTES')}
        env.update(CUDA_VISIBLE_DEVICES=gpu.target(task)[1] if descriptor is not None or deferred else '',
                   TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
                   PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                   PYTHONPATH=str(c.ROOT)+':'+str(c.ROOT/'src')+':'+str(c.ROOT/'scripts'),
                   OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', NUMBA_NUM_THREADS='8')
        for key in ('XDG_CACHE_HOME', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'HF_HOME',
                    'STDPOPSIM_CACHE', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH'):
            env[key] = str(cache/key.lower())
        if descriptor is not None:
            env.update(ARGGNN_PHYSICAL_GPU=str(gpu.target(task)[0]), ARGGNN_MAX_CUDA_RESERVED_BYTES=str(task['max_cuda_reserved_bytes']),
                       ARGGNN_READABLE_GPU_FD=str(descriptor), ARGGNN_READABLE_JOB_REF=json.dumps(ref),
                       PYTORCH_NVML_BASED_CUDA_CHECK='1')
        if deferred:env['PYTORCH_NVML_BASED_CUDA_CHECK']='1'
        deadline = time.monotonic() + task['max_seconds']
        env['ARGGNN_WORKFLOW_DEADLINE'] = str(deadline)
        read_fd, write_fd = os.pipe()
        env['ARGGNN_READABLE_STARTUP_FD'] = str(read_fd)
        command = [str(c.contained_path(call['python'])), '-B', '-s', '-m', api.MODULE,
                   '--child', '--job-ref', json.dumps(ref)]
        api.save(output/'plaintext_command.json', dict(argv=command, cwd=str(c.ROOT)), exclusive=True)
        inherited = (read_fd,) if descriptor is None else (read_fd,) + gpu.inherited(descriptor)
        cancelled = []
        handlers = {number: signal.signal(number, lambda n, frame: cancelled.append(n))
                    for number in (signal.SIGTERM, signal.SIGINT)}
        process = tracker = None
        error = None
        try:
            with (output/'stage.log').open('xb') as log:
                try:
                    process = subprocess.Popen(command, cwd=c.ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True, pass_fds=inherited)
                    tracker = OwnedGroup(process.pid)
                    api.save(output/'supervision_identity.json', dict(job_ref=ref, child_identity=tracker.root), exclusive=True)
                    if descriptor is not None:gpu.set_root(descriptor,tracker.root)
                    os.close(read_fd)
                    read_fd = None
                    os.write(write_fd, b'R')
                    os.close(write_fd)
                    write_fd = None
                    while process.poll() is None:
                        api.need(api.current_policy()['approved_workflow_ref'] == job['plan_ref'], 'Release withdrawn')
                        api.need(not cancelled and time.monotonic() < deadline, 'Worker cancelled or timed out')
                        api.need(tracker.measure() <= task['max_rss_bytes'], 'Worker PSS cap exceeded')
                        api.need(lifecycle.scope_bytes(output) <= task['max_output_bytes'], 'Worker output cap exceeded')
                        if descriptor is not None:
                            gpu.monitor(job, ref, descriptor, tracker)
                        elif deferred:
                            gpu.monitor_deferred(job,ref,tracker)
                        time.sleep(2)
                except BaseException as exc:
                    error = type(exc).__name__ + ': ' + str(exc)
                finally:
                    for fd in (read_fd, write_fd):
                        if fd is not None:
                            os.close(fd)
                    if tracker is not None:
                        tracker.stop_and_wait(process)
                    elif process is not None:
                        process.terminate()
                        process.wait()
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
        direct = None if tracker is None else tracker.direct_wait
        drained = tracker is not None and not tracker.capture()
        retained = lifecycle.scope_bytes(output)
        result = dict(actual_direct_wait=bool(direct), actual_direct_returncode=None if direct is None else direct['actual_returncode'],
                      owned_process_group_absent=drained, supervisor_error=error, output_bytes=retained,
                      job_ref=ref, automatic_retry=False, supervision=None if tracker is None else tracker.receipt())
        api.save(output/'actual_wait.json', result, exclusive=True)
        api.need(result['actual_direct_returncode'] == 0 and drained and error is None
                 and retained <= task['max_output_bytes'] and (output/'result.json').is_file(),
                 'Scientific worker failed: ' + str(error))
        return result
