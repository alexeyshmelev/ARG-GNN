"""Supply ROOT's reviewed admissions to the existing fit, without restarting it."""
import fcntl
import math
import time
from .candidate_common import ROOT, GIB, read_json, read_ref, write_json, file_ref, identity, gpu_inventory, need
from ..storage import archive_supervisor_v1 as archives

REQUEST = dict(path='research/repairs/benchmark_recovery_20261020/training/fit_request_v1.json',
               sha256='67042c208e26d075b38ba0efc095fb4b311243005447648cf63e029eaa0d864d')
STORAGE = dict(path='research/repairs/benchmark_recovery_20261020/storage/compression_plan_v3.json',
               sha256='09dfb4934286a9668120dc426b10b7a2d6272b3da4ad4b8aaedaa77288480151')
LAUNCH = '.runtime/benchmarking/recovery_20261020_treeloom_fit_v1/launch.json'


def finite_values(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_values(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_values(item) for item in value)
    return True


def main():
    request = read_ref(REQUEST)
    output = ROOT / request['output_directory']
    launch, _ = read_json(LAUNCH)
    need(launch['request_ref'] == REQUEST, 'Exact existing launch required')
    supervisor = launch['supervisor_identity']
    child, _ = read_json(output / 'child_identity.json')
    worker = child['worker_identity']
    need(child['supervisor_identity'] == supervisor, 'Exact supervisor and child required')
    config, config_ref, _ = archives.read_config(STORAGE['path'], approved=True)
    need(config_ref == STORAGE and config['concurrent_ROOT_preparation']['current_actual_request_ref'] == REQUEST
         and config['concurrent_ROOT_preparation']['max_rss_bytes'] == 512 * GIB,
         'Full continued RAM allocation must already be reserved')
    records = ROOT / '.runtime/benchmarking/recovery_20261020_treeloom_fit_v1'
    deadline = time.monotonic() + 12 * 3600
    with (records / 'gate_controller.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.monotonic() < deadline:
            need(not (output / 'supervisor_result.json').exists(), 'Scientific fit ended before continuation')
            need(identity(supervisor['pid']) == supervisor and identity(worker['pid']) == worker,
                 'Existing fit process generation changed')
            if (output / 'continue_gate.json').exists():
                return
            if not (output / 'ready_for_gpu.json').exists():
                time.sleep(10)
                continue
            ready, _ = read_json(output / 'ready_for_gpu.json')
            need(ready['request_ref'] == REQUEST and ready['worker_identity'] == worker,
                 'GPU readiness must belong to this exact existing fit')
            if not (output / 'gpu_gate.json').exists():
                with archives.locked(archives.LOCK):
                    sample = gpu_inventory(request['gpu_uuid'])
                    if sample['free_bytes'] >= request['minimum_free_gpu_bytes']:
                        admission = dict(request_ref=REQUEST, worker_identity=worker,
                            physical_gpu=0, gpu_uuid=request['gpu_uuid'],
                            max_cuda_reserved_bytes=request['max_cuda_reserved_bytes'],
                            actual_free_sample=sample, controller_identity=identity(),
                            controller_source_ref=file_ref(__file__),
                            policy='actual shared free memory with 16 GiB allocator cap and monitored 8 GiB floor',
                            other_processes_modified=False, separate_native_allocator_cap_guaranteed=False)
                        reference = write_json(output / 'root_gpu_admission.json', admission)
                        write_json(output / 'gpu_gate.json', dict(schema='treeloom_candidate_gate_v1',
                            gate='gpu_gate.json', allow=True, request_ref=REQUEST, worker_identity=worker,
                            gpu_uuid=request['gpu_uuid'], max_cuda_reserved_bytes=request['max_cuda_reserved_bytes'],
                            admission_ref=reference, expires_unix=time.time()+180))
                time.sleep(10)
                continue
            if not (output / 'qualification.json').exists():
                time.sleep(10)
                continue
            qualification_ref = file_ref(output / 'qualification.json')
            qualification = read_ref(qualification_ref)
            need(qualification['request_ref'] == REQUEST and qualification['worker_identity'] == worker
                 and qualification['status'] == 'completed_first_update'
                 and qualification['optimizer_updates'] == 1
                 and len(qualification['all_49_Adam_counters']) == 49
                 and set(qualification['all_49_Adam_counters']) == {1}
                 and qualification['initial_RSS_bound_passed'] is True
                 and finite_values(qualification['result'])
                 and file_ref(qualification['checkpoint_ref']['path']) == qualification['checkpoint_ref'],
                 'A genuine finite complete-parent update and retained checkpoint are required')
            with archives.locked(archives.LOCK):
                config, config_ref, _ = archives.read_config(STORAGE['path'], approved=True)
                state, _ = read_json(ROOT / config['records_directory'] / 'state.json')
                capacity = archives.capacity(config, state)
                if not capacity['ram_allowed'] or not capacity['storage_allowed']:
                    write_json(records / 'continuation_capacity_wait.json',
                               dict(qualification_ref=qualification_ref, capacity=capacity), replace=True)
                else:
                    admission = dict(request_ref=REQUEST, worker_identity=worker,
                        max_rss_bytes=512*GIB, storage_config_ref=config_ref, capacity=capacity,
                        controller_identity=identity(), controller_source_ref=file_ref(__file__),
                        first_training_update_retained=True)
                    reference = write_json(output / 'root_cpu_admission.json', admission)
                    write_json(output / 'continue_gate.json', dict(schema='treeloom_candidate_gate_v1',
                        gate='continue_gate.json', allow=True, request_ref=REQUEST, worker_identity=worker,
                        qualification_ref=qualification_ref, allow_updates_through=600,
                        max_rss_bytes=512*GIB, observation_cache_bytes=352*GIB,
                        training_headroom_bytes=128*GIB, cpu_admission_ref=reference,
                        expires_unix=time.time()+180))
                    write_json(records / 'gate_controller_completed.json',
                               dict(request_ref=REQUEST, qualification_ref=qualification_ref,
                                    continuation_gate_ref=file_ref(output/'continue_gate.json')))
                    return
            time.sleep(30)
    raise RuntimeError('Admission controller reached its finite deadline')


if __name__ == '__main__':
    main()
