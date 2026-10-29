"""Frozen American continuation; direct waits, no replay, GPU release before CPU score."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from arggnn.benchmarking.paper_campaign import runtime as r, processes

BASE = 'research/joint_evaluation_20261029'
ROOT_RECORDS = BASE + '/root/american_continue_run_v1'
FREEZE = BASE + '/american/freeze_v2.json'
RESULTS = 'experiments/results/joint_american_20261029'
LOCK_PATH = '.runtime/benchmarking/joint_root_successor_gpu3.lock'
PYTHON = str(r.ROOT / '.runtime/env/bin/python3.11')
MODULE = 'research.joint_evaluation_20261029.american.pipeline_v2'
ENVIRONMENT = dict(PYTHONPATH='.runtime/vendor/nvml_profile_20261028:src:.',
    PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, process_start_ticks=fields[19], state=fields[0])
    except FileNotFoundError:
        return None


def alive(expected):
    value = identity(expected['pid'])
    return value is not None and value['process_start_ticks'] == expected['process_start_ticks'] and value['state'] != 'Z'


def immutable(filename, value):
    destination = r.path(filename)
    need(not destination.exists(), 'Immutable receipt already exists: ' + str(destination))
    return r.save(destination, value)


def heldout(protocol):
    return [f'argformer_american_validation_{i:03d}' for i in range(5)] + [f'argformer_american_test_{i:03d}' for i in range(10)]


def expected_stages(protocol):
    common = ['--freeze', FREEZE]
    def spec(name, module, args, products, gpu):
        return dict(name=name, command=[PYTHON, '-B', '-u', '-m', module] + args,
                    products=[RESULTS + '/' + p for p in products], gpu=gpu)
    stages = [spec('gpu_qualification', 'research.joint_evaluation_20261029.american.qualify_gpu_v2',
                   common + ['--gpu', '3'], ['gpu_qualification_v2.json'], True),
              spec('fit', MODULE, ['train'] + common + ['--gpu', '3'],
                   ['fit_v2/product.json', 'fit_v2/final_state.pt'], True)]
    for parent in heldout(protocol):
        stages.append(spec('predict_' + parent, MODULE, ['predict'] + common + ['--gpu', '3', '--parent', parent],
            ['predictions_v2/' + parent + '/' + name for name in ('product.json', 'predictions.npz', 'logits.npz')], True))
    stages.append(spec('score', MODULE, ['score'] + common, ['scores_v2/product.json', 'scores_v2/all15_prediction_seal.json'], False))
    return stages


def successful_wait(row):
    need(row.get('actual_wait') is True and row.get('actual_returncode') == 0 and
         row.get('owned_process_group_absent') is True, 'Genuine successful direct wait required')
    need(not processes.session_members(row['pid']), 'Previously waited worker session still active')


def validate_wait(row, spec, plan_ref):
    successful_wait(row)
    need(row.get('products_complete') is True and row['command'] == spec['command'] and
         row['plan_ref'] == plan_ref, 'Exact successful stage binding required')
    need(row['products'] == [r.ref(p) for p in spec['products']], 'Exact complete immutable stage products required')


def stage(folder, spec, plan_ref, gpu_lock=None):
    directory = folder / spec['name']
    directory.mkdir(exist_ok=True)
    previous = directory / 'actual_wait.json'
    if previous.exists():
        validate_wait(r.read(r.ref(previous)), spec, plan_ref)
        return r.ref(previous)
    need(not (directory / 'started.json').exists() and not (directory / 'worker.log').exists(),
         'Prior partial stage is preserved and must never be replayed')
    need(not any(r.path(p).exists() for p in spec['products']), 'Unattributed output already exists; never replay')
    need(not spec['gpu'] or gpu_lock is not None, 'GPU stages require inherited lifetime shared lock')
    env = os.environ.copy()
    env.update(ENVIRONMENT, CUDA_VISIBLE_DEVICES=r.GPU_UUIDS[3] if spec['gpu'] else '')
    with (directory / 'worker.log').open('xb') as log:
        worker = subprocess.Popen(spec['command'], cwd=r.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=() if gpu_lock is None else (gpu_lock.fileno(),))
        row = dict(plan_ref=plan_ref, pid=worker.pid, process_start_ticks=identity(worker.pid)['process_start_ticks'],
                   command=spec['command'], started_unix=time.time())
        immutable(directory / 'started.json', row)
        code = worker.wait()
        while processes.session_members(worker.pid):
            time.sleep(1)
    refs, error = [], None
    if code == 0:
        try:
            refs = [r.ref(p) for p in spec['products']]
        except Exception as exc:
            error = str(exc)
    row.update(actual_wait=True, actual_returncode=code, owned_process_group_absent=True,
        finished_unix=time.time(), products=refs, products_complete=code == 0 and error is None)
    if error:
        row['product_error'] = error
    receipt = immutable(previous, row)
    validate_wait(row, spec, plan_ref)
    return receipt


def cache_state(plan):
    wait_path = r.path(plan['cache_actual_wait_path'])
    started = r.read(plan['cache_started_ref'])
    job = r.read(plan['cache_job_ref'])
    need(started['job_ref'] == plan['cache_job_ref'] and started['command'] == job['argv'], 'Exact existing cache job')
    if wait_path.exists():
        row = r.read(r.ref(wait_path))
        successful_wait(row)
        need(all(row[k] == started[k] for k in ('pid', 'process_start_ticks', 'command', 'job_ref')),
             'Cache terminal identity differs from actual launch')
        need(row['products'] == [r.ref(p) for p in job['products']], 'Exact complete cache product')
        return r.ref(wait_path)
    # Worker exit can precede terminal receipt publication; its exact supervisor must survive that interval.
    if not alive(plan['cache_controller_identity']):
        if wait_path.exists():
            return cache_state(plan)
        raise ValueError('Cache controller disappeared or PID was reused without terminal wait; preserved, no replay')
    value = identity(started['pid'])
    need(value is None or value['process_start_ticks'] == started['process_start_ticks'], 'Cache worker PID was reused')
    return None


def cache_product(plan, protocol):
    product = r.read(r.ref(RESULTS + '/cache_build_v2/product.json'))
    expected = {p for p, row in protocol['parents'].items() if row['split'] == 'train'}
    need(product['complete'] and product['freeze_ref'] == plan['freeze_ref'] and
         product['cache_namespace'] == protocol['cache_namespace'] and set(product['parents']) == expected,
         'Complete exact20-parent TRAIN cache required')
    need(sum(p['windows'] for p in product['parents'].values()) == 158476 and
         all(p['targets'] == 10000 for p in product['parents'].values()) and
         product['query_ancestry_truth_opened'] is False, 'Full158476 windows and200000 canonical targets')
    need(0 < product['actual_scope_bytes'] <= product['conservatively_reserved_bytes'] <= 63 * 2**30,
         'Completed cache within admitted63GiB')
    need(set(product['largest_TRAIN_windows']) == {'nodes', 'SNP_rows', 'prepared_tensor_bytes'} and
         all(len(g) == 16 for g in product['largest_TRAIN_windows'].values()), 'Full16-window global capacity groups')
    return product


def validate_product(spec, plan, protocol):
    row = r.read(r.ref(spec['products'][0]))
    need(row['freeze_ref'] == plan['freeze_ref'], 'Product belongs to exact freeze')
    if spec['name'] == 'gpu_qualification':
        cache = cache_product(plan, protocol)
        need(row['passed'] and row['physical_gpu'] == 3 and row['full_minibatch_windows'] == 16 and
             row['largest_across_all_158476_TRAIN_windows'] is True and row['sampled_only'] is False and
             row['heldout_observations_or_truth_opened'] is False and row['benchmark_checkpoint_written'] is False and
             row['complete_TRAIN_cache_ref'] == r.ref(RESULTS + '/cache_build_v2/product.json') and
             0 < row['maximum_peak_reserved_bytes'] <= plan['minimum_free_gpu_bytes'], 'Full TRAIN capacity gate failed')
        need({g['selection']: g['windows'] for g in row['groups']} == cache['largest_TRAIN_windows'] and
             len(row['groups']) == 3 and all(len(g['losses']) == 2 for g in row['groups']), 'All exact largest16 groups and two updates')
        return
    need(row['complete'] and row['model_id'] == protocol['model_id'], 'Complete correct-model output')
    if spec['name'] == 'fit':
        training = r.read(row['training_plan_ref'])
        need(row['schema'] == 'american_joint_fixed_final_checkpoint_v1' and row['epochs'] == 13 and
             row['target_presentations'] == 2600000 and row['heldout_observations_or_truth_opened'] is False and
             row['checkpoint_ref'] == r.ref(spec['products'][1]) and row['updates'] == training['planned_updates'] and
             training['windows'] == 158476 and training['targets'] == 200000, 'Fixed full13epoch checkpoint')
    elif spec['name'].startswith('predict_'):
        parent = spec['name'][8:]
        fit = r.read(r.ref(RESULTS + '/fit_v2/product.json'))
        need(row['schema'] == 'american_joint_native3_prediction_v1' and row['parent_id'] == parent and
             row['split'] == protocol['parents'][parent]['split'] and row['rows'] == 10000 and
             row['class_order'] == ['EUR', 'EAS', 'AFR'] and row['query_ancestry_truth_opened'] is False and
             row['checkpoint_ref'] == fit['checkpoint_ref'] and row['prediction_ref'] == r.ref(spec['products'][1]) and
             row['logits_ref'] == r.ref(spec['products'][2]) and
             row['focal_registry_ref'] == protocol['parents'][parent]['binding']['focal_registry_ref'], 'Canonical complete prediction')
        from research.joint_evaluation_20261029.american.common import load_npz
        from arggnn.american_c3_evaluate_v1 import verify_prediction
        verify_prediction(load_npz(row['prediction_ref']), load_npz(row['focal_registry_ref']))
    else:
        need(row['schema'] == 'american_joint_common_native3_scores_v1' and
             row['all15_predictions_sealed_before_truth'] is True and row['probabilities_modified'] is False and
             set(row['parents']) == set(heldout(protocol)), 'Full15-parent unchanged native3 score')


def native_ready(plan, protocol):
    # This barrier is read-only and never opens truth, training inputs or deleted native scratch.
    from arggnn.benchmarking.paper_american.scoring import corrected_argformer_seal
    from arggnn.benchmarking.american_scoring import _seal
    from arggnn.american_c3_evaluate_v1 import verify_prediction
    from research.joint_evaluation_20261029.american.common import load_npz
    refs = []
    for parent in heldout(protocol):
        binding = plan['native_score_gate'][parent]
        wait_path = r.path(binding['argformer_wait_path'])
        product_path = r.path(binding['argformer_completion_path'])
        if not wait_path.exists():
            return None
        wait_ref = r.ref(wait_path)
        wait = r.read(wait_ref)
        successful_wait(wait)
        need(product_path.exists(), 'Successful native wait lacks its required prediction product')
        result = r.read(wait['result_ref'])
        product_ref = r.ref(product_path)
        need(result['scientific_success'] is True and result['value']['completion_ref'] == product_ref,
             'Native prediction wait authenticates the exact completion')
        focal = r.read(binding['focal_preparation_ref'])
        need(focal['focals_ref'] == protocol['parents'][parent]['binding']['focal_registry_ref'], 'Same canonical focal support')
        need(focal['parent_id'] == parent and focal['query_ancestry_truth_opened'] is False, 'Exact public focal preparation')
        records = load_npz(focal['focals_ref'])
        native = corrected_argformer_seal(parent, product_ref, None, focal)
        flare = _seal('flare', parent, binding['flare_ref'], None, focal)
        for product in (native, flare):
            verify_prediction(load_npz(product['prediction_ref']), records)
        refs.append(dict(parent_id=parent, argformer_wait_ref=wait_ref, argformer=product_ref,
                         flare=binding['flare_ref'], prediction_refs=[native['prediction_ref'], flare['prediction_ref']]))
    return refs


def validate_plan(plan):
    from research.joint_evaluation_20261029.american.common_v2 import frozen
    for reference in plan['source_refs']:
        need(r.ref(reference['path']) == reference, 'Frozen continuation source changed')
    need(plan['freeze_ref'] == r.ref(FREEZE), 'Exact American freeze')
    seal, protocol = frozen(FREEZE)
    config = r.read(plan['successor_v2_config_ref'])
    template = r.read(config['successor_template_ref'])
    need(template['required_root_model_chains'] == ['american_joint', 'argmix_joint', 'european_four_joint'],
         'All three root GPU priority chains required')
    need(plan['gpu_lock_path'] == LOCK_PATH and plan['records_directory'] == ROOT_RECORDS and
         plan['stages'] == expected_stages(protocol), 'Exact stable mutex, records scope and frozen18-stage chain')
    need(plan['minimum_free_gpu_bytes'] == 76 * 2**30 and plan['minimum_host_available_bytes'] == 128 * 2**30,
         'Explicit GPU capacity and owned64GiB plus64GiB host headroom')
    need(set(plan['native_score_gate']) == set(heldout(protocol)), 'All15 native parent bindings')
    return protocol


def host_capacity(plan):
    ram = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) * 1024
    return ram >= plan['minimum_host_available_bytes']


def capacity(plan):
    output = subprocess.check_output(['nvidia-smi', '--id=' + r.GPU_UUIDS[3],
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15)
    uuid, free = [v.strip() for v in output.strip().split(',')]
    need(uuid == r.GPU_UUIDS[3], 'Exact physical GPU3 identity')
    apps = subprocess.check_output(['nvidia-smi', '--id=' + r.GPU_UUIDS[3],
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    return not apps.strip() and int(free) * 2**20 >= plan['minimum_free_gpu_bytes'] and host_capacity(plan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan_ref = r.ref(args.plan)
    plan = r.read(plan_ref)
    protocol = validate_plan(plan)
    if args.check_only:
        print(json.dumps(dict(valid=True, plan_ref=plan_ref, stages=len(plan['stages']),
                             cache_terminal=cache_state(plan) is not None)), flush=True)
        return
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    folder = r.path(plan['records_directory'])
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'controller.lock').open('a+b') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        def status(value, detail):
            nonlocal previous
            if (value, detail) != previous:
                r.save(folder / 'status.json', dict(status=value, detail=detail, plan_ref=plan_ref,
                       pid=os.getpid(), time_unix=time.time()))
                print(value + ': ' + detail, flush=True)
                previous = (value, detail)
        cache_wait = cache_state(plan)
        while cache_wait is None:
            status('waiting', 'Exact full TRAIN cache controller is still preparing or publishing its real wait')
            time.sleep(20)
            cache_wait = cache_state(plan)
        cache_product(plan, protocol)
        results = []
        with r.path(LOCK_PATH).open('a+b') as gpu_lock:
            while True:
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    status('waiting', 'Another authorized root GPU3 chain holds the stable shared lock')
                    time.sleep(20)
            for spec in plan['stages'][:-1]:
                if not (folder / spec['name'] / 'actual_wait.json').exists():
                    while not capacity(plan):
                        status('waiting', 'Awaiting empty physicalGPU3 and registered GPU/host capacity')
                        time.sleep(20)
                status('running', spec['name'])
                results.append(stage(folder, spec, plan_ref, gpu_lock))
                validate_product(spec, plan, protocol)
            last = r.read(results[-1])
            completed = dict(schema='joint_american_GPU_chain_completed_v1', model_id=protocol['model_id'],
                plan_ref=plan_ref, successor_v2_config_ref=plan['successor_v2_config_ref'],
                stage_actual_wait_refs=results, cache_actual_wait_ref=cache_wait,
                actual_wait=True, actual_returncode=0, owned_process_group_absent=True, products_complete=True,
                gpu_qualification_fit_and_all15_predictions_complete=True, native_score_pending=True,
                pid=last['pid'], process_start_ticks=last['process_start_ticks'])
            if (folder / 'completed.json').exists():
                need(r.read(r.ref(folder / 'completed.json')) == completed, 'Exact existing GPU-chain completion')
            else:
                immutable(folder / 'completed.json', completed)
        # GPU3 is free here, before native completion waits that can depend on successor GPU work.
        native = native_ready(plan, protocol)
        while native is None:
            status('waiting', 'GPU chain complete; waiting for all15 authentic ARGformer and FLARE predictions')
            time.sleep(30)
            native = native_ready(plan, protocol)
        if not (folder / 'native_prediction_barrier.json').exists():
            immutable(folder / 'native_prediction_barrier.json', dict(plan_ref=plan_ref, parents=native, truth_opened=False))
        else:
            need(r.read(r.ref(folder / 'native_prediction_barrier.json'))['parents'] == native, 'Native products changed after sealing')
        while not host_capacity(plan):
            status('waiting', 'GPU lock released; awaiting128GiB host capacity for CPU score')
            time.sleep(20)
        status('running', 'score')
        score_wait = stage(folder, plan['stages'][-1], plan_ref)
        validate_product(plan['stages'][-1], plan, protocol)
        if not (folder / 'scores_completed.json').exists():
            immutable(folder / 'scores_completed.json', dict(plan_ref=plan_ref, gpu_chain_ref=r.ref(folder / 'completed.json'),
                score_actual_wait_ref=score_wait, native_prediction_barrier_ref=r.ref(folder / 'native_prediction_barrier.json')))
        status('completed', 'Frozen GPU chain and15-parent CPU scoring complete')


if __name__ == '__main__':
    main()
