"""Root-run CUDA initialization qualification; one original TRAIN batch only.

No production source, checkpoint, configuration or existing receipt is changed.
The parent holds the established GPU3 mutex through every actual child wait.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/joint_evaluation_20261029/review_finish/recovery_20261030'
OUTPUT = BASE / 'native_argmix_cuda_smoke_run_v1'
LOCK = ROOT / '.runtime/benchmarking/joint_root_successor_gpu3.lock'
ENV = ROOT / '.runtime/official_argmix_abi210_v1/cu129/env'
UUID = 'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786'
RECIPE = ROOT / 'research/benchmark_rebuild_20261024/argmix/paper_argmix_recipe.json'
FIT = ROOT / 'experiments/paper_argmix_1611_rebuild_20261024_v1/fit_inputs/fit_inputs.json'
PINS = {
    'src/arggnn/benchmarking/paper_argmix/paper_argmix_train.py': '8dba581094c5ea2a82352f8ab1e33fa3442309bce1b0c87473eb9cdba3711c2a',
    'src/arggnn/benchmarking/paper_argmix/paper_argmix_model.py': '760eee9c215023dd15d00a4e66ce4a9e14094db853328eab557d850f4b66bf40',
    'src/arggnn/benchmarking/paper_argmix/paper_argmix_gpu.py': '81230787bbc560b82cab46c973b46b571d1ad63395f88e7757c2e488c7bcc0e6',
    str(RECIPE.relative_to(ROOT)): '8e6b5edf7adea653bdae1b8a8e7009892344821afcaaa5ca0e869069aa605564',
    str(FIT.relative_to(ROOT)): '51e0aa21d160086d053e7f7c32294582f280271ccc37e5254444de414a8baa43',
}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def check_boundary():
    need(Path.cwd() == ROOT, 'Run only from the project root')
    need(Path(sys.prefix).resolve() == ENV.resolve(), 'Use the exact original native runtime')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == UUID, 'Expose only physical GPU3 by UUID')
    need(os.environ.get('ARGGNN_PHYSICAL_GPU') == '3', 'Physical GPU3 binding required')
    need(os.environ.get('ARGGNN_GPU_MEMORY_POLICY') == 'full_device', 'Original full-device policy required')
    need(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Do not create bytecode')
    for name, expected in PINS.items():
        need(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, 'Pinned input changed: ' + name)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def child(case, lock_fd):
    check_boundary()
    held = os.fstat(lock_fd)
    target = LOCK.stat()
    need((held.st_dev, held.st_ino) == (target.st_dev, target.st_ino), 'Inherited established GPU3 lock required')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    import numpy as np
    import torch
    from arggnn.benchmarking.paper_argmix.paper_argmix_gpu import validate_gpu_binding
    from arggnn.benchmarking.paper_argmix import paper_argmix_train as native
    from arggnn.benchmarking.paper_argmix import paper_argmix_generator as g
    binding = validate_gpu_binding()
    need(torch.cuda.device_count() == 1, 'Exactly one admitted CUDA device required')
    need(not torch.cuda.is_initialized(), 'Each case requires a fresh CUDA subprocess')
    config = json.loads(RECIPE.read_text())
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    result = dict(case=case, binding=binding, torch_version=torch.__version__, cuda_build=torch.version.cuda)
    if case == 'early_init':
        torch.cuda.init()
    torch.manual_seed(config['training']['seed'])
    rng = np.random.Generator(np.random.PCG64(config['training']['seed']))
    if case == 'preinit':
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError as error:
            need('Invalid device argument' in str(error), 'Unexpected preinit exception: ' + str(error))
            result.update(expected_failure_reproduced=True, error=str(error), initialized=torch.cuda.is_initialized())
            save(OUTPUT / (case + '.json'), result)
            return
        raise RuntimeError('Original preinit reset unexpectedly succeeded; investigate before claiming the repair')
    if case == 'late_init':
        torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    model, model_spec = native.build_model(config, 'equation_corrected')
    order = rng.permutation(500000)
    result.update(reset_succeeded=True, initialized=torch.cuda.is_initialized(), model_spec=model_spec,
                  initial_model_sha256=model_digest(model), order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                  CPU_rng_sha256=tensor_digest(torch.get_rng_state()), CUDA_rng_sha256=tensor_digest(torch.cuda.get_rng_state(0)))
    if case == 'early_init':
        fit = json.loads(FIT.read_text())
        need(len(fit['train']) == 5, 'All five original TRAIN parents required')
        corpus = native.CompactCorpus([g.verify_ref(row['features']) for row in fit['train']],
            [g.verify_ref(row['labels']) for row in fit['train']], config_ref=g.file_ref(RECIPE), allowed_splits={'train'})
        need(corpus.examples == 500000 and all(n == 100000 for n in corpus.counts), 'Unchanged full TRAIN corpus required')
        options = config['training']
        need(options['batch_size'] == 256, 'Original registered batch size required')
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], betas=tuple(options['betas']),
            eps=options['eps'], weight_decay=options['weight_decay'])
        started = time.monotonic()
        def guard(required_write_bytes=0):
            need(required_write_bytes == 0, 'No training-product writes allowed in smoke')
            need(time.monotonic() - started <= 300, 'Smoke batch exceeded its qualification deadline')
            need(sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file()) < 8 * 2**20, 'Smoke metadata cap exceeded')
        measured = native.epoch(model, corpus, 256, device, order=order[:256], optimizer=optimizer,
            clip_norm=options['clip_grad_norm'], guard=guard)
        torch.cuda.synchronize(device)
        need(measured['examples'] == 256, 'Exactly one full original batch required')
        need(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Finite updated model required')
        need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Finite gradients required')
        need(all(bool(torch.isfinite(value).all()) for state in optimizer.state.values()
                 for value in state.values() if isinstance(value, torch.Tensor)), 'Finite optimizer state required')
        steps = sorted({float(state['step']) for state in optimizer.state.values()})
        need(steps == [1.0], 'Exactly one optimizer update per used parameter required')
        result.update(TRAIN_batch=measured, optimizer_steps=steps, finite_forward_backward_optimizer=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            updated_model_sha256=model_digest(model), production_checkpoint_written=False,
            validation_or_TEST_opened=False, training_parents=[part['parent_id'] for part in corpus.refs])
    save(OUTPUT / (case + '.json'), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('preinit', 'late_init', 'early_init'))
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.case:
        need(args.lock_fd is not None, 'Child requires inherited GPU3 lock descriptor')
        child(args.case, args.lock_fd)
        return
    check_boundary()
    need(not OUTPUT.exists(), 'Preserve existing qualification attempt; no replay')
    with LOCK.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(['nvidia-smi', '--id=3', '--query-gpu=uuid,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        need(gpu[0].strip() == UUID and int(gpu[1]) == 0 and int(gpu[2]) * 2**20 >= 16 * 2**30, 'Empty GPU3 with original admission headroom required')
        contexts = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
        need(not any(line.split(',')[0].strip() == UUID for line in contexts.splitlines()), 'GPU3 compute context present')
        OUTPUT.mkdir()
        waits = []
        for case in ('preinit', 'late_init', 'early_init'):
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--case', case, '--lock-fd', str(lock.fileno())]
            with (OUTPUT / (case + '.log')).open('xb') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
                start_ticks = (Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
                code = process.wait()
            sessions = subprocess.check_output(['ps', '-eo', 'pid=,sid=,stat='], text=True)
            absent = not any(int(line.split()[1]) == process.pid and not line.split()[2].startswith('Z') for line in sessions.splitlines())
            wait = dict(case=case, command=command, pid=process.pid, process_start_ticks=start_ticks,
                actual_wait=True, actual_returncode=code, owned_process_group_absent=absent)
            save(OUTPUT / (case + '_actual_wait.json'), wait)
            waits.append(wait)
            need(code == 0 and absent, 'Qualification child failed; preserve attempt and stop')
        late = json.loads((OUTPUT / 'late_init.json').read_text())
        early = json.loads((OUTPUT / 'early_init.json').read_text())
        comparisons = {key: late[key] == early[key] for key in
            ('model_spec', 'initial_model_sha256', 'order_sha256', 'CPU_rng_sha256', 'CUDA_rng_sha256')}
        need(all(comparisons.values()), 'CUDA initialization placement changed registered initialization or ordering')
        save(OUTPUT / 'completed.json', dict(schema='native_argmix_cuda_initialization_smoke_v1', complete=True,
            exact_native_runtime=str(ENV), physical_gpu=3, source_ref=dict(path=str(Path(__file__).relative_to(ROOT)),
            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), source_pins=PINS, actual_waits=waits,
            initialization_and_order_comparisons=comparisons, original_preinit_failure_reproduced=True,
            initialized_reset_succeeded=True, one_original_TRAIN_batch_succeeded=True, validation_or_TEST_opened=False,
            benchmark_training_claimed=False, production_checkpoint_written=False))
        print(json.dumps(dict(complete=True, receipt=str((OUTPUT / 'completed.json').relative_to(ROOT))), indent=2))


if __name__ == '__main__':
    main()
