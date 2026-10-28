"""Bounded GPU forward/backward profile of frozen predictors, no optimizer steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import traceback
import sys
import uuid
import time

import setproctitle
setproctitle.setproctitle('AI Auto Research Pipeline Test')
import numpy as np
import pynvml
import torch
from torch.nn import functional as F
from research.redesign_20261028.architecture import window_model_frozen_v1 as v1
from research.redesign_20261028.architecture import window_model_focal_executor_v2 as v2

BASE = Path('/workspace-SR003.nfs2/ARG-GNN/research/redesign_20261028')
UUID = 'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v1','v2'], required=True)
    parser.add_argument('--arm', choices=['no_cross','structure'], required=True)
    parser.add_argument('--batch-size', type=int, choices=[4,16], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == UUID
    protocol = json.loads((BASE/'root/gpu_profile_protocol.json').read_text())
    assert sha(v1.__file__) == protocol['v1_sha256']
    assert sha(v2.__file__) == protocol['v2_sha256']
    output = BASE/'root/gpu_profiles'/f'{args.arm}_{args.version}_b{args.batch_size}'
    output.mkdir(parents=True, exist_ok=False)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(UUID)
    foreign_before = [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                      if p.pid != os.getpid()]
    assert not foreign_before, 'Measurement device has another compute process, preserve it and defer profiling'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(101)
    manifest_path = BASE/'diagnostic_packets/manifest.json'
    assert sha(manifest_path) == protocol['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['fit'][:args.batch_size]
    for row in rows:
        assert sha(manifest_path.parent/row['packet_path']) == row['packet_sha256']
        assert sha(manifest_path.parent/row['label_path']) == row['label_sha256']
    begin = time.perf_counter()
    observations = v1.collate_observations([v1.load_observations(manifest_path.parent/row['packet_path']) for row in rows])
    original = v1.prepare_observations(observations)
    batch = original if args.version == 'v1' else v2.prepare_executor(original)
    preparation = time.perf_counter() - begin
    labels = torch.tensor([int(np.load(manifest_path.parent/row['label_path'],allow_pickle=False)) for row in rows])
    module = v1 if args.version == 'v1' else v2
    model = module.WindowTreeLoom(width=32, classes=6)
    checkpoint = BASE/'root/learning_runs'/args.arm/'final_state.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    device = torch.device('cuda:0')
    properties = torch.cuda.get_device_properties(device)
    assert uuid.UUID(str(properties.uuid).removeprefix('GPU-')) == uuid.UUID(UUID.removeprefix('GPU-'))
    begin = time.perf_counter()
    batch = batch.to(device)
    model = model.to(device)
    labels = labels.to(device)
    torch.cuda.synchronize()
    transfer = time.perf_counter() - begin
    process_name = pynvml.nvmlSystemGetProcessName(os.getpid())
    if isinstance(process_name,bytes): process_name=process_name.decode()
    assert process_name == 'AI Auto Research Pipeline Test', process_name
    samples = []
    sampling_errors = []
    stop = threading.Event()
    def sample_device():
        try:
            while not stop.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                foreign = [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid != os.getpid()]
                samples.append({'monotonic':time.perf_counter(),'gpu_percent':util.gpu,'foreign_compute_pids':foreign,
                                'memory_controller_percent':util.memory,'device_memory_used':memory.used})
                stop.wait(0.1)
        except Exception as error:
            sampling_errors.append(repr(error))
    sampler = threading.Thread(target=sample_device,daemon=True)
    sampler.start()
    model.train()
    durations = []
    try:
        for iteration in range(protocol['warmup_repetitions']+protocol['measured_repetitions']):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            logits = model(batch, mode=args.arm)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begin
            if iteration >= protocol['warmup_repetitions']:
                durations.append(elapsed)
            assert torch.isfinite(logits).all() and torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        gradients = np.concatenate([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().cpu().reshape(-1).numpy()
                                    for p in model.parameters()])
        np.savez(output/'numerics.npz',logits=logits.detach().cpu().numpy(), gradients=gradients,
                 gradient_present=np.array([p.grad is not None for p in model.parameters()]))
    finally:
        stop.set()
        sampler.join()
    foreign_after = [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                     if p.pid != os.getpid()]
    result = {
        'schema':'bounded_gpu_forward_backward_profile_v1','version':args.version,'arm':args.arm,
        'batch_size':args.batch_size,'physical_gpu':3,'gpu_uuid':UUID,'process_title':process_name,
        'pid':os.getpid(),'torch_version':torch.__version__,
        'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'protocol_sha256':sha(BASE/'root/gpu_profile_protocol.json'),
        'harness_sha256':sha(__file__),'model_v1_sha256':sha(v1.__file__),'executor_v2_sha256':sha(v2.__file__),
        'checkpoint_sha256':sha(checkpoint),'manifest_sha256':sha(manifest_path),
        'target_ids':[(row['parent_id'],row['example_index']) for row in rows],
        'nodes':len(original.parent),'tree_occurrences':len(original.leaves),'site_occurrences':len(original.site_genotype),
        'preparation_seconds':preparation,'transfer_seconds':transfer,
        'forward_backward_seconds':durations,'median_seconds':statistics.median(durations),
        'min_seconds':min(durations),'max_seconds':max(durations),'profile_loss':float(loss.detach()),
        'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),
        'device_activity_samples':samples,'sampling_errors':sampling_errors,'foreign_compute_pids_before':foreign_before,'foreign_compute_pids_after':foreign_after,
        'optimizer_updates':0,'accuracy_evaluated':False,
        'controlled_device_activity_qualified':bool(samples) and not sampling_errors and not foreign_after and not any(row['foreign_compute_pids'] for row in samples),
        'scope':'Repeated fixed-batch profile. Excludes dataset streaming, optimizer, gradient reset and numerical checks. Peaks cover transfer, warmup and measured iterations. Device samples are short-window aggregate activity, not sustained useful training utilization.'}
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    total_output = sum(path.stat().st_size for path in (BASE/'root/gpu_profiles').rglob('*') if path.is_file())
    assert total_output <= protocol['max_output_bytes']
    print(json.dumps({k:result[k] for k in ['version','arm','batch_size','median_seconds','allocated_peak_bytes','reserved_peak_bytes','profile_loss']}),flush=True)
    pynvml.nvmlShutdown()


if __name__=='__main__':
    try:
        main()
    except Exception:
        failures = BASE/'root/gpu_profiles'
        failures.mkdir(parents=True,exist_ok=True)
        (failures/f'failure_{os.getpid()}.json').write_text(json.dumps({'argv':sys.argv,'traceback':traceback.format_exc()},indent=2)+'\n')
        raise
