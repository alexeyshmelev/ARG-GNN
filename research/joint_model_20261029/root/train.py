"""Registered site-batched ancestry fitting with canonical target weighting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time
import traceback
import uuid

import setproctitle
setproctitle.setproctitle('AI Auto Research Pipeline Test')
import numpy as np
import pynvml
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from research.joint_model_20261029.data.site_dataset import CachedSiteDataset, collate_sites
from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel
from research.joint_model_20261029.data.loader_shutdown import register_worker_resource_cleanup, shutdown_loaders

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/joint_model_20261029'
RUN_CONTEXT = {}
GPUS = {3:'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786',
        4:'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71',
        5:'GPU-3db85505-d061-e45d-d154-e7e8c1cad497'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def initialize_worker(worker_id):
    torch.set_num_threads(1)
    setproctitle.setproctitle('AI Agent Research Data')
    register_worker_resource_cleanup()


def make_loader(dataset, batch_sites, workers, seed, shuffle):
    return DataLoader(dataset, batch_size=batch_sites, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      num_workers=workers, persistent_workers=bool(workers),
                      prefetch_factor=2 if workers else None, pin_memory=True,
                      collate_fn=collate_sites, worker_init_fn=initialize_worker,
                      multiprocessing_context='spawn' if workers else None,
                      drop_last=False)


def scores(logits, labels, classes):
    loss = F.cross_entropy(logits, labels, reduction='sum').item()
    pred = logits.argmax(1)
    confusion = torch.bincount(labels * classes + pred, minlength=classes**2).reshape(classes, classes)
    c = confusion.to(torch.float64)
    denominator = c.sum(0) + c.sum(1)
    f1 = torch.where(denominator > 0, 2*c.diag()/denominator, torch.zeros_like(denominator))
    return dict(targets=len(labels), accuracy=float(c.diag().sum()/len(labels)),
                cross_entropy=loss/len(labels), macro_f1=float(f1.mean()),
                confusion=confusion.tolist(), true_counts=confusion.sum(1).tolist(),
                predicted_counts=confusion.sum(0).tolist())


@torch.no_grad()
def evaluate(model, loader, device, mode, expected_ids, classes, output=None):
    start = time.perf_counter()
    model.eval()
    logits, labels, identities = [], [], []
    for cpu in loader:
        batch = cpu.to(device, non_blocking=True)
        value = model(batch.observations, mode=mode)
        if not torch.isfinite(value).all().item():
            raise FloatingPointError('Nonfinite evaluation logits')
        logits.append(value.cpu())
        labels.append(batch.labels.cpu())
        identities.extend(cpu.canonical_ids)
    value = torch.cat(logits)
    truth = torch.cat(labels)
    assert len(truth) == len(expected_ids)
    assert len(set(identities)) == len(expected_ids)
    assert set(identities) == set(expected_ids), 'Evaluation IDs differ from registered canonical support'
    result = scores(value, truth, classes)
    result['seconds'] = time.perf_counter() - start
    result['canonical_identity_set_sha256'] = hashlib.sha256(json.dumps(sorted(identities), separators=(',',':')).encode()).hexdigest()
    if output is not None:
        np.savez_compressed(output, logits=value.numpy(), labels=truth.numpy(),
                            parent_id=np.array([x[0] for x in identities]),
                            example_index=np.array([x[1] for x in identities], dtype=np.int64))
    return result


class Activity:
    def __init__(self, handle):
        self.handle = handle
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        self.phase = 'startup'
        self.thread = threading.Thread(target=self.run, daemon=True)
    def run(self):
        try:
            while not self.stop.is_set():
                u = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                m = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                pids = [x.pid for x in pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)]
                self.samples.append(dict(monotonic=time.perf_counter(), phase=self.phase,
                                         gpu_percent=u.gpu, memory_percent=u.memory,
                                         device_bytes=m.used, compute_pids=pids))
                self.stop.wait(0.5)
        except Exception as error:
            self.errors.append(repr(error))
    def finish(self):
        self.stop.set()
        self.thread.join()
        return dict(samples=self.samples, errors=self.errors)


def main():
    launch_begin=time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--arm', choices=['no_cross','leaf_only','structure'], required=True)
    parser.add_argument('--stage', choices=['development','benchmark','profile'], required=True)
    parser.add_argument('--gpu', type=int, choices=[3,4,5], required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--batch-sites', type=int)
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    assert '/' not in args.run_name and '..' not in args.run_name
    gpu_uuid = GPUS[args.gpu]
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == gpu_uuid
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    assert protocol['live_launch_allowed'] is True, 'Prospective design is not a launch protocol'
    assert args.stage in protocol['allowed_stages'], 'Stage outside frozen launch scope'
    for source in protocol['source_refs']:
        assert digest(ROOT/source['path']) == source['sha256'], source['path']
    assert digest(protocol['manifest']) == protocol['manifest_sha256']
    assert protocol['test_used_for_selection'] is False
    if args.stage == 'benchmark':
        assert protocol['benchmark_training_authorized'] is True
        assert args.arm == protocol['selected_arm']
    batch_sites = args.batch_sites or protocol['batch_sites']
    assert batch_sites > 0
    if args.stage != 'profile':
        assert batch_sites == protocol['batch_sites']
    output = BASE/'runs'/args.run_name
    output.mkdir(parents=True, exist_ok=False)
    RUN_CONTEXT.update(output=str(output), phase='initialization', canonical_ids=[], site_keys=[])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(protocol['model_seed'])
    np.random.seed(protocol['model_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
    free = pynvml.nvmlDeviceGetMemoryInfo(handle).free
    if free < protocol['minimum_free_gpu_bytes']:
        raise RuntimeError('Insufficient free memory, existing processes are preserved')
    device = torch.device('cuda:0')
    actual_uuid = str(torch.cuda.get_device_properties(device).uuid).removeprefix('GPU-')
    assert uuid.UUID(actual_uuid) == uuid.UUID(gpu_uuid.removeprefix('GPU-'))
    process_name = pynvml.nvmlSystemGetProcessName(os.getpid())
    if isinstance(process_name,bytes): process_name=process_name.decode()
    assert process_name == 'AI Auto Research Pipeline Test'
    fit_split = 'full_train' if args.stage == 'benchmark' else 'fit'
    ds = CachedSiteDataset(protocol['manifest'], split=fit_split,cache_bytes=protocol['worker_cache_bytes'],mode=args.arm)
    loader = make_loader(ds,batch_sites,protocol['workers'],protocol['batch_seed'],True)
    dev = None
    dev_loader = None
    if args.stage == 'development':
        dev = CachedSiteDataset(protocol['manifest'], split='development',cache_bytes=protocol['worker_cache_bytes'],mode=args.arm)
        dev_loader = make_loader(dev,batch_sites,protocol['workers'],protocol['batch_seed'],False)
    model = JointAncestryModel(**protocol['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'],
                                 weight_decay=protocol['weight_decay'],
                                 betas=tuple(protocol['betas']), eps=protocol['epsilon'])
    init = b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())
    epochs = protocol['benchmark_epochs'] if args.stage == 'benchmark' else protocol['epochs']
    if args.stage == 'profile': epochs=protocol['profile_epochs']
    planned_updates = epochs*((len(ds)+batch_sites-1)//batch_sites)
    started = dict(stage=args.stage,arm=args.arm,pid=os.getpid(),gpu=args.gpu,gpu_uuid=gpu_uuid,
                   process_title=process_name,protocol_sha256=digest(protocol_path),
                   harness_sha256=digest(__file__),manifest_sha256=protocol['manifest_sha256'],
                   initialization_sha256=hashlib.sha256(init).hexdigest(),
                   model_parameters=sum(x.numel() for x in model.parameters()),
                   fit_sites=ds.site_count,fit_targets=ds.target_count,
                   batch_sites=batch_sites,workers=protocol['workers'],worker_cache_bytes=protocol['worker_cache_bytes'],
                   planned_epochs=epochs,planned_updates=planned_updates,
                   started_unix=time.time(),startup_seconds=time.perf_counter()-launch_begin,device_name=torch.cuda.get_device_name(device),
                   optimizer=dict(name='AdamW',learning_rate=protocol['learning_rate'],
                                  betas=protocol['betas'],epsilon=protocol['epsilon'],
                                  weight_decay=protocol['weight_decay'],gradient_clip=protocol['gradient_clip']),
                   precision=dict(learned_dtype='float32',tf32_matmul=False,tf32_cudnn=False),
                   profile_warmup_epochs=1 if args.stage=='profile' else None)
    save_json(output/'started.json',started)
    activity = Activity(handle)
    activity.thread.start()
    begin = time.perf_counter()
    updates = 0
    presentations = 0
    target_order = hashlib.sha256()
    epoch_records = []
    phase_seconds = dict(iterator_startup=0.,loader_wait=0.,transfer=0.,forward_loss_check=0.,backward_clip=0.,optimizer=0.)
    events_pending = []
    epoch_loss = torch.zeros((),device=device)
    epoch_hits = torch.zeros((),device=device,dtype=torch.int64)
    clipped_steps = torch.zeros((),device=device,dtype=torch.int64)
    maximum_gradient_norm = torch.zeros((),device=device)
    try:
        for epoch in range(1,epochs+1):
            model.train()
            activity.phase = 'training'
            RUN_CONTEXT['phase'] = 'training'
            epoch_begin = time.perf_counter()
            epoch_loss.zero_()
            epoch_hits.zero_()
            epoch_targets = 0
            epoch_sites = 0
            activity.phase = 'loader_startup'
            iterator_begin = time.perf_counter()
            iterator = iter(loader)
            iterator_seconds = time.perf_counter() - iterator_begin
            phase_seconds['iterator_startup'] += iterator_seconds
            first_batch_seconds = None
            while True:
                wait_begin = time.perf_counter()
                try: cpu = next(iterator)
                except StopIteration: break
                phase_seconds['loader_wait'] += time.perf_counter()-wait_begin
                if first_batch_seconds is None:
                    first_batch_seconds = time.perf_counter()-epoch_begin
                    activity.phase = 'training'
                RUN_CONTEXT.update(canonical_ids=cpu.canonical_ids,site_keys=cpu.site_keys)
                assert cpu.population_sites == ds.site_count and cpu.population_targets == ds.target_count
                assert abs(cpu.loss_scale-ds.site_count/(cpu.site_count*ds.target_count))<1e-14
                markers = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
                markers[0].record()
                batch = cpu.to(device, non_blocking=True)
                markers[1].record()
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch.observations,mode=args.arm)
                raw_loss = F.cross_entropy(logits,batch.labels,reduction='sum')
                loss = raw_loss*batch.loss_scale
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Nonfinite training objective')
                markers[2].record()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'],error_if_nonfinite=True)
                clipped_steps.add_(norm>protocol['gradient_clip'])
                maximum_gradient_norm = torch.maximum(maximum_gradient_norm,norm.detach())
                markers[3].record()
                optimizer.step()
                markers[4].record()
                events_pending.append(markers)
                updates += 1
                count = len(batch.labels)
                presentations += count
                epoch_targets += count
                epoch_sites += cpu.site_count
                epoch_loss.add_(raw_loss.detach())
                epoch_hits.add_((logits.detach().argmax(1)==batch.labels).sum())
                target_order.update(json.dumps(cpu.canonical_ids,separators=(',',':')).encode())
                if updates%protocol['log_every_updates']==0 or updates==1:
                    torch.cuda.synchronize()
                    for points in events_pending:
                        for index,key in enumerate(['transfer','forward_loss_check','backward_clip','optimizer']):
                            phase_seconds[key]+=points[index].elapsed_time(points[index+1])/1000
                    events_pending.clear()
                    record=dict(update=updates,epoch=epoch,target_presentations=presentations,
                                epoch_target_presentations=epoch_targets,
                                online_training_cross_entropy=float(epoch_loss)/epoch_targets,
                                online_training_accuracy=float(epoch_hits)/epoch_targets,
                                elapsed_seconds=time.perf_counter()-begin,
                                phase_seconds=phase_seconds.copy(),
                                preclip_gradient_norm=float(norm),clipped_steps=int(clipped_steps),
                                max_preclip_gradient_norm=float(maximum_gradient_norm))
                    with (output/'progress.jsonl').open('a') as f:
                        f.write(json.dumps(record,allow_nan=False)+'\n')
                    save_json(output/'status.json',dict(status='training',**record))
                    print(json.dumps(record,allow_nan=False),flush=True)
            torch.cuda.synchronize()
            for points in events_pending:
                for index,key in enumerate(['transfer','forward_loss_check','backward_clip','optimizer']):
                    phase_seconds[key]+=points[index].elapsed_time(points[index+1])/1000
            events_pending.clear()
            epoch_end = time.perf_counter()
            activity.phase = 'checkpoint'
            epoch_record=dict(epoch_start_monotonic=epoch_begin,epoch_end_monotonic=epoch_end,
                              iterator_seconds=iterator_seconds,first_batch_seconds=first_batch_seconds,epoch=epoch,updates=updates,targets=epoch_targets,sites=epoch_sites,
                              training_seconds=epoch_end-epoch_begin,
                              online_training_cross_entropy=float(epoch_loss)/epoch_targets,
                              online_training_accuracy=float(epoch_hits)/epoch_targets)
            assert epoch_targets==ds.target_count and epoch_sites==ds.site_count
            if args.stage!='profile':
                temporary=output/'checkpoint.pt.tmp'
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                                epoch=epoch,updates=updates,protocol_sha256=digest(protocol_path)),temporary)
                temporary.replace(output/'checkpoint.pt')
                if dev_loader is not None and (epoch%protocol['evaluate_every_epochs']==0 or epoch==epochs):
                    activity.phase='development_evaluation'
                    epoch_record['development']=evaluate(model,dev_loader,device,args.arm,dev.canonical_ids,
                                                         protocol['model']['classes'],
                                                         output/'final_development.npz' if epoch==epochs else None)
            epoch_records.append(epoch_record)
            save_json(output/'epochs.json',epoch_records)
        if args.stage!='profile':
            activity.phase='fit_evaluation'
            final_fit=evaluate(model,loader,device,args.arm,ds.canonical_ids,protocol['model']['classes'],output/'final_fit.npz')
            torch.save(model.state_dict(),output/'final_state.pt')
        else: final_fit=None
        result=dict(**started,completed_unix=time.time(),optimizer_updates=updates,
                    target_presentations=presentations,target_order_sha256=target_order.hexdigest(),
                    elapsed_seconds=time.perf_counter()-begin,phase_seconds=phase_seconds,
                    epoch_records=epoch_records,final_fit=final_fit,
                    clipped_steps=int(clipped_steps),max_preclip_gradient_norm=float(maximum_gradient_norm),
                    allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                    reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                    parent_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    test_evaluated=False,benchmark_score_claimed=False,
                    activity_scope='Device-wide sampled utilization, inspect compute PIDs before attribution. CPU loader wait and CUDA event times can overlap and must not be added as exclusive wall-time fractions.')
        save_json(output/'result.json',result)
        save_json(output/'status.json',dict(status='completed',updates=updates,elapsed_seconds=result['elapsed_seconds']))
    except Exception:
        save_json(output/'failure.json',dict(traceback=traceback.format_exc(),updates=updates,target_presentations=presentations,context=RUN_CONTEXT))
        save_json(output/'status.json',dict(status='failed',updates=updates))
        raise
    finally:
        activity.phase = 'loader_shutdown'
        try:
            save_json(output/'loader_shutdown.json',shutdown_loaders(loader,dev_loader))
        except Exception:
            save_json(output/'loader_shutdown_failure.json',dict(traceback=traceback.format_exc()))
            raise
        finally:
            save_json(output/'activity.json',activity.finish())
            pynvml.nvmlShutdown()


if __name__=='__main__':
    try:
        main()
    except Exception:
        if 'output' in RUN_CONTEXT:
            failure=Path(RUN_CONTEXT['output'])/'failure.json'
            if not failure.exists():
                save_json(failure,dict(traceback=traceback.format_exc(),context=RUN_CONTEXT))
        raise
    finally:
        try: pynvml.nvmlShutdown()
        except pynvml.NVMLError: pass
