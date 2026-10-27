"""ROOT-owned CPU qualification: one serial pair and one adaptive concurrent pair.

No TRAIN truth, model, optimizer, GPU initialization or production state changes.
The coordinator starts two fresh visible Python workers sequentially. It never
signals a process; the outer ROOT guardian owns any exceptional family cleanup.
"""
import argparse
from concurrent.futures import TimeoutError as FutureTimeout
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .preparation_evidence_v2 import bounded_signature, array_layout, validate_signature, SCHEMA as SIGNATURE_SCHEMA

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/efficiency_20261027/execution'
SPEC=BASE/'TWO_PARENT_COMPARISON_SPEC_v1.json'
MANIFEST=BASE/'source_pins_two_parent_comparison_v2.json'
ARMS=('current_serial_pair','adaptive_concurrent_pair')


def need(ok,message):
    if not ok:raise ValueError(message)


def ref(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path,value):
    raw=json.dumps(value,indent=2,allow_nan=False).encode()
    need(len(raw)<=8*2**20,'Bounded qualification metadata')
    with Path(path).open('xb') as handle:handle.write(raw+b'\n')


def identity(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def verify(digest):
    need(ref(MANIFEST)['sha256']==digest,'Exact reviewed comparison manifest')
    rows=json.loads(MANIFEST.read_text())['source_refs']
    need(all(ref(row['path'])==row for row in rows),'Immutable comparison dependencies changed')
    return ref(MANIFEST)


def restore_CPU_random_state(saved):
    import random
    import numpy as np
    import torch
    random.setstate(saved['python'])
    state=saved['numpy']
    np.random.set_state((state['name'],np.asarray(state['keys'],dtype=np.uint32),
                       state['position'],state['has_gauss'],state['cached_gaussian']))
    torch.set_rng_state(saved['torch_CPU'])
    need(not torch.cuda.is_initialized(),'CPU comparison never restores/initializes a CUDA RNG')


def CPU_random_signature():
    import pickle,random
    import numpy as np
    import torch
    return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),
                          torch.get_rng_state().numpy().tobytes()))).hexdigest()


class FamilyMonitor:
    """Read-only actual own descendants, PSS/RSS and PID/start identity samples."""
    def __init__(self,output,maximum_bytes,seconds):
        import threading
        self.output=output;self.maximum=maximum_bytes;self.seconds=seconds
        self.start=time.monotonic();self.stop=threading.Event();self.lock=threading.RLock()
        self.error=None;self.known={};self.samples=0;self.journal_bytes=0
        self.peaks=dict(main_RSS_bytes=0,family_PSS_or_RSS_bytes=0,encoder_children=0,all_children=0)
        self.thread=threading.Thread(target=self._run,name='TreeLoom_pair_family_sampler',daemon=True)

    def sample(self):
        import psutil
        process=psutil.Process();rows=[]
        for member in [process]+process.children(recursive=True):
            try:
                memory=member.memory_full_info();ticks=identity(member.pid)
                if ticks is None:continue
                command=member.cmdline()
                tracker=any('multiprocessing.resource_tracker' in part for part in command)
                role='main' if member.pid==os.getpid() else ('resource_tracker' if tracker else 'encoder_worker')
                rows.append(dict(pid=member.pid,start_ticks=ticks,role=role,RSS_bytes=memory.rss,
                    PSS_or_RSS_bytes=getattr(memory,'pss',memory.rss),measure='PSS' if hasattr(memory,'pss') else 'RSS'))
            except (psutil.NoSuchProcess,psutil.ZombieProcess):continue
        main=sum(r['RSS_bytes'] for r in rows if r['role']=='main')
        total=sum(r['PSS_or_RSS_bytes'] for r in rows)
        workers=sum(r['role']=='encoder_worker' for r in rows)
        sample=dict(elapsed_seconds=time.monotonic()-self.start,main_RSS_bytes=main,
            family_PSS_or_RSS_bytes=total,encoder_children=workers,all_children=len(rows)-1,processes=rows)
        with self.lock:
            for key in self.peaks:self.peaks[key]=max(self.peaks[key],sample[key])
            for row in rows:self.known[(row['pid'],row['start_ticks'])]=dict(pid=row['pid'],start_ticks=row['start_ticks'],role=row['role'])
            self.samples+=1
            raw=(json.dumps(sample,separators=(',',':'))+'\n').encode()
            self.journal_bytes+=len(raw)
            need(self.journal_bytes<=16*2**20,'Bounded process-family sample journal')
            with (self.output/'family_samples.jsonl').open('ab') as handle:handle.write(raw)
            need(main<=self.maximum and total<=self.maximum,'Actual own main/family memory exceeded declared diagnostic cap')
            need(workers<=8,'At most eight live encoder workers')
        return sample

    def _run(self):
        try:
            while not self.stop.is_set():self.sample();self.stop.wait(2.)
        except BaseException as error:
            with self.lock:self.error=error

    def check(self,stage):
        with self.lock:error=self.error
        if error is not None:raise RuntimeError('Own family monitor failed at '+stage) from error
        need(time.monotonic()-self.start<=self.seconds,'Bounded complete-pair worker lifetime')

    def start_sampling(self):self.sample();self.thread.start()

    def finish(self):
        self.stop.set();self.thread.join()
        self.sample();self.check('monitor_finish')
        with self.lock:
            return dict(peaks=dict(self.peaks),samples=self.samples,
                tracked_processes=list(self.known.values()),sample_interval_seconds=2,
                read_only=True,signals_sent=0,resource_tracker_counted_separately=True)


def run_arm(args,output,spec):
    # Environment setup and hooks happen once before either producer can start.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import gc,threading
    import torch
    import setproctitle
    from arggnn.benchmarking import contracts as c
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
    from research.efficiency_20261027.integration_candidate.memory_v1 import LayoutMemoryGovernor,LayoutAwareParents
    from research.efficiency_20261027.execution.adaptive_prefetch_v2 import AdaptiveMemoryGovernor,AdaptiveCompactParents,EncoderCredits
    from research.preprocessing_optimization_20261026.integration import hooks
    from research.efficiency_20261027.execution import adaptive_encoder_hooks_v2
    from research.reviews.workflow_progress_diagnosis_20261018.treeloom.resident_pool_v2 import exact_signature,restore_bundle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    need(args.arm in ARMS and not torch.cuda.is_initialized(),'One CPU-only comparison arm')
    output.mkdir(exist_ok=False)
    monitor=FamilyMonitor(output,spec['max_actual_family_bytes'],spec['max_arm_seconds'])
    monitor.start_sampling();pool=None
    events=[];event_lock=threading.RLock();timing={};clock=[None]
    def rss():return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    def check(stage):
        monitor.check(stage)
        need(rss()<=spec['max_actual_family_bytes'],'Actual main RSS cap at '+stage)
        need(not torch.cuda.is_initialized(),'No CUDA initialization during observation-only qualification')
        if pool is not None and args.arm==ARMS[1]:pool.raise_if_failed()
    def event(name,detail):
        row=dict(event=name,observed_unix=time.time(),elapsed_seconds=None if clock[0] is None else time.perf_counter()-clock[0],**detail)
        with event_lock:
            events.append(row)
            need(len(events)<=100,'Bounded two-parent progress')
            if name=='compact_parent_ready':timing[detail['parent_id']]['ready_seconds']=row['elapsed_seconds']
    try:
        need(ref(spec['checkpoint_ref']['path'])==spec['checkpoint_ref'],'Exact immutable update268 checkpoint')
        saved=torch.load(ROOT/spec['checkpoint_ref']['path'],map_location='cpu',weights_only=True,mmap=True)
        need(saved['complete_parent_updates']==spec['start_updates']==268,'Frozen parent-selection boundary')
        registry=fresh.FreshParentRegistry.from_ref(spec['parent_registry_ref'])
        schedule=[p for epoch in range(1,31) for p in registry.epoch_order(epoch)]
        parents=list(dict.fromkeys(schedule[268:]))[:2]
        need(parents==spec['parents'],'Exact earliest two distinct registered parents')
        need(saved['observation_barrier_ref']==spec['observation_barrier_ref'], 'Exact checkpoint observation barrier')
        barrier=c.read_ref(spec['observation_barrier_ref'])
        need(barrier['all20_observations_sealed_before_truth'] and not barrier['private_truth_opened']
             and set(barrier['parents'])==set(registry.ids('train')), 'Full original observation-only barrier')
        for parent in registry.ids('train'):c.read_ref(barrier['parents'][parent]['plan_ref'])
        for parent in parents:
            need(spec['seals'][parent]==barrier['parents'][parent],'Original complete observation seal')
            need(spec['plans'][parent]==dict(c.read_ref(barrier['parents'][parent]['plan_ref']),plan_ref=barrier['parents'][parent]['plan_ref']),
                 'Original complete observation plan')
        rt=fresh.FreshTreeLoom(registry)
        inputs=spec['observed_inputs']
        need(all(set(item)==set(fresh.INPUT_REFS) for item in inputs.values()),'Only observation inputs, no truth refs')
        external={p:{'bank.genotype':c.read_ref(inputs[p]['observation_ref'])['genotypes_ref']} for p in parents}
        bound=c.read_ref(spec['bound_ref'])
        bounds={row['parent_id']:row['complete_compact_cache_upper_bound_bytes']+160000 for row in bound['parents']}
        def prepare(parent,*,resource_check):
            timing[parent]['actual_prepare_start_seconds']=time.perf_counter()-clock[0]
            item=inputs[parent]
            state=load_observation(item['observation_ref'],item['focal_registry_ref'],dict(
                native_preparation_ref=item['native_preparation_ref'],class_order=list(fresh.CLASSES),
                expected_haplotypes=1600,reference_counts=[200,200,200],
                input_reference_class_mask=None,input_reference_sample_indices=None))
            timing[parent]['loaded_seconds']=time.perf_counter()-clock[0]
            bundle,meta=rt.prepare_observation(state,split='train',resource_check=resource_check,
                **item,max_nodes=400000000,max_holds=100000,max_episode_payload_bytes=8*2**30)
            del state
            plan,seal=spec['plans'][parent],spec['seals'][parent]
            need(bundle.parent_id==parent and bundle.sequence_length_bp==30000000
                 and bundle.native_holds==plan['native_holds'] and bundle.original_sites==plan['original_sites']
                 and bundle.focal_identity_sha256==seal['focal_identity_sha256']
                 and all(getattr(bundle,key)==seal[key]==item[key] for key in fresh.INPUT_REFS),
                 'Complete native parent retains every original input/focal/native count')
            timing[parent]['prepared_seconds']=time.perf_counter()-clock[0]
            return bundle,meta
        if args.arm==ARMS[0]:
            governor=LayoutMemoryGovernor(maximum_bytes=spec['governor_bytes'],rss_bytes=rss,
                resource_check=check,reserve_bytes=spec['phase_reserve_bytes'])
            bindings=hooks.install_tree(preparation_workers=4,max_rss_bytes=spec['governor_bytes'])
            pool=LayoutAwareParents(layout_governor=governor,parent_bounds=bounds,input_refs=inputs,
                external_arrays=external,max_cache_bytes=spec['compact_pool_cap_bytes'],
                max_rss_bytes=spec['governor_bytes'],phase_reserve_bytes=spec['phase_reserve_bytes'],
                rss_bytes=rss,resource_check=check,event=event,prepare=prepare)
        else:
            governor=AdaptiveMemoryGovernor(maximum_bytes=spec['governor_bytes'],rss_bytes=rss,
                resource_check=check,reserve_bytes=spec['phase_reserve_bytes'])
            credits=EncoderCredits(maximum_bytes=spec['governor_bytes'])
            bindings=adaptive_encoder_hooks_v2.install_tree(credits)
            pool=AdaptiveCompactParents(schedule=schedule,start_cursor=268,parent_bounds=bounds,input_refs=inputs,
                external_arrays=external,max_cache_bytes=spec['compact_pool_cap_bytes'],governor=governor,
                credits=credits,resource_check=check,event=event,prepare=prepare)
        restore_CPU_random_state(saved['random_state']);before=CPU_random_signature()
        del saved
        gc.collect()
        write(output/'configuration.json',dict(arm=args.arm,spec_ref=ref(SPEC),source_manifest_ref=ref(MANIFEST),
            inputs=inputs,parents=parents,modeled_foreground_reserve_bytes=spec['modeled_foreground_reserve_bytes'],
            actual_training_executed=False,preprocessing_bindings=bindings))
        def wait_future(future,parent):
            while True:
                try:
                    need(future.result(timeout=1)==parent,'Exact demanded preparation future')
                    return
                except FutureTimeout:
                    if future.done():raise
                    check('awaiting_complete_parent')
        clock[0]=time.perf_counter();deferred=[]
        # Retain the original foreground headroom in both admission paths, but
        # perform no learned work. This isolates actual preparation supply.
        with governor.phase('qualification_foreground_reserve_only',spec['modeled_foreground_reserve_bytes']):
            if args.arm==ARMS[0]:
                for parent in parents:
                    timing[parent]=dict(request_seconds=time.perf_counter()-clock[0])
                    pool.prefetch(parent);wait_future(pool.future,parent)
            else:
                for parent in parents:
                    timing[parent]=dict(request_seconds=time.perf_counter()-clock[0])
                    if not pool._submit(parent,optional=True):deferred.append(parent)
                for parent in parents:
                    if parent in deferred:
                        need(pool._submit(parent,optional=False),'Deferred parent remains once-only and required')
                    wait_future(pool.futures[parent],parent)
            makespan=time.perf_counter()-clock[0]
        # No signature pass is included in either timed preparation interval.
        write(output/'timed_preparation.json',dict(arm=args.arm, parents=parents, timing=timing,
            pair_prepare_pack_seconds=makespan, optional_deferrals=deferred, events=events,
            source_manifest_ref=ref(MANIFEST), spec_ref=ref(SPEC)))
        signatures={};signature_refs={};restore_times={}
        for parent in parents:
            check('post_timing_exact_restore')
            record=pool.records[parent];start=time.perf_counter()
            with governor.phase('qualification_exact_restore',record['read_peak_upper_bound_bytes']):
                bundle,meta=restore_bundle(record,input_refs=inputs[parent],resource_check=check,verify_integrity=False)
                signature=exact_signature(bundle,meta,resource_check=check)
                signatures[parent]=bounded_signature(signature,array_metadata=array_layout(bundle),resource_check=check)
                del signature
                signature_path=output/(parent+'.signature.json')
                write(signature_path,signatures[parent])
                signature_refs[parent]=ref(signature_path)
                counts=dict(native_holds=bundle.native_holds,original_sites=bundle.original_sites,
                            native_nodes=int(bundle.bank.parent.numel()),focals=int(bundle.focal_hold.numel()))
                del bundle,meta;gc.collect()
            restore_times[parent]=dict(seconds=time.perf_counter()-start,counts=counts)
        after=CPU_random_signature();need(before==after,'Preparation, compact reuse and signatures must not mutate global CPU RNG')
        hooks.verify_bindings(bindings)
        receipt=pool.receipt();pool.close();pool=None
        import multiprocessing
        need(not multiprocessing.active_children(),'Every encoder ProcessPoolExecutor must reap its own workers')
        memory=monitor.finish()
        need(memory['peaks']['encoder_children']<=(4 if args.arm==ARMS[0] else 8),'Exact per-arm worker maximum')
        verify(args.source_pins_sha256)
        write(output/'completed.json',dict(status='passed',arm=args.arm,parents=parents,timing=timing,
            pair_prepare_pack_seconds=makespan,supply_seconds_per_parent=makespan/2,
            optional_deferrals=deferred,signatures=signatures,signature_refs=signature_refs,
            signature_schema=SIGNATURE_SCHEMA,post_timing_restore_signature=restore_times,
            pool_receipt=receipt,family_monitor=memory,events=events,CPU_RNG_sha256=before,
            global_CPU_RNG_unchanged=True,all_encoder_workers_reaped=True,
            CUDA_initialized=False,truth_opened=False,model_or_optimizer_executed=False,
            production_changed=False,source_manifest_ref=ref(MANIFEST)))
    except BaseException as error:
        write(output/'failed.json',dict(type=type(error).__name__,error=str(error),events=events))
        raise
    finally:
        if pool is not None:pool.close()
        monitor.stop.set();monitor.thread.join()


def run_comparison(args,output,spec):
    output.mkdir(exist_ok=False)
    commands=[];rows=[]
    for arm in ARMS:
        command=[str(ROOT/'.runtime/env/bin/python3.11'),'-B','-m',
                 'research.efficiency_20261027.execution.compare_two_parent_preparation_v2',
                 '--output',str(output/arm),'--source-pins-sha256',args.source_pins_sha256,'--arm',arm]
        commands.append(command)
        with (output/(arm+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='src:.',
                         OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
            ticks=identity(child.pid)
            write(output/(arm+'_launch.json'),dict(pid=child.pid,start_ticks=ticks,argv=command))
            code=child.wait()
        write(output/(arm+'_actual_wait.json'),dict(pid=child.pid,start_ticks=ticks,actual_wait=True,returncode=code))
        need(code==0,'Exact isolated arm failed; no automatic retry')
        row=json.loads((output/arm/'completed.json').read_text())
        deadline=time.monotonic()+15
        tracked=row['family_monitor']['tracked_processes']
        while any(identity(item['pid'])==item['start_ticks'] for item in tracked):
            need(time.monotonic()<deadline,'An exact tracked arm process remains after its actual wait')
            time.sleep(.1)
        need(row['status']=='passed','Complete arm evidence required')
        write(output/(arm+'_family_absent.json'),dict(all_tracked_PID_start_identities_absent=True,tracked=tracked))
        rows.append(row)
    baseline,candidate=rows
    need(baseline['parents']==candidate['parents']==spec['parents'],'Same complete parent consumption order')
    for row in rows:
        need(set(row['signatures'])==set(row['signature_refs'])==set(spec['parents']),
             'All original parents require bounded signature receipts')
        for parent in spec['parents']:
            validate_signature(row['signatures'][parent])
            need(ref(row['signature_refs'][parent]['path'])==row['signature_refs'][parent]
                 and json.loads((ROOT/row['signature_refs'][parent]['path']).read_text())==row['signatures'][parent],
                 'Durable exact bounded signature receipt required')
    need(baseline['signatures']==candidate['signatures'],'Every restored array/scalar/provenance signature must be identical')
    need(baseline['CPU_RNG_sha256']==candidate['CPU_RNG_sha256'],'Same restored CPU random state per arm')
    verify(args.source_pins_sha256)
    write(output/'completed.json',dict(status='passed',schema='TreeLoom_two_parent_preparation_comparison_v2',
        spec_ref=ref(SPEC),source_manifest_ref=ref(MANIFEST),commands=commands,
        arms=[{key:row[key] for key in ('arm','parents','pair_prepare_pack_seconds','supply_seconds_per_parent',
              'timing','optional_deferrals','family_monitor','pool_receipt','signature_refs','signature_schema')} for row in rows],
        signature_encoding='canonical_complete_JSON_SHA256_v2',
        exact_serial_concurrent_signatures=True,saved_CPU_RNG_unchanged=True,
        pair_speed_ratio=baseline['pair_prepare_pack_seconds']/candidate['pair_prepare_pack_seconds'],
        actual_waits_and_all_tracked_families_absent=True,training_executed=False,truth_opened=False,
        limitations=['One fixed-order serial then concurrent pair; NFS/cache/order effects may bias throughput.',
                    'The128GiB foreground reservation is arithmetic only; no training contention was measured.',
                    'Real preparation pair success does not authorize production deployment or establish training speedup.']))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-pins-sha256',required=True)
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();verify(args.source_pins_sha256)
    output=Path(args.output).resolve()
    need(output.is_relative_to(ROOT/'research/efficiency_20261027/root/runs') and not output.exists(),
         'Fresh ROOT-owned output required')
    spec=json.loads(SPEC.read_text())
    if args.arm:run_arm(args,output,spec)
    else:run_comparison(args,output,spec)


if __name__=='__main__':main()
