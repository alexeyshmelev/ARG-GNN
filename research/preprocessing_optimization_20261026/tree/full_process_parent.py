"""Full observed parent parity with bounded four-process native-tree preparation."""
from pathlib import Path
from functools import partial
import hashlib
import json
import os
import resource
import threading
import time
import torch
import numpy as np
import setproctitle
from research.preprocessing_optimization_20261026.integration.full_parent import signature

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/preprocessing_optimization_20261026/tree'



def json_scalar(value):
    """Convert numeric diagnostic scalars without accepting arbitrary objects."""
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(f'Unsupported qualification JSON value: {type(value).__name__}')


def write_record_atomic(target,record):
    """Serialize completely before exposing a completed qualification artifact."""
    payload=json.dumps(record,indent=2,allow_nan=False,default=json_scalar)
    if target.exists():
        raise FileExistsError(target)
    temporary=target.with_suffix(target.suffix+'.tmp')
    with temporary.open('x') as handle:
        handle.write(payload+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def owned_memory():
    """Only this qualification process and its actual process descendants."""
    pending=[os.getpid()];seen=set();rows=[]
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        seen.add(pid)
        try:
            rss=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            for task in Path(f'/proc/{pid}/task').iterdir():
                try:pending.extend(map(int,(task/'children').read_text().split()))
                except FileNotFoundError:pass
        except FileNotFoundError:continue
        rows.append(dict(pid=pid,rss_bytes=rss))
    return rows


def main():
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as runtime
    from research.benchmark_rebuild_20261024.american.treeloom_speed import static_preparation_v2 as preparation
    from research.preprocessing_optimization_20261026.integration.hooks import install_tree
    from research.preprocessing_optimization_20261026.tree import process_observations
    install_tree()
    events=[]
    def observe(event,detail):events.append(dict(event=event,**detail))
    preparation.encode_block=partial(process_observations.encode_block,preparation_workers=4,
        tree_chunk_size=128,max_parallel_bytes=8*2**30,max_chunk_bytes=64*2**20,
        preparation_observer=observe)
    plan=json.loads((ROOT/'configs/benchmarks/paper_campaign_resume_20261026_v1.json').read_text())
    task=next(t for t in plan['tasks'] if t['id']=='paper_american_treeloom_fit')
    arguments=task['call']['spec']['arguments']
    registry=runtime.FreshParentRegistry.from_ref(arguments['parent_registry_ref'])
    rt=runtime.FreshTreeLoom(registry)
    parent='argformer_american_train_000';item=arguments['train_inputs'][parent]
    marks=[];last=[None];start=time.monotonic();peaks=dict(parent=0,descendants=0,total=0)
    stopped=threading.Event();failure=[]
    def monitor():
        while not stopped.is_set():
            rows=owned_memory()
            main_rss=sum(row['rss_bytes'] for row in rows if row['pid']==os.getpid())
            child_rss=sum(row['rss_bytes'] for row in rows if row['pid']!=os.getpid())
            peaks['parent']=max(peaks['parent'],main_rss)
            peaks['descendants']=max(peaks['descendants'],child_rss)
            peaks['total']=max(peaks['total'],main_rss+child_rss)
            if main_rss>128*2**30 or child_rss>8*2**30 or main_rss+child_rss>136*2**30:
                failure.append('Complete-parent128GiB plus process/IPC8GiB reserve exceeded')
            stopped.wait(1.)
    watcher=threading.Thread(target=monitor,name='TreeLoom_qualification_memory',daemon=True);watcher.start()
    def check(stage):
        if failure:raise MemoryError(failure[0])
        if time.monotonic()-start>7200:raise TimeoutError('Two-hour full observed-parent qualification bound')
        if stage!=last[0]:
            row=dict(stage=stage,elapsed=time.monotonic()-start,rss_bytes=owned_memory()[0]['rss_bytes'])
            marks.append(row);print(json.dumps(row),flush=True);last[0]=stage
    try:
        state=load_observation(item['observation_ref'],item['focal_registry_ref'],dict(
            native_preparation_ref=item['native_preparation_ref'],class_order=list(runtime.CLASSES),
            expected_haplotypes=1600,reference_counts=[200,200,200],
            input_reference_class_mask=None,input_reference_sample_indices=None))
        loaded=time.monotonic()
        bundle,metadata=rt.prepare_observation(state,split='train',resource_check=check,
            **{k:item[k] for k in runtime.INPUT_REFS},max_nodes=400000000,
            max_holds=100000,max_episode_payload_bytes=8*2**30)
        finished=time.monotonic()
        record=dict(variant='process4',parent=parent,load_seconds=loaded-start,
            prepare_seconds=finished-loaded,total_seconds=finished-start,
            observed_inputs={k:item[k] for k in runtime.INPUT_REFS},bundle=signature(bundle),
            metadata=signature(metadata),marks=marks,preparation_events=events,peak_rss_bytes=peaks,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            truth_opened=False,model_executed=False,full_parent=True)
        check('signature_complete')
        paths=[OUT/name for name in ('kernels.cpp','kernels.py','libstructural.so','observations.py',
            'episodes.py','cut.py','process_observations.py','full_process_parent.py')]
        record['source_refs']=[dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
        reference=ROOT/'research/preprocessing_optimization_20261026/integration/full_reference.json'
        if reference.exists():
            original=json.loads(reference.read_text())
            assert record['bundle']==original['bundle'],'Full bundle hash mismatch'
            assert record['metadata']==original['metadata'],'Full metadata mismatch'
            record['reference_comparison']='bitwise identical complete bundle and metadata'
        else:record['reference_comparison']='reference still running, comparison pending'
        stopped.set();watcher.join()
        record['peak_rss_bytes']=dict(peaks)
        record['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary={key:record[key] for key in ('variant','parent','load_seconds','prepare_seconds',
            'total_seconds','reference_comparison','peak_rss_bytes','max_rss_kib','source_refs',
            'truth_opened','model_executed','full_parent')}
        summary['detail_report_pending']=True
        print(json.dumps(summary,allow_nan=False,default=json_scalar),flush=True)
        write_record_atomic(OUT/'full_process_parent_summary.json',summary)
        target=OUT/'full_process_parent.json'
        write_record_atomic(target,record)
        print(json.dumps(dict(completed=str(target),prepare_seconds=finished-loaded,
            reference_comparison=record['reference_comparison'],peaks=peaks),
            allow_nan=False,default=json_scalar),flush=True)
    finally:
        stopped.set();watcher.join()

if __name__=='__main__':main()
