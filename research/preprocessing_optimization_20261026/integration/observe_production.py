"""Bounded read-only sampling of the one authorized resumed TreeLoom worker."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/preprocessing_optimization_20261026/integration'
LEAF=ROOT/'experiments/results/paper_american_20261024/treeloom_fit/native/exact_preprocessing_from_update241_optimization_attempt5'
SCOPE=LEAF.parent.parent
LEADER=1036598
TITLE='AI Auto Research Pipeline Test'


def read_json(path):
    if not path.exists():return None
    if path.stat().st_size>4*2**20:raise ValueError('Unexpected observation metadata size')
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return None


def reference(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def process_rows():
    todo=[LEADER]
    seen=set()
    rows=[]
    while todo:
        pid=todo.pop()
        if pid in seen:continue
        seen.add(pid)
        if len(seen)>16:raise ValueError('Unexpected own child count')
        proc=Path('/proc')/str(pid)
        try:
            fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
            if int(fields[3])!=LEADER or proc.joinpath('cwd').resolve()!=ROOT:
                raise ValueError('Own worker session or project directory changed')
            status={line.split(':',1)[0]:line.split(':',1)[1].strip()
                    for line in proc.joinpath('status').read_text().splitlines() if ':' in line}
            command=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
            threads=[]
            for task in proc.joinpath('task').iterdir():
                todo.extend(map(int,task.joinpath('children').read_text().split()))
                threads.append(dict(tid=int(task.name),wait_channel=task.joinpath('wchan').read_text().strip()))
            rows.append(dict(pid=pid,ppid=int(status['PPid']),start_ticks=fields[19],
                process_title=command if command==TITLE else command[:180],
                OS_threads=int(status['Threads']),state=status['State'],thread_wait_states=threads))
        except FileNotFoundError:
            continue
    return rows


def query_gpu():
    commands=[['nvidia-smi','-i','3','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],
        ['nvidia-smi','-i','3','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']]
    outputs=[subprocess.check_output(command,text=True,timeout=10).strip() for command in commands]
    gpu=outputs[0].split(', ')
    apps=[]
    for line in outputs[1].splitlines():
        if not line:continue
        pid,name,memory=line.split(', ',2)
        apps.append(dict(pid=int(pid),name=name,used_MiB=int(memory)))
    return dict(physical_gpu=int(gpu[0]),uuid=gpu[1],utilization_percent=int(gpu[2]),used_MiB=int(gpu[3]),
        compute_contexts=apps,raw_gpu_query=outputs[0],raw_compute_query=outputs[1])


def main():
    start=time.time()
    samples=OUT/'production_observations.jsonl'
    if samples.exists():raise FileExistsError(samples)
    worker_sample=None
    prepared_sample=None
    initial_start=None
    with samples.open('x') as log:
        while time.time()-start<2700:
            rows=process_rows()
            leader=next((row for row in rows if row['pid']==LEADER),None)
            if leader is None:raise RuntimeError('Authorized TreeLoom leader exited before monitored milestone')
            if initial_start is None:initial_start=leader['start_ticks']
            if leader['start_ticks']!=initial_start:raise RuntimeError('Leader identity changed')
            gpu=query_gpu()
            progress_path=LEAF/'progress.jsonl'
            progress=[json.loads(line) for line in progress_path.read_text().splitlines()] if progress_path.exists() else []
            ready=[row for row in progress if row['event']=='compact_parent_ready']
            training=read_json(LEAF/'training_live.json')
            resource=read_json(SCOPE/'resource_progress.json')
            sample=dict(observed_unix=time.time(),processes=rows,gpu3=gpu,latest_progress=progress[-5:],
                compact_ready=ready,training_live=training,resource_progress=resource)
            log.write(json.dumps(sample,sort_keys=True)+'\n')
            log.flush()
            workers=[row for row in rows if row['pid']!=LEADER and row['process_title']==TITLE]
            worker_pids={row['pid'] for row in workers}
            child_contexts=[item for item in gpu['compute_contexts'] if item['pid'] in worker_pids]
            if len(workers)==4 and not child_contexts and worker_sample is None:
                worker_sample=sample
                print(json.dumps(dict(event='four_production_CPU_workers_observed',pids=sorted(worker_pids),
                    OS_thread_counts=[row['OS_threads'] for row in workers],
                    no_worker_GPU3_context=True,observed_unix=sample['observed_unix'])),flush=True)
            if ready and prepared_sample is None:
                prepared_sample=sample
                print(json.dumps(dict(event='real_full_parent_prepared',ready=ready[-1],
                    observed_unix=sample['observed_unix'])),flush=True)
            owned=[item for item in gpu['compute_contexts'] if item['pid']==LEADER]
            if ready and training and owned and owned[0]['used_MiB']>1024 and gpu['utilization_percent']>0 and worker_sample:
                result=dict(status='pass',scope='One real resumed production parent preparation and active GPU training milestone, not benchmark completion or accuracy evaluation',
                    leader_pid=LEADER,leader_start_ticks=initial_start,physical_gpu=3,
                    attempt_leaf=str(LEAF.relative_to(ROOT)),resumed_complete_updates=241,
                    observed_unix=time.time(),first_four_worker_sample=worker_sample,
                    first_prepared_parent_sample=prepared_sample,active_training_sample=sample,
                    worker_thread_interpretation='Observed OS thread counts are reported literally. Frozen initializer enforces one Torch compute thread and at most one thread per native pool. Additional OS threads are not claimed to be extra compute-pool threads.',
                    child_CUDA_scope='No observed preparation worker has a GPU3 compute context in the sampled nvidia-smi query. Other GPUs were not queried.',
                    source_refs=[reference(OUT/'release.json'),reference(OUT/'migration/resource592/amendment_receipt.json'),
                        reference(ROOT/'research/preprocessing_optimization_20261026/tree/process_observations.py'),reference(Path(__file__).resolve())],
                    observation_log_ref=reference(samples),model_or_scoring_changed=False,
                    full_training_complete=False,ancestry_accuracy_claimed=False)
                target=OUT/'production_verification.json'
                with target.open('x') as handle:
                    json.dump(result,handle,indent=2,allow_nan=False)
                    handle.write('\n')
                print(json.dumps(dict(event='production_training_verified',reference=reference(target),
                    utilization_percent=gpu['utilization_percent'],owned_GPU_MiB=owned[0]['used_MiB'])),flush=True)
                return
            time.sleep(45)
    raise TimeoutError('Finite production milestone observation window ended')


if __name__=='__main__':main()
