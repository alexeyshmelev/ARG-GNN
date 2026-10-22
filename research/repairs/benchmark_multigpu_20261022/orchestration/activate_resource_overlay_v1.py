"""Expand controller RAM reservations without modifying or stopping existing workers."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
from research.repairs.benchmark_multigpu_20261022.expansion import prepare_overlay_v1 as prepare
from research.repairs.benchmark_multigpu_20261022.expansion import controller_overlay_v1 as overlay

q=overlay.queue
c=overlay.c
BASE=c.ROOT/'research/repairs/benchmark_multigpu_20261022'
PLAN='configs/benchmarks/executable_workflow_20261022_multigpu_v2.json'
RECORDS='.runtime/benchmarking/workflow_20261022_parallel_resource_overlay_v1'
MODULE='research.repairs.benchmark_multigpu_20261022.expansion.controller_overlay_v1'


def main():
    receipt_path=BASE/'orchestration/resource_overlay_launch_v1.json'
    q.need(not receipt_path.exists(),'One actual launch only')
    ref=c.file_ref(c.contained_path(PLAN))
    base=q.load_plan(ref,approved=True)
    with (c.ROOT/'.runtime/locks/benchmark_recovery_capacity_20261020.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=c.read_json(c.ROOT/base['records_directory']/'state.json')
        active=sorted(key for key,entry in state['tasks'].items() if entry['status']=='running')
        q.need(bool(active) and set(active)<={'AM_treeloom_fit','C4__ancestralpaths_european_train_000__release_archive'},'Only the exact known active workers may be inherited')
        policy=c.read_json(c.POLICY_PATH)
        q.need(policy['approved_parallel_workflow_ref']==ref and policy['physical_gpus']==[5]
            and policy['additional_physical_gpus']==[2,3], 'Preserved canonical authorities required')
        before=q.save(BASE/'orchestration/policy_before_resource_overlay_v1.json',policy,exclusive=True)
        policy.update(parallel_workflow_drain_ref=ref,
            parallel_workflow_drain_reason='Preserve original live workers and move future dispatch to the RAM-expanded controller view')
        q.save(c.POLICY_PATH,policy)
        config_ref=prepare.prepare(ref,str(BASE/'expansion/actual_v1'),RECORDS,active)
        config,base,snapshot=overlay.check_config(config_ref)
        view=overlay.plan_view(config,base,snapshot)
        initial=q.support.initial_state(view,ref)
        projection=q.storage_projection(view,initial)
        q.need(projection<980000000000,'Original980GB ceiling remains mandatory')
        q.need(view['max_parallel_rss_bytes']+view['host_ram_margin_bytes']<=overlay.memtotal_bytes(),'Physical RAM margin required')
        policy=c.read_json(c.POLICY_PATH)
        policy.update(approved_parallel_resource_overlay_ref=config_ref,
            parallel_resource_overlay_released_at_utc=datetime.now(timezone.utc).isoformat(),
            parallel_resource_overlay_max_parallel_rss_bytes=view['max_parallel_rss_bytes'])
        q.save(c.POLICY_PATH,policy)
        directory=c.contained_path(RECORDS);directory.mkdir(parents=True,exist_ok=True)
        q.need(not (directory/'state.json').exists(),'Fresh resource controller state required')
        command=[str(c.ROOT/'.runtime/env/bin/python'),'-B','-s','-m',MODULE,'--config-ref',json.dumps(config_ref,sort_keys=True),'--execute']
        with (directory/'controller.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=c.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        stat=Path(f'/proc/{process.pid}/stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        receipt=dict(launched_at_utc=datetime.now(timezone.utc).isoformat(),controller_pid=process.pid,
            controller_start_ticks=int(fields[19]),config_ref=config_ref,base_plan_ref=ref,command=command,
            preserved_active_task_ids=active,max_parallel_rss_bytes=view['max_parallel_rss_bytes'],
            storage_projection_bytes=projection,storage_ceiling_bytes=980000000000,
            original_workers_unchanged=True,processes_signalled=False,
            policy_before_ref=before,launcher_source_ref=c.file_ref(Path(__file__)))
        q.save(receipt_path,receipt,exclusive=True)
        print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__=='__main__':main()
