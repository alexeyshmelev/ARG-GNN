"""Activate only the new parallel queue while the old GPU5 MLM continues."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
from research.repairs.benchmark_multigpu_20261022.runtime import readable_workflow_v2 as q

c=q.c
BASE=c.ROOT/'research/repairs/benchmark_multigpu_20261022/orchestration'
PLAN='configs/benchmarks/executable_workflow_20261022_multigpu_v2.json'
MODULE='research.repairs.benchmark_multigpu_20261022.runtime.readable_workflow_v2'


def main():
    q.need(not (BASE/'launch_receipt_v2.json').exists(),'One launch only, inspect existing receipt')
    ref=c.file_ref(c.contained_path(PLAN))
    plan=q.load_plan(ref)
    directory=c.contained_path(plan['records_directory'])
    directory.mkdir(parents=True,exist_ok=True)
    q.need(not (directory/'state.json').exists(),'Fresh parallel controller required')
    with (c.ROOT/'.runtime/locks/benchmark_recovery_capacity_20261020.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        policy=c.read_json(c.POLICY_PATH)
        q.need(policy['physical_gpus']==[5] and policy['additional_physical_gpus']==[2,3]
            and policy['user_authorized_physical_gpus']==[2,3,5], 'Exact combined user authorization required')
        before=q.save(BASE/'policy_before_parallel_launch_v2.json',policy,exclusive=True)
        policy.update(approved_parallel_workflow_ref=ref,
            parallel_workflow_released_at_utc=datetime.now(timezone.utc).isoformat(),
            parallel_workflow_release_reason='User authorized GPUs 2 and 3. Preserve live GPU 5 training, completed work and original science. Activate operational repairs and independent native parent queues.',
            parallel_workflow_prior_policy_ref=before)
        q.save(c.POLICY_PATH,policy)
        q.load_plan(ref,approved=True)
        command=[str(c.ROOT/'.runtime/env/bin/python'),'-B','-s','-m',MODULE,'--plan-ref',json.dumps(ref,sort_keys=True),'--execute']
        with (directory/'controller.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=c.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        stat=Path(f'/proc/{process.pid}/stat').read_text()
        fields=stat[stat.rfind(')')+2:].split()
        receipt=dict(launched_at_utc=datetime.now(timezone.utc).isoformat(),plan_ref=ref,command=command,
            controller_pid=process.pid,controller_start_ticks=int(fields[19]),authorized_physical_gpus=[2,3,5],
            old_GPU5_worker_preserved=True,unrelated_processes_signalled=False,
            policy_ref=c.file_ref(c.POLICY_PATH),launcher_source_ref=c.file_ref(Path(__file__)))
        q.save(BASE/'launch_receipt_v2.json',receipt,exclusive=True)
        print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__=='__main__':main()
