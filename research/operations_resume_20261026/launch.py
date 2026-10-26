"""Launch the reviewed continuation once, with readable argv and isolated logs."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time
from arggnn.benchmarking.paper_campaign import runtime as r

plan_ref = r.read(r.ref('research/operations_resume_20261026/preflight.json'))['plan_ref']
plan = r.read(plan_ref)
records = r.path(plan['records_directory'])
records.mkdir(parents=True, exist_ok=True)
receipt = r.path('research/operations_resume_20261026/launch.json')
if receipt.exists() or (records / 'state.json').exists():
    raise ValueError('Continuation has already been dispatched; inspect existing process and state')
for source in plan['controller_source_refs']:
    if r.ref(source['path']) != source:
        raise ValueError('Controller source changed before launch')
lock = r.path('.runtime/benchmarking/paper_campaign_global.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
fcntl.flock(lock, fcntl.LOCK_UN)
lock.close()
command = [str(r.path('.runtime/env/bin/python')), '-B', '-s', '-u', '-m',
           'arggnn.benchmarking.paper_campaign.scheduler', '--plan', plan_ref['path']]
env = os.environ.copy()
env.update(PYTHONPATH=str(r.ROOT/'src')+':'+str(r.ROOT), PYTHONDONTWRITEBYTECODE='1',
           PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
           OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
log_path = records/'controller.log'
with log_path.open('xb') as log:
    child = subprocess.Popen(command, cwd=r.ROOT, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    ticks = (Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    git = subprocess.run(['git','rev-parse','HEAD'],cwd=r.ROOT,check=True,capture_output=True,text=True).stdout.strip()
    r.save(receipt, dict(schema='paper_campaign_resume_launch_v1',pid=child.pid,
           start_ticks=ticks, command=command, plan_ref=plan_ref,
           launch_unix=time.time(), git_commit=git, physical_gpus=[3,4,5],
           log_path=str(log_path.relative_to(r.ROOT)), previous_successes_reused=32))
print(receipt.read_text())
