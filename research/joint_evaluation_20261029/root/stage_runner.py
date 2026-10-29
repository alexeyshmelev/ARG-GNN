"""Plaintext argv execution with genuine waits and immutable stage receipts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from arggnn.benchmarking.paper_campaign import runtime as r, processes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    job_ref=r.ref(args.job)
    job=r.read(job_ref)
    for source in job['source_refs']:
        if r.ref(source['path']) != source:
            raise ValueError('Source changed before stage launch')
    folder=r.path(job['records_directory'])
    if not folder.is_relative_to(r.ROOT/'research/joint_evaluation_20261029/root'):
        raise ValueError('Bounded root record scope required')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'stage.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'started.json').exists() or (folder/'actual_wait.json').exists():
            raise ValueError('No silent replay of a previous stage')
        command=job['argv']
        if not isinstance(command,list) or command[0]!=str(r.ROOT/'.runtime/env/bin/python3.11') or any(x in command for x in ('-c','-EncodedCommand')):
            raise ValueError('Visible project Python module argv required')
        env=os.environ.copy()
        env.update(job['environment'])
        with (folder/'worker.log').open('xb') as log:
            worker=subprocess.Popen(command,cwd=r.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start_ticks=(Path('/proc')/str(worker.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
            row=dict(job_ref=job_ref,pid=worker.pid,process_start_ticks=start_ticks,started_unix=time.time(),command=command)
            r.save(folder/'started.json',row)
            code=worker.wait()
            while not processes.drain(worker):
                time.sleep(1)
        products=[]
        if code==0:
            for filename in job['products']:
                products.append(r.ref(filename))
        row.update(actual_wait=True,actual_returncode=code,owned_process_group_absent=True,finished_unix=time.time(),products=products)
        r.save(folder/'actual_wait.json',row)
        if code:
            raise SystemExit(code)


if __name__=='__main__':main()
