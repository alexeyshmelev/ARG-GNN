"""Launch the separately reviewed GPU admission continuation once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.argmix.recovery_20261030_gpu_capacity_v1 import continue_v1 as worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    review_ref=r.ref(args.review)
    review=r.read(review_ref)
    assert review['passed'] is True
    continuation_ref=review['continuation_ref']
    assert r.ref(worker.__file__)==review['continuation_source_ref']
    spec,manifest,plan,prefix,binding,minimum_host=worker.validate(continuation_ref)
    assert worker.c.host_available_bytes()>=minimum_host
    uuid,available=worker.c.gpu_capacity(3,spec['minimum_free_gpu_bytes'])
    assert available, 'Leave existing GPU users untouched'
    folder=worker.HERE
    for name in ['root_launch.json','controller.log','started.json']:
        assert not (folder/name).exists(), 'Previous continuation exists; inspect instead of relaunching'
    command=[str(r.ROOT/'.runtime/env/bin/python3.11'),'-B','-s','-u','-m',
             'research.joint_evaluation_20261029.argmix.recovery_20261030_gpu_capacity_v1.continue_v1',
             '--continuation',str(continuation_ref['path'])]
    if not args.apply:
        print(json.dumps({'valid':True,'command':command,'continuation_ref':continuation_ref,'signals_sent':0}))
        return
    with (folder/'controller.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=r.ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    receipt=dict(pid=child.pid,process_start_ticks=ticks,command=command,continuation_ref=continuation_ref,
                 review_ref=review_ref,launcher_ref=r.ref(__file__),started_utc=datetime.now(timezone.utc).isoformat(),
                 worker_signals_sent=0,completed_prefix_stages_repeated=0)
    with (folder/'root_launch.json').open('x') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(receipt),flush=True)
    time.sleep(2)
    assert child.poll() is None, 'Continuation exited; inspect preserved log'


if __name__=='__main__':main()
