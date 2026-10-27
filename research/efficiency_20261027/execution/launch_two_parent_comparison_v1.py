"""Visible ROOT-only launch plan; review-only by default, no GPU work in worker."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from .compare_two_parent_preparation_v1 import ROOT,SPEC,verify,need

GUARDIAN='research/efficiency_20261027/root/run_guarded_profile_with_peer.py'
LIVE_PEER='research/efficiency_20261027/root/runs/upward_residency_full_gpu5_v2'


def argv_for(args):
    directory=(ROOT/args.directory).resolve()
    need(directory.is_relative_to(ROOT/'research/efficiency_20261027/root/runs') and not directory.exists(),
         'Fresh ROOT comparison directory')
    peers=list(dict.fromkeys(args.peer_directory))
    need(LIVE_PEER in peers,'Preserve ROOT active full-residency peer growth accounting')
    need(all((ROOT/p).resolve().is_relative_to(ROOT/'research/efficiency_20261027/root/runs') for p in peers),
         'Explicit ROOT peer directories')
    spec=json.loads(SPEC.read_text())
    need(spec['max_actual_family_bytes']==288*2**30,'Reviewed diagnostic family cap')
    command=['.runtime/env/bin/python3.11','-B',GUARDIAN,'--directory',args.directory,
        '--gpu',str(args.gpu),'--host-gib','288','--device-gib','0','--wait-seconds','86400',
        '--run-seconds','12000']
    for peer in peers:command.extend(['--peer-directory',peer])
    command.extend(['--','.runtime/env/bin/python3.11','-B','-m',
        'research.efficiency_20261027.execution.compare_two_parent_preparation_v1',
        '--output',str(directory/'data'),'--source-pins-sha256',args.source_pins_sha256])
    return command


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--gpu',type=int,choices=(3,4,5),default=5)
    parser.add_argument('--peer-directory',action='append',required=True)
    parser.add_argument('--source-pins-sha256',required=True)
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();pins=verify(args.source_pins_sha256)
    need(Path.cwd()==ROOT,'Authorized project root required')
    command=argv_for(args)
    print(json.dumps(dict(mode='launch' if args.launch else 'review_only',argv=command,
        visible_command=shlex.join(command),source_manifest_ref=pins,
        GPU_work=False,ROOT_guardian_owns_exceptional_family_cleanup=True),indent=2),flush=True)
    if args.launch:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH='src:.',PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),check=True)


if __name__=='__main__':main()
