"""Standalone full European cohort8 qualification worker for ROOT's guardian.

ROOT alone launches and chooses admission. This worker performs CPU preparation
before a late, exact predecessor gate and genuine ROOT-scope shared GPU lease.
It does not edit campaign records, launch subprocess workers, or run a fit.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from .qualification_gate_v1 import ROOT,RUNS,PREDECESSOR,GPU_UUID,need,contained,reference,read,predecessor_identity,predecessor_release

MODULE='research.efficiency_20261027.architecture.european_cohort_train.qualify_cli_v1'
HERE=Path(__file__).parent
SOURCE_MANIFEST_REF=dict(path='research/efficiency_20261027/architecture/european_cohort_train/source_pins_v2.json',sha256='56203f33d6acd884b4ee2221745956ac2592e6b47160ff577607adc6a9b1eaa0')
SOURCE_REVIEW_REF=dict(path='research/efficiency_20261027/fidelity/EUROPEAN_COHORT8_TRAIN_V2_SOURCE_REVIEW.json',sha256='2e044448dce28bd9138efde0f9225a3ae7c75487494c9ecb0f2c0c6352164f37')
GIB=2**30


def verify_wrapper(digest,review_path,review_digest):
    manifest,manifest_ref=read(HERE/'qualification_cli_pins_v1.json',digest)
    need(manifest.get('schema')=='European_cohort8_qualification_CLI_sources_v1','Exact wrapper manifest')
    refs=manifest['source_refs']
    paths=[row['path'] for row in refs]
    need(len(set(paths))==len(paths) and SOURCE_MANIFEST_REF in refs and SOURCE_REVIEW_REF in refs
         and str((HERE/'qualify_cli_v1.py').relative_to(ROOT)) in paths
         and str((HERE/'qualification_gate_v1.py').relative_to(ROOT)) in paths,'Complete unique reviewed wrapper/source bindings')
    for ref in refs:need(reference(ref['path'])==ref,'Reviewed wrapper/source changed: '+ref['path'])
    review,review_ref=read(review_path,review_digest)
    need(review.get('schema')=='European_cohort8_qualification_CLI_review_v1' and review.get('passed') is True
         and review.get('wrapper_manifest_ref')==manifest_ref,'Exact independently passed wrapper review')
    return manifest_ref,review_ref


def validate_guardian_records(request,launch,args,argv,pid,start_ticks,parent_pid,parent_ticks):
    need(request.get('command')==launch.get('actual_argv'),'Same visible ROOT request and actual launch')
    command=request['command']
    need(type(command) is list and len(command)>=4,'Explicit visible command vector')
    peers=args.required_peer_directory
    need(type(request.get('peer_directories')) is list and request['peer_directories']==peers
         and len(set(peers))==len(peers) and str(PREDECESSOR.relative_to(ROOT)) in peers,
         'Exact ROOT-selected peer reservations including the full-residency predecessor')
    for peer in peers:need(contained(peer).parent==RUNS,'Every peer is an exact ROOT run directory')
    need((ROOT/command[0]).resolve()==(ROOT/'.runtime/env/bin/python3.11').resolve()
         and command[1:]==['-B','-m',MODULE,*argv],'Exact standalone project-Python command')
    need(launch.get('pid')==pid and launch.get('process_start_ticks')==start_ticks
         and request.get('supervisor_pid')==parent_pid and request.get('supervisor_start_ticks')==parent_ticks,
         'Exact live ROOT guardian and scientific child identities')
    need(request.get('physical_gpu')==5 and request.get('gpu_uuid')==GPU_UUID
         and request.get('host_cap_bytes')==args.family_cap_bytes
         and request.get('device_budget_bytes')==args.device_soft_bytes
         and request.get('output_cap_bytes')==args.output_cap_bytes,
         'CLI resource bounds must equal ROOT guardian admitted bounds')
    need(args.family_cap_bytes>=512*GIB and args.output_cap_bytes>=64*2**20
         and 0<args.device_soft_bytes and 0<args.max_seconds<=864000
         and args.max_seconds>=args.gpu_wait_seconds>0,'Original512GiB worker and finite ROOT family/device/wall/storage bounds')


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-pins-sha256',required=True)
    parser.add_argument('--source-review-sha256',required=True)
    parser.add_argument('--wrapper-pins-sha256',required=True)
    parser.add_argument('--wrapper-review-path',required=True)
    parser.add_argument('--wrapper-review-sha256',required=True)
    parser.add_argument('--family-cap-bytes',type=int,required=True)
    parser.add_argument('--device-soft-bytes',type=int,required=True)
    parser.add_argument('--output-cap-bytes',type=int,required=True)
    parser.add_argument('--max-seconds',type=int,required=True)
    parser.add_argument('--gpu-wait-seconds',type=int,required=True)
    parser.add_argument('--predecessor-completed-sha256')
    parser.add_argument('--required-peer-directory',action='append',default=[])
    args=parser.parse_args(argv)
    need(args.source_pins_sha256==SOURCE_MANIFEST_REF['sha256'] and args.source_review_sha256==SOURCE_REVIEW_REF['sha256'],
         'Explicit exact European v2 source manifest and source-review SHA256')
    wrapper_ref,wrapper_review_ref=verify_wrapper(args.wrapper_pins_sha256,args.wrapper_review_path,args.wrapper_review_sha256)
    predecessor_identity()
    output=contained(args.output);directory=output.parent
    need(output.parent.parent==RUNS and output.name=='data' and not output.exists()
         and directory.is_dir() and not directory.is_symlink(),'Fresh data output in an existing ROOT guardian run directory')
    from research.efficiency_20261027.root import run_guarded_profile_with_peer as guardian
    start=time.monotonic()
    while not (directory/'launch.json').exists():
        need(time.monotonic()-start<60,'ROOT launch receipt did not appear')
        time.sleep(.1)
    request,request_ref=read(directory/'request.json');launch,launch_ref=read(directory/'launch.json')
    pid,parent=os.getpid(),os.getppid()
    ticks,parent_ticks=guardian.identity(pid),guardian.identity(parent)
    validate_guardian_records(request,launch,args,argv,pid,ticks,parent,parent_ticks)
    need(os.getsid(pid)==os.getpgid(pid)==pid,'Exact ROOT-owned process session')
    need(os.environ.get('ARGGNN_PHYSICAL_GPU')=='5' and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID
         and os.environ.get('ARGGNN_GPU_MEMORY_POLICY')=='full_device'
         and 'ARGGNN_READABLE_GPU_FD' not in os.environ and 'ARGGNN_MAX_CUDA_RESERVED_BYTES' not in os.environ,
         'ROOT full-device environment before explicit deferred handoff')
    # The guardian remains unchanged. This child now enters the reviewed CPU-only
    # preparation contract, and restores acquired identity only in acquire_gpu0.
    del os.environ['ARGGNN_PHYSICAL_GPU']
    os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='1'
    import torch
    need(not torch.cuda.is_initialized(),'No CUDA before late acquisition')
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.backends.cudnn.benchmark=False
    from arggnn.benchmarking.paper_campaign.processes import pss
    from . import driver_v2 as driver
    from research.repairs.benchmark_multigpu_20261022.runtime.queue_gpu_v1 import physical_path
    runtime=driver.read_small(driver.ORIGINAL_JOB_REF)['call']['spec']['scientific_spec']
    need(runtime['memory_envelope']['max_rss_bytes']==512*GIB,'Original European worker envelope')
    spec=driver.build_spec(mode='qualification',output_directory=str(output),physical_gpu=5,gpu_uuid=GPU_UUID,
                           source_review_ref=SOURCE_REVIEW_REF)
    driver.validate(spec)
    lease={'physical':None,'logical':None,'gate':None,'acquired':False}
    last_check=[-float('inf')];peak_family=[0]
    def write(name,value):
        need(output.is_dir() and '/' not in name,'Bounded wrapper metadata in actual driver output')
        data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        need(len(data)<=2*2**20 and guardian.scope_bytes(directory)+len(data)<=args.output_cap_bytes,
             'Exact wrapper metadata fits the ROOT output reservation')
        with (output/name).open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        return reference(output/name)
    def check(stage):
        elapsed=time.monotonic()-start
        need(elapsed<=args.max_seconds,'European qualification wall bound at '+str(stage))
        need(driver.memory._current_rss_bytes()<=512*GIB,'Original512GiB main RSS ceiling')
        if torch.cuda.is_initialized():
            need(lease['acquired'] and torch.cuda.memory_reserved(0)<=args.device_soft_bytes,
                 'Late acquired CUDA and finite ROOT-selected soft reserved ceiling')
        else:need(not lease['acquired'] or lease['logical'] is not None,'Acquired lease identity')
        if elapsed-last_check[0]<5:return
        last_check[0]=elapsed
        need(os.getppid()==parent and guardian.identity(parent)==parent_ticks,'Actual ROOT guardian remains alive')
        need(reference(directory/'request.json')==request_ref and reference(directory/'launch.json')==launch_ref,
             'Exact ROOT admission identity unchanged')
        value=pss(pid);peak_family[0]=max(peak_family[0],value)
        need(value<=args.family_cap_bytes,'Complete owned-session PSS ceiling')
        need(guardian.scope_bytes(directory)<=args.output_cap_bytes,'Whole run output ceiling')
        if lease['acquired']:
            held=os.fstat(lease['physical']);named=physical_path(5).stat()
            need((held.st_dev,held.st_ino)==(named.st_dev,named.st_ino),'Held actual physical GPU5 lease inode unchanged')
    def acquire_gpu0():
        need(not lease['acquired'] and not torch.cuda.is_initialized(),'Single late GPU acquisition before CUDA')
        waiting=time.monotonic();last_print=-float('inf')
        while True:
            check('European_wait_for_exact_predecessor')
            gate=predecessor_release(args.predecessor_completed_sha256)
            free=guardian.gpu_free(5) if gate is not None else None
            # The historical AP full-device job requested64GiB free. The ROOT
            # soft ceiling is a separate explicit operational budget, not an
            # American two-bank estimate or a measured European peak proof.
            required=max(args.device_soft_bytes+2*GIB,64*GIB)
            if gate is not None and free>=required:
                descriptor=os.open(physical_path(5),os.O_RDWR|os.O_NOFOLLOW)
                try:fcntl.flock(descriptor,fcntl.LOCK_SH|fcntl.LOCK_NB)
                except BlockingIOError:os.close(descriptor)
                else:
                    lease['physical']=descriptor
                    need(stat.S_ISREG(os.fstat(descriptor).st_mode),'Ordinary existing physical GPU5 lease inode')
                    break
            elapsed=time.monotonic()-waiting
            if elapsed-last_print>=30:
                print(json.dumps(dict(event='European_GPU5_deferred',seconds=elapsed,predecessor_reaped=gate is not None,
                    free_GPU_bytes=free,required_free_GPU_bytes=required,CUDA_initialized=False)),flush=True)
                last_print=elapsed
            need(elapsed<args.gpu_wait_seconds,'Exact predecessor/free-capacity/physical-lease wait expired')
            time.sleep(5)
        # Recheck after obtaining the actual shared physical inode. No campaign
        # lease record is forged or modified; this is ROOT's opportunistic scope.
        need(predecessor_release(args.predecessor_completed_sha256)==gate and guardian.gpu_free(5)>=required,
             'Exact predecessor and current capacity remain valid under physical lease')
        raw=subprocess.check_output(['nvidia-smi','--id=5','--query-gpu=uuid,memory.total',
            '--format=csv,noheader,nounits'],text=True,timeout=15)
        uuid,mib=[part.strip() for part in raw.strip().split(',')]
        total=int(mib)*2**20
        need(uuid==GPU_UUID and args.device_soft_bytes+2*GIB<=total,'ROOT GPU soft ceiling plus margin fits actual GPU5')
        row=dict(schema='European_ROOT_opportunistic_GPU_lease_v1',physical_gpu=5,gpu_uuid=GPU_UUID,
            admission_kind='ROOT_guardian_shared_physical_lease_with_finite_soft_limit',physical_lease='shared',
            campaign_reservation_created=False,request_ref=request_ref,launch_ref=launch_ref,
            source_manifest_ref=SOURCE_MANIFEST_REF,wrapper_manifest_ref=wrapper_ref,root_pid=pid,root_start_ticks=ticks,
            supervisor_pid=parent,supervisor_start_ticks=parent_ticks,physical_fd=lease['physical'],
            device_soft_bytes=args.device_soft_bytes,minimum_free_bytes=required,observed_free_bytes=free,
            physical_total_bytes=total,predecessor=gate)
        write('root_gpu_lease.json',row)
        logical=os.open(output/'root_gpu_lease.json',os.O_RDWR|os.O_NOFOLLOW)
        lease['logical']=logical
        fcntl.flock(logical,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ.update(ARGGNN_PHYSICAL_GPU='5',ARGGNN_READABLE_GPU_FD=str(logical))
        lease.update(acquired=True,gate=gate)
        write('qualification_wrapper_configuration.json',dict(arguments=vars(args),wrapper_manifest_ref=wrapper_ref,
            wrapper_review_ref=wrapper_review_ref,source_manifest_ref=SOURCE_MANIFEST_REF,source_review_ref=SOURCE_REVIEW_REF,
            actual_ROOT_request_ref=request_ref,actual_ROOT_launch_ref=launch_ref,predecessor=gate,
            worker_RSS_bytes=512*GIB,CPU_threads=8,GPU_lease=row))
        return row
    try:
        result=driver.run_qualification(spec,resource_check=check,acquire_gpu0=acquire_gpu0)
        check('European_qualification_wrapper_complete')
        need(lease['acquired'],'Complete qualification used actual late GPU lease')
        write('qualification_wrapper_completed.json',dict(schema='European_cohort8_qualification_CLI_completion_v1',
            passed=True,qualification_ref=result['completion_ref'],source_manifest_ref=SOURCE_MANIFEST_REF,
            wrapper_manifest_ref=wrapper_ref,wrapper_review_ref=wrapper_review_ref,predecessor=lease['gate'],
            measured_peak_session_PSS_bytes=peak_family[0],physical_gpu=5,gpu_uuid=GPU_UUID,
            actual_ROOT_request_ref=request_ref,actual_ROOT_launch_ref=launch_ref,
            own_actual_wait_and_reaping_still_required=True))
        print(json.dumps(dict(event='European_full_parent_qualification_completed',qualification_ref=result['completion_ref'])),flush=True)
    finally:
        for key in ('logical','physical'):
            if lease[key] is not None:os.close(lease[key]);lease[key]=None
        os.environ.pop('ARGGNN_READABLE_GPU_FD',None)
        os.environ.pop('ARGGNN_PHYSICAL_GPU',None)

if __name__=='__main__':main()
