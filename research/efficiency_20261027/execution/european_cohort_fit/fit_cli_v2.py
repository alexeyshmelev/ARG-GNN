"""Standalone fixed90-update European cohort8 FIT worker for ROOT's guardian.

ROOT alone launches and chooses admission. This worker performs CPU preparation
before a late, exact successful qualification gate and genuine ROOT-scope shared GPU lease.
Optional exact continuation restores all RNG streams and the next parent, within the original90-update budget.
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
from .fit_gate_v2 import ROOT,RUNS,GPU_UUID,need,contained,reference,read,qualification_release,resume_release
from .progress_v2 import snapshot

MODULE='research.efficiency_20261027.execution.european_cohort_fit.fit_cli_v2'
HERE=Path(__file__).parent
SOURCE_MANIFEST_REF=dict(path='research/efficiency_20261027/architecture/european_cohort_train/source_pins_v2.json',sha256='56203f33d6acd884b4ee2221745956ac2592e6b47160ff577607adc6a9b1eaa0')
SOURCE_REVIEW_REF=dict(path='research/efficiency_20261027/fidelity/EUROPEAN_COHORT8_TRAIN_V2_SOURCE_REVIEW.json',sha256='2e044448dce28bd9138efde0f9225a3ae7c75487494c9ecb0f2c0c6352164f37')
GIB=2**30


def verify_wrapper(digest,review_path,review_digest):
    manifest,manifest_ref=read(HERE/'fit_cli_pins_v2.json',digest)
    need(manifest.get('schema')=='European_cohort8_FIT_CLI_sources_v2','Exact wrapper manifest')
    refs=manifest['source_refs']
    paths=[row['path'] for row in refs]
    need(len(set(paths))==len(paths) and SOURCE_MANIFEST_REF in refs and SOURCE_REVIEW_REF in refs
         and str((HERE/'fit_cli_v2.py').relative_to(ROOT)) in paths
         and str((HERE/'fit_gate_v2.py').relative_to(ROOT)) in paths,'Complete unique reviewed wrapper/source bindings')
    for ref in refs:need(reference(ref['path'])==ref,'Reviewed wrapper/source changed: '+ref['path'])
    review,review_ref=read(review_path,review_digest)
    need(review.get('schema')=='European_cohort8_FIT_CLI_review_v2' and review.get('passed') is True
         and review.get('wrapper_manifest_ref')==manifest_ref,'Exact independently passed wrapper review')
    need(review.get('resume_restore_CPU_equivalence_passed') is True,'Independent restore-equivalence source review required')
    return manifest_ref,review_ref


def validate_guardian_records(request,launch,args,argv,pid,start_ticks,parent_pid,parent_ticks):
    need(request.get('command')==launch.get('actual_argv'),'Same visible ROOT request and actual launch')
    command=request['command']
    need(type(command) is list and len(command)>=4,'Explicit visible command vector')
    peers=args.required_peer_directory
    need(type(request.get('peer_directories')) is list and request['peer_directories']==peers
         and len(set(peers))==len(peers),
         'Exact ROOT-selected current peer reservations')
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
    need(args.family_cap_bytes>=512*GIB and args.output_cap_bytes==2*GIB
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
    parser.add_argument('--qualification-path',required=True)
    parser.add_argument('--qualification-sha256',required=True)
    parser.add_argument('--fit-release-path',required=True)
    parser.add_argument('--fit-release-sha256',required=True)
    parser.add_argument('--resume-driver-pins-sha256',required=True)
    parser.add_argument('--resume-checkpoint-path')
    parser.add_argument('--resume-checkpoint-sha256')
    parser.add_argument('--required-peer-directory',action='append',default=[])
    args=parser.parse_args(argv)
    need(args.source_pins_sha256==SOURCE_MANIFEST_REF['sha256'] and args.source_review_sha256==SOURCE_REVIEW_REF['sha256'],
         'Explicit exact European v2 source manifest and source-review SHA256')
    wrapper_ref,wrapper_review_ref=verify_wrapper(args.wrapper_pins_sha256,args.wrapper_review_path,args.wrapper_review_sha256)
    initial_gate=qualification_release(args.qualification_path,args.qualification_sha256,
        args.fit_release_path,args.fit_release_sha256)
    resume_gate=resume_release(args.resume_checkpoint_path,args.resume_checkpoint_sha256)
    output=contained(args.output);directory=output.parent
    need(output.parent.parent==RUNS and output.name=='data' and not output.exists()
         and directory.is_dir() and not directory.is_symlink(),'Fresh data output in an existing ROOT guardian run directory')
    from . import run_guarded_fit_v1 as guardian
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
    # CPU observation preparation precedes actual late physical GPU acquisition.
    del os.environ['ARGGNN_PHYSICAL_GPU']
    os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='1'
    import torch
    need(not torch.cuda.is_initialized(),'No CUDA before late acquisition')
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.backends.cudnn.benchmark=False
    from arggnn.benchmarking.paper_campaign.processes import pss
    from research.efficiency_20261027.architecture.european_cohort_train import driver_v3 as driver
    from research.repairs.benchmark_multigpu_20261022.runtime.queue_gpu_v1 import physical_path
    need(driver.source_manifest_ref()['sha256']==args.resume_driver_pins_sha256,'Exact resume driver source manifest')
    runtime=driver.read_small(driver.ORIGINAL_JOB_REF)['call']['spec']['scientific_spec']
    need(runtime['memory_envelope']['max_rss_bytes']==512*GIB,'Original European worker envelope')
    spec=driver.build_spec(mode='fit',output_directory=str(output),physical_gpu=5,gpu_uuid=GPU_UUID,
        source_review_ref=SOURCE_REVIEW_REF,qualification_ref=initial_gate['qualification_ref'],
        fit_release_ref=initial_gate['fit_release_ref'],
        resume_checkpoint_ref=None if resume_gate is None else resume_gate['resume_checkpoint_ref'])
    driver.validate(spec)
    lease={'physical':None,'logical':None,'gate':None,'acquired':False}
    last_check=[-float('inf')];peak_family=[0];checkpoint_cache={}
    def write(name,value):
        need(output.is_dir() and '/' not in name,'Bounded wrapper metadata in actual driver output')
        data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        need(len(data)<=2*2**20 and guardian.scope_bytes(directory)+len(data)<=args.output_cap_bytes,
             'Exact wrapper metadata fits the ROOT output reservation')
        with (output/name).open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        return reference(output/name)
    def progress(stage):
        if not output.is_dir():return
        value=snapshot(output,checkpoint_cache)
        value.update(stage=str(stage),observed_unix=time.time(),source_manifest_ref=SOURCE_MANIFEST_REF,
            wrapper_manifest_ref=wrapper_ref,actual_ROOT_request_ref=request_ref,actual_ROOT_launch_ref=launch_ref)
        data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        need(len(data)<=2*2**20 and guardian.scope_bytes(directory)+len(data)<=args.output_cap_bytes,
             'Bounded durable progress fits whole-run allowance including temporary replacement')
        temporary=output/'fit_progress.tmp'
        with temporary.open('wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,output/'fit_progress.json')
    def check(stage):
        elapsed=time.monotonic()-start
        need(elapsed<=args.max_seconds,'European fixed90 FIT wall bound at '+str(stage))
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
        progress(stage)
        if lease['acquired']:
            held=os.fstat(lease['physical']);named=physical_path(5).stat()
            need((held.st_dev,held.st_ino)==(named.st_dev,named.st_ino),'Held actual physical GPU5 lease inode unchanged')
    def acquire_gpu0():
        need(not lease['acquired'] and not torch.cuda.is_initialized(),'Single late GPU acquisition before CUDA')
        waiting=time.monotonic();last_print=-float('inf')
        while True:
            check('European_wait_for_qualified_fit_GPU')
            gate=qualification_release(args.qualification_path,args.qualification_sha256,args.fit_release_path,args.fit_release_sha256)
            need(gate==initial_gate,'Exact qualification/release evidence unchanged')
            need(resume_release(args.resume_checkpoint_path,args.resume_checkpoint_sha256)==resume_gate,
                 'Exact prior FIT family and selected checkpoint remain unchanged')
            free=guardian.gpu_free(5)
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
                print(json.dumps(dict(event='European_GPU5_deferred',seconds=elapsed,qualification_reaped=gate is not None,
                    free_GPU_bytes=free,required_free_GPU_bytes=required,CUDA_initialized=False)),flush=True)
                last_print=elapsed
            need(elapsed<args.gpu_wait_seconds,'Qualified FIT free-capacity/physical-lease wait expired')
            time.sleep(5)
        # Recheck after obtaining the actual shared physical inode. No campaign
        # lease record is forged or modified; this is ROOT's opportunistic scope.
        need(qualification_release(args.qualification_path,args.qualification_sha256,args.fit_release_path,args.fit_release_sha256)==gate and guardian.gpu_free(5)>=required,
             'Exact qualification and current capacity remain valid under physical lease')
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
            physical_total_bytes=total,qualification_gate=gate,resume_gate=resume_gate)
        write('root_gpu_lease.json',row)
        logical=os.open(output/'root_gpu_lease.json',os.O_RDWR|os.O_NOFOLLOW)
        lease['logical']=logical
        fcntl.flock(logical,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ.update(ARGGNN_PHYSICAL_GPU='5',ARGGNN_READABLE_GPU_FD=str(logical))
        lease.update(acquired=True,gate=gate)
        write('fit_wrapper_configuration.json',dict(arguments=vars(args),wrapper_manifest_ref=wrapper_ref,
            wrapper_review_ref=wrapper_review_ref,source_manifest_ref=SOURCE_MANIFEST_REF,source_review_ref=SOURCE_REVIEW_REF,
            actual_ROOT_request_ref=request_ref,actual_ROOT_launch_ref=launch_ref,qualification_gate=gate,
            worker_RSS_bytes=512*GIB,CPU_threads=8,GPU_lease=row,resume_gate=resume_gate,
            resume_driver_source_manifest_ref=driver.source_manifest_ref()))
        return row
    try:
        result=driver.run_fit(spec,resource_check=check,acquire_gpu0=acquire_gpu0)
        check('European_fit_wrapper_complete')
        progress('completed_fixed90updates')
        need(lease['acquired'],'Complete fixed90 FIT used actual late GPU lease')
        write('fit_wrapper_completed.json',dict(schema='European_cohort8_FIT_CLI_completion_v2',
            passed=True,completion_ref=result['completion_ref'],checkpoint_ref=result['checkpoint_ref'],
            optimizer_updates=result['optimizer_updates'],completed_epochs=result['completed_epochs'],
            fresh_fit_only=False,resume_supported=True,source_manifest_ref=SOURCE_MANIFEST_REF,
            resumed_from_ref=result['resumed_from_ref'],resumed_after_updates=result['resumed_after_updates'],
            resume_checkpoint_ref=result['resume_checkpoint_ref'],resume_driver_source_manifest_ref=driver.source_manifest_ref(),
            wrapper_manifest_ref=wrapper_ref,wrapper_review_ref=wrapper_review_ref,qualification_gate=lease['gate'],
            measured_peak_session_PSS_bytes=peak_family[0],physical_gpu=5,gpu_uuid=GPU_UUID,
            actual_ROOT_request_ref=request_ref,actual_ROOT_launch_ref=launch_ref,
            own_actual_wait_and_reaping_still_required=True))
        print(json.dumps(dict(event='European_fixed90_fit_completed',completion_ref=result['completion_ref'],checkpoint_ref=result['checkpoint_ref'])),flush=True)
    finally:
        for key in ('logical','physical'):
            if lease[key] is not None:os.close(lease[key]);lease[key]=None
        os.environ.pop('ARGGNN_READABLE_GPU_FD',None)
        os.environ.pop('ARGGNN_PHYSICAL_GPU',None)

if __name__=='__main__':main()
