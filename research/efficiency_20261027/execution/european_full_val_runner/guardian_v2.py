"""Narrow CPU guardian using existing ROOT admission and reviewed owned-family tools.

Only ROOT invokes this entrypoint. No GPU is acquired. One fixed aggregate
attempt is allowed;all data,logs and failed evidence remain charged and retained.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity,current_memory,campaign_growth,peer_growth,write
from .aggregate_guard_v2 import measure
from .native_supervisor_v2 import process_table,owned_family,signal_owned,reap_adopted
from .contracts_v2 import ROOT,AGGREGATE,MODULE,HOST_CAP,RESERVATION,OPERATING,MAX_SECONDS,GIB,need,ref,read,verify_sources,validate_release,retained_attempts,RETAINED_SHORT_RESERVATION,PROJECT_CAP,PROJECT_ADMISSION_CAP,owner_authorization,ROOT_STATUS_MAX_AGE_SECONDS,GROWTH_INTERVAL_SECONDS

HOST_MARGIN=64*GIB
OUTER_LOG_CAP=8*2**20


def capture_family(leader,known,supervisor_pid,table=None):
    table=process_table() if table is None else table
    family=owned_family(table,leader,known)
    # This guardian is a subreaper and starts exactly one scientific child.
    # Newly adopted descendants remain owned after a native session or worker
    # exits. The exact generation is captured before any pidfd signal.
    for pid,row in table.items():
        if row['ppid']==supervisor_pid and row['start']>=leader['start']:
            family[pid]=row
    family=owned_family(table,leader,{**known,**family})
    known.update(family)
    return family


def wait_owned(child,leader,known,*,terminate):
    waits=[];started=time.monotonic();sent_term=False
    while True:
        family=capture_family(leader,known,os.getpid())
        waits.extend(reap_adopted(family,child.pid))
        family=capture_family(leader,known,os.getpid())
        if not family:
            code=child.wait(timeout=5)
            return dict(actual_returncode=code,actual_wait=True,owned_group_absent=True,owned_family_absent=True,
                pid=child.pid,process_start_ticks=str(leader['start']),adopted_waits=waits,
                tracked_family=[{key:row[key] for key in ('pid','start','ppid','session','pgrp')} for row in known.values()])
        if not terminate:return None
        if not sent_term:signal_owned(family,signal.SIGTERM);sent_term=True
        if time.monotonic()-started>=10:signal_owned(family,signal.SIGKILL)
        need(time.monotonic()-started<20,'Exact owned recursive family failed to drain')
        child.poll();time.sleep(.1)


def combined_storage(retained_bytes):
    _,current_retained=retained_attempts()
    need(current_retained==retained_bytes,'Retained failed short scope bytes changed during full validation')
    current=measure(AGGREGATE)
    return dict(current,retained_short_charged_bytes=retained_bytes,
                retained_short_reservation_in_other_bytes=RETAINED_SHORT_RESERVATION)



def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-pins-sha256',required=True)
    parser.add_argument('--source-review-path',required=True)
    parser.add_argument('--source-review-sha256',required=True)
    parser.add_argument('--execution-release-path',required=True)
    parser.add_argument('--execution-release-sha256',required=True)
    parser.add_argument('--peer-directory',action='append',default=[])
    args=parser.parse_args(argv)
    need(Path.cwd()==ROOT,'Explicit project root')
    review_ref=dict(path=args.source_review_path,sha256=args.source_review_sha256)
    manifest_ref,_=verify_sources(args.source_pins_sha256,review_ref)
    release_ref=dict(path=args.execution_release_path,sha256=args.execution_release_sha256)
    release=validate_release(read(release_ref),manifest_ref,review_ref,args.peer_directory)
    retained_ref,retained_bytes=retained_attempts()
    other=release.get('other_project_reserved_bytes')
    need(type(other) is int and other>=0 and release.get('reservation_scope_excluded_from_other_bytes') is True,
         'ROOT exact other project commitments excluding this full validation reservation')
    peer_output=0
    for peer in args.peer_directory:
        request=read(ref(ROOT/peer/'request.json'));amount=request.get('output_cap_bytes')
        need(type(amount) is int and amount>=0,'Every explicit ROOT peer declares output allowance')
        peer_output+=amount
    need(other>=peer_output+RETAINED_SHORT_RESERVATION,
         'Other project commitments cover peers plus the retained short16GiB allowance')
    need(not AGGREGATE.exists(),'One fresh fixed aggregate attempt only;failed evidence is never retried or removed')
    AGGREGATE.mkdir()
    started=time.monotonic();created=time.time();deadline=created+MAX_SECONDS
    worker_args=['--source-pins-sha256',args.source_pins_sha256,'--source-review-path',args.source_review_path,
        '--source-review-sha256',args.source_review_sha256,'--execution-release-path',args.execution_release_path,
        '--execution-release-sha256',args.execution_release_sha256]
    command=['.runtime/env/bin/python3.11','-B','-m',MODULE,*worker_args]
    request=dict(schema='European_full_VAL_ROOT_request_v1',command=command,host_cap_bytes=HOST_CAP,
        output_cap_bytes=RESERVATION,operating_stop_bytes=OPERATING,max_seconds=MAX_SECONDS,CUDA_VISIBLE_DEVICES='',
        aggregate_directory=str(AGGREGATE.relative_to(ROOT)),peer_directories=args.peer_directory,
        other_project_reserved_bytes=other,retained_attempts_ref=retained_ref,retained_charged_bytes=retained_bytes,
        ROOT_status_max_age_seconds=ROOT_STATUS_MAX_AGE_SECONDS,growth_refresh_interval_seconds=GROWTH_INTERVAL_SECONDS,
        supervisor_pid=os.getpid(),supervisor_start_ticks=identity(os.getpid()),
        source_manifest_ref=manifest_ref,independent_source_review_ref=review_ref,execution_release_ref=release_ref,
        created_unix=created,deadline_unix=deadline)
    write(AGGREGATE/'request.json',request)
    def cancelled(signum,frame):raise KeyboardInterrupt('ROOT fullVAL guardian received signal '+str(signum))
    signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled)
    child=leader=None;known={};log=None;peak=0;last_growth=-float('inf');growth=0;details=[]
    peak_admission_seconds=0.;last_status_unix=None;peak_status_gap_seconds=0.
    def admission():
        nonlocal growth,details,last_growth,peak_admission_seconds
        need(ref(release_ref['path'])==release_ref,'ROOT execution release changed')
        need(not (AGGREGATE/'execution_hold').exists(),'ROOT execution hold active')
        need(time.monotonic()-started<MAX_SECONDS,'Original eight-day total admission+run wall bound')
        scan_started=time.monotonic();scan_started_unix=time.time()
        state,growth,details=campaign_growth()
        for peer in args.peer_directory:
            amount,detail=peer_growth(peer);growth+=amount;details.append(detail)
        committed=state['project_reserved_bytes']+other+RESERVATION
        need(committed<=PROJECT_ADMISSION_CAP,
             'ROOT campaign+other+full validation exceeds owner-authorized project admission ceiling')
        need(time.monotonic()-started<MAX_SECONDS and not (AGGREGATE/'execution_hold').exists(),
             'Original eight-day wall or ROOT hold after slow admission scan')
        last_growth=time.monotonic();scan_seconds=last_growth-scan_started
        peak_admission_seconds=max(peak_admission_seconds,scan_seconds)
        return dict(admission_started_unix=scan_started_unix,admission_finished_unix=time.time(),
             admission_seconds=scan_seconds,peak_admission_seconds=peak_admission_seconds,
             growth_refresh_interval_seconds=GROWTH_INTERVAL_SECONDS,ROOT_status_max_age_seconds=ROOT_STATUS_MAX_AGE_SECONDS,
             retained_charged_bytes=retained_bytes,campaign_project_reserved_bytes=state['project_reserved_bytes'],other_project_reserved_bytes=other,
            own_aggregate_reservation_bytes=RESERVATION,total_committed_bytes=committed,
            project_admission_ceiling_bytes=PROJECT_ADMISSION_CAP,global_project_ceiling_bytes=PROJECT_CAP,
             owner_storage_authorization_ref=owner_authorization())
    try:
        while True:
            budget=admission();available=current_memory();required=HOST_CAP+growth+HOST_MARGIN
            status=dict(observed_unix=time.time(),host_available_bytes=available,required_host_bytes=required,
                live_growth_bytes=growth,workers=details,**budget)
            if available>=required:
                write(AGGREGATE/'status.json',dict(status='admitted',**status));last_status_unix=status['observed_unix'];break
            write(AGGREGATE/'status.json',dict(status='waiting',reason='Existing campaign/peer growth and64GiB margin',**status))
            time.sleep(min(GROWTH_INTERVAL_SECONDS,max(0,deadline-time.time())))
        libc=ctypes.CDLL(None,use_errno=True)
        need(libc.prctl(36,1,0,0,0)==0,'Existing Linux subreaper process-ownership mechanism')
        environment=dict(os.environ,PYTHONPATH='src:.',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',PATH='/usr/bin:/bin')
        for key in ('ARGGNN_PHYSICAL_GPU','ARGGNN_READABLE_GPU_FD','ARGGNN_MAX_CUDA_RESERVED_BYTES','ARGGNN_GPU_MEMORY_POLICY'):
            environment.pop(key,None)
        log=(AGGREGATE/'worker.stdout.log').open('xb')
        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        leader=process_table().get(child.pid)
        need(leader is not None and leader['session']==leader['pgrp']==child.pid,'Exact newly owned worker generation')
        known={child.pid:leader}
        write(AGGREGATE/'launch.json',dict(pid=child.pid,process_start_ticks=str(leader['start']),actual_argv=command,
            admitted_snapshot=status,started_unix=time.time()))
        while child.poll() is None:
            family=capture_family(leader,known,os.getpid());table=process_table()
            rss=table.get(os.getpid(),{}).get('rss_bytes',HOST_CAP)+sum(row['rss_bytes'] for row in family.values() if row['state']!='Z')
            peak=max(peak,rss);storage=combined_storage(retained_bytes)
            need(rss<HOST_CAP,'Complete recursive owned family128GiB RSS ceiling')
            need(storage['charged_bytes']<OPERATING,'Full validation aggregate operating stop')
            need((AGGREGATE/'worker.stdout.log').stat().st_size<OUTER_LOG_CAP,'Finite captured outer log allowance')
            need(current_memory()>=HOST_MARGIN,'Host memory pressure')
            need(time.monotonic()-started<MAX_SECONDS and not (AGGREGATE/'execution_hold').exists(),'Original eight-day wall or ROOT hold')
            if time.monotonic()-last_growth>=GROWTH_INTERVAL_SECONDS:
                budget=admission()
                need(current_memory()+rss>=HOST_CAP+growth+HOST_MARGIN,'Campaign/peer growth needs capacity')
            observed_unix=time.time();status_gap=observed_unix-last_status_unix
            peak_status_gap_seconds=max(peak_status_gap_seconds,status_gap)
            write(AGGREGATE/'status.json',dict(status='running',observed_unix=observed_unix,pid=child.pid,
                status_gap_seconds=status_gap,peak_status_gap_seconds=peak_status_gap_seconds,
                process_start_ticks=str(leader['start']),elapsed_seconds=time.monotonic()-started,
                recursive_family_RSS_bytes=rss,peak_sampled_family_RSS_bytes=peak,storage=storage,**budget))
            last_status_unix=observed_unix
            time.sleep(1)
        terminal=wait_owned(child,leader,known,terminate=False)
        need(terminal is not None,'Worker exited with remaining owned recursive family')
        terminal.update(finished_unix=time.time(),peak_sampled_family_RSS_bytes=peak)
        write(AGGREGATE/'actual_wait.json',terminal)
        need(terminal['actual_returncode']==0,'Observed-only VAL worker failed')
        write(AGGREGATE/'status.json',dict(status='finished',actual_returncode=0,observed_unix=time.time()))
    except BaseException as error:
        if child is not None:
            if leader is None:
                child.kill();code=child.wait()
                write(AGGREGATE/'actual_wait.json',dict(actual_wait=True,actual_returncode=code,
                    identity_capture_failed=True,owned_family_absence_not_established=True))
            else:
                terminal=wait_owned(child,leader,known,terminate=True)
                terminal.update(finished_unix=time.time(),peak_sampled_family_RSS_bytes=peak)
                if not (AGGREGATE/'actual_wait.json').exists():write(AGGREGATE/'actual_wait.json',terminal)
        write(AGGREGATE/'failure.json',dict(error=repr(error),observed_unix=time.time(),failed_evidence_retained=True))
        raise
    finally:
        if log is not None:log.close()

if __name__=='__main__':main()
