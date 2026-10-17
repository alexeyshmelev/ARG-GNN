"""Actual finite pidfd supervisor for frozen bounded TreeLoom; inactive without exact ROOT input."""
from pathlib import Path
import argparse,hashlib,json,os,select,signal,subprocess,sys,time
from . import contract_v1 as c
from . import borrowed_gpu0_v1 as borrowed
PIDFD={'path':'research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py','sha256':'61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca'}
OWNED={'path':'research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py','sha256':'93d71acc40f1037d0b37acb2ca0859300ea4bd9741409e9bd00ca21bf100d633'}
SOURCE=c.HERE+'/parent_v1.py'
OWNED_ROSTER_CAP=64
def put(cap,name,value):
    directory=c.local(cap['ROOT_record_directory']);c.need(directory.is_dir(),'Actual admitted own AUX directory')
    items=list(directory.iterdir());c.need(len(items)<64 and all(p.is_file()and not p.is_symlink()for p in items),'Flat own compact records')
    data=c.encoded(value)+b'\n';charged=4096+sum(p.stat().st_size+4096 for p in items)
    c.need(len(data)<=65536 and charged+len(data)+4096<=2*2**20,'Own2MiB parent/AUX metadata bound')
    path=directory/name;fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as f:c.need(f.write(data)==len(data),'Complete parent metadata');f.flush();os.fsync(f.fileno())
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path.relative_to(c.ROOT)),sha256=hashlib.sha256(data).hexdigest())
def execute_v1(record):
    cap,_,_,storage=c.validate_cap_v1(record);os.sched_setaffinity(0,cap["cpu_affinity"]);g=c.guard_v1(cap,storage)
    c.need(cap['output_directory'].startswith('experiments/results/TreeLoom_frozen_bounded_')and cap['worker_auxiliary_scope'].startswith('.runtime/tmp/TreeLoom_frozen_bounded_'),'Dedicated new one-shot scientific scopes')
    out=c.local(cap['output_directory']);aux=c.local(cap['worker_auxiliary_scope'])
    c.need(not out.exists()and not aux.exists()and out.parent.is_dir()and aux.parent.is_dir(),'Fresh absent primary and AUX; no reuse')
    g.budget_check_v9(storage,boundary=True,required_write_bytes=131072)
    out.mkdir(mode=0o700);aux.mkdir(mode=0o700)
    c.put(cap,'attempt_consumed.json',dict(input_ref=record,automatic_retry=False))
    physical_fd,logical_fd,ticket=borrowed.attach_v1(cap,cap)
    ticket_ref=c.put(cap,'borrowed_GPU0_ticket.json',ticket)

    c.need(Path.cwd()==c.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0,'Fresh actual ROOT -B -s parent')
    c.load(PIDFD,'pidfd_backend_v1');owned=c.load(OWNED,'_empty_owner_original_owned_lifecycle')
    sys.modules['pidfd_backend_v1'].self_probe_v1()
    c.need(not owned._children(os.getpid()),'Parent has no unrelated children')
    base=c.load(cap['base_pool_ref'],'_empty_owner_original_subreaper');base.enable_child_subreaper()
    cancelled=[None];handlers={s:signal.signal(s,lambda signum,frame:cancelled.__setitem__(0,signum))for s in(signal.SIGTERM,signal.SIGINT)}
    gate_r,gate_w=os.pipe();proc=tree=seed=None;raw_wait=None;failure=None;tail=b'';eof=False;launch=None;cleanup_errors=[];adopted_waits=[];adopted_wait_count=0;adopted_waits_truncated=False
    started=time.monotonic()
    try:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=c.GPU0_UUID,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
            PYTHONPATH=str(c.ROOT)+':'+str(c.ROOT/'src')+':'+str(c.ROOT/'scripts'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            ARGGNN_PHYSICAL_GPU='0',ARGGNN_GPU_LEASE=str(c.ROOT/'.runtime/locks/gpu_0.lock'),ARGGNN_GPU_LEASE_FD=str(physical_fd),
            downstream_LOGICAL_LOCK_FD=str(logical_fd),ARGGNN_JOB_NAME=ticket['physical_lease_metadata']['job'])
        command=[str(c.local(c.read(cap['runtime_ref'])['python'])),'-B','-s','-m',c.HERE.replace('/','.')+'.worker_v1','--input-ref',json.dumps(record),'--ticket-ref',json.dumps(ticket_ref),'--gate-fd',str(gate_r)]
        proc=subprocess.Popen(command,cwd=c.ROOT,env=env,start_new_session=True,pass_fds=(gate_r,physical_fd,logical_fd),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
        os.close(gate_r);gate_r=None;os.set_blocking(proc.stdout.fileno(),False)
        seed=owned._stat(proc.pid);c.need(seed is not None and seed['ppid']==os.getpid()and seed['group']==seed['session']==proc.pid,'Exact owned new-session child')
        tree=owned.OwnedTree(proc.pid)
        launch=put(cap,'actual_launch.json',dict(schema='ROOT_frozen_TreeLoom_actual_launch_v1',scientific_input_ref=record,worker_pid=proc.pid,worker_start_ticks=seed['start'],parent_pid=os.getpid(),gate_closed=True,owned=tree.evidence(),command=command))
        c.validate_cap_v1(record);c.need(cancelled[0]is None,'No launch after ROOT cancellation')
        os.write(gate_w,b'G');os.close(gate_w);gate_w=None
        while True:
            c.need(cancelled[0]is None,'ROOT cancellation requires owned cleanup')
            c.need(time.time()<min(cap['drain_expires_unix']-30,__import__('datetime').datetime.fromisoformat(cap['created_at_utc']).timestamp()+cap['caps']['max_seconds']),'Retain30s aggregate owned cleanup')
            status=os.waitid(os.P_PID,proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            tree.refresh();c.need(len(tree.known)<=OWNED_ROSTER_CAP,'Finite exact-owned controller plus inventory-helper roster')
            rss=0
            for row in tree.known.values():
                if owned._same(owned._stat(row['pid']),row):
                    try:rss+=int(Path('/proc/%d/statm'%row['pid']).read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                    except(FileNotFoundError,ProcessLookupError):pass
            c.need(rss<=cap['caps']['max_cpu_rss_bytes'],'Original full owned process RSS cap')

            ready,_,_=select.select([proc.stdout],[],[],.2)
            if ready:
                block=os.read(proc.stdout.fileno(),65536)
                if block:tail=(tail+block)[-16384:]
                else:eof=True
            if status is not None:break
    except BaseException as error:failure=type(error).__name__+': '+str(error)[:2000]
    finally:
        for fd in(gate_r,gate_w):
            if fd is not None:
                try:os.close(fd)
                except OSError:pass
        cleanup_started=time.monotonic();deadline=cleanup_started+30
        if proc is not None:
            def note_cleanup(error):
                nonlocal failure
                message=type(error).__name__+': '+str(error)[:200]
                if len(cleanup_errors)<8:cleanup_errors.append(message)
                failure=failure or 'Owned cleanup observation failed: '+message
            while time.monotonic()<deadline:
                try:
                    if raw_wait is None:
                        waited,status=os.waitpid(proc.pid,os.WNOHANG)
                        if waited==proc.pid:raw_wait=status;proc.returncode=os.waitstatus_to_exitcode(status)
                except BaseException as error:note_cleanup(error)
                try:
                    if tree is not None:
                        tree.refresh();c.need(len(tree.known)<=OWNED_ROSTER_CAP,'Finite exact-owned cleanup roster')
                except BaseException as error:note_cleanup(error)
                # Never let a refresh error or a consumed owner wait suppress retained pidfd cleanup.
                rows=list(tree.known.values())if tree is not None else([seed]if seed else[])
                for row in rows[:OWNED_ROSTER_CAP]:
                    if time.monotonic()>=deadline:break
                    try:owned.signal_identity(row,signal.SIGTERM if time.monotonic()-cleanup_started<10 else signal.SIGKILL)
                    except BaseException as error:note_cleanup(error)
                if raw_wait is not None:
                    try:
                        while time.monotonic()<deadline:
                            try:adopted_pid,adopted_status=os.waitpid(-1,os.WNOHANG)
                            except ChildProcessError:break
                            if adopted_pid==0:break
                            adopted_wait_count+=1
                            if len(adopted_waits)<OWNED_ROSTER_CAP:
                                adopted_returncode=os.waitstatus_to_exitcode(adopted_status)
                                adopted_waits.append(dict(pid=adopted_pid,status=adopted_status,returncode=adopted_returncode))
                                if adopted_returncode!=0:failure=failure or 'Owned inventory helper returned nonzero: '+str(adopted_returncode)
                            else:
                                adopted_waits_truncated=True;failure=failure or 'Owned adopted wait roster exceeded64'
                    except BaseException as error:note_cleanup(error)
                try:
                    try:
                        block=os.read(proc.stdout.fileno(),65536)
                        if block:tail=(tail+block)[-16384:]
                        else:eof=True
                    except BlockingIOError:pass
                    if raw_wait is not None and tree is not None and tree.identities_absent()and not owned._children(os.getpid())and eof:break
                except BaseException as error:note_cleanup(error)
                time.sleep(min(.05,max(0.,deadline-time.monotonic())))
            if proc.stdout is not None:proc.stdout.close()
        for s,v in handlers.items():signal.signal(s,v)
        os.close(logical_fd);os.close(physical_fd)
    drained=False;owned_evidence=None
    try:
        if tree is not None:owned_evidence=tree.evidence()
        drained=raw_wait is not None and tree is not None and len(tree.known)<=OWNED_ROSTER_CAP and not adopted_waits_truncated and tree.identities_absent()and not owned._children(os.getpid())and eof
    except BaseException as error:
        message=type(error).__name__+': '+str(error)[:200]
        if len(cleanup_errors)<8:cleanup_errors.append(message)
        failure=failure or 'Final owned absence observation failed: '+message
        if tree is not None:owned_evidence=dict(retained_known_identities=list(tree.known.values())[:OWNED_ROSTER_CAP],final_observation_uncertain=True)
    outcome=dict(schema='ROOT_frozen_TreeLoom_actual_direct_wait_v1',scientific_input_ref=record,actual_launch_ref=launch,
        worker_pid=proc.pid if proc is not None else None,worker_start_ticks=seed['start']if seed else None,
        actual_raw_wait=raw_wait,actual_direct_returncode=os.waitstatus_to_exitcode(raw_wait)if raw_wait is not None else None,
        all_owned_processes_drained=drained,pipe_eof=eof,cleanup_errors=cleanup_errors,failure=failure,
        actual_tool_outer_terminal_consumed=False,actual_tool_outer_exit_code=None,historical_old_drain_proven=False,
        new_scientific_score=False,storage_credit_claimed=False,pidfd_only_cleanup=True,adopted_waits=adopted_waits,adopted_wait_count=adopted_wait_count,adopted_waits_truncated=adopted_waits_truncated,owned_evidence=owned_evidence,owner_subreaper_enabled_before_spawn=True,
        elapsed_seconds=time.monotonic()-started,log_tail=tail.decode('utf-8',errors='replace'))
    result=put(cap,'actual_wait.json',outcome);print(json.dumps(dict(actual_wait_ref=result,**outcome),sort_keys=True),flush=True)
    return 0 if drained and outcome['actual_direct_returncode']==0 and failure is None else 1
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--capability-ref');a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE',operations=0)));return 0
    c.need(a.capability_ref,'Exact ROOT frozen scientific input ref');return execute_v1(json.loads(a.capability_ref))
if __name__=='__main__':raise SystemExit(main())
