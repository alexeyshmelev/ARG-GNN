"""ROOT-only finite preparation supervisor, reusing the existing owned CPU framework."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,ctypes,hashlib,json,os,signal,stat,subprocess,sys,time
from research.reviews.American_C3_VAL003004_current_Census_preparation_source_20261016_v1 import root_contract_v1 as c
ROOT=c.ROOT;HERE=c.REL;SOURCE=c.DRIVER;PIDFD=c.PIDFD;OWNED=c.OWNED
need=c.require;local=c.local;ref=c.ref;read=c.read;load=c.load;put=c.put;current=c.current;guard=c.guard
SLOT_FD=None;CAPS={};OUTPUT=PARENT=CONFIG=None
def input_spec(record):
    global CAPS,OUTPUT,PARENT,CONFIG
    s,q,cfg=c.input_spec(record);CAPS=c.CAPS;OUTPUT=s["output_directory"];PARENT=c.PARENT;CONFIG=s["refs"]["config"]
    return s,q,cfg
def worker_python(q):
    runtime=read(q["runtime_ref"])
    expected=".runtime/envs/biological_tsinfer_v1" if q["stage"]=="infer" else ".runtime/env"
    need(runtime["prefix"]==str(ROOT/expected),"Exact existing qualified interpreter prefix")
    return str(ROOT/expected/"bin/python")
def sample_scope(s):
    totals={}
    for path in [s["output_directory"],s["worker_auxiliary_scope"]]:
        p=local(path);total=entries=0
        if p.exists():
            for directory,dirs,files in os.walk(p,followlinks=False):
                st=Path(directory).lstat();need(stat.S_ISDIR(st.st_mode),"Canonical charged directory");total+=st.st_blocks*512
                for name in dirs+files:
                    st=(Path(directory)/name).lstat();entries+=1
                    need(entries<=8192 and (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)),"Bounded regular charged scope; no links/specials")
                    if stat.S_ISREG(st.st_mode):need(st.st_nlink==1,"No charged hardlinks");total+=max(st.st_size,st.st_blocks*512)
        totals[path]=total
    need(totals[s["worker_auxiliary_scope"]]<=CAPS["aux_bytes"] and sum(totals.values())<=CAPS["reservation_bytes"],"Original full reservation and bounded AUX")
    tmp=local(s["worker_auxiliary_scope"])/"tmp"
    scratch=0
    if tmp.exists():
        for directory,dirs,files in os.walk(tmp,followlinks=False):
            scratch+=Path(directory).lstat().st_blocks*512
            for name in files:
                st=(Path(directory)/name).lstat();scratch+=max(st.st_size,st.st_blocks*512)
    need(scratch<=CAPS["private_work_bytes"],"Original private-work allowance")
    return totals
def boundary(g,s,required=0,target="output"):
    current(s,150);totals=sample_scope(s)
    need(sum(totals.values())+required+8192<=CAPS["reservation_bytes"],"Whole charged phase cap before metadata write")
    if target=="aux":need(totals[s["worker_auxiliary_scope"]]+required+8192<=CAPS["aux_bytes"],"AUX prospective bound")
    g.budget_check_v9(s["storage"],boundary=True,required_write_bytes=required+8192)
    current(s,150)
def validate_completion(value,q):
    need(value["status"]==c.STATUSES[q["phase"]],"Exact unchanged producer completed status")
    if q["stage"]=="infer":
        m=c.observation(value["observation_ref"],q["parent_id"])
        need(m["sites_and_sample_identities_verified"] is True and m["source_observation_ref"]==q["prior"]["observation_ref"],"Full inferred original sites/identities")
    else:need(value["parent_id"]==q["parent_id"],"Exact original parent")
    if q["stage"]=="native_paths":
        need(value["query_focal_count"]==10000 and value["query_ancestry_truth_opened"] is False
            and value["native_split_seed"]==42 and value["native_train_fraction"]==.9,"Original full paths and prespecified label-blind focals")
def admit(request_ref):
    q,caps,_=c.validate_request(request_ref);packet,output=c.phase_paths(c.PHASE)
    need(not packet.exists() and not local(output).exists(),"Fresh exclusive source packet/output; no replay")
    g=c.original_guard()
    implementation=ref(g.budget.SOURCE_REL);launcher=ref(g.budget.LAUNCHER_REL)
    with g.budget.v4.mutex(ROOT):state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
    need(q["prior"]["closed_key"] in state[1],"Actual predecessor CLOSED in current Census state")
    packet.mkdir();config_ref=c.save(packet,"config.json",c.make_config(q,caps))
    sources={path:ref(path)["sha256"] for path in c.source_paths()}
    closure=c.save(packet,"source_and_input_closure.json",dict(schema="American_C3_original_preparation_closure_v1",
        source_sha256=sources,root_stage_request_ref=request_ref,bindings=q["bindings"],native_MLM_dependency=False))
    consumers=c.unique([*state[3].values(),*c.direct_roles(request_ref,q,config_ref,closure,q["runtime_ref"])])
    job="American-C3-val"+c.PARENT[-3:]+"-full30mb-"+c.STAGE+"-current-Census-v1";aux=packet/"worker_aux"
    reserve=dict(key="H041/"+job,output=str(ROOT/output),auxiliary_scopes=[str(aux)],
        reserved_bytes=c.CAPS["reservation_bytes"],consumer_bindings=consumers)
    authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=q["expires_at_utc"])
    ar=c.save(packet,"reserve_authority.json",authority)
    receipt=g.apply_reserve_v1(ar,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    rr=c.save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,
        job_id=job,output_directory=output,reserved_new_bytes=c.CAPS["reservation_bytes"],reservation=receipt["value"],
        initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=q["expires_at_utc"],completed_overlay_policy_ref=c.POLICY_REF,
        budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.GUARD_REF))
    storage_ref=c.save(packet,"storage.json",storage)
    spec=dict(schema="American_C3_current_Census_active_preparation_input_v1",status="active",root_authorizer="root_agent",
        execution_authorized=True,GPU_authorized=False,parent_id=c.PARENT,phase=c.PHASE,stage=c.STAGE,job_id=job,
        output_directory=output,worker_auxiliary_scope=str(aux.relative_to(ROOT)),root_stage_request_ref=request_ref,
        root_driver_ref=ref(SOURCE),main_source_ref=ref(c.ENTRY),source_sha256=sources,transitive_source_closure_ref=closure,
        caps_ref=q["caps_ref"],refs=dict(config=config_ref,runtime=q["runtime_ref"],storage=storage_ref),
        runtime_ref=q["runtime_ref"],storage=storage,created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
        query_truth_access_authorized=False,TEST_access_authorized=False,GNN_optimization_authorized=False)
    ir=c.save(packet,"input.json",spec)
    c.save(packet,"admitted.json",dict(input_ref=ir,storage=storage,reserve_receipt_ref=rr,driver_ref=ref(SOURCE),
        consumer_binding_count=len(consumers),scientific_execution_started=False))
    return dict(input_ref=ir,consumer_binding_count=len(consumers))

def execute(r):
    need(os.getpid()==os.getpgrp()==os.getsid(0),"ROOT supervisor must use a dedicated actual session")
    started=time.monotonic();s,q,cfg=input_spec(r);g=guard(True)
    need(Path.cwd()==ROOT and Path(sys.executable).resolve()==(ROOT/".runtime/env/bin/python").resolve(),"Original CPU supervisor runtime")
    need(not local(OUTPUT).exists() and not local(s["worker_auxiliary_scope"]).exists(),"Fresh one-shot execution only")
    g.validate_storage_v9(s,dict(output_cap_bytes=CAPS["reservation_bytes"]),s["storage"],source_ref=ref(SOURCE),config_ref=CONFIG,runtime_ref=s["runtime_ref"])
    aux=local(s["worker_auxiliary_scope"]);g.budget_check_v9(s["storage"],boundary=True,required_write_bytes=32768)
    aux.mkdir();(aux/"tmp").mkdir()
    put(aux/"attempt_consumed.json",dict(input_ref=r,consumed_utc=datetime.now(timezone.utc).isoformat(),one_shot=True))
    load(PIDFD,"pidfd_backend_v1");owned=load(OWNED,"_European_outer_owned")
    sys.modules["pidfd_backend_v1"].self_probe_v1()
    need(not owned._children(os.getpid()),"No unrelated children")
    need(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,"Owned subreaper")
    gate_read,gate_write=os.pipe();proc=tree=seed=None;status=None;failure=None;tail=bytearray();eof=False;nonblocking=False;cancel=[None]
    old={sig:signal.signal(sig,lambda n,f:cancel.__setitem__(0,n)) for sig in (signal.SIGINT,signal.SIGTERM)}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]),
        TMPDIR=str(aux/"tmp"),TMP=str(aux/"tmp"),TEMP=str(aux/"tmp"),OMP_NUM_THREADS="8",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="8",ARGGNN_C3_PREPARATION_SLOT_FD=str(SLOT_FD),ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(c.POLICY_REF))
    deadline=min(started+CAPS["wall_seconds"]-300,time.monotonic()+datetime.fromisoformat(s["expires_at_utc"]).timestamp()-time.time()-300)
    try:
        proc=subprocess.Popen([worker_python(q),"-B","-m",HERE.replace("/",".")+".worker_entry_v1",
            "--worker-ref",json.dumps(r),"--gate-fd",str(gate_read)],cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(gate_read,SLOT_FD),bufsize=0)
        os.close(gate_read);gate_read=None;os.set_blocking(proc.stdout.fileno(),False);nonblocking=True
        seed=owned._stat(proc.pid);tree=owned.OwnedTree(proc.pid)
        put(aux/"launch.json",dict(pid=proc.pid,input_ref=r,ownership=tree.evidence(),gate_closed=True))
        need(time.monotonic()<deadline and cancel[0] is None,"Work window before gate")
        os.write(gate_write,b"G");os.close(gate_write);gate_write=None
        while time.monotonic()<deadline and cancel[0] is None:
            event=os.waitid(os.P_PID,proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            tree.refresh();need(len(tree.known)<=64,"Bounded own process roster")
            rss=0
            for row in tree.known.values():
                if owned._same(owned._stat(row["pid"]),row):
                    try:rss+=int(Path("/proc/%d/statm"%row["pid"]).read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
                    except (FileNotFoundError,ProcessLookupError):pass
            need(rss<=CAPS["rss_bytes"],"Owned RSS fail-stop")
            current(s,300);sample_scope(s)
            try:data=os.read(proc.stdout.fileno(),65536)
            except BlockingIOError:data=None
            if data==b"":eof=True
            elif data:tail.extend(data);del tail[:-CAPS["log_bytes"]]
            if event is not None:break
            time.sleep(.2)
        else:failure="Cancellation or scientific work cutoff"
    except BaseException as exc:failure=type(exc).__name__+": "+str(exc)[:2000]
    finally:
        for fd in (gate_read,gate_write):
            if fd is not None:
                try:os.close(fd)
                except OSError:pass
        stop=time.monotonic();end=stop+CAPS["cleanup_seconds"];absence=False;groups=[];adopted=[];errors=[]
        while proc is not None and time.monotonic()<end:
            try:
                if status is None:
                    pid,actual=os.waitpid(proc.pid,os.WNOHANG)
                    if pid==proc.pid:status=actual;proc.returncode=os.waitstatus_to_exitcode(actual)
                if tree is not None:tree.refresh();need(len(tree.known)<=64,"Bounded cleanup roster")
                rows=list(tree.known.values())[:64] if tree is not None else ([seed] if seed else [])
                for row in rows:
                    if time.monotonic()>=end:break
                    owned.signal_identity(row,signal.SIGTERM if time.monotonic()-stop<10 else signal.SIGKILL)
                if status is not None:
                    for _ in range(64):
                        if time.monotonic()>=end:break
                        try:pid,actual=os.waitpid(-1,os.WNOHANG)
                        except ChildProcessError:break
                        if not pid:break
                        if len(adopted)<64:adopted.append(dict(pid=pid,status=actual,returncode=os.waitstatus_to_exitcode(actual)))
                        else:failure=failure or "Adopted roster overflow"
                for _ in range(4):
                    if time.monotonic()>=end or not nonblocking:break
                    try:data=os.read(proc.stdout.fileno(),65536)
                    except BlockingIOError:break
                    if not data:eof=True;break
                    tail.extend(data);del tail[:-CAPS["log_bytes"]]
                if status is not None and tree is None:break
                if status is not None and tree is not None and time.monotonic()<end:
                    absence=tree.identities_absent() and not owned._children(os.getpid())
                    if absence and eof:
                        groups=[]
                        for pgid in sorted({x["group"] for x in tree.known.values()}):
                            gone=None
                            if time.monotonic()<end:
                                try:os.killpg(pgid,0)
                                except ProcessLookupError:gone=True
                                else:gone=False
                            groups.append(dict(pgid=pgid,absent=gone))
                        if groups and all(x["absent"] is True for x in groups):break
            except BaseException as exc:
                failure=failure or "Cleanup observation failed"
                if len(errors)<16:errors.append(type(exc).__name__+": "+str(exc)[:500])
            time.sleep(min(.05,max(0,end-time.monotonic())))
        if proc is not None and proc.stdout is not None:proc.stdout.close()
        for sig,handler in old.items():signal.signal(sig,handler)
    rc=None if status is None else os.waitstatus_to_exitcode(status)
    drained=status is not None and absence and eof and bool(groups) and all(x["absent"] is True for x in groups)
    if not drained:failure=failure or "Unresolved owned drain/wait"
    if rc!=0 or any(x["returncode"]!=0 for x in adopted):failure=failure or "Genuine nonzero/unknown child wait"
    observation=dict(schema="American_C3_full_original_CPU_actual_wait_v1",input_ref=r,pid=None if proc is None else proc.pid,
        actual_direct_worker_wait=status is not None,actual_OS_wait_status=status,returncode=rc,
        all_owned_processes_and_groups_drained=drained,group_observations=groups,adopted_waits=adopted,
        operational_failure=failure,cleanup_errors=errors,outer_truth_scored=False,automatic_close_or_credit=False)
    print(json.dumps(observation,sort_keys=True),flush=True)
    if not drained or time.time()+150>=datetime.fromisoformat(s["expires_at_utc"]).timestamp():
        print(json.dumps(dict(retained_worker_tail=bytes(tail[-8192:]).decode(errors="replace"),no_future_stage_authorized=True)),flush=True)
        return dict(status="FAILED_UNRESOLVED_OR_EVIDENCE_MARGIN",returncode=1,actual_wait_ref=None,observation=observation)
    try:
        # Genuine drain precedes every new source/accounting/product inspection.
        boundary(g,s,32768,target="aux")
        cp=ref(OUTPUT+"/completed.json");completed=read(cp);validate_completion(completed,q)
        observation.update(completed_ref=cp,phase=q["phase"],parent_id=PARENT,actual_direct_launcher_wait=status is not None,
            known_owned_pids=sorted(tree.known) if tree is not None else [],known_owned_process_groups=[x["pgid"] for x in groups],
            supervisor_pid=os.getpid(),supervisor_process_group=os.getpgrp())
        wr=put(aux/"actual_wait.json",observation)
        boundary(g,s,len(tail)*6+8192,target="aux");put(aux/"worker_log.json",dict(tail=bytes(tail).decode(errors="replace"),retained_raw_bytes=len(tail)),cap=128*2**20)
        return dict(status="PREPARED" if failure is None else "FAILED",returncode=0 if failure is None else 1,
            actual_wait_ref=wr,completed_ref=cp,outer_tool_terminal_consumed=False,outer_truth_scored=False)
    except BaseException as exc:
        print(json.dumps(dict(actual_wait_observation=observation,evidence_failure=str(exc)[:2000])),flush=True)
        return dict(status="FAILED_EVIDENCE",returncode=1)

def close(record,approval_ref):
    # A genuine completed close may follow the original execution expiry. This
    # fresh close authority grants no further execution and never extends it.
    s=read(record);q=read(s["root_stage_request_ref"]);c.select(q["parent_id"],q["stage"]);packet,_=c.phase_paths(c.PHASE)
    need(record==ref(packet/"input.json") and s["root_driver_ref"]==ref(SOURCE)
        and q["adapter_manifest_ref"]==ref(c.MANIFEST),"Exact original admitted completed stage/source")
    wr=ref(packet/"worker_aux/actual_wait.json");wait=read(wr);approval=read(approval_ref)
    terminal_ref=ref(packet/"outer_terminal.root_actual_v1.json");terminal=read(terminal_ref)
    need(terminal["root_authorizer"]=="root_agent" and terminal["actual_terminal_consumed"] is True
        and terminal["actual_exit_code"]==0 and terminal["completed_ref"]==wait["completed_ref"] and terminal["direct_wait_ref"]==wr,
        "Actual ROOT consumed successful outer terminal")
    need(approval["root_authorizer"]=="root_agent" and approval["active"] is True
        and approval["input_ref"]==record and approval["actual_wait_ref"]==wr and approval["outer_terminal_ref"]==terminal_ref
        and approval["all_prior_launch_resume_write_capabilities_revoked"] is True
        and approval["all_genuinely_owned_PIDs_and_groups_listed"] is True,"Actual complete ROOT close/revocation approval")
    need(wait["returncode"]==0 and wait["operational_failure"] is None and wait["actual_direct_worker_wait"] is True
        and wait["all_owned_processes_and_groups_drained"] is True,"Actual successful direct wait and complete drain")
    pids=approval["root_owned_pids"];groups=approval["root_owned_process_groups"]
    need(set(wait["known_owned_pids"])|{wait["supervisor_pid"]}<=set(pids)
        and set(wait["known_owned_process_groups"])|{wait["supervisor_process_group"]}<=set(groups)
        and 1<=len(set(pids))==len(pids)<=64 and 1<=len(set(groups))==len(groups)<=64,"Complete bounded known-owned roster")
    observations=[]
    for pid in pids:
        try:os.kill(pid,0)
        except ProcessLookupError:observations.append(dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=datetime.now(timezone.utc).isoformat()))
        else:raise ValueError("Owned PID alive/reused; preserve and block")
    for pgid in groups:
        try:os.killpg(pgid,0)
        except ProcessLookupError:continue
        raise ValueError("Owned group alive/reused; preserve and block")
    g=c.original_guard();impl=ref(g.budget.SOURCE_REL);launcher=ref(g.budget.LAUNCHER_REL);own=s["storage"]["reservation"]
    with g.budget.v4.mutex(ROOT):state=g._state_locked(impl["sha256"],launcher["sha256"],fresh=True)
    need(state[0].get(own["key"])==own and own["key"] not in state[1],"Exact open original reservation/prefix; no rebase/replay")
    folder=packet/"closed";need(not folder.exists(),"No close replay");folder.mkdir()
    launch_ref=ref(packet/"worker_aux/launch.json");launch=read(launch_ref)
    observation=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",
        key=own["key"],reservation_sha256=g.budget.digest(own),storage_ref=ref(packet/"admitted.json"),launch_ref=launch_ref,
        direct_wait_ref=wr,completed_ref=wait["completed_ref"],launcher_pid=launch["pid"],
        all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
        completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
        external_NFS_writers_outside_observation=True,pids=pids,observations=observations,
        root_owned_process_groups=groups,root_approval_ref=approval_ref)
    obsref=c.save(folder,"root_observation.json",observation)
    cp=read(wait["completed_ref"]);identity=dict(status=cp["status"])
    if "parent_id" in cp:identity["parent_id"]=c.PARENT
    request=dict(key=own["key"],reservation_sha256=g.budget.digest(own),storage_ref=ref(packet/"admitted.json"),storage_locator=["storage"],
        launch_ref=launch_ref,launch_pid_locator=["pid"],launch_identity=dict(pid=launch["pid"],input_ref=record),
        direct_wait_ref=wr,wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
        wait_identity=dict(pid=wait["pid"],returncode=0,actual_direct_worker_wait=True,
            all_owned_processes_and_groups_drained=True,completed_ref=wait["completed_ref"],operational_failure=None),
        completed_ref=wait["completed_ref"],completed_identity=identity,observation_ref=obsref)
    authority=g.operation_draft_v1("close_completed",request,expected_before=g._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
    ar=c.save(folder,"close_authority.json",authority)
    result=g.apply_close_completed_v1(ar,expected_source_sha256=impl["sha256"],expected_launcher_sha256=launcher["sha256"])
    return c.save(folder,"close_receipt.json",result)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--parent",choices=c.PARENTS);p.add_argument("--stage",choices=c.STAGES)
    mode=p.add_mutually_exclusive_group();mode.add_argument("--bind",action="store_true");mode.add_argument("--admit",action="store_true")
    mode.add_argument("--execute",action="store_true");mode.add_argument("--close",action="store_true")
    p.add_argument("--review-ref");p.add_argument("--request-ref");p.add_argument("--input-ref");p.add_argument("--approval-ref")
    a=p.parse_args(argv)
    if a.bind:result=c.bind(a.parent,a.stage,json.loads(a.review_ref))
    elif a.admit:result=admit(json.loads(a.request_ref))
    elif a.execute:
        global SLOT_FD
        record=json.loads(a.input_ref);input_spec(record)
        with c.cpu_slot() as fd:SLOT_FD=fd;result=execute(record)
    elif a.close:result=close(json.loads(a.input_ref),json.loads(a.approval_ref))
    else:result=dict(status="SOURCE_ONLY_INACTIVE",parents=list(c.PARENTS),stages=list(c.STAGES),maximum_CPU_producers=2,execution=False)
    print(json.dumps(result,sort_keys=True),flush=True)
    return result.get("returncode",0) if isinstance(result,dict) else 0
if __name__=="__main__":raise SystemExit(main())
