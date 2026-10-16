"""Inactive ROOT admission and finite pidfd-owned supervisor for full matched CPU training."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ctypes,difflib,hashlib,json,os,signal,stat,subprocess,sys,time
from research.reviews.American_TreeLoom_matched_CPU_execution_source_20261017_v5 import root_contract_v1 as c
ROOT=c.ROOT;HERE=c.HERE;SOURCE=c.DRIVER;PIDFD=c.PIDFD;OWNED=c.OWNED
need=c.require;local=c.local;ref=c.ref;read=c.read;load=c.load;put=c.put;current=c.current;guard=c.guard
CAPS={};OUTPUT=PARENT=CONFIG=None
def input_spec(record):
 global CAPS,OUTPUT,PARENT,CONFIG
 s,q,cfg=c.input_spec(record);CAPS=c.CAPS;OUTPUT=s["output_directory"];PARENT=c.PARENT;CONFIG=s["refs"]["config"]
 return s,q,cfg
def validate_completion(value,q):
 need(value["status"]==c.STATUSES[c.PHASE]and value["parent_id"]==c.PARENT,"Exact matched CPU completed envelope")
 result=read(value["result_ref"])
 need(result["status"]=="completed_TreeLoom_saved600_warmstart_matched_training"
  and result["additional_updates_per_arm"]==600 and result["new_saved_logical_optimizer_updates_total"]==1200
  and result["epochs"]==30 and result["trained_parent_ids"]==list(c.TRAIN)
  and result["train_sequence_lengths_bp"]==[2000000]*20
  and result["first_step_timing_inside_full_budget"]is True and result["standalone_token_update_probe"]is False
  and all(len(rows)==6 for rows in result["checkpoint_candidates"].values())
  and len(result["checkpoint_candidates"])==2 and result["TEST_deferred"]is True,"Real full600additional EACHarm/1200total; no token qualification")
def admit(request_ref):
 q,caps,_=c.validate_request(request_ref);packet=local(q["packet_directory"]);output=q["output_directory"]
 need(set(q["cpu_ids"])<=os.sched_getaffinity(0),"ROOT supplies32actually available CPU identities")
 need(not packet.exists()and not local(output).exists(),"Fresh exclusive one-shot packet/output")
 g=guard(False);preparation=local(q["preparation_directory"]);stem=packet.name
 cfgref=put(preparation/(stem+".config.json"),c.make_config(q,caps));sources=c.sources()["source_sha256"]
 closure=put(preparation/(stem+".source_and_input_closure.json"),dict(source_sha256=sources,root_stage_request_ref=request_ref))
 implementation=ref(g.budget.SOURCE_REL);launcher=ref(g.budget.LAUNCHER_REL)
 with g.budget.v4.mutex(ROOT):state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
 consumers=g.consumer_bindings_v1(list(state[3].values()),c.direct_roles(request_ref,q,cfgref,closure))
 job="TreeLoom-matched-CPU-"+packet.name
 reserve=dict(key="H060/"+job,output=str(ROOT/output),auxiliary_scopes=[str(packet)],
  reserved_bytes=caps["reservation_bytes"],consumer_bindings=consumers)
 # Exact original read-only schema/type/hash preflight over every returned/current role.
 # This precedes authority creation and reservation accounting mutation.
 paths=g.budget.v4.scope_spec(ROOT,reserve)
 need(reserve["key"] not in state[0] and reserve["key"] not in g._policy()["never_credit_keys"]
  and not any(p.exists() for p in paths),"New key and fresh exclusive paths")
 authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3])
 authority.update(active=True,expires_at_utc=q["expires_at_utc"])
 ar=put(preparation/(stem+".reserve_authority.json"),authority)
 receipt=g.apply_reserve_v1(ar,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
 need(not packet.exists() and not local(output).exists(),"Reserved target paths remain absent before exclusive creation")
 packet.mkdir()
 rr=put(packet/"reserve_receipt.json",receipt)
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,
  job_id=job,output_directory=output,reserved_new_bytes=caps["reservation_bytes"],reservation=receipt["value"],
  initial_ledger_pins=receipt["initial_ledger_pins"],created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
  completed_overlay_policy_ref=c.POLICY_REF,storage_policy_selector_ref=c.SELECTOR_REF,
  budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.GUARD_REF))
 sr=put(packet/"storage.json",storage)
 s=dict(schema="TreeLoom_matched_CPU_admitted_input_v1",status="active",active=True,root_authorizer="root_agent",execution_authorized=True,
  GPU_authorized=False,physical_gpu=None,parent_id=c.PARENT,phase=c.PHASE,stage="train",job_id=job,packet_directory=q["packet_directory"],
  output_directory=output,worker_auxiliary_scope=q["packet_directory"]+"/worker_aux",root_stage_request_ref=request_ref,
  root_driver_ref=ref(SOURCE),main_source_ref=ref(c.ENTRY),source_sha256=sources,transitive_source_closure_ref=closure,
  refs=dict(config=cfgref,runtime=q["runtime_ref"],storage=sr),runtime_ref=q["runtime_ref"],storage=storage,
  created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"])
 ir=put(packet/"input.json",s);c.input_spec(ir)
 put(packet/"admitted.json",dict(input_ref=ir,driver_ref=ref(SOURCE),storage=storage,reserve_receipt_ref=rr,
  consumer_binding_count=len(consumers),scientific_execution_started=False))
 return dict(input_ref=ir,consumer_binding_count=len(consumers),admission_only=True,
  execution_argv=["setsid",str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",HERE.replace("/","." )+".root_driver_v1","--execute-ref",json.dumps(ir)])

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
    packet=local(s["worker_auxiliary_scope"]).parent
    admin=packet.lstat().st_blocks*512
    for child in packet.iterdir():
        if child==local(s["worker_auxiliary_scope"]):continue
        st=child.lstat();need(stat.S_ISREG(st.st_mode) and st.st_nlink==1,"Flat regular administrative metadata")
        admin+=max(st.st_size,st.st_blocks*512)
    need(admin<=CAPS["root_packet_bytes"],"Bounded root packet metadata")
    totals[str(packet.relative_to(ROOT))]=admin
    need(sum(totals.values())<=CAPS["reservation_bytes"],"Output+workerAUX+rootmetadata whole charge")
    return totals

def boundary(g,s,required=0,target="output"):
    current(s,150);totals=sample_scope(s)
    need(sum(totals.values())+required+8192<=CAPS["reservation_bytes"],"Whole charged phase cap before metadata write")
    if target=="aux":need(totals[s["worker_auxiliary_scope"]]+required+8192<=CAPS["aux_bytes"],"AUX prospective bound")
    g.budget_check_v9(s["storage"],boundary=True,required_write_bytes=required+8192)
    current(s,150)

def execute(r):
    need(os.getpid()==os.getpgrp()==os.getsid(0),"ROOT supervisor must use a dedicated actual session")
    started=time.monotonic();s,q,cfg=input_spec(r);g=guard(True);c.verify_actual_roles(s,q,g)
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
        TMPDIR=str(aux/"tmp"),TMP=str(aux/"tmp"),TEMP=str(aux/"tmp"),OMP_NUM_THREADS="32",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="32",ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(c.POLICY_REF))
    for name,key in (("cache","XDG_CACHE_HOME"),("config","XDG_CONFIG_HOME"),("data","XDG_DATA_HOME"),("torch","TORCH_HOME"),
        ("cuda","CUDA_CACHE_PATH"),("inductor","TORCHINDUCTOR_CACHE_DIR"),("triton","TRITON_CACHE_DIR"),
        ("matplotlib","MPLCONFIGDIR"),("huggingface","HF_HOME"),("numba","NUMBA_CACHE_DIR")):
        env[key]=str(aux/"tmp"/name)
    env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1",WANDB_MODE="disabled")
    deadline=min(started+CAPS["wall_seconds"]-300,time.monotonic()+datetime.fromisoformat(s["expires_at_utc"]).timestamp()-time.time()-300)
    try:
        proc=subprocess.Popen([worker_python(q),"-B","-m",HERE.replace("/",".")+".worker_entry_v1",
            "--worker-ref",json.dumps(r),"--gate-fd",str(gate_read)],cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(gate_read,),bufsize=0)
        os.close(gate_read);gate_read=None;os.set_blocking(proc.stdout.fileno(),False);nonblocking=True
        seed=owned._stat(proc.pid);tree=owned.OwnedTree(proc.pid)
        put(aux/"launch.json",dict(pid=proc.pid,input_ref=r,ownership=tree.evidence(),gate_closed=True))
        need(time.monotonic()<deadline and cancel[0] is None,"Work window before gate")
        os.write(gate_write,b"G");os.close(gate_write);gate_write=None
        while time.monotonic()<deadline and cancel[0] is None:
            event=os.waitid(os.P_PID,proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            tree.refresh();need(len(tree.known)<=64,"Bounded own process roster")
            rss=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
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
        cp=None
        if local(OUTPUT+"/completed.json").is_file():
            cp=ref(OUTPUT+"/completed.json");completed=read(cp);validate_completion(completed,q)
        else:failure=failure or "No genuine completed matched training envelope"
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

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group();m.add_argument("--admit-ref");m.add_argument("--execute-ref")
 a=p.parse_args(argv)
 if not(a.admit_ref or a.execute_ref):
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False,training_updates=0)));return 0
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"Exact ROOT -B -s repository interpreter")
 result=admit(json.loads(a.admit_ref))if a.admit_ref else execute(json.loads(a.execute_ref))
 print(json.dumps(result,sort_keys=True),flush=True);return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())
