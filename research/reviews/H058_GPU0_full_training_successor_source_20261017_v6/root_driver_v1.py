"""Fixed H058 ROOT admission and original owned supervisor. Default inert."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,collections,ctypes,fcntl,hashlib,importlib,io,itertools,json,os,resource,selectors,signal,stat,subprocess,sys,tempfile,time
from research.reviews.H058_GPU0_full_training_successor_source_20261017_v5.common_v1 import *
SOURCE=DRIVER

def mutation_memo_v1(g,q):
    spec=read(ref(HERE+"/dependencies.v1.json"))["mutation_memo"]
    pins=read(ref(MANIFEST))["source_sha256"]
    for key in ("source_ref","source_manifest_ref","source_handoff_ref","independent_review_ref","independent_review_manifest_ref"):
        r=spec[key];require(pins.get(r["path"])==r["sha256"],"Existing memo/review transitive closure")
        raw(r)
    review=read(spec["independent_review_ref"])
    require(review.get("verdict")=="SOURCE_PASS" and review.get("source_only") is True
        and review.get("source_ref")==spec["source_ref"],"Exact existing memo SOURCE_PASS")
    cache=[dict(path=p,sha256=s)for p,s in sorted(pins.items())if Path(p).suffix in(".py",".sh")]
    module=load(spec["source_ref"],"_H058_GPU0_existing_ROOT_mutation_memo")
    return module.RootMutationMemo(g,integration_manifest_ref=ref(MANIFEST),
        adapter_ref=spec["source_ref"],cache_refs=cache)

def admit_v1(request_ref):
    require(Path.cwd()==ROOT and sys.dont_write_bytecode,"Exact ROOT cwd/no bytecode")
    q,cfg,producer,protocol,manifest=validate_request_v1(request_ref)
    packet=local(q["packet_directory"]);output=local(q["output_directory"])
    require(not packet.exists() and not output.exists(),"Fresh exclusive full-parent admission/output; no replay")
    g=guard_v1(consumer=False);staging=local(q["preparation_directory"])
    runtime=save(staging,q["instance_id"]+".runtime.json",dict(schema="H058_fixed_ROOT_runtime_v1",root_authorizer="root_agent",
        job_id=q["job_id"],phase=q["phase"],device="cuda" if q["phase"]=="train" else "cpu",
        physical_gpu=0 if q["phase"]=="train" else None,output_directory=q["output_directory"],
        executable=str(ROOT/".runtime/env/bin/python"),process_title="AI Auto Research Pipeline Test",
        runtime_equivalence_ref=cfg["runtime_equivalence_ref"]))
    original=read(protocol.H050_INPUT_REF)
    closure=save(staging,q["instance_id"]+".consumer_manifest.json",dict(schema="H058_fixed_exact_transitive_closure_v1",
        root_request_ref=request_ref,science_handoff_ref=q["science_handoff_ref"],
        scientific_manifest_ref=cfg["source_manifest_ref"],adapter_manifest_ref=ref(MANIFEST),
        producer_ref=producer,source_sha256={r["path"]:r["sha256"] for r in source_refs_v1()+manifest["refs"]},
        original_H050_input_ref=protocol.H050_INPUT_REF,
        original_observation_entry_refs=original["observation_entry_refs"],
        original_observation_cache_source_refs=original["observation_cache_source_refs"],
        phase_target_bindings_ref=protocol.H050_TRAIN_BINDINGS_REF if q["phase"]=="train" else protocol.H050_VAL_BINDINGS_REF,
        config_ref=q["config_ref"],runtime_ref=runtime,direct_attachment_authority_ref=q.get("direct_attachment_authority_ref"),
        train_completion_ref=q.get("train_completion_ref"),train_actual_wait_ref=q.get("train_actual_wait_ref"),
        train_outer_terminal_ref=q.get("train_outer_terminal_ref")))
    implementation=ref(g.budget.SOURCE_REL);launcher=ref(g.budget.LAUNCHER_REL)
    memo=mutation_memo_v1(g,q)
    state=memo.read_state_v1(expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    needed=[SELECTOR_REF,ref(MANIFEST),ref(DRIVER),ref(CONTEXT),ref(DIRECT),request_ref,q["config_ref"],
        q["science_handoff_ref"],q["source_review_ref"],cfg["source_manifest_ref"],producer,runtime,closure,BASEFIT]
    needed+=[GUARD,POLICY,REPAIR,RENEWAL,CERT,MEMO,NAMESPACE_MEMO,V4,PIDFD,OWNED,BASE_POOL,GPU0_GUARD]
    if q["phase"]=="train":
        from research.reviews.H058_GPU0_full_training_successor_source_20261017_v5 import direct_gpu0_v1 as direct
        needed+=[q["direct_attachment_authority_ref"],direct.CURRENT_CAP,q["native_resource_disposition_ref"]]
    consumers=unique([*state[3].values(),*needed])
    auxiliary=q["packet_directory"]+"/worker_aux"
    reserve=dict(key="H058/"+q["job_id"],output=str(output),auxiliary_scopes=[str(packet)],
        reserved_bytes=CAPS["reservation_bytes"],consumer_bindings=consumers)
    authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=q["expires_at_utc"])
    authority_ref=save(staging,q["instance_id"]+".reserve_authority.json",authority)
    receipt=memo.reserve_v1(authority_ref,expected_source_sha256=implementation["sha256"],
        expected_launcher_sha256=launcher["sha256"])
    require(not packet.exists() and not output.exists(),"Charged scopes absent through actual reserve")
    packet.mkdir()
    receipt_ref=save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=g.MODE,job_id=q["job_id"],output_directory=q["output_directory"],
        reserved_new_bytes=CAPS["reservation_bytes"],reservation=receipt["value"],
        initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=q["expires_at_utc"],
        completed_overlay_policy_ref=POLICY,storage_policy_selector_ref=SELECTOR_REF,budget_components=dict(implementation=implementation,
            launcher=launcher,consumer_adapter=GUARD))
    storage_ref=save(packet,"storage.json",storage)
    s=dict(schema="H058_full_parent_ROOT_input_v1",status="active",root_authorizer="root_agent",
        execution_authorized=True,hypothesis_id="H058",phase=q["phase"],job_id=q["job_id"],
        output_directory=q["output_directory"],worker_auxiliary_scope=auxiliary,
        root_stage_request_ref=request_ref,producer_ref=producer,main_source_ref=ref(DRIVER),
        root_context_module_ref=ref(CONTEXT),consumer_manifest_ref=closure,
        source_sha256={r["path"]:r["sha256"] for r in source_refs_v1()+manifest["refs"]},
        refs=dict(config=q["config_ref"],runtime=runtime,storage=storage_ref),
        created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
        physical_gpu=0 if q["phase"]=="train" else None,device="cuda" if q["phase"]=="train" else "cpu",
        max_cpu_rss_bytes=cfg["max_cpu_rss_bytes"],max_cuda_reserved_bytes=cfg["max_cuda_reserved_bytes"])
    input_ref=save(packet,"input.json",s);validate_input_v1(input_ref)
    admitted_ref=save(packet,"admitted.json",dict(schema="H058_fixed_ROOT_admitted_v1",input_ref=input_ref,
        reserve_receipt_ref=receipt_ref,consumer_binding_count=len(consumers),
        scientific_execution_started=False,physical_attachment_started=False,
        whole_scope_reservation_bytes=CAPS["reservation_bytes"],scientific_output_cap_bytes=CAPS["scientific_output_bytes"]))
    return dict(input_ref=input_ref,admitted_ref=admitted_ref,
        one_shot_exec_argv=[str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",
            HERE.replace("/","." )+".root_driver_v1","--execute","--input-ref",
            json.dumps(input_ref,separators=(",",":"))])

class _RAMLog:
    def __init__(self,cap):self.cap=cap;self.parts=collections.deque();self.kept=0;self.total=0
    def add(self,data):
        self.total+=len(data);self.parts.append(data);self.kept+=len(data)
        while self.parts and self.kept>self.cap:
            first=self.parts.popleft();over=self.kept-self.cap
            if len(first)>over:self.parts.appendleft(first[over:]);self.kept-=over
            else:self.kept-=len(first)
    def record(self):
        return dict(tail=b"".join(self.parts).decode("utf-8",errors="replace"),
            total_pipe_bytes=self.total,retained_raw_bytes=self.kept,cap_bytes=self.cap,
            discarded_earlier_bytes=self.total-self.kept,buffered_in_RAM_until_actual_drain=True)

def _subreaper():
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(36,1,0,0,0)==0,"Linux subreaper needed to adopt/drain native DataLoader descendants")

def _children():
    raw=Path("/proc/self/task/%d/children"%os.getpid()).read_text().strip()
    return [int(p) for p in raw.split()] if raw else []

def rss_identity_v1(owned,identity):
    before=owned._stat(identity['pid'])
    if before is None:return 0
    require(owned._same(before,identity),'RSS observation identity changed')
    try:pages=int(Path('/proc/%d/statm'%identity['pid']).read_text().split()[1])
    except (FileNotFoundError,ProcessLookupError):return 0
    after=owned._stat(identity['pid'])
    if after is None:return 0
    require(owned._same(after,identity),'RSS observation identity changed')
    return pages*os.sysconf('SC_PAGE_SIZE')

def worker_v1(input_ref,gate_fd,attachment_ref):
    require(os.read(gate_fd,1)==b"G","No science before owned parent launch receipt")
    os.close(gate_fd)
    s,q,cfg,storage,protocol=validate_input_v1(input_ref)
    require_work_window_v2(s)
    require(Path.cwd()==ROOT and os.getpid()==os.getpgrp()==os.getsid(0)
        and sys.dont_write_bytecode,"Actual dedicated owned worker")
    paths=worker_env_paths_v1(s)
    require(all(os.environ.get(k)==v and Path(v).resolve()==Path(v) and Path(v).is_dir()
        for k,v in paths.items()),"All temp/cache/config paths inside charged AUX")
    tempfile.tempdir=paths["TMPDIR"]
    require(tempfile.gettempdir()==paths["TMPDIR"],"No host temporary fallback")
    if s["phase"]=="train":
        require(attachment_ref["path"]==s["worker_auxiliary_scope"]+"/attachment.json","Exact actual attachment receipt")
        ticket=read(attachment_ref)
        require(ticket["job_id"]==s["job_id"] and ticket["borrowed_physical_inode"] is True
            and ticket["direct_attachment_authority_ref"]==q["direct_attachment_authority_ref"],"Actual separately admitted attachment")
        from research.reviews.H058_GPU0_full_training_successor_source_20261017_v5 import direct_gpu0_v1 as direct
        direct.install_before_torch_v1(s,q,ticket)
    else:require(attachment_ref is None and os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU evaluation no physical attachment")
    from research.reviews.H058_GPU0_full_training_successor_source_20261017_v5 import context_factory_v1
    ctx=context_factory_v1.build_context_v1(input_ref,s)
    package=cfg["producer_ref"]["path"].removesuffix(".py").replace("/",".")
    producer=importlib.import_module(package)
    require(Path(producer.__file__).resolve()==local(cfg["producer_ref"]["path"]),"Exact actual reviewed scientific package")
    producer.run_context_v1(ctx,input_ref)
    ctx["guard"]()
    return 0

def emit_capacity_failure_v3(sample):
    """Outer-tool evidence before owned stop; no pre-stop filesystem/accounting wait."""
    payload=json.dumps(sample,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()
    require(len(payload)<=4096,"Bounded failing GPU sample")
    print(json.dumps(dict(H058_capacity_failure_before_owned_stop=sample),sort_keys=True,ensure_ascii=True),flush=True)


def execute_v1(input_ref):
    stage_started=time.monotonic()
    require(Path.cwd()==ROOT and sys.dont_write_bytecode,"Exact ROOT cwd/no bytecode")
    s,q,cfg,storage,protocol=validate_input_v1(input_ref)
    require(Path(sys.executable).resolve()==(ROOT/".runtime/env/bin/python").resolve(),"Original pinned runtime interpreter")
    require_work_window_v2(s)
    expires_unix=datetime.fromisoformat(s["expires_at_utc"]).timestamp()
    work_deadline=min(stage_started+CAPS["max_seconds"]-CAPS["work_cutoff_margin_seconds"],
        time.monotonic()+expires_unix-time.time()-CAPS["work_cutoff_margin_seconds"])
    g=guard_v1()
    g.validate_storage_v9(s,dict(output_cap_bytes=CAPS["reservation_bytes"]),storage,
        source_ref=ref(DRIVER),config_ref=s["refs"]["config"],runtime_ref=s["refs"]["runtime"])
    out,aux=local(s["output_directory"]),local(s["worker_auxiliary_scope"])
    require(not out.exists() and not aux.exists(),"Fresh exclusive science output/AUX, no replay")
    paths=worker_env_paths_v1(s)
    directories=sorted({Path(v) for v in paths.values()},key=lambda p:(len(p.parts),str(p)))
    g.budget_check_v9(storage,boundary=True,required_write_bytes=(2+len(directories))*8192)
    aux.mkdir()
    for directory in directories:directory.mkdir()
    last_census=charged_census_v1(s);peak_scope=sum(last_census.values())
    load(PIDFD,"pidfd_backend_v1");owned=load(OWNED,"_H058_original_full_parent_owned_lifecycle")
    sys.modules["pidfd_backend_v1"].self_probe_v1()
    require(not owned._children(os.getpid()),"Supervisor has no unrelated children");_subreaper()
    parent_identity=owned._stat(os.getpid());logs=_RAMLog(CAPS["log_bytes"]);selector=selectors.DefaultSelector()
    proc=tree=leader_seed=None;failure=cancelled=None;peak=0;adopted=[];original={}
    pipe_nonblocking=pipe_eof=gate_released=False
    raw_status=None;cleanup_errors=[];identity_absence=None;observations=[]
    gate_read=gate_write=physical_fd=logical_fd=None;attachment_ref=None;start=stage_started
    capacity_samples=0;last_capacity=None;next_capacity=0.
    failed_capacity_sample=None;pre_stop_capacity_emitted=False;capacity_emit_error=None
    def stop(sig,frame):
        nonlocal cancelled
        cancelled=sig
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):original[sig]=signal.signal(sig,stop)
        env=os.environ.copy()
        env.pop("PYTHONHOME",None)
        env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
            PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]),
            OMP_NUM_THREADS="32" if s["phase"]=="evaluate" else "4",
            MKL_NUM_THREADS="32" if s["phase"]=="evaluate" else "4",
            OPENBLAS_NUM_THREADS="32" if s["phase"]=="evaluate" else "4",
            NUMEXPR_NUM_THREADS="32" if s["phase"]=="evaluate" else "4",
            ARGGNN_LOGICAL_JOB_NAME=s["job_id"],
            ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(POLICY),
            ARGGNN_BUDGET_V9_SOURCE_SHA256=storage["budget_components"]["implementation"]["sha256"],
            ARGGNN_BUDGET_V9_LAUNCHER_SHA256=storage["budget_components"]["launcher"]["sha256"])
        env.update(paths)
        if s["phase"]=="train":
            from research.reviews.H058_GPU0_full_training_successor_source_20261017_v5 import direct_gpu0_v1 as direct
            physical_fd,logical_fd,ticket=direct.attach_v1(s,q)
            attachment_ref=compact_v1(s,"attachment.json",ticket,g,storage)
            env.update(CUDA_VISIBLE_DEVICES=GPU0_UUID,ARGGNN_PHYSICAL_GPU="0",
                ARGGNN_GPU_LEASE=str(local(".runtime/locks/gpu_0.lock")),
                ARGGNN_GPU_LEASE_FD=str(physical_fd),H058_LOGICAL_LOCK_FD=str(logical_fd),
                ARGGNN_JOB_NAME=ticket["physical_lease_metadata"]["job"])
            last_capacity=ticket["capacity_at_attachment"];capacity_samples=1
        else:
            env["CUDA_VISIBLE_DEVICES"]=""
            for name in ("ARGGNN_PHYSICAL_GPU","ARGGNN_GPU_LEASE","ARGGNN_GPU_LEASE_FD","H058_LOGICAL_LOCK_FD","ARGGNN_JOB_NAME"):
                env.pop(name,None)
        gate_read,gate_write=os.pipe()
        command=[str(ROOT/".runtime/env/bin/python"),"-B","-m",
            "research.reviews.H058_GPU0_full_training_successor_source_20261017_v5.root_driver_v1",
            "--worker-input-ref",json.dumps(input_ref,separators=(",",":")),"--gate-fd",str(gate_read)]
        if attachment_ref is not None:command+=["--attachment-ref",json.dumps(attachment_ref,separators=(",",":"))]
        inherited=tuple(fd for fd in (gate_read,physical_fd,logical_fd) if fd is not None)
        require(time.monotonic()<work_deadline,"No child launch inside cleanup/evidence reserve")
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            bufsize=0,start_new_session=True,pass_fds=inherited)
        os.close(gate_read);gate_read=None
        # Make pipe reads nonblocking before any ownership operation can fail.
        os.set_blocking(proc.stdout.fileno(),False);pipe_nonblocking=True
        leader_seed=owned._stat(proc.pid)
        require(leader_seed is not None and leader_seed["ppid"]==os.getpid()
            and leader_seed["group"]==leader_seed["session"]==proc.pid,"Actual direct new-session identity seed")
        tree=owned.OwnedTree(proc.pid)
        compact_v1(s,"launch.json",dict(schema="H058_fixed_full_parent_actual_child_launch_v1",
            pid=proc.pid,parent_pid=os.getpid(),input_ref=input_ref,command=command,
            ownership=tree.evidence(),gate_still_closed=True,attachment_ref=attachment_ref,
            borrowed_physical_inode=attachment_ref is not None,controller_inheritance_claimed=False),g,storage)
        selector.register(proc.stdout,selectors.EVENT_READ)
        require(cancelled is None,"Cancelled before gate release")
        os.write(gate_write,b"G");gate_released=True;os.close(gate_write);gate_write=None

        while True:
            require(cancelled is None,"ROOT cancellation signal "+str(cancelled))
            require(time.monotonic()<work_deadline,"Pre-expiry scientific work cutoff; cleanup/evidence reserve retained")
            terminal=os.waitid(os.P_PID,proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            tree.refresh()
            require(len(tree.known)<=64,"Bounded fixed H058 owned process identities")
            peak=max(peak,rss_identity_v1(owned,parent_identity)+
                sum(rss_identity_v1(owned,r) for r in tree.known.values()))
            require(peak<=CAPS["max_cpu_rss_bytes"],"Combined owned RSS cap")
            last_census=charged_census_v1(s);peak_scope=max(peak_scope,sum(last_census.values()))
            if s["phase"]=="train" and time.monotonic()>=next_capacity:
                last_capacity=direct.capacity_v1(q,direct.base_v1(),worker_pid=proc.pid)
                direct.check_attachment_v1(q,ticket,physical_fd,logical_fd)
                capacity_samples+=1;next_capacity=time.monotonic()+2
            # Direct exit transfers immediately to one finite cleanup allowance.
            if terminal is not None:
                if any(pid!=proc.pid for pid in tree.live()):
                    failure=failure or "Direct worker exited with live owned descendants"
                break
            for key,_ in selector.select(timeout=.2):
                data=os.read(key.fileobj.fileno(),65536)
                if data:logs.add(data)
                else:pipe_eof=True;selector.unregister(key.fileobj)
    except BaseException as exc:
        failure=failure or type(exc).__name__+": "+str(exc)[:2000]
        if getattr(exc,"capacity_sample",None) is not None:
            failed_capacity_sample=exc.capacity_sample
            try:
                emit_capacity_failure_v3(failed_capacity_sample)
                pre_stop_capacity_emitted=True
            except BaseException as emit_exc:
                capacity_emit_error=type(emit_exc).__name__+": "+str(emit_exc)[:512]
    finally:
        # A single aggregate allowance covers direct OS wait, all known-owned
        # PIDFD TERM/KILL, descendant waits, pipe drain and absence observations.
        cleanup_started=time.monotonic()
        cleanup_deadline=cleanup_started+CAPS["cleanup_seconds"]
        for fd in (gate_read,gate_write):
            if fd is not None:
                try:os.close(fd)
                except OSError as exc:
                    if len(cleanup_errors)<16:cleanup_errors.append(type(exc).__name__+": "+str(exc)[:1000])
        def retain_pipe():
            nonlocal pipe_eof
            if proc is None or proc.stdout is None or not pipe_nonblocking:return
            for _ in range(4):
                try:data=os.read(proc.stdout.fileno(),65536)
                except BlockingIOError:break
                if not data:pipe_eof=True;break
                logs.add(data)
        def note_cleanup(exc):
            nonlocal failure
            message=type(exc).__name__+": "+str(exc)[:1000]
            failure=failure or "Owned cleanup observation failed: "+message
            if len(cleanup_errors)<16:cleanup_errors.append(message)
        if proc is not None:
            while time.monotonic()<cleanup_deadline:
                try:
                    if raw_status is None:
                        waited,status=os.waitpid(proc.pid,os.WNOHANG)
                        require(waited in (0,proc.pid),"Only actual direct child OS wait")
                        if waited==proc.pid:
                            raw_status=status;proc.returncode=os.waitstatus_to_exitcode(status)
                    if tree is not None:
                        tree.refresh();require(len(tree.known)<=64,"Bounded fixed H058 cleanup identities")
                except BaseException as exc:note_cleanup(exc)
                # Known identities remain signalable even when tree.refresh
                # failed. Never signal a guessed PID, group or unrelated job.
                rows=list(itertools.islice(tree.known.values(),64)) if tree is not None else ([leader_seed] if leader_seed else [])
                signum=signal.SIGTERM if time.monotonic()-cleanup_started<10 else signal.SIGKILL
                for row in rows[:64]:
                    if time.monotonic()>=cleanup_deadline:break
                    try:owned.signal_identity(row,signum)
                    except BaseException as exc:note_cleanup(exc)
                if raw_status is not None:
                    while time.monotonic()<cleanup_deadline:
                        try:pid,status=os.waitpid(-1,os.WNOHANG)
                        except ChildProcessError:break
                        except BaseException as exc:note_cleanup(exc);break
                        if pid==0:break
                        adopted.append(dict(pid=pid,status=status,returncode=os.waitstatus_to_exitcode(status)))
                        if len(adopted)>64:
                            failure=failure or "More than64 adopted H058 wait identities";break
                try:
                    retain_pipe()
                    if raw_status is not None and tree is None:
                        # Without the required ownership object, never infer
                        # complete drain or authorize any later stage.
                        break
                    if raw_status is not None and tree is not None and time.monotonic()<cleanup_deadline:
                        identity_absence=tree.identities_absent() and not owned._children(os.getpid())
                        if identity_absence and pipe_eof:
                            observations=[]
                            for group in sorted({r["group"] for r in list(itertools.islice(tree.known.values(),64))[:64]}):
                                gone=None
                                if time.monotonic()<cleanup_deadline:
                                    try:os.killpg(group,0)
                                    except ProcessLookupError:gone=True
                                    except OSError as exc:note_cleanup(exc)
                                    else:gone=False
                                observations.append(dict(pgid=group,absent=gone,
                                    method="targeted ESRCH within aggregate cleanup deadline" if gone is not None
                                        else "not asserted; aggregate cleanup deadline exhausted"))
                            if observations and all(r["absent"] is True for r in observations):
                                break
                except BaseException as exc:note_cleanup(exc)
                time.sleep(min(.05,max(0.,cleanup_deadline-time.monotonic())))
            if time.monotonic()<cleanup_deadline:
                try:retain_pipe()
                except BaseException as exc:note_cleanup(exc)
            if proc.stdout is not None:
                try:proc.stdout.close()
                except OSError as exc:note_cleanup(exc)
        selector.close()
        for sig,handler in original.items():signal.signal(sig,handler)
        cleanup_finished=time.monotonic()
    rc=os.waitstatus_to_exitcode(raw_status) if raw_status is not None else None
    if proc is None:failure=failure or "No worker spawned"
    if raw_status is None:failure=failure or "Direct OS wait unknown after finite cleanup"
    # Retain only observations genuinely made inside the single cleanup
    # allowance. No process/group probe is started after that deadline.
    if not observations:
        rows=list(itertools.islice(tree.known.values(),64))[:64] if tree is not None else ([leader_seed] if leader_seed else [])
        observations=[dict(pgid=group,absent=None,method="not asserted; owned cleanup incomplete")
            for group in sorted({r["group"] for r in rows})]
    drained=raw_status is not None and identity_absence is True and pipe_eof         and all(r["absent"] is True for r in observations)
    if not drained:failure=failure or "Finite cleanup ended without genuine wait and complete owned absence"
    if rc is not None and rc!=0:failure=failure or "Actual worker exit "+str(rc)
    if any(r["returncode"]!=0 for r in adopted):failure=failure or "Nonzero actual adopted descendant wait"
    # Closing our borrowed OFD never unlocks the old controller's distinct OFD.
    # If child cleanup is unresolved, its inherited logical descriptor is not
    # declared released and no storage close/credit or later work is permitted.
    for fd in (physical_fd,logical_fd):
        if fd is not None:
            try:os.close(fd)
            except OSError as exc:failure=failure or "Attachment descriptor close: "+str(exc)[:1000]
    if tree is not None and len(tree.known)<=64:
        ownership=tree.evidence()
    else:
        ownership=dict(owned_tree_constructed=tree is not None,
            actual_direct_identity_seed=leader_seed,complete_owned_roster_known=False,
            retained_owned_identities=[dict(pid=row["pid"],start=row["start"])
                for row in itertools.islice(tree.known.values(),64)] if tree is not None else [],
            owned_roster_truncated=tree is not None,signals_use_pidfd_only=True)
    observation=dict(schema="H058_fixed_full_parent_bounded_cleanup_observation_v2",input_ref=input_ref,
        pid=proc.pid if proc is not None else None,returncode=rc,actual_OS_wait_status=raw_status,
        actual_direct_worker_wait=raw_status is not None,all_owned_processes_and_groups_drained=drained,
        owned_identity_absence=identity_absence,group_observations=observations,ownership=ownership,
        adopted_waits=adopted,operational_failure=failure,gate_released=gate_released,pipe_EOF_observed=pipe_eof,
        cleanup_errors=cleanup_errors,cleanup_deadline_seconds=CAPS["cleanup_seconds"],
        cleanup_seconds=cleanup_finished-cleanup_started,elapsed_seconds=time.monotonic()-start,
        work_cutoff_margin_seconds=CAPS["work_cutoff_margin_seconds"],
        terminal_evidence_margin_seconds=CAPS["terminal_evidence_margin_seconds"],
        actual_authority_seconds_remaining=expires_unix-time.time(),
        no_future_stage_authorized=failure is not None or not drained,
        storage_close_or_credit_authorized=False,
        borrowed_physical_inode=attachment_ref is not None,controller_inheritance_claimed=False,
        failing_GPU_capacity_sample=failed_capacity_sample,
        failing_GPU_sample_emitted_to_outer_stdout_before_owned_stop=pre_stop_capacity_emitted,
        failing_GPU_sample_pre_stop_emission_error=capacity_emit_error)
    # Stdout preserves genuine/unknown OS-wait evidence even when unresolved
    # writers or expired authority prohibit another strict filesystem write.
    print(json.dumps(dict(H058_bounded_cleanup_observation=observation),sort_keys=True),flush=True)
    evidence_ready=time.time()+CAPS["terminal_evidence_margin_seconds"]<expires_unix
    if not drained or raw_status is None or not evidence_ready:
        if not evidence_ready:failure=failure or "Terminal-evidence reserve no longer available"
        partial_log=logs.record()
        print(json.dumps(dict(H058_unresolved_cleanup_log=dict(tail=partial_log["tail"][-8192:],
            total_pipe_bytes=partial_log["total_pipe_bytes"],retained_raw_bytes=partial_log["retained_raw_bytes"],
            no_filesystem_write_or_later_stage=True)),sort_keys=True),flush=True)
        return dict(status="FAILED_UNRESOLVED_CLEANUP" if not drained else "FAILED_TERMINAL_EVIDENCE_MARGIN",
            completed_ref=None,actual_wait_ref=None,returncode=rc if rc not in (None,0) else 1,
            actual_direct_worker_returncode=rc,actual_direct_worker_wait=raw_status is not None,
            all_owned_processes_and_groups_drained=drained,operational_failure=failure,
            no_future_stage_authorized=True,actual_outer_tool_terminal_still_required=True)
    minimal=completed_ref=wait_ref=None
    try:
        # Quiescent namespace and prospective authority margin established.
        # Minimal honest OS wait precedes every later scientific inspection.
        minimal=compact_v1(s,"actual_wait_minimal.json",observation,g,storage)
        g.budget_check_v9(storage,boundary=True,required_write_bytes=8*2**20)
        completed_path=out/"completed.json"
        completed_ref=ref(str(completed_path.relative_to(ROOT))) if completed_path.is_file() else None
        if completed_ref is None:failure=failure or "No scientific completion"
        log_ref=compact_v1(s,"worker_log.json",logs.record(),g,storage,cap=16*2**20)
        final=charged_census_v1(s);peak_scope=max(peak_scope,sum(final.values()))
        wait_ref=compact_v1(s,"actual_wait.json",dict(observation,
            schema="H058_fixed_full_parent_actual_wait_v2",completed_ref=completed_ref,
            operational_failure=failure,no_future_stage_authorized=failure is not None,
            actual_wait_minimal_ref=minimal,log_ref=log_ref,peak_combined_owned_RSS_bytes=peak,
            sampled_peak_scope_bytes=peak_scope,final_scope_census=final,
            capacity_samples=capacity_samples,last_capacity_sample=last_capacity,
            outer_tool_terminal_consumed=False,automatic_close_or_credit=False),g,storage)
        g.budget_check_v9(storage,boundary=True)
    except BaseException as exc:
        failure=failure or "Strict terminal evidence failed: "+type(exc).__name__+": "+str(exc)[:2000]
        print(json.dumps(dict(H058_terminal_evidence_failed=observation,
            failure=failure,actual_wait_minimal_ref=minimal,no_future_stage_authorized=True),sort_keys=True),flush=True)
    return dict(status="PASS" if failure is None else "FAILED",completed_ref=completed_ref,
        actual_wait_ref=wait_ref,actual_wait_minimal_ref=minimal,
        returncode=0 if failure is None else (rc if rc not in (None,0) else 1),
        actual_direct_worker_returncode=rc,all_owned_processes_and_groups_drained=drained,
        no_future_stage_authorized=failure is not None,actual_outer_tool_terminal_still_required=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument("--admit",action="store_true");actions.add_argument("--execute",action="store_true")
    actions.add_argument("--worker-input-ref")
    parser.add_argument("--request-ref");parser.add_argument("--input-ref")
    parser.add_argument("--gate-fd",type=int);parser.add_argument("--attachment-ref")
    args=parser.parse_args(argv)
    if not(args.admit or args.execute or args.worker_input_ref):
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",execution=False,admission=False)));return 0
    require(sys.platform.startswith("linux") and sys.dont_write_bytecode
        and not any(n in sys.modules for n in ("torch","torch_geometric")),"Fresh Linux stdlib ROOT entry")
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    if args.admit:
        require(args.request_ref is not None,"Exact actual ROOT request")
        result=admit_v1(json.loads(args.request_ref));print(json.dumps(result,sort_keys=True),flush=True);return 0
    if args.worker_input_ref:
        require(args.gate_fd is not None,"Owned launch gate required")
        return worker_v1(json.loads(args.worker_input_ref),args.gate_fd,
            json.loads(args.attachment_ref) if args.attachment_ref else None)
    require(args.input_ref is not None,"Exact actual admitted input")
    result=execute_v1(json.loads(args.input_ref));print(json.dumps(result,sort_keys=True),flush=True)
    return result["returncode"]
if __name__=="__main__":raise SystemExit(main())
