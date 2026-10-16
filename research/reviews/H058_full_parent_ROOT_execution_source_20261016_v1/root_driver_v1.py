"""Fixed H058 ROOT admission and original owned supervisor. Default inert."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,collections,ctypes,fcntl,hashlib,importlib,io,json,os,resource,selectors,signal,stat,subprocess,sys,tempfile,time
from research.reviews.H058_full_parent_ROOT_execution_source_20261016_v1.common_v1 import *
SOURCE=DRIVER

def admit_v1(request_ref):
    require(Path.cwd()==ROOT and sys.dont_write_bytecode,"Exact ROOT cwd/no bytecode")
    q,cfg,producer,protocol,manifest=validate_request_v1(request_ref)
    packet=local(q["packet_directory"]);output=local(q["output_directory"])
    require(not packet.exists() and not output.exists(),"Fresh exclusive full-parent admission/output; no replay")
    g=guard_v1(consumer=False);packet.mkdir()
    runtime=save(packet,"runtime.json",dict(schema="H058_fixed_ROOT_runtime_v1",root_authorizer="root_agent",
        job_id=q["job_id"],phase=q["phase"],device="cuda" if q["phase"]=="train" else "cpu",
        physical_gpu=0 if q["phase"]=="train" else None,output_directory=q["output_directory"],
        executable=str(ROOT/".runtime/env/bin/python"),process_title="AI Auto Research Pipeline Test",
        runtime_equivalence_ref=cfg["runtime_equivalence_ref"]))
    original=read(protocol.H050_INPUT_REF)
    closure=save(packet,"consumer_manifest.json",dict(schema="H058_fixed_exact_transitive_closure_v1",
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
    with g.budget.v4.mutex(ROOT):state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
    needed=[ref(MANIFEST),ref(DRIVER),ref(CONTEXT),ref(DIRECT),request_ref,q["config_ref"],
        q["science_handoff_ref"],cfg["source_manifest_ref"],producer,runtime,closure,BASEFIT]
    needed+=[GUARD,POLICY,REPAIR,RENEWAL,CERT,MEMO,NAMESPACE_MEMO,V4,PIDFD,OWNED,BASE_POOL,GPU0_GUARD]
    if q["phase"]=="train":
        from research.reviews.H058_full_parent_ROOT_execution_source_20261016_v1 import direct_gpu0_v1 as direct
        needed+=[q["direct_attachment_authority_ref"],direct.CURRENT_CAP]
    consumers=unique([*state[3].values(),*needed])
    auxiliary=q["packet_directory"]+"/worker_aux"
    reserve=dict(key="H058/"+q["job_id"],output=str(output),auxiliary_scopes=[str(local(auxiliary))],
        reserved_bytes=CAPS["reservation_bytes"],consumer_bindings=consumers)
    authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=q["expires_at_utc"])
    authority_ref=save(packet,"reserve_authority.json",authority)
    receipt=g.apply_reserve_v1(authority_ref,expected_source_sha256=implementation["sha256"],
        expected_launcher_sha256=launcher["sha256"])
    receipt_ref=save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=g.MODE,job_id=q["job_id"],output_directory=q["output_directory"],
        reserved_new_bytes=CAPS["reservation_bytes"],reservation=receipt["value"],
        initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=q["expires_at_utc"],
        completed_overlay_policy_ref=POLICY,budget_components=dict(implementation=implementation,
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
    return dict(input_ref=input_ref,admitted_ref=admitted_ref)

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
    if not owned._same(owned._stat(identity['pid']),identity):return 0
    try:pages=int(Path('/proc/%d/statm'%identity['pid']).read_text().split()[1])
    except (FileNotFoundError,ProcessLookupError):return 0
    require(owned._same(owned._stat(identity['pid']),identity),'RSS observation identity changed')
    return pages*os.sysconf('SC_PAGE_SIZE')

def worker_v1(input_ref,gate_fd,attachment_ref):
    require(os.read(gate_fd,1)==b"G","No science before owned parent launch receipt")
    os.close(gate_fd)
    s,q,cfg,storage,protocol=validate_input_v1(input_ref)
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
        from research.reviews.H058_full_parent_ROOT_execution_source_20261016_v1 import direct_gpu0_v1 as direct
        direct.install_before_torch_v1(s,q,ticket)
    else:require(attachment_ref is None and os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU evaluation no physical attachment")
    from research.reviews.H058_full_parent_ROOT_execution_source_20261016_v1 import context_factory_v1
    ctx=context_factory_v1.build_context_v1(input_ref,s)
    package=cfg["producer_ref"]["path"].removesuffix(".py").replace("/",".")
    producer=importlib.import_module(package)
    require(Path(producer.__file__).resolve()==local(cfg["producer_ref"]["path"]),"Exact actual reviewed scientific package")
    producer.run_context_v1(ctx,input_ref)
    ctx["guard"]()
    return 0

def execute_v1(input_ref):
    require(Path.cwd()==ROOT and sys.dont_write_bytecode,"Exact ROOT cwd/no bytecode")
    s,q,cfg,storage,protocol=validate_input_v1(input_ref)
    require(Path(sys.executable).resolve()==(ROOT/".runtime/env/bin/python").resolve(),"Original pinned runtime interpreter")
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
    proc=tree=None;failure=cancelled=None;peak=0;stop_at=None;adopted=[];original={}
    gate_read=gate_write=physical_fd=logical_fd=None;attachment_ref=None;start=time.monotonic()
    capacity_samples=0;last_capacity=None;next_capacity=0.
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
            from research.reviews.H058_full_parent_ROOT_execution_source_20261016_v1 import direct_gpu0_v1 as direct
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
            "research.reviews.H058_full_parent_ROOT_execution_source_20261016_v1.root_driver_v1",
            "--worker-input-ref",json.dumps(input_ref,separators=(",",":")),"--gate-fd",str(gate_read)]
        if attachment_ref is not None:command+=["--attachment-ref",json.dumps(attachment_ref,separators=(",",":"))]
        inherited=tuple(fd for fd in (gate_read,physical_fd,logical_fd) if fd is not None)
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            bufsize=0,start_new_session=True,pass_fds=inherited)
        os.close(gate_read);gate_read=None;tree=owned.OwnedTree(proc.pid)
        compact_v1(s,"launch.json",dict(schema="H058_fixed_full_parent_actual_child_launch_v1",
            pid=proc.pid,parent_pid=os.getpid(),input_ref=input_ref,command=command,
            ownership=tree.evidence(),gate_still_closed=True,attachment_ref=attachment_ref,
            borrowed_physical_inode=attachment_ref is not None,controller_inheritance_claimed=False),g,storage)
        os.set_blocking(proc.stdout.fileno(),False);selector.register(proc.stdout,selectors.EVENT_READ)
        require(cancelled is None,"Cancelled before gate release")
        os.write(gate_write,b"G");os.close(gate_write);gate_write=None
        while proc.poll() is None or selector.get_map() or tree.live() or owned._children(os.getpid()):
            tree.refresh()
            peak=max(peak,rss_identity_v1(owned,parent_identity)+sum(rss_identity_v1(owned,r) for r in tree.known.values()))
            try:
                last_census=charged_census_v1(s);peak_scope=max(peak_scope,sum(last_census.values()))
                if s["phase"]=="train" and time.monotonic()>=next_capacity:
                    # Only the fixed H058 worker may create a CUDA context.
                    last_capacity=direct.capacity_v1(q,direct.base_v1(),worker_pid=proc.pid)
                    direct.check_attachment_v1(q,ticket,physical_fd,logical_fd)
                    capacity_samples+=1;next_capacity=time.monotonic()+2
            except Exception as exc:failure=failure or "Fixed scope/lease/capacity: "+type(exc).__name__+": "+str(exc)[:1000]
            if peak>CAPS["max_cpu_rss_bytes"]:failure=failure or "Combined owned RSS cap exceeded"
            if time.monotonic()-start>CAPS["max_seconds"]:failure=failure or "ROOT24h deadline exceeded"
            if datetime.now(timezone.utc)>=datetime.fromisoformat(s["expires_at_utc"]):failure=failure or "Actual ROOT authority expired"
            if cancelled is not None:failure=failure or "ROOT cancellation signal "+str(cancelled)
            if proc.poll() is not None and tree.live():failure=failure or "Direct worker exited with live descendants"
            if failure:
                if stop_at is None:stop_at=time.monotonic()
                tree.signal_all(signal.SIGKILL if time.monotonic()-stop_at>=20 else signal.SIGTERM)
            for key,_ in selector.select(timeout=.2):
                data=os.read(key.fileobj.fileno(),65536)
                if data:logs.add(data)
                else:selector.unregister(key.fileobj)
            if proc.poll() is not None:
                proc.wait()
                while True:
                    try:pid,status=os.waitpid(-1,os.WNOHANG)
                    except ChildProcessError:break
                    if pid==0:break
                    adopted.append(dict(pid=pid,status=status))
    except BaseException as exc:failure=failure or type(exc).__name__+": "+str(exc)[:2000]
    finally:
        for fd in (gate_read,gate_write):
            if fd is not None:os.close(fd)
        if proc is not None:
            if tree is None:rc=proc.wait()
            else:
                while proc.poll() is None or tree.live() or owned._children(os.getpid()):
                    tree.signal_all(signal.SIGKILL)
                    if proc.poll() is not None:
                        proc.wait()
                        while True:
                            try:pid,status=os.waitpid(-1,os.WNOHANG)
                            except ChildProcessError:break
                            if pid==0:break
                            adopted.append(dict(pid=pid,status=status))
                    for key,_ in selector.select(timeout=.1):
                        data=os.read(key.fileobj.fileno(),65536)
                        if data:logs.add(data)
                        else:selector.unregister(key.fileobj)
                rc=proc.wait()
            if proc.stdout is not None:
                while True:
                    try:data=os.read(proc.stdout.fileno(),65536)
                    except BlockingIOError:break
                    if not data:break
                    logs.add(data)
                proc.stdout.close()
        else:rc=None
        selector.close()
        for sig,handler in original.items():signal.signal(sig,handler)
    # The H058 logical lock is retained through the complete actual drain.
    # The borrowed descriptor is never unlocked: close cannot release the old
    # controller's distinct original exclusive open-file-description lock.
    if proc is None or tree is None:
        for fd in (physical_fd,logical_fd):
            if fd is not None:os.close(fd)
        raise RuntimeError("No fabricated launch/wait/ownership evidence; attachment closed without worker success")

    absent=tree.identities_absent() and not owned._children(os.getpid())
    observations=[]
    for group in sorted({r["group"] for r in tree.known.values()}):
        try:os.killpg(group,0)
        except ProcessLookupError:gone=True
        else:gone=False
        observations.append(dict(pgid=group,absent=gone,method="targeted os.killpg(pgid,0) ESRCH"))
    drained=absent and all(r["absent"] for r in observations)
    for fd in (physical_fd,logical_fd):
        if fd is not None:os.close(fd)
    require(absent,"No science output inspection before owned identities absent")
    if not drained:failure=failure or "Owned group absence not established"
    if rc!=0:failure=failure or "Actual worker exit "+str(rc)
    if any(not os.WIFEXITED(r["status"]) or os.WEXITSTATUS(r["status"])!=0 for r in adopted):
        failure=failure or "Nonzero adopted descendant wait"
    minimal=compact_v1(s,"actual_wait_minimal.json",dict(schema="H058_fixed_full_parent_actual_direct_wait_minimal_v1",
        pid=proc.pid,returncode=rc,actual_direct_worker_wait=True,input_ref=input_ref,
        all_owned_processes_and_groups_drained=drained,group_observations=observations,
        operational_failure=failure,success_not_yet_asserted=True),g,storage)
    g.budget_check_v9(storage,boundary=True,required_write_bytes=8*2**20)
    completed_path=out/"completed.json"
    completed_ref=ref(str(completed_path.relative_to(ROOT))) if completed_path.is_file() else None
    if completed_ref is None:failure=failure or "No scientific completion"
    log_ref=compact_v1(s,"worker_log.json",logs.record(),g,storage,cap=16*2**20)
    final=charged_census_v1(s);peak_scope=max(peak_scope,sum(final.values()))
    wait_ref=compact_v1(s,"actual_wait.json",dict(schema="H058_fixed_full_parent_actual_wait_v1",
        input_ref=input_ref,pid=proc.pid,returncode=rc,actual_direct_worker_wait=True,
        completed_ref=completed_ref,all_owned_processes_and_groups_drained=drained,
        group_observations=observations,ownership=tree.evidence(),adopted_waits=adopted,
        operational_failure=failure,actual_wait_minimal_ref=minimal,log_ref=log_ref,
        peak_combined_owned_RSS_bytes=peak,elapsed_seconds=time.monotonic()-start,
        sampled_peak_scope_bytes=peak_scope,final_scope_census=final,
        capacity_samples=capacity_samples,last_capacity_sample=last_capacity,
        borrowed_physical_inode=attachment_ref is not None,controller_inheritance_claimed=False,
        outer_tool_terminal_consumed=False,automatic_close_or_credit=False),g,storage)
    g.budget_check_v9(storage,boundary=True)
    return dict(status="PASS" if failure is None else "FAILED",completed_ref=completed_ref,
        actual_wait_ref=wait_ref,returncode=0 if failure is None else (rc or 1),
        actual_outer_tool_terminal_still_required=True)

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
